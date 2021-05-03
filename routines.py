import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from model import get_model_from_name
from data import get_dataloader
import sys
PATH_TO_CIFAR = "./cifar/"
sys.path.append(PATH_TO_CIFAR)
import train as cifar_train
import copy
from advertorch.attacks import LinfPGDAttack
from advertorch.context import ctx_noparamgrad_and_eval

# Compute fisher matrix for FedCurvAT; on adversarially perturbed local data
def compute_fisher_matrix(args, network, optimizer, cifar_criterion, train_loader, adversary=None):
    network.eval()

    ut_local = {}
    vt_local = {}
    params = {n: p for n, p in network.named_parameters() if p.requires_grad}
    for n in params:
        ut_local[n] = torch.zeros(params[n].shape, requires_grad=False).cuda(args.gpu_id)
        vt_local[n] = torch.zeros(params[n].shape, requires_grad=False).cuda(args.gpu_id)

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.gpu_id!=-1:
            data = data.cuda(args.gpu_id)
            target = target.cuda(args.gpu_id)
        optimizer.zero_grad()  
        if args.adversarial_training != 0:
            with ctx_noparamgrad_and_eval(network):
                data = adversary.perturb(data, target)

        output = network(data)
        loss = cifar_criterion(output, target)     
        loss.backward()
                    
        for n in params:
            ut_local[n].data += params[n].grad.data ** 2 / len(train_loader)

    for n in params:
        vt_local[n].data = ut_local[n].data*params[n].data

    for n in params:
        ut_local[n] = ut_local[n].cpu()
        vt_local[n] = vt_local[n].cpu()

    return ut_local, vt_local

def penalty(network, ut_local, ut_global, vt_local, vt_global, lb, weight_coefficient):
    params = {n: p for n, p in network.named_parameters() if p.requires_grad}
    l2 = 0
    l3 = 0
    for n in params:
        l2 += (lb*params[n]*params[n]*(ut_global[n] - weight_coefficient*ut_local[n])).sum()
        l3 -= (2*lb*params[n]*(vt_global[n] - weight_coefficient*vt_local[n])).sum()
    #print(l2, l3)
    return l2 + l3

def get_trained_model(args, id, random_seed, train_loader, test_loader, ut_local=None, ut_global=None, vt_local=None, vt_global=None, lb=0.0, network=None, comm_round=0):
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    if args.gpu_id!=-1:
        device = 'cuda'
    else:
        device = 'cpu'
    if network is None:
        network = get_model_from_name(args, idx=id)
        if args.model_name == 'vgg9':
            checkpoint=torch.load('initialization.pth', map_location=torch.device(device))
        elif args.model_name == 'nin':
            checkpoint = torch.load('initialization_nin.pth', map_location=torch.device(device))
        elif args.model_name == 'vgg16':
            checkpoint = torch.load('initialization_vgg11_batchnorm.pth', map_location=torch.device(device))

        network.load_state_dict(checkpoint)
        print("SAME INITIALIZATION")

    params = {n: p for n, p in network.named_parameters() if p.requires_grad}
    
    if ut_local is None:
        ut_global = {}
        vt_global = {}
        ut_local = {}
        vt_local = {}
        for n in params:
            ut_global[n] = torch.zeros(params[n].shape, requires_grad=False)
            vt_global[n] = torch.zeros(params[n].shape, requires_grad=False)            
            ut_local[n] = torch.zeros(params[n].shape, requires_grad=False)
            vt_local[n] = torch.zeros(params[n].shape, requires_grad=False)

    optimizer = optim.SGD(network.parameters(), lr=args.learning_rate,
                          momentum=args.momentum, weight_decay=1e-4)
    print("LEARNING RATE : ", args.learning_rate)
    print("MOMENTUM : ", args.momentum)
    print("LAMBDA : ", lb) 
    print("AVDERSARIAL TRAINING : ", args.adversarial_training)
    adversary = LinfPGDAttack(
        network, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8.0 / 255.0,
        nb_iter=10, eps_iter=2.0 / 255.0, rand_init=True, clip_min=0.0,
        clip_max=1.0, targeted=False)

    cifar_criterion = torch.nn.CrossEntropyLoss()
    if args.gpu_id!=-1:
        network = network.cuda(args.gpu_id)
    log_dict = {}
    log_dict['train_losses'] = []
    log_dict['train_counter'] = []
    log_dict['test_losses'] = []
    # log_dict['test_counter'] = [i * len(test_loader.dataset) for i in range(args.n_epochs + 1)]
    # print(list(network.parameters()))
    acc = test(args, network, test_loader, log_dict)

    for n in ut_local:
        ut_local[n] = ut_local[n].cuda(args.gpu_id)
        vt_local[n] = vt_local[n].cuda(args.gpu_id)
        ut_global[n] = ut_global[n].cuda(args.gpu_id)
        vt_global[n] = vt_global[n].cuda(args.gpu_id)

    for epoch in range(1, args.n_epochs + 1):
        print("Epoch %d"%epoch, flush=True)
        train(args, network, optimizer, cifar_criterion, train_loader, 
            ut_local, ut_global, vt_local, vt_global, lb,
            log_dict, epoch, model_id=str(id), adversary = adversary)
        #if epoch%5==0:
        #    acc = test(args, network, test_loader, log_dict)
    adv_acc = 0.0
    if args.adversarial_training != 0:
        adv_acc = test_adv(args, network, test_loader, log_dict, adversary=adversary) 
    torch.save(network.state_dict(), '{}/model_{}_{}_{}.pth'.format(args.save_dir, args.model_name, str(id), comm_round))
    #torch.save(optimizer.state_dict(), '{}/optimizer_{}_{}_{}.pth'.format(args.save_dir, args.model_name, str(id), epoch))

    for n in ut_local:
        ut_local[n] = ut_local[n].cpu()
        vt_local[n] = vt_local[n].cpu()
        ut_global[n] = ut_global[n].cpu()
        vt_global[n] = vt_global[n].cpu()

    ut_local_new, vt_local_new = compute_fisher_matrix(args, network, optimizer, cifar_criterion, train_loader, adversary=adversary)
    return network, acc, adv_acc, (ut_local_new, vt_local_new)

def check_freezed_params(model, frozen):
    flag = True
    for idx, param in enumerate(model.parameters()):
        if idx >= len(frozen):
            return flag

        flag = flag and (param.data == frozen[idx].data).all()

    return flag

def get_intmd_retrain_model(args, random_seed, network, aligned_wts, train_loader, test_loader):
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    num_params_aligned = len(aligned_wts)
    for idx, param in enumerate(network.parameters()):
        if idx < num_params_aligned:
            param.requires_grad = False

    print("number of layers that are intmd retrained ", len(list(network.parameters()))-num_params_aligned)
    optimizer = optim.SGD(network.parameters(), lr=args.learning_rate * args.intmd_retrain_lrdec,
                          momentum=args.momentum)
    log_dict = {}
    log_dict['train_losses'] = []
    log_dict['train_counter'] = []
    log_dict['test_losses'] = []
    # log_dict['test_counter'] = [i * len(test_loader.dataset) for i in range(args.n_epochs + 1)]
    # print(list(network.parameters()))
    acc = test(args, network, test_loader, log_dict)
    for epoch in range(1, args.intmd_retrain_epochs + 1):
        train(args, network, optimizer, train_loader, log_dict, epoch, model_id=str(id))
        acc = test(args, network, test_loader, log_dict)

    print("Finally accuracy of model {} after intermediate retraining for {} epochs with lr decay {} is {}".format(
        random_seed, args.intmd_retrain_epochs, args.intmd_retrain_lrdec, acc
    ))

    assert check_freezed_params(network, aligned_wts)
    return network

def get_trained_data_separated_model(args, id, local_train_loader, local_test_loader, test_loader, base_net=None):
    torch.backends.cudnn.enabled = False
    if base_net is not None:
        network = copy.deepcopy(base_net)
    else:
        network = get_model_from_name(args, idx=id)
    optimizer = optim.SGD(network.parameters(), lr=args.learning_rate,
                          momentum=args.momentum)
    if args.gpu_id!=-1:
        network = network.cuda(args.gpu_id)
    log_dict = {}
    log_dict['train_losses'] = []
    log_dict['train_counter'] = []
    log_dict['local_test_losses'] = []
    log_dict['test_losses'] = []
    # log_dict['test_counter'] = [i * len(test_loader.dataset) for i in range(args.n_epochs + 1)]
    # print(list(network.parameters()))
    acc = test(args, network, test_loader, log_dict)
    local_acc = test(args, network, local_test_loader, log_dict, is_local=True)
    for epoch in range(1, args.n_epochs + 1):
        train(args, network, optimizer, local_train_loader, log_dict, epoch, model_id=str(id))
        acc = test(args, network, test_loader, log_dict)
        local_acc = test(args, network, local_test_loader, log_dict, is_local=True)
    return network, acc, local_acc

def get_retrained_model(args, train_loader, test_loader, old_network, tensorboard_obj=None, nick='', start_acc=-1, retrain_seed=-1):
    torch.backends.cudnn.enabled = False
    if args.retrain_lr_decay > 0:
        args.retrain_lr = args.learning_rate / args.retrain_lr_decay
        print('optimizer_learning_rate is ', args.retrain_lr)
    if retrain_seed!=-1:
        torch.manual_seed(retrain_seed)
        
    optimizer = optim.SGD(old_network.parameters(), lr=args.retrain_lr,
                              momentum=args.momentum)
    log_dict = {}
    log_dict['train_losses'] = []
    log_dict['train_counter'] = []
    log_dict['test_losses'] = []
    # log_dict['test_counter'] = [i * len(train_loader.dataset) for i in range(args.n_epochs + 1)]

    acc = test(args, old_network, test_loader, log_dict)
    print("check accuracy once again before retraining starts: ", acc)

    if tensorboard_obj is not None and start_acc != -1:
        tensorboard_obj.add_scalars('test_accuracy_percent/', {nick: start_acc},
                                    global_step=0)
        assert start_acc == acc


    best_acc = -1
    for epoch in range(1, args.retrain + 1):
        train(args, old_network, optimizer, train_loader, log_dict, epoch)
        acc, loss = test(args, old_network, test_loader, log_dict, return_loss=True)

        if tensorboard_obj is not None:
            assert nick != ''
            tensorboard_obj.add_scalars('test_loss/', {nick: loss}, global_step=epoch)
            tensorboard_obj.add_scalars('test_accuracy_percent/', {nick: acc}, global_step=epoch)

        print("At retrain epoch the accuracy is : ", acc)
        best_acc = max(best_acc, acc)

    return old_network, best_acc

def get_pretrained_model(args, path, data_separated=False, idx=-1):
    model = get_model_from_name(args, idx=idx)

    if args.gpu_id != -1:
        state = torch.load(
            path,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, 'cuda:' + str(args.gpu_id))
            ),
        )
    else:
        state = torch.load(
            path,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
            ),
        )

    model_state_dict = state #['model_state_dict']
   
    #if 'test_accuracy' not in state:
    #   state['test_accuracy'] = -1

    #if 'epoch' not in state:
    #    state['epoch'] = -1

    #if not data_separated:
    #    print("Loading model at path {} which had accuracy {} and at epoch {}".format(path, state['test_accuracy'],
    #                                                                                  state['epoch']))
    #else:
    #    print("Loading model at path {} which had local accuracy {} and overall accuracy {} for choice {} at epoch {}".format(path,
    #        state['local_test_accuracy'], state['test_accuracy'], state['choice'], state['epoch']))

    model.load_state_dict(model_state_dict)

    if args.gpu_id != -1:
        model = model.cuda(args.gpu_id)
    return model

    #if not data_separated:
    #    return model, state['test_accuracy']
    #else:
    #    return model, state['test_accuracy'], state['local_test_accuracy']

def train(args, network, optimizer, cifar_criterion, train_loader, ut_local, ut_global, vt_local, vt_global, lb, log_dict, epoch, model_id=-1, adversary=None):
    weight_coefficient = 1/args.num_models
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #print(batch_idx)
        if args.gpu_id!=-1:
            data = data.cuda(args.gpu_id)
            target = target.cuda(args.gpu_id)            

        optimizer.zero_grad()

        if args.adversarial_training != 0:
            with ctx_noparamgrad_and_eval(network):
                data = adversary.perturb(data, target)

        output = network(data)
        loss = cifar_criterion(output, target) + penalty(network, ut_local, ut_global, vt_local, vt_global, lb, weight_coefficient)             
        loss.backward()
        optimizer.step()
    batch_idx = len(train_loader)
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
           epoch, len(train_loader.dataset), len(train_loader.dataset),
           100. * batch_idx / len(train_loader), loss.item()))
    log_dict['train_losses'].append(loss.item())
    log_dict['train_counter'].append(
           (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

    assert args.exp_name == "exp_" + args.timestamp

def test_adv(args, network, test_loader, log_dict, debug=False, return_loss=False, is_local=False, adversary=None):
    network.eval()
    test_loss = 0
    correct = 0
    if is_local:
        print("\n--------- Testing in local mode ---------")
    else:
        print("\n--------- Testing in global mode ---------")

    if args.dataset.lower() == 'cifar10':
        cifar_criterion = torch.nn.CrossEntropyLoss()

    if adversary is None:
        adversary = LinfPGDAttack(
        network, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8.0 / 255.0,
        nb_iter=10, eps_iter=2.0 / 255.0, rand_init=True, clip_min=0.0,
        clip_max=1.0, targeted=False)

    #   with torch.no_grad():
    for data, target in test_loader:
        if args.gpu_id!=-1:
            data = data.cuda(args.gpu_id)
            target = target.cuda(args.gpu_id)

        with ctx_noparamgrad_and_eval(network):
            data = adversary.perturb(data, target)

        output = network(data)
        if debug:
            print("output is ", output)

        if args.dataset.lower() == 'cifar10':
            # mnist models return log_softmax outputs, while cifar ones return raw values!
            test_loss += cifar_criterion(output, target).item()
        elif args.dataset.lower() == 'mnist':
            test_loss += F.nll_loss(output, target, size_average=False).item()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

    print("size of test_loader dataset: ", len(test_loader.dataset))
    test_loss /= len(test_loader.dataset)
    print('\nAdv Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    ans = (float(correct) * 100.0) / len(test_loader.dataset)

    if not return_loss:
        return ans
    else:
        return ans, test_loss

def test(args, network, test_loader, log_dict, debug=False, return_loss=False, is_local=False):
    network.eval()
    test_loss = 0
    correct = 0
    if is_local:
        print("\n--------- Testing in local mode ---------")
    else:
        print("\n--------- Testing in global mode ---------")

    if args.dataset.lower() == 'cifar10':
        cifar_criterion = torch.nn.CrossEntropyLoss()

    #   with torch.no_grad():
    for data, target in test_loader:
        if args.gpu_id!=-1:
            data = data.cuda(args.gpu_id)
            target = target.cuda(args.gpu_id)

        output = network(data)
        if debug:
            print("output is ", output)

        if args.dataset.lower() == 'cifar10':
            # mnist models return log_softmax outputs, while cifar ones return raw values!
            test_loss += cifar_criterion(output, target).item()
        elif args.dataset.lower() == 'mnist':
            test_loss += F.nll_loss(output, target, size_average=False).item()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

    print("size of test_loader dataset: ", len(test_loader.dataset))
    test_loss /= len(test_loader.dataset)
    if is_local:
        string_info = 'local_test'
    else:
        string_info = 'test'
    log_dict['{}_losses'.format(string_info)].append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    ans = (float(correct) * 100.0) / len(test_loader.dataset)

    if not return_loss:
        return ans
    else:
        return ans, test_loss

def train_data_separated_models(args, local_train_loaders, local_test_loaders, test_loader, choices):
    networks = []
    local_accuracies = []
    accuracies = []
    base_nets = []
    base_net = get_model_from_name(args, idx=0)
    base_nets.append(base_net)
    if args.diff_init or args.width_ratio!=1:
        base_nets.append(get_model_from_name(args, idx=1))
    else:
        base_nets.append(base_net)

    for i in range(args.num_models):
        print("\nTraining model {} on its separate data \n ".format(str(i)))
        network, acc, local_acc = get_trained_data_separated_model(args, i,
                                           local_train_loaders[i], local_test_loaders[i], test_loader, base_nets[i])
        networks.append(network)
        accuracies.append(acc)
        local_accuracies.append(local_acc)
        if args.dump_final_models:
            save_final_data_separated_model(args, i, network, local_acc, acc, choices[i])
    return networks, accuracies, local_accuracies


def train_models(args, train_loader_array, test_loader, ut_local_array=None, vt_local_array=None, ut_global=None, vt_global=None, lb=0.0, initial_model=None, checkpoint_models=None, comm_round=0):
    networks = []
    accuracies = []
    adv_accuracies = []
    ut_local_new_array = []
    vt_local_new_array = []
    for i in range(args.num_models):
        if checkpoint_models is not None:
            print("CHECKPOINT HERE")
            network = checkpoint_models[i]
            network, acc, adv_acc, (ut_local_new, vt_local_new) = get_trained_model(args, i, i, train_loader_array[i], test_loader,
                ut_local=ut_local_array[i], vt_local=vt_local_array[i], ut_global=ut_global, vt_global=vt_global, lb=lb,
                network=network, comm_round=comm_round)
        elif initial_model is not None:
            network = copy.deepcopy(initial_model)
            network, acc, adv_acc, (ut_local_new, vt_local_new) = get_trained_model(args, i, i, train_loader_array[i], test_loader, 
                ut_local=ut_local_array[i], vt_local=vt_local_array[i], ut_global=ut_global, vt_global=vt_global, lb=lb,
                network=network, comm_round=comm_round)
        else:
            # ut arrays are also None
            network, acc, adv_acc, (ut_local_new, vt_local_new) = get_trained_model(args, i, i, train_loader_array[i], test_loader, comm_round=comm_round)
        networks.append(network)
        accuracies.append(acc)
        adv_accuracies.append(adv_acc)
        ut_local_new_array.append(ut_local_new)
        vt_local_new_array.append(vt_local_new)
        if args.dump_final_models:
            save_final_model(args, i, network, acc)
    return networks, accuracies, adv_accuracies, (ut_local_new_array, vt_local_new_array)

def save_final_data_separated_model(args, idx, model, local_test_accuracy, test_accuracy, choice):
    path = os.path.join(args.result_dir, args.exp_name, 'model_{}'.format(idx))
    os.makedirs(path, exist_ok=True)
    import time
    args.ckpt_type = 'final'
    time.sleep(1)  # workaround for RuntimeError('Unknown Error -1') https://github.com/pytorch/pytorch/issues/10577
    torch.save({
        'args': vars(args),
        'epoch': args.n_epochs,
        'local_test_accuracy': local_test_accuracy,
        'test_accuracy': test_accuracy,
        'choice': str(choice),
        'model_state_dict': model.state_dict(),
    }, os.path.join(path, '{}.checkpoint'.format(args.ckpt_type))
    )


def save_final_model(args, idx, model, test_accuracy):
    path = os.path.join(args.result_dir, args.exp_name, 'model_{}'.format(idx))
    os.makedirs(path, exist_ok=True)
    import time
    args.ckpt_type = 'final'
    time.sleep(1)  # workaround for RuntimeError('Unknown Error -1') https://github.com/pytorch/pytorch/issues/10577
    torch.save({
        'args': vars(args),
        'epoch': args.n_epochs,
        'test_accuracy': test_accuracy,
        'model_state_dict': model.state_dict(),
    }, os.path.join(path, '{}.checkpoint'.format(args.ckpt_type))
    )

def retrain_models(args, old_networks, train_loader_array, test_loader, config, tensorboard_obj=None, initial_acc=None, nicks=None):
    accuracies = []
    retrained_networks = []
    # nicks = []

    # assert len(old_networks) >= 4

    for i in range(len(old_networks)):
        nick = nicks[i]
        # if i == len(old_networks) - 1:
        #     nick = 'naive_averaging'
        # elif i == len(old_networks) - 2:
        #     nick = 'geometric'
        # else:
        #     nick = 'model_' + str(i)
        # nicks.append(nick)
        print("Retraining model : ", nick)

        if initial_acc is not None:
            start_acc = initial_acc[i]
        else:
            start_acc = -1
        if args.dataset.lower()[0:7] == 'cifar10':

            # if args.reinit_trainloaders:
            #     print('reiniting trainloader')
            #     retrain_loader, _ = cifar_train.get_dataset(config, no_randomness=args.no_random_trainloaders)
            # else:
            retrain_loader = train_loader_array[i]

            output_root_dir = "{}/{}_models_ensembled/".format(args.baseroot, (args.dataset).lower())
            output_root_dir = os.path.join(output_root_dir, args.exp_name, nick)
            os.makedirs(output_root_dir, exist_ok=True)

            retrained_network, acc = cifar_train.get_retrained_model(args, retrain_loader, test_loader, old_networks[i], config, output_root_dir, tensorboard_obj=tensorboard_obj, nick=nick, start_acc=initial_acc[i])
            
        elif args.dataset.lower() == 'mnist':

            # if args.reinit_trainloaders:
            #     print('reiniting trainloader')
            #     retrain_loader, _ = get_dataloader(args, no_randomness=args.no_random_trainloaders)
            # else:
            retrain_loader = train_loader_array[i]
                
            start_acc = initial_acc[i]
            retrained_network, acc = get_retrained_model(args, retrain_loader, test_loader, old_network=old_networks[i], tensorboard_obj=tensorboard_obj, nick=nick, start_acc=start_acc, retrain_seed=args.retrain_seed)
        retrained_networks.append(retrained_network)
        accuracies.append(acc)
    return retrained_networks, accuracies


def intmd_retrain_models(args, old_networks, aligned_wts, train_loader, test_loader, config, tensorboard_obj=None, initial_acc=None):
    accuracies = []
    retrained_networks = []
    # nicks = []

    # assert len(old_networks) >= 4

    for i in range(len(old_networks)):

        nick = 'intmd_retrain_model_' + str(i)
        print("Retraining model : ", nick)

        if initial_acc is not None:
            start_acc = initial_acc[i]
        else:
            start_acc = -1
        if args.dataset.lower() == 'cifar10':

            output_root_dir = "{}/{}_models_ensembled/".format(args.baseroot, (args.dataset).lower())
            output_root_dir = os.path.join(output_root_dir, args.exp_name, nick)
            os.makedirs(output_root_dir, exist_ok=True)

            retrained_network, acc = cifar_train.get_retrained_model(args, train_loader, test_loader, old_networks[i], config, output_root_dir, tensorboard_obj=tensorboard_obj, nick=nick, start_acc=start_acc)

        elif args.dataset.lower() == 'mnist':
            # start_acc = initial_acc[i]
            retrained_network, acc = get_intmd_retrain_model(args, train_loader, test_loader, old_network=old_networks[i], tensorboard_obj=tensorboard_obj, nick=nick, start_acc=start_acc)
        retrained_networks.append(retrained_network)
        accuracies.append(acc)
    return retrained_networks, accuracies
