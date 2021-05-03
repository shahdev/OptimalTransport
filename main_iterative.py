import parameters
from data import get_dataloader
import routines
import baseline
import wasserstein_ensemble
import os
import utils
import numpy as np
import sys
import torch
import math

PATH_TO_CIFAR = "./cifar/"
sys.path.append(PATH_TO_CIFAR)
import train as cifar_train
from tensorboardX import SummaryWriter

if __name__ == '__main__':

    print("------- Setting up parameters -------")
    args = parameters.get_parameters()
    print("The parameters are: \n", args)

    if args.deterministic:
        # torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # loading configuration
    config, second_config = utils._get_config(args)
    args.config = config
    args.second_config = second_config


    # get dataloaders
    print("------- Obtain dataloaders -------")
    train_loader_array, test_loader = get_dataloader(args)
    retrain_loader_array, _ = get_dataloader(args, no_randomness=args.no_random_trainloaders)

    print("------- Training independent models -------")
    initial_model = None

    ut_local_array = None
    vt_local_array = None
    ut_global = None
    vt_global = None

    lb = args.lb
    E0 = args.n_epochs
    gamma = args.gamma
    F = args.F
    E_min = args.E_min
    for comm_round in range(args.num_comm_rounds):
        args.n_epochs = max(E_min, math.ceil(E0 * args.gamma ** int(comm_round / args.F)))
        print("LOCAL TRAINING EPOCHS : ", args.n_epochs)
        models, accuracies, adv_accuracies, (ut_local_array, vt_local_array) = routines.train_models(args, train_loader_array, test_loader, ut_local_array=ut_local_array, vt_local_array=vt_local_array, ut_global=ut_global, vt_global=vt_global, lb=lb, initial_model=initial_model, comm_round=comm_round)
        
        ut_global = {}
        vt_global = {}
        for n in ut_local_array[0]:
            ut_global[n] = torch.mean(torch.stack([x[n] for x in ut_local_array]), axis=0)
            vt_global[n] = torch.mean(torch.stack([x[n] for x in vt_local_array]), axis=0)

        print("Communication Round: ", comm_round)

        for name, param in models[0].named_parameters():
            print(f'layer {name} has #params ', param.numel())

        import time
        # second_config is not needed here as well, since it's just used for the dataloader!
        print("Activation Timer start")
        st_time = time.perf_counter()
        activations = utils.get_model_activations(args, models, train_loader_array=train_loader_array, config=config)
        end_time = time.perf_counter()
        setattr(args, 'activation_time', end_time - st_time)
        print("Activation Timer ends")

        for idx, model in enumerate(models):
            setattr(args, f'params_model_{idx}', utils.get_model_size(model))

        # if args.ensemble_iter == 1:
        #
        # else:
        #     # else just recompute activations inside the method iteratively
        #     activations = None


        # set seed for numpy based calculations
        NUMPY_SEED = 100
        np.random.seed(NUMPY_SEED)

        # run geometric aka wasserstein ensembling
        print("------- Geometric Ensembling -------")
        # Deprecated: wasserstein_ensemble.geometric_ensembling(models, train_loader, test_loader)


        print("Timer start")
        st_time = time.perf_counter()
        
        geometric_acc1, geometric_model1 = wasserstein_ensemble.geometric_ensembling_modularized_compare(args, [models[0], models[1]], train_loader_array, test_loader, activations, mode='2_networks')
        geometric_acc, geometric_model, local_models = wasserstein_ensemble.geometric_ensembling_modularized_compare(args, models, train_loader_array, test_loader, activations, mode='all_networks')

        log_dict = {}
        geometric_adv_acc = routines.test_adv(args, geometric_model, test_loader, log_dict) 
        end_time = time.perf_counter()
        print("Timer ends")
        setattr(args, 'geometric_time', end_time - st_time)
        args.params_geometric = utils.get_model_size(geometric_model)

        print("Time taken for geometric ensembling is {} seconds".format(str(end_time - st_time)))
        # run baselines

        print("------- Naive ensembling of weights -------")
        naive_acc, naive_model = baseline.naive_ensembling(args, models, test_loader)
        naive_adv_acc = routines.test_adv(args, naive_model, test_loader, log_dict) 
        final_results_dic = {}
        if args.save_result_file != '':            
            results_dic = {}
            results_dic['exp_name'] = args.exp_name

            for idx, acc in enumerate(accuracies):
                results_dic['model{}_acc'.format(idx)] = acc

            for idx, acc in enumerate(adv_accuracies):
                results_dic['model{}_adv_acc'.format(idx)] = acc

            results_dic['geometric_acc'] = geometric_acc
            results_dic['geometric_adv_acc'] = geometric_adv_acc
            results_dic['naive_acc'] = naive_acc
            results_dic['naive_adv_acc'] = naive_adv_acc
            results_dic['geometric_acc1'] = geometric_acc1
            if args.eval_aligned:
                results_dic['model0_aligned'] = args.model0_aligned_acc

            results_dic['geometric_time'] = args.geometric_time

            final_results_dic[comm_round] = results_dic
            utils.save_results_params_csv(
                args.save_result_file,
                final_results_dic,
                args
            )

            print('----- Saved results at {} ------'.format(args.save_result_file))
            print(results_dic)


        print("FYI: the parameters were: \n", args)
        #if geometric_acc > naive_acc and geometric_adv_acc > naive_adv_acc:
        initial_model = geometric_model #Set the model for next round of training
        print("OPTIMAL TRANSPORT FUSION")
        #else:
        #    initial_model = naive_model
        #    print("NAIVE AVERAGE FUSION")

        ut_local_array = []
        vt_local_array = []
        for idx in range(len(local_models)):
            ut_local, vt_local = routines.compute_fisher_matrix(args, local_models[idx], train_loader_array[idx])
            ut_local_array.append(ut_local)
            vt_local_array.append(vt_local)

        ut_global = {}
        vt_global = {}
        for n in ut_local_array[0]:
            ut_global[n] = torch.mean(torch.stack([x[n] for x in ut_local_array]), axis=0)
            vt_global[n] = torch.mean(torch.stack([x[n] for x in vt_local_array]), axis=0)

        torch.save(initial_model.state_dict(),  '{}/global_model_{}.pth'.format(args.save_dir, comm_round))
