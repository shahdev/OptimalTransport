import ot
import torch
import numpy as np
import routines
from model import get_model_from_name
import utils
from ground_metric import GroundMetric
import math
import sys
import compute_activations

def cost_matrix(x, y, p=2):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
    return c

def get_histogram(args, idx, cardinality, layer_name, activations=None, return_numpy = True, float64=False):
    if activations is None:
        # returns a uniform measure
        if not args.unbalanced:
            print("returns a uniform measure of cardinality: ", cardinality)
            return np.ones(cardinality)/cardinality
        else:
            return np.ones(cardinality)
    else:
        # return softmax over the activations raised to a temperature
        # layer_name is like 'fc1.weight', while activations only contains 'fc1'
        print(activations[idx].keys())
        unnormalized_weights = activations[idx][layer_name.split('.')[0]]
        print("For layer {},  shape of unnormalized weights is ".format(layer_name), unnormalized_weights.shape)
        unnormalized_weights = unnormalized_weights.squeeze()
        assert unnormalized_weights.shape[0] == cardinality

        if return_numpy:
            if float64:
                return torch.softmax(unnormalized_weights / args.softmax_temperature, dim=0).data.cpu().numpy().astype(
                    np.float64)
            else:
                return torch.softmax(unnormalized_weights / args.softmax_temperature, dim=0).data.cpu().numpy()
        else:
            return torch.softmax(unnormalized_weights / args.softmax_temperature, dim=0)


def get_wassersteinized_layers_modularized(args, networks, activations=None, eps=1e-7, test_loader=None, base_model=0, fuse_layer_start_idx=0):
    '''
    Two neural networks that have to be averaged in geometric manner (i.e. layerwise).
    The 1st network is aligned with respect to the other via wasserstein distance.
    Also this assumes that all the layers are either fully connected or convolutional *(with no bias)*

    :param networks: list of networks
    :param activations: If not None, use it to build the activation histograms.
    Otherwise assumes uniform distribution over neurons in a layer.
    :return: list of layer weights 'wassersteinized'
    '''

    # simple_model_0, simple_model_1 = networks[0], networks[1]
    # simple_model_0 = get_trained_model(0, model='simplenet')
    # simple_model_1 = get_trained_model(1, model='simplenet')

    avg_aligned_layers = []
    # cumulative_T_var = None    

    if args.gpu_id==-1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu_id))

    for x in networks[0].named_parameters():
        if 'bias' in x[0]:
            bias = True
            break
    if bias:
        layer_step = 2        
    else:
        layer_step = 1

    num_layers = int(len(list(networks[0].parameters()))/layer_step)

    num_models = len(networks)

    aligned_models = []


    for k in range(0, num_models):
        aligned_layers = []
        orig_model = []
        layer_shapes = []

        T_var = None
        # print(list(networks[0].parameters()))
        previous_layer_shape = None
        ground_metric_object = GroundMetric(args)

        if args.eval_aligned:
            model0_aligned_layers = []
        # for idx, ((layer0_name, fc_layer0_weight), (layer1_name, fc_layer1_weight)) in \
        #         enumerate(zip(networks[k].named_parameters(), networks[base_model].named_parameters())):
        for idx in range(num_layers):
            layer0_name, fc_layer0_weight = list(networks[k].named_parameters())[2*idx]
            layer1_name, fc_layer1_weight = list(networks[base_model].named_parameters())[2*idx]

            if bias:
                bias_layer0_name, bias_fc_layer0_weight = list(networks[k].named_parameters())[2*idx+1]

            assert fc_layer0_weight.shape == fc_layer1_weight.shape
            print("Previous layer shape is ", previous_layer_shape)
            previous_layer_shape = fc_layer1_weight.shape

            mu_cardinality = fc_layer0_weight.shape[0]
            nu_cardinality = fc_layer1_weight.shape[0]

            # mu = np.ones(fc_layer0_weight.shape[0])/fc_layer0_weight.shape[0]
            # nu = np.ones(fc_layer1_weight.shape[0])/fc_layer1_weight.shape[0]
            if idx < fuse_layer_start_idx:
                t_fc0_model = fc_layer0_weight.data
                layer_shape = fc_layer0_weight.shape
                if bias:
                    bias_t_fc0_model = bias_fc_layer0_weight 
                    bias_layer_shape = bias_fc_layer0_weight.shape
            else:
                layer_shape = fc_layer0_weight.shape
                if bias:
                    bias_layer_shape = bias_fc_layer0_weight.shape

                if len(layer_shape) > 2:
                    is_conv = True
                    # For convolutional layers, it is (#out_channels, #in_channels, height, width)
                    fc_layer0_weight_data = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], fc_layer0_weight.shape[1], -1)
                    fc_layer1_weight_data = fc_layer1_weight.data.view(fc_layer1_weight.shape[0], fc_layer1_weight.shape[1], -1)
                    if bias:
                        bias_fc_layer0_weight_data =  bias_fc_layer0_weight.data                        
                else:
                    is_conv = False
                    fc_layer0_weight_data = fc_layer0_weight.data
                    fc_layer1_weight_data = fc_layer1_weight.data
                    if bias:
                        bias_fc_layer0_weight_data =  bias_fc_layer0_weight.data

                if bias:
                        bias_aligned_wt = bias_fc_layer0_weight_data

                if idx == fuse_layer_start_idx:
                    if is_conv:
                        M = ground_metric_object.process(fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1),
                                        fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
                        # M = cost_matrix(fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1),
                        #                 fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
                    else:
                        # print("layer data is ", fc_layer0_weight_data, fc_layer1_weight_data)
                        M = ground_metric_object.process(fc_layer0_weight_data, fc_layer1_weight_data)
                        # M = cost_matrix(fc_layer0_weight, fc_layer1_weight)

                    aligned_wt = fc_layer0_weight_data
                else:

                    print("shape of layer: model 0", fc_layer0_weight_data.shape)
                    print("shape of layer: model 1", fc_layer1_weight_data.shape)
                    print("shape of previous transport map", T_var.shape)
                    # aligned_wt = None, this caches the tensor and causes OOM
                    if is_conv:
                        T_var_conv = T_var.unsqueeze(0).repeat(fc_layer0_weight_data.shape[2], 1, 1)
                        aligned_wt = torch.bmm(fc_layer0_weight_data.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)
                        M = ground_metric_object.process(
                            aligned_wt.contiguous().view(aligned_wt.shape[0], -1),
                            fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
                        )
                    else:
                        if fc_layer0_weight.data.shape[1] != T_var.shape[0]:
                            # Handles the switch from convolutional layers to fc layers
                            fc_layer0_unflattened = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], T_var.shape[0], -1).permute(2, 0, 1)
                            aligned_wt = torch.bmm(
                                fc_layer0_unflattened,
                                T_var.unsqueeze(0).repeat(fc_layer0_unflattened.shape[0], 1, 1)
                            ).permute(1, 2, 0)
                            aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)

                        else:
                            # print("layer data (aligned) is ", aligned_wt, fc_layer1_weight_data)
                            aligned_wt = torch.matmul(fc_layer0_weight.data, T_var)
                        # M = cost_matrix(aligned_wt, fc_layer1_weight)
                        M = ground_metric_object.process(aligned_wt, fc_layer1_weight)
                        print("ground metric is ", M)

                if idx == (num_layers - 1):
                    print("Simple averaging of last layer weights. NO transport map needs to be computed")
                    aligned_layers.append(aligned_wt)
                    layer_shapes.append(layer_shape)
                    if bias:
                        aligned_layers.append(bias_aligned_wt)
                        layer_shapes.append(bias_layer_shape)
                    break

                if args.importance is None or (idx == num_layers - 1):
                    mu = get_histogram(args, 0, mu_cardinality, layer0_name)
                    nu = get_histogram(args, 1, nu_cardinality, layer1_name)
                else:
                    # mu = _get_neuron_importance_histogram(args, aligned_wt, is_conv)
                    mu = _get_neuron_importance_histogram(args, fc_layer0_weight_data, is_conv)
                    nu = _get_neuron_importance_histogram(args, fc_layer1_weight_data, is_conv)
                    print(mu, nu)
                    assert args.proper_marginals

                cpuM = M.data.cpu().numpy()
                if args.exact:
                    T = ot.emd(mu, nu, cpuM)
                else:
                    T = ot.bregman.sinkhorn(mu, nu, cpuM, reg=args.reg)
                # T = ot.emd(mu, nu, log_cpuM)

                if args.gpu_id!=-1:
                    T_var = torch.from_numpy(T).cuda(args.gpu_id).float()
                else:
                    T_var = torch.from_numpy(T).float()

                # torch.set_printoptions(profile="full")
                print("the transport map is ", T_var)
                # torch.set_printoptions(profile="default")

                if args.correction:
                    if not args.proper_marginals:
                        # think of it as m x 1, scaling weights for m linear combinations of points in X
                        if args.gpu_id != -1:
                            # marginals = torch.mv(T_var.t(), torch.ones(T_var.shape[0]).cuda(args.gpu_id))  # T.t().shape[1] = T.shape[0]
                            marginals = torch.ones(T_var.shape[0]).cuda(args.gpu_id) / T_var.shape[0]
                        else:
                            # marginals = torch.mv(T_var.t(),
                            #                      torch.ones(T_var.shape[0]))  # T.t().shape[1] = T.shape[0]
                            marginals = torch.ones(T_var.shape[0]) / T_var.shape[0]
                        marginals = torch.diag(1.0/(marginals + eps))  # take inverse
                        T_var = torch.matmul(T_var, marginals)
                    else:
                        # marginals_alpha = T_var @ torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)
                        marginals_beta = T_var.t() @ torch.ones(T_var.shape[0], dtype=T_var.dtype).to(device)

                        marginals = (1 / (marginals_beta + eps))
                        print("shape of inverse marginals beta is ", marginals_beta.shape)
                        print("inverse marginals beta is ", marginals_beta)

                        T_var = T_var * marginals
                        # i.e., how a neuron of 2nd model is constituted by the neurons of 1st model
                        # this should all be ones, and number equal to number of neurons in 2nd model
                        print(T_var.sum(dim=0))
                        # assert (T_var.sum(dim=0) == torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)).all()

                if args.debug:
                    if idx == (num_layers - 1):
                        print("there goes the last transport map: \n ", T_var)
                    else:
                        print("there goes the transport map at layer {}: \n ".format(idx), T_var)

                    print("Ratio of trace to the matrix sum: ", torch.trace(T_var) / torch.sum(T_var))

                print("Ratio of trace to the matrix sum: ", torch.trace(T_var) / torch.sum(T_var))
                print("Here, trace is {} and matrix sum is {} ".format(torch.trace(T_var), torch.sum(T_var)))
                setattr(args, 'trace_sum_ratio_{}'.format(layer0_name), (torch.trace(T_var) / torch.sum(T_var)).item())

                if args.past_correction:
                    print("this is past correction for weight mode")
                    print("Shape of aligned wt is ", aligned_wt.shape)
                    print("Shape of fc_layer0_weight_data is ", fc_layer0_weight_data.shape)
                    t_fc0_model = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1))
                    if bias:
                        bias_t_fc0_model = torch.matmul(T_var.t(), bias_aligned_wt)
                else:
                    t_fc0_model = torch.matmul(T_var.t(), fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1))
                    if bias:
                        bias_t_fc0_model = torch.matmul(T_var.t(), bias_aligned_wt)

            aligned_layers.append(t_fc0_model)
            layer_shapes.append(layer_shape) 
            if bias:
                aligned_layers.append(bias_t_fc0_model)
                layer_shapes.append(bias_layer_shape)

        aligned_models.append(aligned_layers)        
    
    for layer_num in range(len(aligned_models[0])):
        layer_shape = layer_shapes[layer_num]
        for x in aligned_models:
            if len(layer_shape) > 2:
                x[layer_num] = x[layer_num].view(layer_shape)

        geometric_fc = sum([x[layer_num] for x in aligned_models])/num_models
        # if len(layer_shape) > 2 and layer_shape != geometric_fc.shape:
        #     geometric_fc = geometric_fc.view(layer_shape)
        avg_aligned_layers.append(geometric_fc)

    return avg_aligned_layers, aligned_models

def print_stats(arr, nick=""):
    print(nick)
    print("summary stats are: \n max: {}, mean: {}, min: {}, median: {}, std: {} \n".format(
        arr.max(), arr.mean(), arr.min(), np.median(arr), arr.std()
    ))

def get_activation_distance_stats(activations_0, activations_1, layer_name=""):
    if layer_name != "":
        print("In layer {}: getting activation distance statistics".format(layer_name))
    M = cost_matrix(activations_0, activations_1) ** (1/2)
    mean_dists =  torch.mean(M, dim=-1)
    max_dists = torch.max(M, dim=-1)[0]
    min_dists = torch.min(M, dim=-1)[0]
    std_dists = torch.std(M, dim=-1)

    print("Statistics of the distance from neurons of layer 1 (averaged across nodes of layer 0): \n")
    print("Max : {}, Mean : {}, Min : {}, Std: {}".format(torch.mean(max_dists), torch.mean(mean_dists), torch.mean(min_dists), torch.mean(std_dists)))

def update_model(args, model, new_params, test=False, test_loader=None, reversed=False, idx=-1):

    updated_model = get_model_from_name(args, idx=idx)
    if args.gpu_id != -1:
        updated_model = updated_model.cuda(args.gpu_id)

    layer_idx = 0
    model_state_dict = model.state_dict()

    print("len of model_state_dict is ", len(model_state_dict.items()))
    print("len of new_params is ", len(new_params))

    for key, value in model_state_dict.items():
        print("updated parameters for layer ", key)
        model_state_dict[key] = new_params[layer_idx]
        layer_idx += 1
        if layer_idx == len(new_params):
            break


    updated_model.load_state_dict(model_state_dict)

    if test:
        log_dict = {}
        log_dict['test_losses'] = []
        final_acc = routines.test(args, updated_model, test_loader, log_dict)
        print("accuracy after update is ", final_acc)
    else:
         final_acc = None

    return updated_model, final_acc

def _check_activation_sizes(args, acts0, acts1):
    if args.width_ratio == 1:
        return acts0.shape == acts1.shape
    else:
        return acts0.shape[-1]/acts1.shape[-1] == args.width_ratio

def process_activations(args, activations, layer0_name, layer1_name):
    activations_0 = activations[0][layer0_name.replace('.' + layer0_name.split('.')[-1], '')].squeeze(1)
    activations_1 = activations[1][layer1_name.replace('.' + layer1_name.split('.')[-1], '')].squeeze(1)

    # assert activations_0.shape == activations_1.shape
    _check_activation_sizes(args, activations_0, activations_1)

    if args.same_model != -1:
        # sanity check when averaging the same model (with value being the model index)
        assert (activations_0 == activations_1).all()
        print("Are the activations the same? ", (activations_0 == activations_1).all())

    if len(activations_0.shape) == 2:
        activations_0 = activations_0.t()
        activations_1 = activations_1.t()
    elif len(activations_0.shape) > 2:
        reorder_dim = [l for l in range(1, len(activations_0.shape))]
        reorder_dim.append(0)
        print("reorder_dim is ", reorder_dim)
        activations_0 = activations_0.permute(*reorder_dim).contiguous()
        activations_1 = activations_1.permute(*reorder_dim).contiguous()

    return activations_0, activations_1

def _reduce_layer_name(layer_name):
    # print("layer0_name is ", layer0_name) It was features.0.weight
    # previous way assumed only one dot, so now I replace the stuff after last dot
    return layer_name.replace('.' + layer_name.split('.')[-1], '')

def _get_layer_weights(layer_weight, is_conv):
    if is_conv:
        # For convolutional layers, it is (#out_channels, #in_channels, height, width)
        layer_weight_data = layer_weight.data.view(layer_weight.shape[0], layer_weight.shape[1], -1)
    else:
        layer_weight_data = layer_weight.data

    return layer_weight_data

def _process_ground_metric_from_acts(args, is_conv, ground_metric_object, activations):
    print("inside refactored")
    if is_conv:
        if not args.gromov:
            M0 = ground_metric_object.process(activations[0].view(activations[0].shape[0], -1),
                                             activations[1].view(activations[1].shape[0], -1))
        else:
            M0 = ground_metric_object.process(activations[0].view(activations[0].shape[0], -1),
                                              activations[0].view(activations[0].shape[0], -1))
            M1 = ground_metric_object.process(activations[1].view(activations[1].shape[0], -1),
                                              activations[1].view(activations[1].shape[0], -1))

        print("# of ground metric features is ", (activations[0].view(activations[0].shape[0], -1)).shape[1])
    else:
        if not args.gromov:
            M0 = ground_metric_object.process(activations[0], activations[1])
        else:
            M0 = ground_metric_object.process(activations[0], activations[0])
            M1 = ground_metric_object.process(activations[1], activations[1])

    if args.gromov:
        return M0, M1
    else:
        return M0, None


def _custom_sinkhorn(args, mu, nu, cpuM):
    if not args.unbalanced:
        if args.sinkhorn_type == 'normal':
            T = ot.bregman.sinkhorn(mu, nu, cpuM, reg=args.reg)
        elif args.sinkhorn_type == 'stabilized':
            T = ot.bregman.sinkhorn_stabilized(mu, nu, cpuM, reg=args.reg)
        elif args.sinkhorn_type == 'epsilon':
            T = ot.bregman.sinkhorn_epsilon_scaling(mu, nu, cpuM, reg=args.reg)
        elif args.sinkhorn_type == 'gpu':
            T, _ = utils.sinkhorn_loss(cpuM, mu, nu, gpu_id=args.gpu_id, epsilon=args.reg, return_tmap=True)
        else:
            raise NotImplementedError
    else:
        T = ot.unbalanced.sinkhorn_knopp_unbalanced(mu, nu, cpuM, reg=args.reg, reg_m=args.reg_m)
    return T


def _sanity_check_tmap(T):
    if not math.isclose(np.sum(T), 1.0, abs_tol=1e-7):
        print("Sum of transport map is ", np.sum(T))
        raise Exception('NAN inside Transport MAP. Most likely due to large ground metric values')

def _get_updated_acts_v0(args, layer_shape, aligned_wt, model0_aligned_layers, networks, test_loader, layer_names):
    '''
    Return the updated activations of the 0th model with respect to the other one.

    :param args:
    :param layer_shape:
    :param aligned_wt:
    :param model0_aligned_layers:
    :param networks:
    :param test_loader:
    :param layer_names:
    :return:
    '''
    if layer_shape != aligned_wt.shape:
        updated_aligned_wt = aligned_wt.view(layer_shape)
    else:
        updated_aligned_wt = aligned_wt

    updated_model0, _ = update_model(args, networks[0], model0_aligned_layers + [updated_aligned_wt], test=True,
                                     test_loader=test_loader, idx=0)
    updated_activations = utils.get_model_activations(args, [updated_model0, networks[1]],
                                                      config=args.config,
                                                      layer_name=_reduce_layer_name(layer_names[0]), selective=True)

    updated_activations_0, updated_activations_1 = process_activations(args, updated_activations,
                                                                       layer_names[0], layer_names[1])
    return updated_activations_0, updated_activations_1

def _get_updated_acts_v1(args, networks, test_loader, layer_names):
    '''
    Return the updated activations of the 0th model with respect to the other one.

    :param args:
    :param layer_shape:
    :param aligned_wt:
    :param model0_aligned_layers:
    :param networks:
    :param test_loader:
    :param layer_names:
    :return:
    '''
    updated_activations = utils.get_model_activations(args, networks,
                                                      config=args.config)

    updated_activations_0, updated_activations_1 = process_activations(args, updated_activations,
                                                                       layer_names[0], layer_names[1])
    return updated_activations_0, updated_activations_1

def _check_layer_sizes(args, layer_idx, shape1, shape2, num_layers):
    if args.width_ratio == 1:
        return shape1 == shape2
    else:
        if args.dataset == 'mnist':
            if layer_idx == 0:
                return shape1[-1] == shape2[-1] and (shape1[0]/shape2[0]) == args.width_ratio
            elif layer_idx == (num_layers -1):
                return (shape1[-1]/shape2[-1]) == args.width_ratio and shape1[0] == shape2[0]
            else:
                ans = True
                for ix in range(len(shape1)):
                    ans = ans and shape1[ix]/shape2[ix] == args.width_ratio
                return ans
        elif args.dataset[0:7] == 'Cifar10':
            assert args.second_model_name is not None
            if layer_idx == 0 or layer_idx == (num_layers -1):
                return shape1 == shape2
            else:
                if (not args.reverse and layer_idx == (num_layers-2)) or (args.reverse and layer_idx == 1):
                    return (shape1[1] / shape2[1]) == args.width_ratio
                else:
                    return (shape1[0]/shape2[0]) == args.width_ratio


def _compute_marginals(args, T_var, device, eps=1e-7):
    if args.correction:
        if not args.proper_marginals:
            # think of it as m x 1, scaling weights for m linear combinations of points in X
            marginals = torch.ones(T_var.shape)
            if args.gpu_id != -1:
                marginals = marginals.cuda(args.gpu_id)

            marginals = torch.matmul(T_var, marginals)
            marginals = 1 / (marginals + eps)
            print("marginals are ", marginals)

            T_var = T_var * marginals

        else:
            # marginals_alpha = T_var @ torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)
            marginals_beta = T_var.t() @ torch.ones(T_var.shape[0], dtype=T_var.dtype).to(device)

            marginals = (1 / (marginals_beta + eps))
            print("shape of inverse marginals beta is ", marginals_beta.shape)
            print("inverse marginals beta is ", marginals_beta)

            T_var = T_var * marginals
            # i.e., how a neuron of 2nd model is constituted by the neurons of 1st model
            # this should all be ones, and number equal to number of neurons in 2nd model
            print(T_var.sum(dim=0))
            # assert (T_var.sum(dim=0) == torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)).all()

        print("T_var after correction ", T_var)
        print("T_var stats: max {}, min {}, mean {}, std {} ".format(T_var.max(), T_var.min(), T_var.mean(),
                                                                     T_var.std()))
    else:
        marginals = None

    return T_var, marginals

def _get_current_layer_transport_map(args, mu, nu, M0, M1, idx, layer_shape, eps=1e-7, layer_name=None):

    if not args.gromov:
        cpuM = M0.data.cpu().numpy()
        if args.exact:
            T = ot.emd(mu, nu, cpuM)
        else:
            T = _custom_sinkhorn(args, mu, nu, cpuM)

        if args.print_distances:
            ot_cost = np.multiply(T, cpuM).sum()
            print(f'At layer idx {idx} and shape {layer_shape}, the OT cost is ', ot_cost)
            if layer_name is not None:
                setattr(args, f'{layer_name}_layer_{idx}_cost', ot_cost)
            else:
                setattr(args, f'layer_{idx}_cost', ot_cost)
    else:
        cpuM0 = M0.data.cpu().numpy()
        cpuM1 = M1.data.cpu().numpy()

        assert not args.exact
        T = ot.gromov.entropic_gromov_wasserstein(cpuM0, cpuM1, mu, nu, loss_fun=args.gromov_loss, epsilon=args.reg)

    if not args.unbalanced:
        _sanity_check_tmap(T)

    if args.gpu_id != -1:
        T_var = torch.from_numpy(T).cuda(args.gpu_id).float()
    else:
        T_var = torch.from_numpy(T).float()

    if args.tmap_stats:
        print(
        "Tmap stats (before correction) \n: For layer {}, frobenius norm from the joe's transport map is {}".format(
            layer0_name, torch.norm(T_var - torch.ones_like(T_var) / torch.numel(T_var), p='fro')
        ))

    print("shape of T_var is ", T_var.shape)
    print("T_var before correction ", T_var)

    return T_var

def _get_neuron_importance_histogram(args, layer_weight, is_conv, eps=1e-9):
    print('shape of layer_weight is ', layer_weight.shape)
    if is_conv:
        layer = layer_weight.contiguous().view(layer_weight.shape[0], -1).cpu().numpy()
    else:
        layer = layer_weight.cpu().numpy()
    
    if args.importance == 'l1':
        importance_hist = np.linalg.norm(layer, ord=1, axis=-1).astype(
                    np.float64) + eps
    elif args.importance == 'l2':
        importance_hist = np.linalg.norm(layer, ord=2, axis=-1).astype(
                    np.float64) + eps
    else:
        raise NotImplementedError

    if not args.unbalanced:
        importance_hist = (importance_hist/importance_hist.sum())
        print('sum of importance hist is ', importance_hist.sum())
    # assert importance_hist.sum() == 1.0
    return importance_hist

def get_network_from_param_list(args, param_list, test_loader):

    print("using independent method")
    new_network = get_model_from_name(args, idx=1)
    if args.gpu_id != -1:
        new_network = new_network.cuda(args.gpu_id)

    # check the test performance of the network before
    log_dict = {}
    log_dict['test_losses'] = []
    routines.test(args, new_network, test_loader, log_dict)

    # set the weights of the new network
    # print("before", new_network.state_dict())
    print("len of model parameters and avg aligned layers is ", len(list(new_network.parameters())),
          len(param_list))
    assert len(list(new_network.parameters())) == len(param_list)

    layer_idx = 0
    model_state_dict = new_network.state_dict()

    print("len of model_state_dict is ", len(model_state_dict.items()))
    print("len of param_list is ", len(param_list))

    for key, value in model_state_dict.items():
        model_state_dict[key] = param_list[layer_idx]
        layer_idx += 1

    new_network.load_state_dict(model_state_dict)

    # check the test performance of the network after
    log_dict = {}
    log_dict['test_losses'] = []
    acc = routines.test(args, new_network, test_loader, log_dict)

    return acc, new_network

def geometric_ensembling_modularized(args, networks, train_loader_array, test_loader, activations=None, base_model=0, fuse_layer_start_idx=0):
    
    avg_aligned_layers, aligned_models = get_wassersteinized_layers_modularized(args, networks, activations, 
        test_loader=test_loader, base_model=base_model, fuse_layer_start_idx=fuse_layer_start_idx)
    acc, new_network = get_network_from_param_list(args, avg_aligned_layers, test_loader)
    return acc, new_network, aligned_models

