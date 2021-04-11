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
    for comm_round in range(args.num_comm_rounds):
        models, accuracies, (ut_local_array, vt_local_array) = routines.train_models(args, train_loader_array, test_loader, ut_local_array=ut_local_array, vt_local_array=vt_local_array, ut_global=ut_global, vt_global=vt_global, lb=lb, initial_model=initial_model)
        
        ut_global = {}
        vt_global = {}
        for n in ut_local_array[0]:
            ut_global[n] = torch.mean(torch.stack([x[n] for x in ut_local_array]))
            vt_global[n] = torch.mean(torch.stack([x[n] for x in vt_local_array]))

        print("Communication Round: ", comm_round)
        # if args.debug:
        #     print(list(models[0].parameters()))

        if args.same_model!=-1:
            print("Debugging with same model")
            model, acc = models[args.same_model], accuracies[args.same_model]
            models = [model, model]
            accuracies = [acc, acc]

        for name, param in models[0].named_parameters():
            print(f'layer {name} has #params ', param.numel())

        import time
        # second_config is not needed here as well, since it's just used for the dataloader!
        print("Activation Timer start")
        st_time = time.perf_counter()
        activations = utils.get_model_activations(args, models, config=config)
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

        geometric_acc, geometric_model = wasserstein_ensemble.geometric_ensembling_modularized(args, models, train_loader_array, test_loader, activations)
        
        end_time = time.perf_counter()
        print("Timer ends")
        setattr(args, 'geometric_time', end_time - st_time)
        args.params_geometric = utils.get_model_size(geometric_model)

        print("Time taken for geometric ensembling is {} seconds".format(str(end_time - st_time)))
        # run baselines
        print("------- Prediction based ensembling -------")
        prediction_acc = baseline.prediction_ensembling(args, models, test_loader)

        print("------- Naive ensembling of weights -------")
        naive_acc, naive_model = baseline.naive_ensembling(args, models, test_loader)

        final_results_dic = {}
        if args.save_result_file != '':            
            results_dic = {}
            results_dic['exp_name'] = args.exp_name

            for idx, acc in enumerate(accuracies):
                results_dic['model{}_acc'.format(idx)] = acc

            results_dic['geometric_acc'] = geometric_acc
            results_dic['prediction_acc'] = prediction_acc
            results_dic['naive_acc'] = naive_acc

            # Additional statistics
            results_dic['geometric_gain'] = geometric_acc - max(accuracies)
            results_dic['geometric_gain_%'] = ((geometric_acc - max(accuracies))*100.0)/max(accuracies)
            results_dic['prediction_gain'] = prediction_acc - max(accuracies)
            results_dic['prediction_gain_%'] = ((prediction_acc - max(accuracies)) * 100.0) / max(accuracies)
            results_dic['relative_loss_wrt_prediction'] = results_dic['prediction_gain_%'] - results_dic['geometric_gain_%']

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

        initial_model = geometric_model #Set the model for next round of training
