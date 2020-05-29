from infer import *

from model import InteractionValueNet
from resnet import InteractionResNet
import torch.nn.functional as F

from data_loader import *

import pickle

import glob

from sklearn.metrics import r2_score
import time

__BEST_MODELS__ = {
    'vgg_adam_300':{'best_idx':[3,89,131,198,248], 'model_file_prefix':'20191009_vgg_adam_300/vgg_*.pt', 'arch': InteractionValueNet},
    'vgg_sgd_300':{'best_idx':[103,131,135,155,208], 'model_file_prefix':'20191010_vgg_sgd_300/vgg_*.pt', 'arch': InteractionValueNet},
    'res_adam':{'best_idx':[4,28,31], 'model_file_prefix':'20191014_res_adam/resnet_*.pt', 'arch': InteractionResNet}
}

def find_param(logdata, idx):
    for paramset in logdata:
        if paramset['idx'] == idx:
            return paramset
    return None

def test_model(test_data, paramsets, model_files, train_mean, train_std, density=0, arch=InteractionResNet, include_all=True, verbose=False):
    test_results = []
    for model_file in model_files:
        test_rst = dict()

        split_names = model_file.rsplit("/",1)[1].split("_")

        test_rst['model_file'] = model_file
        test_rst['idx'] = int(split_names[3])
        test_rst['cv_idx'] = int(split_names[4])

        test_rst['model_type'] = 'res_combine_all'
        test_rst['density'] = density

        param = find_param(paramsets, test_rst['idx'])
        if param is None:
            print('[ERR]')
            print("param:", param, model_file)
            continue
        param['train_mean'] = train_mean
        param['train_std'] = train_std

        ## for testing resnet models
        if 'prob_drop' in param: del param['prob_drop']
        
        try:
            stime = time.time()
            test_loss, percent, predictions, labels = infer(
                arch=arch,
                model_file=model_file,
                test_data=test_data,
                hyperparam=param,
                use_cuda=True,
                to_tensor=ToTensor2D,
                loss_fn=F.mse_loss,
                verbose=verbose
            )
            etime = time.time()
            test_rst['test_loss'] = test_loss
            test_rst['test_accuracy'] = percent
            test_rst['test_r2score'] = r2_score(labels, predictions)
            test_rst['time'] = etime-stime
        except Exception as e:
            print('[ERR]', e)
            print("param:", param, model_file)
            test_rst['test_loss'] = -1
            test_rst['test_accuracy'] = -1
            test_rst['test_r2score'] = -1
            test_rst['time'] = -1
        
        if include_all:
            test_rst['predictions'] = predictions
            test_rst['labels'] = labels
        
        test_results.append(test_rst)

        print("[{}/{}] {:.4f} ({:.1f}%, {:.2f}) {}".format(test_rst['idx'], test_rst['cv_idx'],
            test_rst['test_loss'], test_rst['test_accuracy'], test_rst['test_r2score'], test_rst['time']))
    return test_results

def test_all_combine_models(include_all=False):
    '''
        include_all: include all predictions and target labels
    '''
    # with open('20191019_res_train_combine_all/res_adam_all.pk', 'rb') as f:
    with open('20191022_result_train_all_density/res_adam_all.pk', 'rb') as f:
        log_data = pickle.load(f)
    test_results = []

    ################################################################
    # model_files
    ################################################################
    model_files = glob.glob('20191022_result_train_all_density/20191022_result/*.pt')

    files = ["ibm_data_by_density_{}.pkl".format(d) for d in range(10,80,10)]
    _, _, train_mean, train_std = get_train_test_split_from_multiple_files(files, test_split=0.2, random_seed=42)
    print('train_mean:', train_mean, 'train_std:', train_std)

    for density in range(10,80,10):
        print("--"*50)
        print("# density:", density)
        print("--"*50)
        
        _, test_data, _, _ = get_train_test_split('ibm_data_by_density_{}.pkl'.format(density), test_split=0.2, random_seed=42)

        for model_file in model_files:
            test_rst = dict()

            split_names = model_file.rsplit("/",1)[1].split("_")

            test_rst['model_file'] = model_file
            test_rst['idx'] = int(split_names[3])
            test_rst['cv_idx'] = int(split_names[4])

            test_rst['model_type'] = 'res_combine_all'
            test_rst['density'] = density

            param = find_param(log_data, test_rst['idx'])
            if param is None:
                print('[ERR]')
                print("param:", param, model_file)
                continue
            param['train_mean'] = train_mean
            param['train_std'] = train_std

            ## for testing resnet models
            if 'prob_drop' in param: del param['prob_drop']
            
            try:
                stime = time.time()
                test_loss, percent, predictions, labels = infer(
                    arch=InteractionResNet,
                    model_file=model_file,
                    test_data=test_data,
                    hyperparam=param,
                    use_cuda=True,
                    to_tensor=ToTensor2D,
                    loss_fn=F.mse_loss,
                    verbose=False
                )
                etime = time.time()
                test_rst['test_loss'] = test_loss
                test_rst['test_accuracy'] = percent
                test_rst['test_r2score'] = r2_score(labels, predictions)
                test_rst['time'] = etime-stime
            except Exception as e:
                print('[ERR]', e)
                print("param:", param, model_file)
                test_rst['test_loss'] = -1
                test_rst['test_accuracy'] = -1
                test_rst['test_r2score'] = -1
                test_rst['time'] = -1
            
            if include_all:
                test_rst['predictions'] = predictions
                test_rst['labels'] = labels
            
            test_results.append(test_rst)

            print("[{}/{}] {:.4f} ({:.1f}%, {:.2f}) {}".format(test_rst['idx'], test_rst['cv_idx'],
                test_rst['test_loss'], test_rst['test_accuracy'], test_rst['test_r2score'], test_rst['time']))

    with open('20191022_result_train_all_density/combine_all_density_test.pk', 'wb') as f:
        pickle.dump(test_results, f, pickle.HIGHEST_PROTOCOL)


def test_all_best_models():
    '''test all best models again
    '''
    test_results = []
    for model_type in __BEST_MODELS__:
        print("--"*50)
        print("# model_type:", model_type)
        print("--"*50)

        model_info = __BEST_MODELS__[model_type]

        with open('{}.pk'.format(model_type), 'rb') as f:
            log_data = pickle.load(f)

        ################################################################
        # model_files
        ################################################################
        model_file_prefix = model_info['model_file_prefix']
        model_files = glob.glob(model_file_prefix)
        
        _, _, train_mean, train_std = get_train_test_split('ibm_data_by_density_70.pkl', test_split=0.2, random_seed=42)
        
        for density in range(10,70,10):
            print("--"*50)
            print("# density:", density)
            print("--"*50)
            
            _, test_data, _, _ = get_train_test_split('ibm_data_by_density_{}.pkl'.format(density), test_split=1, random_seed=42)

            for model_file in model_files:
                test_rst = dict()

                split_names = model_file.rsplit("/",1)[1].split("_")

                test_rst['model_file'] = model_file
                test_rst['idx'] = int(split_names[1])
                test_rst['cv_idx'] = int(split_names[2])

                ## test best models only
                if test_rst['idx'] not in model_info['best_idx']: continue

                test_rst['model_type'] = model_type
                test_rst['density'] = density

                param = find_param(log_data, test_rst['idx'])
                if param is None:
                    print('[ERR]')
                    print("param:", param, model_file)
                    continue
                param['train_mean'] = train_mean
                param['train_std'] = train_std
                
                try:
                    stime = time.time()
                    test_loss, percent, predictions, labels = infer(
                        arch=model_info['arch'],
                        model_file=model_file,
                        test_data=test_data,
                        hyperparam=param,
                        use_cuda=True,
                        to_tensor=ToTensor2D,
                        loss_fn=F.mse_loss,
                        verbose=False
                    )
                    etime = time.time()
                    test_rst['test_loss'] = test_loss
                    test_rst['test_accuracy'] = percent
                    test_rst['test_r2score'] = r2_score(labels, predictions)
                    test_rst['time'] = etime-stime
                except Exception as e:
                    print('[ERR]', e)
                    print("param:", param, model_file)
                    test_rst['test_loss'] = -1
                    test_rst['test_accuracy'] = -1
                    test_rst['test_r2score'] = -1
                    test_rst['time'] = -1
                
                test_results.append(test_rst)

                print("[{}/{}] {:.4f} ({:.1f}%, {:.2f}) {}".format(test_rst['idx'], test_rst['cv_idx'],
                    test_rst['test_loss'], test_rst['test_accuracy'], test_rst['test_r2score'], test_rst['time']))

    with open('density_test.pk', 'wb') as f:
        pickle.dump(test_results, f, pickle.HIGHEST_PROTOCOL)


def inference_images(model_config_pk, model_files, input_images, fout='results.pk'):

    with open(model_config_pk, 'rb') as f:
        log_data = pickle.load(f)
    
    ################################################################
    # model_files
    ################################################################
    # model_files = glob.glob('20191022_result_train_all_density/20191022_result/*.pt')

    files = ["ibm_data_by_density_{}.pkl".format(d) for d in range(10,80,10)]
    _, _, train_mean, train_std = get_train_test_split_from_multiple_files(files, test_split=0.2, random_seed=42)
    print('train_mean:', train_mean, 'train_std:', train_std)

    _, test_data, _, _ = get_train_test_split(input_images, test_split=1)
    test_results = test_model(test_data, log_data, model_files, train_mean, train_std, arch=InteractionResNet, include_all=True, verbose=False)

    with open(fout, 'wb') as f:
        pickle.dump(test_results, f, pickle.HIGHEST_PROTOCOL)


def inference_by_density(model_config_pk, model_files, density, arch=InteractionResNet, fout='results.pk'):

    with open(model_config_pk, 'rb') as f:
        log_data = pickle.load(f)
    
    train_data_file = "ibm_data_by_density_{}.pkl".format(density)
    _, test_data, train_mean, train_std = get_train_test_split(train_data_file, test_split=0.2, random_seed=42)
    print('train_mean:', train_mean, 'train_std:', train_std)

    test_results = test_model(test_data, log_data, model_files, train_mean, train_std, arch=arch, include_all=True, verbose=False)

    with open(fout, 'wb') as f:
        pickle.dump(test_results, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    for d in range(10,80,10):
        inference_by_density(
            model_config_pk='20191017_result_train_all_density/res_adam_{}.pk'.format(density),
            model_files='20191017_result_train_all_density/20191017_result/resnet{}*.pt'.format(density),
            density=density, arch=InteractionResNet, fout='20191017_result_train_all_density/res_adam_{}_test_result.pk'.format(density),
        )

    '''model inference using the final best (trained by all density levels)'''
    # test_all_combine_models(include_all=True)

    '''model inference for real images'''
    # inference_images(
    #     model_config_pk='20191022_result_train_all_density/res_adam_all.pk',
    #     model_files=glob.glob('20191022_result_train_all_density/20191022_result/*.pt'),
    #     input_images='testing_450by450_step25.pkl',
    #     fout='results_450by450_step25.pk'
    # )
    