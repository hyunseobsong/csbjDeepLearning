import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import ParameterGrid
from scipy.stats.distributions import uniform
import numpy as np

import pickle
import time

from data_loader import *
from model import *
from train import *
from resnet import *

from GPUtil import showUtilization as gpu_usage


__HYPERPARAMS__ = [
    'weight_decay',
    'prob_drop',
    'maxpool_stride',
    'maxpool_kernel_size',
    'lr',
    'fc1_nodes',
    'fc2_nodes',
    'fc3_nodes',
    'conv_stride',
    'conv_kernel_size',
    'conv_config_num',
    'batch_size',
    'momentum' # for SGD
]

__BEST_MODELS__ = {
    "vgg_adam":[[0.001,0.2,2,3,0.0005,500,100,50,1,3,7,64,0],
                [0.01,0.1,2,3,0.001,1000,1000,50,1,3,4,64,0],
                [0.001,0.7,2,5,0.0005,500,100,500,1,5,5,64,0],
                [0,0.1,2,5,0.001,1000,100,50,1,3,5,128,0],
                [0.01,0.3,2,5,0.0005,500,1000,100,1,3,5,128,0]],
    
    "vgg_sgd":[[0.01,0.4,2,2,0.001,1500,100,0,1,3,6,64,0.9],
               [0.001,0.7,2,5,0.01,1500,100,100,1,3,1,64,0.9],
               [0.0001,0.3,2,5,0.005,1000,50,1000,1,3,5,64,0.9],
               [0.001,0.4,2,5,0.01,500,500,1000,1,3,5,128,0.7],
               [0.0001,0.2,2,5,0.001,500,500,0,1,3,5,128,0.9]],
    
    "res_adam":[[0.0001,0,0,0,0.001,0,0,0,1,3,1,128,0],
                [0,0,0,0,0.001,0,0,0,1,3,1,64,0],
                [0,0,0,0,0.001,0,0,0,1,3,1,128,0]]
}

__BEST_MODEL_PARAMS__ = dict()

for n in __BEST_MODELS__:
    sets = __BEST_MODELS__[n]
    __BEST_MODEL_PARAMS__[n] = [{p:sets[j][i] for i,p in enumerate(__HYPERPARAMS__)} for j in range(len(sets))]



def save_logs(all_logs, fout=None):
    print('save logs: {}'.format(fout))
    if fout:
        with open(fout, 'wb') as f:
            pickle.dump(all_logs, f, pickle.HIGHEST_PROTOCOL)

class HyperParamOptimization(object):
    """docstring for HyperParamOptimization"""
    def __init__(self, param_grid, train_data, test_data, user_params):
        super(HyperParamOptimization, self).__init__()
        self.param_grid = param_grid
        self.train_data = train_data
        self.test_loader = get_test_loader(
            ibm_data=test_data,
            to_tensor=user_params['to_tensor'],
            batch_size=user_params['valid_batch_size'],
            mean_for_normalize=user_params['train_mean'],
            std_for_normalize=user_params['train_std'],
            use_cuda=user_params['use_cuda'],
            shuffle=True)
        self.user_params = user_params
        self.random_seed = user_params['random_seed'] if 'random_seed' in user_params else None
        self.model = user_params['model']
        self.param_list = None
    
    def build_random_grid(self, n_iter=100):
        rng = np.random.RandomState(self.random_seed) if self.random_seed else None
        self.param_list = list(ParameterSampler(self.param_grid, n_iter=n_iter, random_state=rng))
        print("num param sets:", len(self.param_list))
        return self.param_list

    def build_grid(self):
        self.param_list = list(ParameterGrid(self.param_grid))
        print("num param sets:", len(self.param_list))
        return self.param_list

    def run(self, early_stopping_patience=10, verbose=False, fout=None, custom_params=None):
        # user can customize the parameter sets
        if custom_params: self.param_list = custom_params

        if self.param_list:
            try:
                stime = time.time()
                all_logs = []
                n_iter = len(self.param_list)
                for i, hyperparams in enumerate(self.param_list):
                    #if i>1: break
                    try:
                        print("-"*100)
                        print(i, hyperparams)
                        print("-"*100)

                        # TODO combine user params with hyper params
                        hyperparams['epochs'] = self.user_params['epochs']
                        hyperparams['save_model'] = self.user_params['save_model']
                        hyperparams['log_interval'] = self.user_params['log_interval']
                        hyperparams['use_cuda'] = self.user_params['use_cuda']
                        
                        hyperparams['conv_config'] = self.user_params['conv_config'][hyperparams['conv_config_num']]

                        if hyperparams['fc3_nodes']==0:
                            hyperparams['dense_config'] = [hyperparams['fc1_nodes'], hyperparams['fc2_nodes']]
                        else:
                            hyperparams['dense_config'] = [hyperparams['fc1_nodes'], hyperparams['fc2_nodes'], hyperparams['fc3_nodes']]
                        
                        cv_performance = []
                        for k in range(self.user_params['kfold']):
                            hyperparams['model_output'] = "{}_{}_{}".format(self.user_params['model_output'], i, k)

                            ibm_dataset, train_loader, valid_loader = get_train_valid_split(
                                ibm_data=self.train_data, to_tensor=self.user_params['to_tensor'], valid_split=self.user_params['valid_split'],
                                batch_size=hyperparams['batch_size'], valid_batch_size=self.user_params['valid_batch_size'],
                                mean_for_normalize=self.user_params['train_mean'], std_for_normalize=self.user_params['train_std'], k_fold_idx=k,
                                use_cuda=self.user_params['use_cuda'], random_seed=self.random_seed
                            )
                            model = self.model(train_loader=train_loader, valid_loader=valid_loader, hyperparams=hyperparams,
                                optimize_fn=self.user_params['optimize_fn'], loss_fn=self.user_params['loss_fn'], random_seed=self.random_seed)

                            gpu_usage()

                            model.train(early_stopping_patience=early_stopping_patience, verbose=verbose)
                            model.test(test_loader=self.test_loader, verbose=verbose)

                            if model.best_performance is None: break
                            
                            model.best_performance['model_output'] = hyperparams['model_output']
                            if verbose: print("Best:", model.best_performance)
                            cv_performance.append(model.best_performance)
                            hyperparams['total_params'] = model.total_params
                            hyperparams['total_size'] = model.total_size

                            torch.cuda.empty_cache()
                            gpu_usage()
                        
                        hyperparams['cv'] = cv_performance
                        hyperparams['idx'] = i
                        
                        all_logs.append(hyperparams)
                        print("-"*100)
                        print("{}/{} Time:{}min".format(i, n_iter, (time.time()-stime)/60))
                        print("-"*100)
                    except Exception as e:
                        print(e)
                save_logs(all_logs, fout)
            except (KeyboardInterrupt, SystemExit):
                save_logs(all_logs, fout)

def run_vggnet(n_params=300, fin="ibm_data_by_density_70.pkl", fout='test.pk', density=70, custom_params=None, optimizer=optim.Adam):
    user_params = {
        'random_seed': 42,
        'valid_batch_size' : 128,
        'valid_split' : .2,
        'kfold' : 5,
        
        'epochs' : 200,
        'early_stopping_patience' : 20,
        'save_model' : True,
        'log_interval' : 200,
        'use_cuda' : True,
        'model_output' : '20191018_result/vgg_{}_d{}'.format(optimizer.__name__, density),
        
        'model': InteractionValueModel,
        'to_tensor': ToTensor2D,
        'optimize_fn': optimizer, # optim.Adam, optim.SGD
        'loss_fn': F.mse_loss,
        
        'conv_config':[[8, 'M', 16, 'M', 32, 'M'],
                       [16, 'M', 32, 'M', 64, 'M'],
                       [8, 8, 'M', 16, 16, 'M', 32, 32, 'M'],
                       [16, 16, 'M', 32, 32, 'M', 64, 64, 'M'],
                       # [8, 8, 8, 8, 'M', 16, 16, 16, 16, 'M', 32, 32, 32, 32, 'M'],
                       # [16, 16, 16, 16, 'M', 32, 32, 32, 32, 'M', 64, 64, 64, 64, 'M'],
                       [8, 'A', 16, 'A', 32, 'A'],
                       [16, 'A', 32, 'A', 64, 'A'],
                       [8, 8, 'A', 16, 16, 'A', 32, 32, 'A'],
                       [16, 16, 'A', 32, 32, 'A', 64, 64, 'A'],
                       # [8, 8, 8, 8, 'A', 16, 16, 16, 16, 'A', 32, 32, 32, 32, 'A'],
                       # [16, 16, 16, 16, 'A', 32, 32, 32, 32, 'A', 64, 64, 64, 64, 'A'],
                      ]
    }

    # param_grid = {
    #     'batch_size' : [64, 128],
    #     'lr' : [0.01, 0.005, 0.001, 0.0005],
    #     'prob_drop' : 0.1*np.array(range(1,10)),
    #     #'momentum' : [0, 0.5, 0.7, 0.9],
    #     'weight_decay' : [0, 0.01, 0.001, 0.0001],
    #     'conv_kernel_size' : [3, 5],
    #     'maxpool_kernel_size' : [2, 3, 5],
    #     'conv_stride' : [1],
    #     'maxpool_stride' : [2],
    #     'fc1_nodes' : [500, 1000, 1500],
    #     'fc2_nodes' : [50, 100, 500, 1000],
    #     'fc3_nodes' : [0, 50, 100, 500, 1000],
    #     # 'conv_config_num': np.array(range(len(user_params['conv_config'])))
    #     'conv_config_num': [5,7]
    # }
    ## for 10-60
    param_grid = {
        'batch_size' : [64, 128],
        'lr' : [0.001, 0.0005],
        'prob_drop' : [0.1, 0.2, 0.3, 0.7],
        #'momentum' : [0, 0.5, 0.7, 0.9],
        'weight_decay' : [0, 0.01, 0.001],
        'conv_kernel_size' : [3, 5],
        'maxpool_kernel_size' : [3, 5],
        'conv_stride' : [1],
        'maxpool_stride' : [2],
        'fc1_nodes' : [500, 1000],
        'fc2_nodes' : [100, 1000],
        'fc3_nodes' : [50, 100, 500],
        'conv_config_num': [5,7]
    }

    print("user_params['conv_config']:", len(user_params['conv_config']))

    test_split = 0.2
    train_data, test_data, train_mean, train_std = get_train_test_split(fin, test_split=test_split, random_seed=user_params['random_seed'])
    user_params['train_mean'] = train_mean
    user_params['train_std'] = train_std

    print('train_mean:', train_mean, 'train_std:', train_std)

    hyperopt = HyperParamOptimization(param_grid, train_data, test_data, user_params)
    hyperopt.build_random_grid(n_params)
    hyperopt.run(early_stopping_patience=user_params['early_stopping_patience'], verbose=False, fout=fout, custom_params=custom_params)

def run_resnet(n_params=300, fin="ibm_data_by_density_70.pkl", fout='test.pk', density=70, custom_params=None, optimizer=optim.Adam):
    user_params = {
        'random_seed': 42,
        'valid_batch_size' : 64,
        'valid_split' : .2,
        'kfold' : 5,
        
        'epochs' : 400,
        'early_stopping_patience' : 40,
        'save_model' : True,
        'log_interval' : 200,
        'use_cuda' : True,
        'model_output' : '20191022_result_train_all_density/resnet_{}_d{}'.format(optimizer.__name__, density),
        
        'model': InteractionResnetModel,
        'to_tensor': ToTensor2D,
        'optimize_fn': optimizer, # optim.Adam, optim.SGD
        'loss_fn': F.mse_loss,

        'conv_config':[
                       # [PreActResNet, PreActBottleneck, [3, 4, 6, 3]],
                       # [PreActResNet, PreActBottleneck, [2, 2, 2, 2]],
                       [PreActResNet, PreActBlock, [3, 4, 6, 3]],
                       [PreActResNet, PreActBlock, [2, 2, 2, 2]],
                       [ResNet, Bottleneck, [3, 4, 6, 3]],
                       [ResNet, Bottleneck, [2, 2, 2, 2]],
                       [ResNet, BasicBlock, [3, 4, 6, 3]],
                       [ResNet, BasicBlock, [2, 2, 2, 2]]
                      ]
    }

    param_grid = {
        'batch_size' : [64, 128],
        'lr' : [0.001], #[0.01, 0.005, 0.001, 0.0005],
        # 'momentum' : 0.1*np.array(range(1,10)),
        # 'weight_decay' : [0, 0.01, 0.001, 0.0001],
        'weight_decay' : [0, 0.0001],
        'conv_kernel_size' : [3],
        'maxpool_kernel_size' : [2],
        'conv_stride' : [1],
        'maxpool_stride' : [2],
        'fc1_nodes' : [500],
        'fc2_nodes' : [50],
        'fc3_nodes' : [0],
        # 'conv_config_num': np.array(range(len(user_params['conv_config'])))
        'conv_config_num': [1]
    }

    print("user_params['conv_config']:", len(user_params['conv_config']))

    test_split = 0.2
    if isinstance(fin,list):
        train_data, test_data, train_mean, train_std = get_train_test_split_from_multiple_files(fin, test_split=test_split, random_seed=user_params['random_seed'])
    else:
        train_data, test_data, train_mean, train_std = get_train_test_split(fin, test_split=test_split, random_seed=user_params['random_seed'])
    user_params['train_mean'] = train_mean
    user_params['train_std'] = train_std

    print('train_mean:', train_mean, 'train_std:', train_std)

    hyperopt = HyperParamOptimization(param_grid, train_data, test_data, user_params)
    if n_params: hyperopt.build_random_grid(n_params)
    else: hyperopt.build_grid()
    hyperopt.run(early_stopping_patience=user_params['early_stopping_patience'], fout=fout, custom_params=custom_params)

if __name__ == "__main__":
    # # training models by density levels
    # for density in range(10,70,10):
    #     print('--'*50)
    #     print('# density:', density)
    #     print('--'*50)
    #     run_vggnet(n_params=300, fin="ibm_data_by_density_{}.pkl".format(density), fout='vgg_adam_300_{}.pk'.format(density), density=density, custom_params=__BEST_MODEL_PARAMS__['vgg_adam'], optimizer=optim.Adam)
    #     run_vggnet(n_params=300, fin="ibm_data_by_density_{}.pkl".format(density), fout='vgg_sgd_300_{}.pk'.format(density), density=density, custom_params=__BEST_MODEL_PARAMS__['vgg_sgd'], optimizer=optim.SGD)
    #     run_resnet(n_params=None, fin="ibm_data_by_density_{}.pkl".format(density), fout='res_adam_{}.pk'.format(density), density=density)

    # 'epochs' : 200,
    # 'early_stopping_patience' : 30,
    # files = ["ibm_data_by_density_{}.pkl".format(d) for d in range(10,70,10)]
    # run_resnet(n_params=None, fin=files, fout='res_adam_all.pk', density=0, custom_params=__BEST_MODEL_PARAMS__['res_adam'], optimizer=optim.Adam)

    # 'epochs' : 400,
    # 'early_stopping_patience' : 40,
    files = ["ibm_data_by_density_{}.pkl".format(d) for d in range(10,80,10)]
    run_resnet(n_params=None, fin=files, fout='res_adam_all.pk', density=0, custom_params=__BEST_MODEL_PARAMS__['res_adam'], optimizer=optim.Adam)