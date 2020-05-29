from abc import abstractmethod

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

from model import InteractionNet


def gaussian_nll(x, mean, ln_var, reduce='mean'):
    '''negative log-likelihood of a Gaussian distribution
    '''
    if reduce not in ('sum', 'mean', 'no'):
        raise ValueError

    x_prec = exponential.exp(-ln_var)
    x_diff = x - mean
    x_power = (x_diff * x_diff) * x_prec * -0.5
    loss = (ln_var + math.log(2 * math.pi)) / 2 - x_power
    if reduce == 'sum':
        return torch.sum(loss)
    elif reduce == 'mean':
        return torch.mean(loss)
    else:
        return loss

class InteractionEnsembleNet(InteractionNet):
    '''[paper] Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles
    '''
    def __init__(self, **kw):
        super().__init__(**kw)
        self.mu_out = nn.Linear(self.dense_config[-1], self.out_nodes)
        self.var_out = nn.Linear(self.dense_config[-1], self.out_nodes)

    def make_dense_layers(self, cfg, h_in, in_channels):
        ## dense layers
        in_nodes = in_channels * h_in * h_in
        layers = []
        for _nodes in cfg:
            layers += [nn.Linear(in_nodes, _nodes), self.activation_fn(True)]
            if self.use_dropout: layers += [nn.Dropout(self.prob_drop)]
            in_nodes = _nodes
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.dense_layers(x)
        mean = self.mu_out(x)
        var = F.softplus(self.var_out(x)) + 1e-6
        return mean, var

    def nll(self, x, t):
        mean, var = self(x)
        return gaussian_nll(t, mean, torch.log(var))

def model_test(arch, conv_config=[16, 'M', 32, 'M', 64, 'M'], dense_config=[1000, 100, 100]):
    net = arch(conv_config=conv_config, dense_config=dense_config)
    print(net)
    y = net(torch.randn(1,1,100,100))
    print('output size:', y.size())


class InteractionEnsembleModules:
    def __init__(self, train_loader, valid_loader, hyperparams, optimize_fn, loss_fn, random_seed):
        # self.batch_size = hyperparams['batch_size']
        self.lr = hyperparams['lr']
        self.momentum = hyperparams['momentum'] if 'momentum' in hyperparams else None
        self.prob_drop=hyperparams['prob_drop']

        self.epochs = hyperparams['epochs']
        # self.valid_batch_size = hyperparams['valid_batch_size']
        # self.valid_split = hyperparams['valid_split']
        self.save_model = hyperparams['save_model']
        self.log_interval = hyperparams['log_interval']
        self.model_output = hyperparams['model_output']
        self.weight_decay = hyperparams['weight_decay'] if 'weight_decay' in hyperparams else 0

        self.num_ensembles = hyperparams['num_ensembles'] if 'num_ensembles' in hyperparams else 5

        self.hyperparams = hyperparams

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimize_fn = optimize_fn
        self.loss_fn = loss_fn

        ################################################################
        # set a random seed
        ################################################################
        seed_torch(random_seed)

        ################################################################
        # cuda setting
        ################################################################
        self.use_cuda = hyperparams['use_cuda'] and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
    
    def train_epoch(self, model, device, train_loader, optimizer, epoch, log_interval, loss_fn, verbose=False):
        model.train()
        avg_loss = 0
        num_samples = 0
        total_samples = len(train_loader.dataset)

        for batch_idx, sample in enumerate(train_loader):
            data, target = sample['matrix'].to(device), sample['interaction'].to(device)
            num_samples += len(data)

            optimizer.zero_grad()
            output = model(data)

            # print("output:", output.shape)
            # print("target:", target.shape)

            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if verbose & (batch_idx % log_interval == 0):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, num_samples, total_samples,
                    100. * num_samples / total_samples, loss.item()))
            avg_loss += loss.item()
        avg_loss /= total_samples
        return avg_loss

    def train(self, early_stopping_patience=10, verbose=False):
        model = self.build_model()
        # initialize the train tracer to track the best model
        train_tracer = TrainTracer(early_stopping_patience=early_stopping_patience, verbose=verbose)

        if not model.isfeasible:
            self.best_performance = None
            return
        
        # if verbose: print(model)

        if self.momentum: optimizer = self.optimize_fn(model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        else: optimizer = self.optimize_fn(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # hist_best = 1e6
        # epoch_tracer = []
        # best_model = None
        # best_performance = None
        stime = time.time()
        # for epoch in tqdm(range(1, self.epochs + 1)):
        for epoch in range(1, self.epochs + 1):
            avg_train_loss = self.train_epoch(model, self.device, self.train_loader, optimizer, epoch, self.log_interval, loss_fn=self.loss_fn, verbose=verbose)
            avg_test_loss, avg_accuracy, _ = self.test_epoch(model, self.device, self.valid_loader, loss_fn=self.loss_fn, verbose=verbose)
            
            train_tracer(epoch, avg_train_loss, avg_test_loss, avg_accuracy, model)
            
            if train_tracer.early_stop:
                print("Early stopping")
                break

        print("Finished! {0} {1:.1f}min".format(
            train_tracer.best,
            (time.time()-stime)/60))

        # self.best_performance = dict()
        # self.best_performance['epoch'] = train_tracer.best.epoch
        # self.best_performance['train_loss'] = train_tracer.best.train_loss
        # self.best_performance['val_loss'] = train_tracer.best.val_loss
        # self.best_performance['accuracy'] = train_tracer.best.accuracy
        self.best_performance = train_tracer.best

        if self.save_model:
            train_tracer.save_best(self.model_output+"_{0:d}_{1:.3f}_{2:.3f}_{3:.1f}.pt".format(
                train_tracer.best.epoch, train_tracer.best.train_loss, train_tracer.best.val_loss, train_tracer.best.accuracy))
        
    def build_model(self):
        return [InteractionEnsembleNet(
            conv_config=self.hyperparams['conv_config'],
            dense_config=self.hyperparams['dense_config'],
            conv_kernel_size=self.hyperparams['conv_kernel_size'],
            conv_stride=self.hyperparams['conv_stride'],
            maxpool_kernel_size=self.hyperparams['maxpool_kernel_size'],
            maxpool_stride=self.hyperparams['maxpool_stride'],
            prob_drop=self.prob_drop
        ).to(self.device) for i in range(self.num_ensembles)]

    def test_epoch(self, model, device, test_loader, loss_fn, verbose=False):
        model.eval()
        test_loss = 0
        correct = 0

        total_samples = len(test_loader.sampler)
        
        with torch.no_grad():
            predictions = np.array([])
            for sample in test_loader:
                data, target = sample['matrix'].to(device), sample['interaction'].to(device)
                output = model(data)
                test_loss += loss_fn(output, target, reduction='sum').item() # sum up batch loss
                if predictions.shape[0]>0:
                    predictions = np.append(predictions, output.data.cpu().numpy(), axis=0)
                else:
                    predictions = output.data.cpu().numpy()
                
                out_sign = (output>0)
                tar_sign = (target>0)
                correct += (((out_sign==tar_sign).sum(dim=1))==2).sum().item()
                
        test_loss /= total_samples
        percent = 100. * correct / total_samples
        if verbose: print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, total_samples, percent))
        return test_loss, percent, predictions