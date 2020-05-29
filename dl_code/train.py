from abc import abstractmethod

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import random
import time

import numpy as np
from tqdm import tqdm

from model import InteractionValueNet, InteractionModeNet1D, InteractionModeNet2D
from resnet import *

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class BestModel:
    """docstring for BestModel"""
    def __init__(self, verbose=False):
        self._epoch = -1
        self._train_loss = np.Inf
        self._val_loss = np.Inf
        self._accuracy = 0
        self._verbose = verbose
    
    def update(self, epoch, train_loss, val_loss, accuracy, model):
        if self._val_loss > val_loss:
            if self._verbose:
                print(f'Validation loss decreased ({self._val_loss:.6f} --> {val_loss:.6f}).  Store model ...')
            self._epoch = epoch
            self._train_loss = train_loss
            self._val_loss = val_loss
            self._accuracy = accuracy
            self._model_state = copy.deepcopy(model.state_dict())
            return True
        else:
            return False

    @property
    def score(self): return -self._val_loss
    @property
    def epoch(self): return self._epoch
    @property
    def train_loss(self): return self._train_loss
    @property
    def val_loss(self): return self._val_loss
    @property
    def accuracy(self): return self._accuracy
    @property
    def model_state(self): return self._model_state

    def save_checkpoint(self, fout='checkpoint.pt'):
        torch.save(self._model_state, fout)

    def __repr__(self):
        return {"epoch": self.epoch, "train_loss": self.train_loss, "val_loss": self.val_loss, "accuracy": self.accuracy}

    def __str__(self):
        return 'BestModel(epoch={}, accuracy={}, train_loss={}, val_loss={})'.format(
            self._epoch, self._accuracy, self._train_loss, self._val_loss)


class TrainTracer:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, early_stopping_patience=7, verbose=False, delta=0):
        """
        Args:
            early_stopping_patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.delta = delta

        self._best_model = BestModel(verbose=verbose)

    def __call__(self, epoch, train_loss, val_loss, accuracy, model):
        score = -val_loss
        if (score + self.delta) < self._best_model.score:
            self.counter += 1
            if self.verbose: print(f'EarlyStopping counter: {self.counter} out of {self.early_stopping_patience}')
            if self.counter >= self.early_stopping_patience:
                self.early_stop = True
        else:
            self._best_model.update(epoch, train_loss, val_loss, accuracy, model)
            self.counter = 0
    
    def save_best(self, fout='checkpoint.pt'):
        self._best_model.save_checkpoint(fout)

    @property
    def best(self):
        return self._best_model
    @property
    def best_model_state(self):
        return self._best_model.model_state
    

class DLTrainModule:
    def __init__(self, train_loader, valid_loader, hyperparams, optimize_fn, loss_fn, random_seed):
        # self.batch_size = hyperparams['batch_size']
        self.lr = hyperparams['lr']
        self.momentum = hyperparams['momentum'] if 'momentum' in hyperparams else None
        self.prob_drop= hyperparams['prob_drop'] if 'prob_drop' in hyperparams else 1

        self.epochs = hyperparams['epochs']
        # self.valid_batch_size = hyperparams['valid_batch_size']
        # self.valid_split = hyperparams['valid_split']
        self.save_model = hyperparams['save_model']
        self.log_interval = hyperparams['log_interval']
        self.model_output = hyperparams['model_output']
        self.weight_decay = hyperparams['weight_decay'] if 'weight_decay' in hyperparams else 0

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
        self.device = "cuda" if self.use_cuda else "cpu"
    
    @abstractmethod
    def test_epoch(self):raise NotImplementedError
    @abstractmethod
    def build_model(self):raise NotImplementedError

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
        self.model = self.build_model(verbose=verbose)
        self.total_params = self.model.total_params
        self.total_size = self.model.total_size

        # initialize the train tracer to track the best model
        train_tracer = TrainTracer(early_stopping_patience=early_stopping_patience, verbose=verbose)

        if not self.model.isfeasible:
            self.best_performance = None
            return
        
        # if verbose: print(self.model)

        if self.momentum: optimizer = self.optimize_fn(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        else: optimizer = self.optimize_fn(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # hist_best = 1e6
        # epoch_tracer = []
        # best_model = None
        # best_performance = None
        stime = time.time()
        # for epoch in tqdm(range(1, self.epochs + 1)):
        for epoch in range(1, self.epochs + 1):
            avg_train_loss = self.train_epoch(self.model, self.device, self.train_loader, optimizer, epoch, self.log_interval, loss_fn=self.loss_fn, verbose=verbose)
            avg_test_loss, avg_accuracy, _ = self.test_epoch(self.model, self.device, self.valid_loader, loss_fn=self.loss_fn, verbose=verbose)
            
            train_tracer(epoch, avg_train_loss, avg_test_loss, avg_accuracy, self.model)
            
            if train_tracer.early_stop:
                print("Early stopping")
                break

        print("Finished! {0} {1:.1f}min".format(
            train_tracer.best,
            (time.time()-stime)/60))

        self.best_performance = dict()
        self.best_performance['epoch'] = train_tracer.best.epoch
        self.best_performance['train_loss'] = train_tracer.best.train_loss
        self.best_performance['val_loss'] = train_tracer.best.val_loss
        self.best_performance['accuracy'] = train_tracer.best.accuracy
        # self.best_performance = train_tracer.best
        self.best_model_state = train_tracer.best.model_state

        ## TODO: only highly accurate ones
        if self.save_model & (self.best_performance['accuracy']>87):
            train_tracer.save_best(self.model_output+"_{0:d}_{1:.3f}_{2:.3f}_{3:.1f}.pt".format(
                train_tracer.best.epoch, train_tracer.best.train_loss, train_tracer.best.val_loss, train_tracer.best.accuracy))
    
    def test(self, test_loader, verbose=False):
        self.model.load_state_dict(self.best_model_state)
        test_loss, test_accuracy, _ = self.test_epoch(self.model, self.device, test_loader, loss_fn=self.loss_fn, verbose=verbose)
        self.best_performance['test_loss'] = test_loss
        self.best_performance['test_accuracy'] = test_accuracy


class InteractionValueModel(DLTrainModule):
    def __init__(self, **kw):
        super().__init__(**kw)

    def build_model(self, verbose=False):
        return InteractionValueNet(
            conv_config=self.hyperparams['conv_config'],
            dense_config=self.hyperparams['dense_config'],
            conv_kernel_size=self.hyperparams['conv_kernel_size'],
            conv_stride=self.hyperparams['conv_stride'],
            maxpool_kernel_size=self.hyperparams['maxpool_kernel_size'],
            maxpool_stride=self.hyperparams['maxpool_stride'],
            prob_drop=self.prob_drop, 
            device=self.device,
            verbose=verbose
        )

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

class InteractionResnetModel(DLTrainModule):
    def __init__(self, **kw):
        super().__init__(**kw)

    def build_model(self, verbose=False):
        return InteractionResNet(
            conv_config=self.hyperparams['conv_config'],
            dense_config=self.hyperparams['dense_config'],
            conv_kernel_size=self.hyperparams['conv_kernel_size'],
            conv_stride=self.hyperparams['conv_stride'],
            maxpool_kernel_size=self.hyperparams['maxpool_kernel_size'],
            maxpool_stride=self.hyperparams['maxpool_stride'],
            device=self.device,
            verbose=verbose
        )

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

class InteractionMode1DModel(DLTrainModule):
    def __init__(self, **kw):
        super().__init__(**kw)

    def build_model(self, verbose=False):
        return InteractionModeNet1D(
            conv_config=self.hyperparams['conv_config'],
            dense_config=self.hyperparams['dense_config'],
            conv_kernel_size=self.hyperparams['conv_kernel_size'],
            conv_stride=self.hyperparams['conv_stride'],
            maxpool_kernel_size=self.hyperparams['maxpool_kernel_size'],
            maxpool_stride=self.hyperparams['maxpool_stride'],
            out_nodes=9,
            prob_drop=self.prob_drop, 
            device=self.device,
            verbose=verbose
        )

    def test_epoch(self, model, device, test_loader, loss_fn, verbose=False):
        model.eval()
        test_loss = 0
        correct = 0

        total_samples = len(test_loader)
        
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
                
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                
        test_loss /= total_samples
        percent = 100. * correct / total_samples
        if verbose: print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, total_samples, percent))
        return test_loss, percent, predictions


class InteractionMode2DModel(DLTrainModule):
    def __init__(self, **kw):
        super().__init__(**kw)

    def build_model(self, verbose=False):
        return InteractionModeNet2D(
            conv_config=self.hyperparams['conv_config'],
            dense_config=self.hyperparams['dense_config'],
            conv_kernel_size=self.hyperparams['conv_kernel_size'],
            conv_stride=self.hyperparams['conv_stride'],
            maxpool_kernel_size=self.hyperparams['maxpool_kernel_size'],
            maxpool_stride=self.hyperparams['maxpool_stride'],
            out_nodes=6,
            prob_drop=self.prob_drop, 
            device=self.device,
            verbose=verbose
        )

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
                
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                # correct += (pred.eq(target.view_as(pred)).sum(dim=2)>1).sum().item()
        
        correct /= 2
        
        test_loss /= total_samples
        percent = 100. * correct / total_samples
        if verbose: print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, total_samples, percent))
        return test_loss, percent, predictions
