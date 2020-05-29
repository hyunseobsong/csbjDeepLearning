import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from model import *
from resnet import *
from data_loader import get_test_loader, get_train_test_split, ToTensor2D

class DLInferenceModule:
    def __init__(self, arch, model_file, test_data, hyperparam, verbose):
        ################################################################
        # cuda setting
        ################################################################
        self.use_cuda = hyperparam['use_cuda'] and torch.cuda.is_available()
        self.device = "cuda" if self.use_cuda else "cpu"
        
        ################################################################
        # inference model and test loader
        ################################################################
        self.inference_model = self.load_model(arch, model_file, hyperparam, verbose)
        self.test_loader = get_test_loader(
            ibm_data=test_data,
            to_tensor=hyperparam['to_tensor'],
            batch_size=hyperparam['valid_batch_size'],
            mean_for_normalize=hyperparam['train_mean'],
            std_for_normalize=hyperparam['train_std'],
            use_cuda=self.use_cuda,
            shuffle=False)

    def load_model(self, arch, model_file, hyperparam, verbose):
        param = {
            'conv_config':hyperparam['conv_config'],
            'dense_config':hyperparam['dense_config'],
            'conv_kernel_size':hyperparam['conv_kernel_size'],
            'conv_stride':hyperparam['conv_stride'],
            'maxpool_kernel_size':hyperparam['maxpool_kernel_size'],
            'maxpool_stride':hyperparam['maxpool_stride'],
            'verbose':verbose,
            'device':self.device,
        }
        if 'prob_drop' in hyperparam: param['prob_drop'] = hyperparam['prob_drop']

        device = torch.device(self.device)
        model = arch(**param)
        model.load_state_dict(torch.load(model_file, map_location=device))
        return model
    
    def test_epoch(self, loss_fn=None, verbose=False):
        self.inference_model.eval()
        test_loss = 0
        correct = 0

        total_samples = len(self.test_loader.sampler)
        
        with torch.no_grad():
            predictions = np.array([])
            labels = np.array([])
            for sample in self.test_loader:
                if loss_fn:
                    data, target = sample['matrix'].to(self.device), sample['interaction'].to(self.device)
                    if verbose: print(sample['interaction'])
                else:
                    data = sample['matrix'].to(self.device)
                
                output = self.inference_model(data)

                if loss_fn:
                    test_loss += loss_fn(output, target, reduction='sum').item() # sum up batch loss

                    out_sign = (output>0)
                    tar_sign = (target>0)
                    correct += (((out_sign==tar_sign).sum(dim=1))==2).sum().item()

                if predictions.shape[0]>0:
                    predictions = np.append(predictions, output.data.cpu().numpy(), axis=0)
                    if loss_fn: labels = np.append(labels, target.data.cpu().numpy(), axis=0)
                else:
                    predictions = output.data.cpu().numpy()
                    if loss_fn: labels = target.data.cpu().numpy()
                
        test_loss /= total_samples
        percent = 100. * correct / total_samples
        if verbose: print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, total_samples, percent))
        return test_loss, percent, predictions, labels


def infer(
    arch,
    model_file,
    test_data,
    hyperparam, 
    use_cuda=True,
    to_tensor=ToTensor2D,
    loss_fn=F.mse_loss,
    verbose=False):
    
    if verbose: print('train_mean:', hyperparam['train_mean'], 'train_std:', hyperparam['train_std'])

    hyperparam['to_tensor'] = to_tensor
    hyperparam['use_cuda'] = use_cuda
    hyperparam['valid_batch_size'] = 64

    infer_module = DLInferenceModule(arch, model_file, test_data, hyperparam, verbose)
    return infer_module.test_epoch(loss_fn=loss_fn, verbose=verbose)
