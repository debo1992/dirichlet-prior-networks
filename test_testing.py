# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 22:20:06 2020

@author: z5217412
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import sys
sys.path.insert(0,r'C:\Users\z5217412\Documents\Thesis\dirichlet-prior-networks')
#sys.path.remove(r'C:\Users\z5217412\Documents\Thesis\dirichlet-prior-networks-master')
from dpn.models import build_model
from dpn.criterions import build_criterion
from dpn.args import add_args
from dpn.constants import DatasetType
from collections import namedtuple
from functional import entropy_from_logits
from functional import Dirichlet
from testSimplexFn import Simplex 
import numpy as np
import matplotlib.pyplot as plt


def train(args, model, criterion, device, train_loader, optimizer, epoch):
    model.train()
    in_domain, out_of_domain = 0, 0

#    dataIn = np.empty([0,2])
#    dataOut = np.empty([0,2])
#    labelsIn = np.array([])
#    labelsOut = np.array([])
#    for batch_idx, samples in enumerate(train_loader):
#        dIn, lIn = samples[DatasetType.InD]
#        dataIn = np.append(dataIn,dIn,axis=0)
#        labelsIn = np.append(labelsIn,lIn)
#        dOut, lOut = samples[DatasetType.OoD]
#        dataOut = np.append(dataOut,dOut,axis=0)
#        labelsOut = np.append(labelsOut,lOut)
    for batch_idx, samples in enumerate(train_loader):
        optimizer.zero_grad()
        def f(dtype):
            data, labels = samples[dtype]
            

            data, labels = data.to(device), labels.to(device)
            net_output = model(data)
#            model.network[0].weight = torch.nn.Parameter(model.network[0].weight)
#            print(net_output['logits'].exp() )
#            net_output['logits'] = net_output['logits'].exp()

            in_domain = (dtype == DatasetType.InD)
            _loss = criterion[dtype](net_output, labels, in_domain=in_domain)
            return _loss

        in_domain_loss = f(DatasetType.InD)
        out_of_domain_loss = f(DatasetType.OoD)
        loss = (in_domain_loss + out_of_domain_loss)
        # OoD samples
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0 and args.log:            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: ind {:.6f} ood {:.6f} net {:6f}'.format(
                epoch, batch_idx * train_loader.batch_size, train_loader.num_samples,
                100. * batch_idx / len(train_loader), in_domain_loss.item(), out_of_domain_loss.item(),
                loss.item()))
            
        in_domain += in_domain_loss
        out_of_domain += out_of_domain_loss
         
    in_domain /= len(train_loader)
    out_of_domain /= len(train_loader)
    return (in_domain+out_of_domain)
    
def test(args, model, criterion, device, test_loader, epoch):
    model.eval()
    in_domain_loss, out_of_domain_loss = 0, 0
    correct = 0
    with torch.no_grad():
        dataIn = np.empty([0,2])
        dataOut = np.empty([0,2])
        labelsIn = np.array([])
        labelsOut = np.array([])
        for batch_idx, samples in enumerate(test_loader):
            dIn, lIn = samples[DatasetType.InD]
            dataIn = np.append(dataIn,dIn,axis=0)
            labelsIn = np.append(labelsIn,lIn)
            dOut, lOut = samples[DatasetType.OoD]
            dataOut = np.append(dataOut,dOut,axis=0)
            labelsOut = np.append(labelsOut,lOut)
            
        for batch_idx, samples in enumerate(test_loader):
#            print(batch_idx)
#            print(samples)
            def f(dtype):
                data, labels = samples[dtype]
                if dtype == DatasetType.InD:
                    data = torch.zeros(3,2)
                data, labels = data.to(device), labels.to(device)
#                plt.scatter(data[:,0],data[:,1])
                
                net_output = model(data)
                in_domain = (dtype == DatasetType.InD)
                _loss = criterion[dtype](net_output, labels, in_domain=in_domain)
#                plot_alpha = net_output['logits']
#                plot_alpha = plot_alpha.detach().numpy()
#                plot_alpha = np.exp(plot_alpha)
#                print(model.network[0].weight)
#                print(model.network[0].bias)
#                print(plot_alpha)
#                predLabels = np.exp(net_output['logits'])
#                alphas, predLabels = predLabels.max(1)
                
                return net_output['logits'], labels, _loss
           

            logits, labels, _in_domain_loss = f(DatasetType.InD)
            in_domain_loss += _in_domain_loss
            pred = logits.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()

            _, _, _out_of_domain_loss = f(DatasetType.OoD)
            logitsOut, _, _out_of_domain_loss = f(DatasetType.OoD)
            out_of_domain_loss += _out_of_domain_loss
            

#            plt.savefig("C:\\Users\\z5217412\\Documents\\Thesis\\dirichlet-prior-networks\\dpn\\Figs\\" + str(IDX)+'.png')
#            plt.show()
                
    in_domain_loss /= len(test_loader)
    out_of_domain_loss /= len(test_loader)

    if args.log:
        print('Epoch {} | Test set: Average loss: ind {:.4f} ood:{:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            epoch,
            in_domain_loss, out_of_domain_loss, correct, test_loader.num_samples,
            100. * correct / test_loader.num_samples))
    return np.exp(logits), (in_domain_loss+out_of_domain_loss)




def inference(logits):
    dirichlet = Dirichlet(logits=logits)
    differential_entropy = dirichlet.differential_entropy()
    mutual_information = dirichlet.mutual_information()
    entropy = entropy_from_logits(logits)
    export =  {
        "entropy": entropy, 
        "mutual_information": mutual_information, 
        "differential_entropy": differential_entropy
    }
    return export

def build_optimizer(args, model):
    return optim.Adam(model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay)

def build_loader(args):
    from dpn.data import dataset
    return dataset[args.dataset](args)

def main(args):
    lr = 1e-2,#[1e-4,3.16e-3,1e-3,3.16e-2,1e-2]
    cnt_lr = 0
    for args.lr in lr:
        sigs = [8,1,2,4,8] 
        cnt = 0
        alpha = torch.empty(len(lr),len(sigs),3)
        for sig in sigs[0:1]:    
            args.sigma = sig
            loader = build_loader(args)
            device = torch.device(args.device)
            model = build_model(args.model)
            model = model.to(device)
            criterion = build_criterion(args)
            optimizer = build_optimizer(args, model)
            lowestLoss = 50
            los = []#np.array(0)
            for epoch in range(1, args.epochs + 1):
                inloss = train(args, model, criterion, device, loader.train, optimizer, epoch)
                alphas,loss = test(args, model, criterion, device, loader.test, epoch)
                los.append(inloss)
                if lowestLoss>(loss):
                    eph = epoch
                    print(eph)
                    lowestLoss = loss
                    alpha[cnt_lr,cnt,:] = alphas[0]
            cnt = cnt + 1
        los = np.array(los)
        plt.plot(los)
        Simplex(alpha[cnt_lr,:,:])
        plt.savefig("C:\\Users\\z5217412\\Documents\\Thesis\\dirichlet-prior-networks\\dpn\\Figs\\testing"+str(cnt_lr)+".png")
        plt.show()
        cnt_lr = cnt_lr + 1
        
    

    return {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "alpha": alpha
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    alpha = main(args)
