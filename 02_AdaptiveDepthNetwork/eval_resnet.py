from __future__ import print_function
from tabnanny import check
from xmlrpc.client import boolean

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import sys
import os
import argparse

import utils_dataload
import timeit

from models.resnet_fpn import ResNet50_ADN_FPN

parser = argparse.ArgumentParser(description='Following arguments are used for the script')
parser.add_argument('--batch_size', default=256, type=int, help='Batch_size')
parser.add_argument('--pretrained', default=None, help='Path of a pretrained model file')
parser.add_argument('--dataset_path', default="/media/data/ILSVRC2012/", help='A path to dataset directory')
parser.add_argument('--skip', action ='store_true', help = "execute base-net")

args = parser.parse_args()

testloader = utils_dataload.get_testdata('ILSVRC2012',args.dataset_path,batch_size=args.batch_size, num_workers=4)

device='cuda'

net = ResNet50_ADN_FPN()
net = net.to(device)

#Eval for models
def evaluation(skip=(False,False,False,False)):
    net.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs, skip=skip)
            
            _, pred = outputs.topk(5, 1, largest=True, sorted=True)

            label_e = targets.view(targets.size(0), -1).expand_as(pred)
            correct = pred.eq(label_e).float()

            correct_top5 += correct[:, :5].sum()
            correct_top1 += correct[:, :1].sum()
            
            total += targets.size(0)
            
    acc_top1 = 100.*correct_top1/total
    acc_top5 = 100.*correct_top5/total

    print(f"Eval_Acc_top1 = {acc_top1:.3f}")
    print(f"Eval_Acc_top5 = {acc_top5:.3f}")

if args.pretrained != None:
    checkpoint = torch.load(args.pretrained)
    #net.load_state_dict(checkpoint['net_state_dict'])

print(args)
if args.skip == True:
    skip = (True, True, True, True)
else:
    skip = (False, False, False, False)  

evaluation(skip = skip)  
