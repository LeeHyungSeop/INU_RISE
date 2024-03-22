from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.functional as F
import sys
import os
import utils_dataload
import timeit

from models.resnet_fpn import ResNet50_ADN_FPN
from models._utils_resnet import IntermediateLayerGetter


def train_one_epoch(net, criterion, criterion_kd, optimizer, trainloader, device, alpha, epoch):
    net.train()
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    correct_top1_skip = 0
    correct_top5_skip = 0
    total_skip = 0
    
    total_batches = len(trainloader)
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # forward pass for super_net
        outputs_full = net(inputs, skip=(False, False, False, False))

        outputs_full_topK, pred_full = outputs_full['model_out'].topk(500, 1, largest=True, sorted=True)
        label_e = targets.view(targets.size(0), -1).expand_as(pred_full[:,0:5])
        correct = pred_full[:,0:5].eq(label_e).float()
        correct_top5 += correct[:, :5].sum()
        correct_top1 += correct[:, :1].sum()        
        total += targets.size(0)
        
        loss_full_acc = criterion(outputs_full['model_out'], targets)

        # forward pass for base_net
        outputs_skip = net(inputs, skip=(True, True, True, True))
        
        _, pred_skip = outputs_skip['model_out'].topk(500, 1, largest=True, sorted=True)
        label_e_skip = targets.view(targets.size(0), -1).expand_as(pred_skip[:,0:5])
        correct = pred_skip[:,0:5].eq(label_e_skip).float()
        correct_top5_skip += correct[:, :5].sum()
        correct_top1_skip += correct[:, :1].sum()        
        total_skip += targets.size(0)

        T = 4  # temperature

        # get feature KD loss
        loss_feature_kd = 0
        for k, _ in outputs_full['features'].items():
            loss_feature_kd += criterion_kd(F.log_softmax(outputs_skip['features'][k]/T, dim=1), F.softmax(outputs_full['features'][k].clone().detach()/T, dim=1)) * T*T
        
        # get softmax KD loss
        outputs_skip_topK = outputs_skip['model_out'].gather(1, pred_full)
        loss_softmax_kd = criterion_kd(F.log_softmax(outputs_skip_topK[:,0:500]/T, dim=1), F.softmax(outputs_full_topK[:,0:500].clone().detach()/T, dim=1)) * T*T
                
        # final loss
        loss = alpha * loss_full_acc + (1.0 - alpha) * (loss_feature_kd + loss_softmax_kd)

        loss.backward()
        optimizer.step()

        if (batch_idx % 100 == 0):
            progress = batch_idx / total_batches
            print(f"Epoch[{epoch}][{progress * 100:.1f}%] loss loss_full_acc loss_feature_kd loss_softmax_kd: {loss:.3f}\t{loss_full_acc:.3f}\t{loss_feature_kd:.3f}\t{loss_softmax_kd:.3f}")
            sys.stdout.flush()
        
    acc_top1 = 100.*correct_top1/total
    acc_top5 = 100.*correct_top5/total
    print(f"[super] Training_Acc_Top1/5 = {acc_top1:.3f}\t{acc_top5:.3f}")

    acc_top1 = 100.*correct_top1_skip/total_skip
    acc_top5 = 100.*correct_top5_skip/total_skip
    print(f"[base] Training_Acc_Top1/5 = {acc_top1:.3f}\t{acc_top5:.3f}")

def train_one_epoch_two_phases(net, criterion, criterion_kd, optimizer, trainloader, device, alpha, epoch):
    net.train()
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    correct_top1_skip = 0
    correct_top5_skip = 0
    total_skip = 0
    
    total_batches = len(trainloader)

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # forward pass for super_net
        outputs_full = net(inputs, skip=(False, False, False, False))

        outputs_full_topK, pred_full = outputs_full['model_out'].topk(500, 1, largest=True, sorted=True)
        label_e = targets.view(targets.size(0), -1).expand_as(pred_full[:,0:5])
        correct = pred_full[:,0:5].eq(label_e).float()
        correct_top5 += correct[:, :5].sum()
        correct_top1 += correct[:, :1].sum()        
        total += targets.size(0)
        
        loss_full = criterion(outputs_full['model_out'], targets)

        loss_full = alpha * loss_full

        loss_full.backward()

        # forward pass for base_net
        outputs_skip = net(inputs, skip=(True, True, True, True))
        
        _, pred_skip = outputs_skip['model_out'].topk(500, 1, largest=True, sorted=True)
        label_e_skip = targets.view(targets.size(0), -1).expand_as(pred_skip[:,0:5])
        correct = pred_skip[:,0:5].eq(label_e_skip).float()
        correct_top5_skip += correct[:, :5].sum()
        correct_top1_skip += correct[:, :1].sum()        
        total_skip += targets.size(0)

        T = 4  # temperature

        # get feature KD loss
        loss_feature_kd = 0
        for k, _ in outputs_full['features'].items():
            loss_feature_kd += criterion_kd(F.log_softmax(outputs_skip['features'][k]/T, dim=1), F.softmax(outputs_full['features'][k].clone().detach()/T, dim=1)) * T*T
        
        # get softmax KD loss
        outputs_skip_topK = outputs_skip['model_out'].gather(1, pred_full)
        loss_softmax_kd = criterion_kd(F.log_softmax(outputs_skip_topK[:,0:500]/T, dim=1), F.softmax(outputs_full_topK[:,0:500].clone().detach()/T, dim=1)) * T*T
                
        loss_skip = (1.0 - alpha) * (loss_feature_kd + loss_softmax_kd)

        loss_skip.backward()

        optimizer.step()

        if (batch_idx % 100 == 0):
            progress = batch_idx / total_batches
            print(f"Epoch[{epoch}][{progress * 100:.1f}%] loss_full loss_skip loss_feature_kd loss_softmax_kd: {loss_full:.3f}\t{loss_skip:.3f}\t{loss_feature_kd:.3f}\t{loss_softmax_kd:.3f}")
            sys.stdout.flush()
        
    acc_top1 = 100.*correct_top1/total
    acc_top5 = 100.*correct_top5/total
    print(f"[super] Training_Acc_Top1/5 = {acc_top1:.3f}\t{acc_top5:.3f}")

    acc_top1 = 100.*correct_top1_skip/total_skip
    acc_top5 = 100.*correct_top5_skip/total_skip
    print(f"[base] Training_Acc_Top1/5 = {acc_top1:.3f}\t{acc_top5:.3f}")


def evaluate(net, testloader, device, skip):
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
            
    # Save checkpoint.
    acc_top1 = 100.*correct_top1/total
    acc_top5 = 100.*correct_top5/total

    return acc_top1, acc_top5
     
        
def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Following arguments are used for the script')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning Rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='Weight decay')
    parser.add_argument('--batch-size', default=128, type=int, help='Batch_size')
    parser.add_argument('--pretrained', default=None, help='Path of a pretrained model file')
    parser.add_argument('--starting-epoch', default=0, type=int, help='An epoch which model training starts')
    parser.add_argument('--dataset-path', default="/media/data/ILSVRC2012/", help='A path to dataset directory')
    parser.add_argument('--epochs', default=150, type=int, metavar='N', help='number of total epochs to run')

    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr-scheduler", default="multisteplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-multi-steps", nargs="+", default=[60,100,140], type=int, help="multi step milestones")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")

    parser.add_argument('--alpha', default=0.5, type=float, help='hyperparameter for sub-paths specialization')
    
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--two-phases",
        dest="two_phases",
        help="two backwards passes for sub-path distillation",
        action="store_true",
    )

    return parser


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    print(f"decay: {len(decay)}")
    print(f"no decay: {len(no_decay)}")
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def main(args):

    args.visible_device = os.environ.get("CUDA_VISIBLE_DEVICES")
    device='cuda'

    print(f"args: {args}")

    net = ResNet50_ADN_FPN()
    net = net.to(device)

    trainloader = utils_dataload.get_traindata('ILSVRC2012',args.dataset_path,batch_size=args.batch_size,download=True, num_workers=16)
    testloader = utils_dataload.get_testdata('ILSVRC2012',args.dataset_path,batch_size=args.batch_size, num_workers=16)
   
    criterion = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean')

    weight_decay = args.weight_decay
    filter_bias_and_bn = True
    if weight_decay and filter_bias_and_bn:
        parameters = add_weight_decay(net, weight_decay)
        weight_decay = 0.
    else:
        parameters = net.parameters()

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "multisteplr":
        main_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_multi_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    if args.pretrained != None:
        checkpoint = torch.load(args.pretrained)
        net.load_state_dict(checkpoint['net_state_dict'])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.starting_epoch = checkpoint['epoch'] + 1
        print(f"load {args.pretrained}")

    alpha = args.alpha

    print("registering layers for resnet")
    if not args.test_only:
        net = IntermediateLayerGetter(net, ['layer1_skippable', 'layer2_skippable', 'layer3_skippable']) 

    # Wrapper for parallelize training
    class MyDataParallel(nn.DataParallel):
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        net = MyDataParallel(net)

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        skip = (False, False, False, False)
        acc_top1, acc_top5 = evaluate(net, testloader, device, skip)
        print("Test_Acc_Top1/5 = %.3f\t%.3f" % (acc_top1, acc_top5))
        return


    best_acc = 0
    best_acc_top5 = 0
    best_epoch = 0

    for epoch in range(args.starting_epoch, args.epochs):
        start = timeit.default_timer()
        
        print(f"[epoch {epoch}]lr: {lr_scheduler.get_last_lr()}")

        if args.two_phases:
            train_one_epoch_two_phases(net, criterion, criterion_kd, optimizer, trainloader, device, alpha, epoch)
        else:
            train_one_epoch(net, criterion, criterion_kd, optimizer, trainloader, device, alpha, epoch)
        
        lr_scheduler.step()
        stop = timeit.default_timer()
        print(f'Time: {stop - start:.3f} seconds')  

        # evaluate super-net
        skip=(False, False, False, False)
        acc_top1, acc_top5 = evaluate(net.model, testloader, device, skip)
        print("[super-net] Test_Acc_Top1/5 = %.3f\t%.3f" % (acc_top1, acc_top5))

        if acc_top1 > best_acc:
            best_acc = acc_top1
            best_acc_top5 = acc_top5
            best_epoch = epoch
        
        # evaluate base-net
        skip=(True, True, True, True)
        acc_top1, acc_top5 = evaluate(net.model, testloader, device, skip)
        print("[base-net] Test_Acc_Top1/5 = %.3f\t%.3f" % (acc_top1, acc_top5))
        
        state = {
            'net_state_dict': net.model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
        }
        if not os.path.isdir('./checkpoint'):
            os.mkdir('./checkpoint')
        
        torch.save(state, './checkpoint/checkpoint.pth')

        if epoch % 10 == 0:
            torch.save(state, './checkpoint/' + "ILSVRC-ResNet50-ADN-FPN"+ args.visible_device + "-epoch" + str(epoch) + '.pth')

    print(f"Best_Acc_top1 at epoch {best_epoch} = {best_acc:.3f}")
    print(f"Best_Acc_top5 at epoch {best_epoch} = {best_acc_top5:.3f}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)

