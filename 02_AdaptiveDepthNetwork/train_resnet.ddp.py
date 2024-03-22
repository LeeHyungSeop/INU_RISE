from __future__ import print_function

import presets
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import sys
import os
import utils_dataload
import timeit
import time
import datetime

from torchvision.transforms.functional import InterpolationMode

from models.resnet_fpn import ResNet50_ADN_FPN
from models._utils_resnet import IntermediateLayerGetter

import utils

def train_one_epoch(net, criterion, criterion_kd, optimizer, trainloader, device, alpha, num_layers, epoch):
    net.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    correct_top1_skip = 0
    correct_top5_skip = 0
    total_skip = 0
    
    header = f"Epoch: [{epoch}]"
    # for batch_idx, (inputs, targets) in enumerate(trainloader):
    for batch_idx, (image, target) in enumerate(metric_logger.log_every(trainloader, 100, header)):

        start_time = time.time()
        image, target = image.to(device), target.to(device)
        optimizer.zero_grad()

        # forward pass for super_net
        outputs_full = net(image, skip=list((False for _ in range(num_layers))))

        outputs_full_topK, pred_full = outputs_full['model_out'].topk(500, 1, largest=True, sorted=True)
        label_e = target.view(target.size(0), -1).expand_as(pred_full[:,0:5])
        correct = pred_full[:,0:5].eq(label_e).float()
        correct_top5 += correct[:, :5].sum()
        correct_top1 += correct[:, :1].sum()        
        total += target.size(0)
        
        loss_full_acc = criterion(outputs_full['model_out'], target)

        loss_full = alpha * loss_full_acc  # wchkang... exp..
        loss_full.backward()

        # forward pass for base_net
        outputs_skip = net(image, skip=list((True for _ in range(num_layers))))
        
        _, pred_skip = outputs_skip['model_out'].topk(500, 1, largest=True, sorted=True)
        label_e_skip = target.view(target.size(0), -1).expand_as(pred_skip[:,0:5])
        correct = pred_skip[:,0:5].eq(label_e_skip).float()
        correct_top5_skip += correct[:, :5].sum()
        correct_top1_skip += correct[:, :1].sum()        
        total_skip += target.size(0)

        T = 4  # temperature

        # get feature KD loss
        loss_feature_kd = 0
        for k, _ in outputs_full['features'].items():
            loss_feature_kd += criterion_kd(F.log_softmax(outputs_skip['features'][k]/T, dim=1), F.softmax(outputs_full['features'][k].clone().detach()/T, dim=1)) * T*T
        
        # get softmax KD loss
        outputs_skip_topK = outputs_skip['model_out'].gather(1, pred_full)
        loss_softmax_kd = criterion_kd(F.log_softmax(outputs_skip_topK[:,0:500]/T, dim=1), F.softmax(outputs_full_topK[:,0:500].clone().detach()/T, dim=1)) * T*T
                
        # final loss
        # loss = alpha * loss_full_acc + (1.0 - alpha) * (loss_feature_kd + loss_softmax_kd)
        loss_skip = (1.0 - alpha) * (loss_feature_kd + loss_softmax_kd)

        loss_skip.backward()
        optimizer.step()

        if (batch_idx == 0):
            print(f"loss loss_full_acc loss_feature_kd loss_softmax_kd: {(loss_full + loss_skip):.3f}\t{loss_full_acc:.3f}\t{loss_feature_kd:.3f}\t{loss_softmax_kd:.3f}")
            sys.stdout.flush()

        acc1, acc5 = utils.accuracy(outputs_full['model_out'], target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=(loss_full+loss_skip).item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["loss_full_acc"].update(loss_full_acc.item(), n=batch_size)
        metric_logger.meters["loss_feature_kd"].update(loss_feature_kd.item(), n=batch_size)
        metric_logger.meters["loss_softmax_kd"].update(loss_softmax_kd.item(), n=batch_size)
        # metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

        acc1_skip, acc5_skip = utils.accuracy(outputs_skip['model_out'], target, topk=(1, 5))
        # metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1_skip"].update(acc1_skip.item(), n=batch_size)
        metric_logger.meters["acc5_skip"].update(acc5_skip.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

        sys.stdout.flush()
    

def evaluate(net, criterion, testloader, device, print_freq=100, log_suffix="", skip=[False, False, False, False], wrapped_model=True):
    net.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = f"{log_suffix}:"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(testloader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = net(image, skip=skip)
            if wrapped_model:
                output = output["model_out"]

            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(testloader.dataset, "__len__")
        and len(testloader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg
      
        
def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Following arguments are used for the script')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning Rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='Weight decay')
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument('--batch-size', default=128, type=int, help='Batch_size')
    parser.add_argument('--pretrained', default=None, help='Path of a pretrained model file')
    parser.add_argument('--starting-epoch', default=0, type=int, help='An epoch which model training starts')
    parser.add_argument('--data-path', default="/media/data/ILSVRC2012/", help='A path to dataset directory')
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
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )

    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")

    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")


    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")

    return parser


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_train from {cache_path}")
        dataset, _ = torch.load(cache_path)
    else:
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        ra_magnitude = args.ra_magnitude
        augmix_severity = args.augmix_severity
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            presets.ClassificationPresetTrain(
                crop_size=train_crop_size,
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
                ra_magnitude=ra_magnitude,
                augmix_severity=augmix_severity,
            ),
        )
        if args.cache_dataset:
            print(f"Saving dataset_train to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_test from {cache_path}")
        dataset_test, _ = torch.load(cache_path)
    else:
        if args.weights and args.test_only:
            weights = torchvision.models.get_weight(args.weights)
            preprocessing = weights.transforms()
        else:
            preprocessing = presets.ClassificationPresetEval(
                crop_size=val_crop_size, resize_size=val_resize_size, interpolation=interpolation
            )

        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            preprocessing,
        )
        if args.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args):

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    print(f"args: {args}")

    net = ResNet50_ADN_FPN()
    net = net.to(device)

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        # collate_fn=collate_fn,
    )
    testloader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    if args.distributed and args.sync_bn:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

    criterion = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean')

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    # if args.transformer_embedding_decay is not None:
    #     for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
    #         custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        net,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

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

    best_acc = 0
    best_acc_top5 = 0

    alpha = args.alpha

    print("registering layers for resnet")
    num_layers = 4
    if not args.test_only:
        net = IntermediateLayerGetter(net, ['layer1_skippable', 'layer2_skippable', 'layer3_skippable']) 

    if args.distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = net.module
    else:
        model_without_ddp = net

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if not args.test_only:
            model_without_ddp.model.load_state_dict(checkpoint["model"])
        else:
            model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.starting_epoch = checkpoint["epoch"] + 1

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        evaluate(net, criterion, testloader, device=device, wrapped_model=False)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.starting_epoch, args.epochs):
        start = timeit.default_timer()
        
        train_one_epoch(net, criterion, criterion_kd, optimizer, trainloader, device, alpha, num_layers, epoch)
        lr_scheduler.step()
        print(f"[epoch {epoch}]lr: {lr_scheduler.get_last_lr()}")
       
        # evaluate base-net
        evaluate(net, criterion, testloader, device, 100, "[base-net test]", skip=list((True for _ in range(num_layers))), wrapped_model=True)
        
        # # evaluate super-net
        evaluate(net, criterion, testloader, device, 100, "[super-net test]", skip=list((False for _ in range(num_layers))), wrapped_model=True)
        
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if epoch % 10 == 0 or epoch > 130:
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
