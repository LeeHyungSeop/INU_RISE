r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3

Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""
import datetime
import os
import time
import os,sys
import math

import presets
import torch
import torch.utils.data
import torchvision
import utils
from torch import nn
from coco_utils import get_coco
from engine import evaluate, train_one_epoch
from group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler
from torchvision.transforms import InterpolationMode
from transforms import SimpleCopyPaste
import models
import torch.nn.functional as F

def copypaste_collate_fn(batch):
    copypaste = SimpleCopyPaste(blending=True, resize_interpolation=InterpolationMode.BILINEAR)
    return copypaste(*utils.collate_fn(batch))

def get_dataset(is_train, args):
    image_set = "train" if is_train else "val"
    num_classes, mode = {"coco": (91, "instances"), "coco_kp": (2, "person_keypoints")}[args.dataset]
    with_masks = "mask" in args.model
    ds = get_coco(
        root=args.data_path,
        image_set=image_set,
        transforms=get_transform(is_train, args),
        mode=mode,
        use_v2=args.use_v2,
        with_masks=with_masks,
    )
    return ds, num_classes

def get_transform(is_train, args):
    if is_train:
        return presets.DetectionPresetTrain(
            data_augmentation=args.data_augmentation, backend=args.backend, use_v2=args.use_v2
        )
    elif args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()
        return lambda img, target: (trans(img), target)
    else:
        return presets.DetectionPresetEval(backend=args.backend, use_v2=args.use_v2)

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    parser.add_argument("--data-path", default="/datasets01/COCO/022719/", type=str, help="dataset path")
    parser.add_argument(
        "--dataset",
        default="coco",
        type=str,
        help="dataset name. Use coco for object detection and instance segmentation and coco_kp for Keypoint detection",
    )
    parser.add_argument("--model", default="retinanet_resnet50_adn_fpn", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=2, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=26, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument(
        "--lr",
        default=0.02,
        type=float,
        help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu",
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--lr-scheduler", default="multisteplr", type=str, help="name of lr scheduler (default: multisteplr)"
    )
    parser.add_argument(
        "--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)"
    )
    parser.add_argument(
        "--lr-steps",
        default=[16, 22],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
    )
    
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    parser.add_argument("--subpath-alpha", default=0.5, type=float, help="sub-paths distillation alpha (default: 0.5)")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn")
    parser.add_argument(
        "--trainable-backbone-layers", default=None, type=int, help="number of trainable layers of backbone"
    )
    parser.add_argument(
        "--data-augmentation", default="hflip", type=str, help="data augmentation policy (default: hflip)"
    )
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

    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-backbone", default=None, type=str, help="the backbone weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # Use CopyPaste augmentation training parameter
    parser.add_argument(
        "--use-copypaste",
        action="store_true",
        help="Use CopyPaste data augmentation. Works only with data-augmentation='lsj'.",
    )

    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")

    return parser



def train_one_epoch_twobackward(
    model, 
    criterion_kd, 
    optimizer, 
    data_loader, 
    device, 
    epoch, 
    args,  
    scaler=None,
    skip_cfg_basenet=None,
    skip_cfg_supernet=None,
    subpath_alpha=0.5,
    ):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for i, (images, targets) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        alpha = subpath_alpha
        
        optimizer.zero_grad()
        # 1. forward pass for super_net
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_full, detection_full = model(images, targets, skip=skip_cfg_supernet)
            losses_full = alpha * (sum(loss for loss in loss_full.values()))
        
        # reduce losses over all GPUs for logging purposes
        loss_full_reduced = utils.reduce_dict(loss_full)
        losses_full_reduced = sum(loss for loss in loss_full_reduced.values())
        
        loss_value_full = losses_full_reduced.item()
        if not math.isfinite(loss_value_full):
            print(f"Loss is {loss_value_full}, stopping training")
            print(loss_full_reduced)
            sys.exit(1)
            
        if scaler is not None:
            scaler.scale(losses_full).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses_full.backward()
            optimizer.step()

        # 2. forward pass for base_net
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_base, detection_base = model(images, targets, skip=skip_cfg_basenet)
            losses_base = alpha * (sum(loss for loss in loss_base.values()))
        
        # reduce losses over all GPUs for logging purposes
        loss_base_reduced = utils.reduce_dict(loss_base)
        losses_base_reduced = sum(loss for loss in loss_base_reduced.values())
        
        loss_value_base = losses_base_reduced.item()
        if not math.isfinite(loss_value_base):
            print(f"Loss is {loss_value_base}, stopping training")
            print(loss_base_reduced)
            sys.exit(1)
            
        # get softmax KD loss
        T = 4  # temperature
        loss_softmax_kd = loss_softmax_kd = criterion_kd(F.log_softmax(detection_full, dim=1), F.softmax(detection_base.clone().detach()/T, dim=1)) * T*T
                
        # final loss
        loss_kd = (1. - alpha) * loss_softmax_kd

        if scaler is not None:
            scaler.scale(loss_kd).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_kd.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            
        if lr_scheduler is not None:
            lr_scheduler.step()
            
        metric_logger.update(loss=losses_full_reduced, **loss_full_reduced)
        metric_logger.update(loss=losses_base_reduced, **loss_base_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

def main(args):
    if args.backend.lower() == "tv_tensor" and not args.use_v2:
        raise ValueError("Use --use-v2 if you want to use the tv_tensor backend.")
    if args.dataset not in ("coco", "coco_kp"):
        raise ValueError(f"Dataset should be coco or coco_kp, got {args.dataset}")
    if "keypoint" in args.model and args.dataset != "coco_kp":
        raise ValueError("Oops, if you want Keypoint detection, set --dataset coco_kp")
    if args.dataset == "coco_kp" and args.use_v2:
        raise ValueError("KeyPoint detection doesn't support V2 transforms yet")

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    # Data loading code
    print("Loading data")

    dataset, num_classes = get_dataset(is_train=True, args=args)
    dataset_test, _ = get_dataset(is_train=False, args=args)

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    train_collate_fn = utils.collate_fn
    if args.use_copypaste:
        if args.data_augmentation != "lsj":
            raise RuntimeError("SimpleCopyPaste algorithm currently only supports the 'lsj' data augmentation policies")

        train_collate_fn = copypaste_collate_fn

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers, collate_fn=train_collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    print("Creating model")
    kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers}
    if args.data_augmentation in ["multiscale", "lsj"]:
        kwargs["_skip_resize"] = True
    if "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh
    
    # 2024.03.20 hslee
    # have to modify this part to use new model --------------------------------------------------------------------------------------
    # if args.model not in ("retinanet_resnet50_fpn", "swin_t", "vit_b_16", "vit_b_32", "efficientnet_v2_s", "efficientnet_b2"):
    if args.model not in models.__dict__.keys():
        print(f"{args.model} is not supported")
        sys.exit()
    model = models.__dict__[args.model]()
    model.to(device)
    model_without_ddp = model
    print(f"model : {model}")
    
    # --------------------------------------------------------------------------------------------------------------------------------      

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # 2024.03.21 @hslee
    criterion_kd = nn.KLDivLoss(reduction='batchmean')
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        
    

    if args.norm_weight_decay is None:
        parameters = [p for p in model.parameters() if p.requires_grad]
    else:
        param_groups = torchvision.ops._utils.split_normalization_params(model)
        wd_groups = [args.norm_weight_decay, args.weight_decay]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        skip_cfg = args.skip_cfg
        if model_without_ddp.num_skippable_stages != len(skip_cfg):
            print(f"Error: {args.model} has {model_without_ddp.num_skippable_stages} skippable stages!")
            return

    

    # 2024.03.20 @hslee
    num_skippable_stages = model_without_ddp.num_skippable_stages
    print(f"num_skippable_stages : {num_skippable_stages}")
    skip_cfg_basenet = [True for _ in range(num_skippable_stages)]
    skip_cfg_supernet = [False for _ in range(num_skippable_stages)]
    print(f"skip_cfg_basenet : {skip_cfg_basenet}")
    print(f"skip_cfg_supernet : {skip_cfg_supernet}")

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
    
        # 2024.03.20 @hslee
        # train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq, scaler)
        train_one_epoch_twobackward(model, criterion_kd, optimizer, data_loader, device, epoch, args, scaler, skip_cfg_basenet, skip_cfg_supernet, subpath_alpha=args.subpath_alpha)

        lr_scheduler.step()
        
        # evaluate after every epoch
        evaluate(model, data_loader_test, device=device, skip=skip_cfg_basenet)
        evaluate(model, data_loader_test, device=device, skip=skip_cfg_supernet)
        
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)


# ampere
'''
    torchrun --nproc_per_node=4 train_custom.py --dataset coco --data-path=/media/data/coco \
    --model retinanet_resnet50_adn_fpn --epochs 26 \
    --batch-size 4 --workers 8 --lr-steps 16 22 \
    --aspect-ratio-group-factor 3 --lr 0.01 \
    --weights-backbone /home/hslee/INU_RISE/02_AdaptiveDepthNetwork/pretrained/resnet50_adn_model_145.pth \
    2>&1 | tee ./logs/log_train_custom.txt
'''

# Desktop
'''
    torchrun --nproc_per_node=1 train_custom.py --dataset coco --data-path=/home/hslee/Desktop/Datasets/COCO \
    --model retinanet_resnet50_adn_fpn --epochs 26 \
    --batch-size 16 --workers 8 --lr-steps 16 22 \
    --aspect-ratio-group-factor 3 --lr 0.01 \
    --weights-backbone /home/hslee/Desktop/Embedded_AI/INU_4-1/RISE/02_AdaptiveDepthNetwork/pretrained/resnet50_adn_model_145.pth \
    2>&1 | tee ./logs/log_train_custom.txt
'''