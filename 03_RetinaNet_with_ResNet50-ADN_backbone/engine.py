import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
import torch.nn.functional as F

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
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

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

def custom_train_one_epoch_onebackward(
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

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    header = f"Epoch: [{epoch}]"
    for i, (images, targets) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        alpha = subpath_alpha
        
        # 2024.03.23 @hslee
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            ''' 2024.03.23 @hslee
            model() returns following :
                - key : 'classification', value : (loss_cls, cls_logits)
                    loss_cls : scalar tensor
                    cls_logits : tensor of shape (bs, 190323, 91)
                - key : 'bbox_regression', value : (loss_bbox, bbox_reg)
                    loss_bbox : scalar tensor
                    bbox_logits : tensor of shape (bs, 190323, 4)
                    
                print(f"loss_dict_super.keys() : {loss_dict_super.keys()}") 
                    # dict_keys(['classification', 'bbox_regression'])
                print(f"loss_dict_super['classification'][0] : {loss_dict_super['classification'][0]}") 
                    # cls_loss : ['classification'][0]
                print(f"loss_dict_super['classification'][1].shape : {loss_dict_super['classification'][1].shape}") 
                    # cls_logits : ['classification'][1]
                print(f"loss_dict_super['bbox_regression'][0] : {loss_dict_super['bbox_regression'][0]}") 
                    # bbox_loss : ['bbox_regression'][0]
                print(f"loss_dict_super['bbox_regression'][1].shape : {loss_dict_super['bbox_regression'][1].shape}") 
                    # bbox_reg : ['bbox_regression'][1]                    
            '''
            
            # 1. forward pass for super_net
            loss_dict_super = model(images, targets, skip=skip_cfg_supernet) # if training 
            # super_net loss
            losses_super = loss_dict_super['classification'][0] + loss_dict_super['bbox_regression'][0]
            losses_super = alpha * losses_super
            # print(f"losses_super : {losses_super}")
            loss_dict_reduced_super = utils.reduce_dict(loss_dict_super)
            losses_reduced_super = loss_dict_reduced_super['classification'][0] + loss_dict_reduced_super['bbox_regression'][0]
            loss_value_super = losses_reduced_super.item()
            if not math.isfinite(loss_value_super):
                print(f"Loss is {loss_value_super}, stopping training")
                sys.exit(1)
        
        
            # 2. forward pass for base_net
            loss_dict_base = model(images, targets, skip=skip_cfg_basenet) # if training 
            optimizer.zero_grad()
            loss_dict_base = model(images, targets, skip=skip_cfg_basenet) # if training 
            real_losses_base = loss_dict_base['classification'][0] + loss_dict_base['bbox_regression'][0]
            # base_net loss (KL divergence)
            # cls shape  : (bs, 190323, 91)
            # bbox shape : (bs, 190323, 4)
            out_cls_super = loss_dict_super['classification'][1]
            out_cls_base = loss_dict_base['classification'][1]
            out_bbox_super = loss_dict_super['bbox_regression'][1]
            out_bbox_base = loss_dict_base['bbox_regression'][1]
            
            T = 4
            loss_cls_kd = criterion_kd(F.log_softmax(out_cls_base/T, dim=2), F.softmax(out_cls_super.clone().detach()/T, dim=2)) * T*T
            loss_bbox_kd = criterion_kd(F.log_softmax(out_bbox_base/T, dim=2), F.softmax(out_bbox_super.clone().detach()/T, dim=2)) * T*T
            print(f"loss_cls_kd : {loss_cls_kd}")
            print(f"loss_bbox_kd : {loss_bbox_kd}")
            losses_base = loss_cls_kd + loss_bbox_kd
            losses_base = (1 - alpha) * losses_base
            print(f"losses_base : {losses_base}")
            if not math.isfinite(losses_base):
                print(f"Loss is {losses_base}, stopping training")
                # print value of parameter
                print(f"parameter : {model.parameters()}")
                sys.exit(1)
            
            # backward with final loss
            loss = losses_super.item() + losses_base
            print(f"(final) loss : {loss}")
            with torch.cuda.amp.autocast(enabled=False):
                if scaler is not None:
                    scaler.scale(loss).backward()
                else : 
                    loss.backward() 
                    ''' 2023.03.24 @hslee
                    RuntimeError: Expected to mark a variable ready only once. 
                        This error is caused by one of the following reasons: 
                            1) Use of a module parameter outside the `forward` function.
                            2) Reused parameters in multiple reentrant backward passes.
                    '''                    
                    optimizer.step()
                
        if lr_scheduler is not None:
            lr_scheduler.step()
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss_super = losses_super)
        metric_logger.update(real_losses_base = real_losses_base)
        metric_logger.update(KLDiv_loss_base = losses_base)
        metric_logger.update(loss=loss)
        
    return metric_logger
          
def custom_train_one_epoch_twobackward(
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

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    header = f"Epoch: [{epoch}]"
    for i, (images, targets) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        alpha = subpath_alpha
        
        # 2024.03.23 @hslee
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            ''' 2024.03.23 @hslee
            model() returns following :
                - key : 'classification', value : (loss_cls, cls_logits)
                    loss_cls : scalar tensor
                    cls_logits : tensor of shape (N, C, H, W)
                - key : 'bbox_regression', value : (loss_bbox, bbox_reg)
                    loss_cls : scalar tensor
                    cls_logits : tensor of shape (N, C, H, W)
                    
                print(f"loss_dict_super.keys() : {loss_dict_super.keys()}") 
                    # dict_keys(['classification', 'bbox_regression'])
                print(f"loss_dict_super['classification'][0] : {loss_dict_super['classification'][0]}") 
                    # cls_loss : ['classification'][0]
                print(f"loss_dict_super['classification'][1].shape : {loss_dict_super['classification'][1].shape}") 
                    # cls_logits : ['classification'][1]
                print(f"loss_dict_super['bbox_regression'][0] : {loss_dict_super['bbox_regression'][0]}") 
                    # bbox_loss : ['bbox_regression'][0]
                print(f"loss_dict_super['bbox_regression'][1].shape : {loss_dict_super['bbox_regression'][1].shape}") 
                    # bbox_reg : ['bbox_regression'][1]                    
            '''
            
            # 1. forward pass for super_net
            loss_dict_super = model(images, targets, skip=skip_cfg_supernet) # if training 
            # super_net loss
            losses_super = loss_dict_super['classification'][0] + loss_dict_super['bbox_regression'][0]
            out_cls_super = loss_dict_super['classification'][1]
            out_bbox_super = loss_dict_super['bbox_regression'][1]
            losses_super = alpha * losses_super
            # print(f"losses_super : {losses_super}")
            # backward with super_net loss
            
            if not math.isfinite(losses_super):
                print(f"losses_super is {losses_super}, stopping training")
                print("Model Parameters:")
                for param in model.parameters():
                    print(param)
                sys.exit(1)
            with torch.cuda.amp.autocast(enabled=False):
                if scaler is not None:
                    scaler.scale(losses_super).backward()
                else:
                    losses_super.backward()
                    optimizer.step()
        
            # 2. forward pass for base_net
            optimizer.zero_grad()
            loss_dict_base = model(images, targets, skip=skip_cfg_basenet) # if training 
            real_losses_base = loss_dict_base['classification'][0] + loss_dict_base['bbox_regression'][0]
            out_cls_base = loss_dict_base['classification'][1]
            out_bbox_base = loss_dict_base['bbox_regression'][1]
            # print(f"real_losses_base : {real_losses_base}")
            
            # base_net loss (KL divergence)
            T = 4
            # cls shape  : (bs, 190323, 91) -> (bs, 190323 * 91)
            # bbox shape : (bs, 190323, 4)  -> (bs, 190323 * 4)
            out_cls_base = out_cls_base.reshape(args.batch_size, -1)
            out_bbox_base = out_bbox_base.reshape(args.batch_size, -1)
            out_cls_super = out_cls_super.reshape(args.batch_size, -1)
            out_bbox_super = out_bbox_super.reshape(args.batch_size, -1)            

            loss_cls_kd = criterion_kd(F.log_softmax(out_cls_base/T, dim=1), F.softmax(out_cls_super.clone().detach()/T, dim=1)) * T*T
            loss_bbox_kd = criterion_kd(F.log_softmax(out_bbox_base/T, dim=1), F.softmax(out_bbox_super.clone().detach()/T, dim=1)) * T*T
            # print(f"loss_cls_kd : {loss_cls_kd}")
            # print(f"loss_bbox_kd : {loss_bbox_kd}")
            kd_losses_base = loss_cls_kd + loss_bbox_kd
            kd_losses_base = (1. - alpha) * kd_losses_base
            # print(f"kd_losses_base : {kd_losses_base}")
            
            if not math.isfinite(kd_losses_base):
                print(f"kd_losses_base is {kd_losses_base}, stopping training")
                print("Model Parameters:")
                for param in model.parameters():
                    print(param)
                sys.exit(1)
                
            # backward with base_net loss
            with torch.cuda.amp.autocast(enabled=False):
                if scaler is not None:
                    scaler.scale(kd_losses_base).backward()
                else:
                    kd_losses_base.backward()
                    optimizer.step()
                    
        
        if lr_scheduler is not None:
            lr_scheduler.step()
            
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss_super = losses_super)
        metric_logger.update(real_losses_base = real_losses_base)
        metric_logger.update(KLDiv_loss_base = kd_losses_base)
        
    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device, skip=None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"[subnet]{skip} Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images, skip=skip)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
