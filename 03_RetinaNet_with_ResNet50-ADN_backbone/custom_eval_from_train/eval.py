import datetime
import os
import time
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import presets
import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
import utils
from coco_utils import get_coco
from engine import evaluate

import argparse

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
    # elif args.weights and args.test_only:
    #     weights = torchvision.models.get_weight(args.weights)
    #     trans = weights.transforms()
    #     return lambda img, target: (trans(img), target)
    else:
        return presets.DetectionPresetEval(backend=args.backend, use_v2=args.use_v2)

def get_dataset(is_train, args):
    image_set = "train" if is_train else "val"
    num_classes, mode = {"coco": (91, "instances"), "coco_kp": (2, "person_keypoints")}[args.dataset]
    with_masks = "mask" in args.model
    ds = get_coco(
        root="/media/data/coco",
        image_set=image_set,
        transforms=get_transform(is_train, args),
        mode=mode,
        with_masks=with_masks,
    )
    return ds, num_classes

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='PyTorch Detection Testing')
    parser.add_argument(
        "--dataset",
        default="coco",
        type=str,
        help="dataset name. Use coco for object detection and instance segmentation and coco_kp for Keypoint detection",
    )
    parser.add_argument('--nproc_per_node', default=4, type=int, help='number of process per node')
    parser.add_argument('--data-path', default='/media/data/coco', help='path to dataset')
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument("--model", default="retinanet_resnet50_fpn", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=2, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--weights", default='/home/hslee/INU_RISE/01_pytorch-reference-retinanet/train/model_25.pth', type=str, help="the weights enum name to load")
    parser.add_argument('--weights-path', default='/home/hslee/INU_RISE/01_pytorch-reference-retinanet/train/model_25.pth', help='path to weights file')

    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")

    args = parser.parse_args(args=[])
    return parser


def main(args) :
    dataset_test, _ = get_dataset(is_train=False, args=args)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=8, sampler=test_sampler, num_workers=8, collate_fn=utils.collate_fn
    )

    reference_retinanet = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=False)
    checkpoint = torch.load(args.weights_path)

    reference_retinanet.load_state_dict(checkpoint['model'])
    reference_retinanet.to(args.device)

    # engine.py > def evaluate(model, data_loader, device):
    torch.backends.cudnn.deterministic = True
    evaluate(reference_retinanet, data_loader_test, device=args.device)
    

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
    
    
'''
torchrun eval.py --nproc_per_node=4 --data-path /media/data/coco --batch-size 2  --test-only 2>&1 | tee ./eval_log.txt
'''    