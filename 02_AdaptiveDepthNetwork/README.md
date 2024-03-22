# Adaptive Depth Networks with Skippable Sub-Paths

This is the official implementation of “Adaptive Depth Networks with Skippable Sub-Paths” that is under submission to ICML 2024. 

The training scripts and models are modified using [pytorch reference training codes](https://github.com/pytorch/vision/tree/main/references/classification)

## Requirements
We conducted experiments under
- python 3.10
- pytorch 1.13, torchvision 0.14, cuda12
- Nvidia 3090x4, 4090x4

## Training

To train ResNet50-ADN on ILSVRC2012, run this command:

```train
torchrun --nproc_per_node=4 train_adn.py --model resnet50 --batch-size 128 \
    --lr-scheduler multisteplr --lr-multi-steps 60 100 140 --epochs 150 \
    --norm-weight-decay 0 --bias-weight-decay 0 \
    --output-dir /home/hslee/INU_RISE/02_AdaptiveDepthNetwork/checkpoint \
    --data-path /media/data/ILSVRC2012/ 2>&1 | tee ./logs/log_resnet50_adn.txt
```

To train Mobilenet-V2-ADN on ILSVRC2012, run this command:
```train
 torchrun --nproc_per_node=2 train_adn.py --model mobilenet_v2 --epochs 300 --lr 0.1 --wd 0.00001 --lr-scheduler multisteplr --lr-multi-steps 150 225 285 --batch-size 128 --norm-weight-decay 0 --bias-weight-decay 0 --output-dir <checkpoint directory> --data-path <ILSVRC2012 data path>
```

To train Swin-t-ADN, run this command:

```train
torchrun --nproc_per_node=4 train_adn.py --model swin_t --epochs 300 --batch-size 256 --opt adamw --lr 0.001 --weight-decay 0.05 --norm-weight-decay 0.0  --bias-weight-decay 0.0 --transformer-embedding-decay 0.0 --lr-scheduler cosineannealinglr --lr-min 0.00001 --lr-warmup-method linear  --lr-warmup-epochs 20 --lr-warmup-decay 0.01 --amp --label-smoothing 0.1 --mixup-alpha 0.8 --clip-grad-norm 5.0 --cutmix-alpha 1.0 --random-erase 0.25 --interpolation bicubic --auto-augment ta_wide --model-ema --ra-sampler --ra-reps 4  --val-resize-size 224 --output-dir <checkpoint directory> --data-path <ILSVRC2012 data path>
```

To train Vit-b-16-ADN, run this command:

```train
torchrun --nproc_per_node=4 train_adn.py --model vit_b_16 --epochs 300 --batch-size 192 --opt adamw --lr 0.00056 --wd 0.2 --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 30 --lr-warmup-decay 0.033 --amp --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 --model-ema --output-dir <checkpoint directory> --data-path <ILSVRC2012 data path>
```

## Evaluation

*Before evaluation, download the pretrained model in the links.*

To evaluate **super-net <FFFF>** of ResNet50-ADN, run:
```eval
python train_adn.py --model resnet50 --test-only --resume checkpoint/model_145.pth --batch-size 256 --skip-cfg False False False False  --data-path /media/data/ILSVRC2012
```

To evaluate **base-net <TTTT>** of ResNet50-ADN, run:
```eval
python train_adn.py --model resnet50 --test-only --resume pretrained/checkpoint_resnet50-epoch146.pth --batch-size 256 --skip-cfg True True True True  --data-path <ILSVRC-2012 data path>
```

To evaluate **super-net <FFFF>** of Swin-T-ADN, run:
```eval
 python train_adn.py --model swin_t --test-only --resume pretrained/checkpoint_swin-t-epoch297.pth --batch-size 256 --skip-cfg False False False False --model-ema --interpolation bicubic --data-path <ILSVRC-2012 data path>
```

To evaluate **base-net <TTTT>** of Swin-T-ADN, run:
```eval
 python train_adn.py --model swin_t --test-only --resume pretrained/checkpoint_swin-t-epoch297.pth --batch-size 256 --skip-cfg True True True True --model-ema --interpolation bicubic --data-path <ILSVRC-2012 data path>
```

To evaluate **super-net <FFFF>** of Vit-b-16-ADN, run:
```eval
python train_adn.py --model vit_b_16 --test-only --resume pretrained/checkpoint_vit-b16-epoch190.pth --batch-size 256 --skip-cfg False False False False --model-ema --data-path <ILSVRC-2012 data path>
```

To evaluate **base-net <FFFF>** of Vit-b-16-ADN, run:
```eval
python train_adn.py --model vit_b_16 --test-only --resume pretrained/checkpoint_vit-b16-epoch190.pth --batch-size 256 --skip-cfg True True True True --model-ema --data-path <ILSVRC-2012 data path>
```

## Results and Pretrained models

Our adpative depth networks achieve the following performance on ILSVRC-2012 validation set. 
Download the pretrained model in the link.

| Model name                | Acc@1  | Acc@5 |  FLOPs   |          |
| ------------------------- |------------- | ----------- | -------- | ------- |
| ResNet50-ADN (super-net) |     75.446%   |   92.896%     |   4.11G  |
| ResNet50-ADN (base-net)   |     76.91%   |   93.44%     |   2.58G  |    