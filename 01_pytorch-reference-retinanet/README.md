# 01_pytorch-reference-retinanet/

* train
    ```
    torchrun --nproc_per_node=4 train.py \
    --dataset coco --data-path=/media/data/coco  \
    --model retinanet_resnet50_fpn --epochs 26 --batch-size 4 --workers 8  \
    --lr-steps 16 22 --aspect-ratio-group-factor 3 --lr 0.01 --weights-backbone ResNet50_Weights.IMAGENET1K_V1 \
    2>&1 | tee ./train_log.txt
    ```