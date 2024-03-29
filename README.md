# INU_RISE

* This repository is dedicated to researching and developing efficient techniques for the object detection task.

</br>

# Requirements

* All experiments are conducted using identical hardware and software resources.
   * **HW** : 
     * **GPU** : NVIDIA GeForce RTX 4090 * 4
   * **SW** :
      * **Ubuntu** : 22.04
      * **python** : 3.11.7
      * **cuda** : 12.1
      * **pytorch** : 2.2.1

</br>

## Plan & Description of the Project
- [X] The key papers that have influenced the topic selection are as follows.
   * [He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.](https://arxiv.org/abs/1512.03385)
   * [Lin, Tsung-Yi, et al. "Feature pyramid networks for object detection." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.](https://arxiv.org/abs/1612.03144)
   * [Lin, Tsung-Yi, et al. "Focal loss for dense object detection." Proceedings of the IEEE international conference on computer vision. 2017.](https://arxiv.org/abs/1708.02002)
   * [Kang, Woochul. "Adaptive Depth Networks with Skippable Sub-Paths." arXiv preprint arXiv:2312.16392 (2023).](https://arxiv.org/abs/2312.16392)


- [X] **`01_PyTorch_RetinaNet/`** : 
</br>I will use RetinaNet for base detector, which has good accessibility for research and development.
</br>([PyTorch provides the reference model RetinaNet for research purposes](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py), making code management easy and convenient.)
   * **./custom_eval_from_train/eval_log.txt** : </br>
   A result of pytorch reference model(resnet50_fpn_retinanet) trained on COCO dataset.
   ![](/images/01_reference_resnet50_fpn_retinanet_result.png)
   


</br>

- [X] **`02_AdaptiveDepthNetwork/`** : 
</br>We will evaluate the performance by applying the adaptive depth network to the RetinaNet backbone network, which is resnet50_fpn.
   * **Things to develop and modify :**
     * Train the backbone network (ResNet50-FPN) using skip-aware self-distillation.
     * switchable BNs
     * Add a skip argument to the forward function of the evaluation.
     * Modify the lateral path of the FPN appropriately for the skipped network.
   * **ResNet50-ADN training result(model_145.pth) :** 
      | Model name                | Acc@1  | Acc@5 |  FLOPs   |
      | ------------------------- |------------- | ----------- | -------- |
      | ResNet50-ADN (super-net) |     76.914%   |   93.438%     |   4.11G  |
      | ResNet50-ADN (base-net)   |     75.446%   |   92.896%     |   2.58G  |    

- [ ] **`03_RetinaNet_with_ResNet50-ADN_backbone/`** :
</br>Replace the backbone of RetinaNet in "01_PyTorch_RetinaNet/" with ResNet50-ADN (base-net) from "02_AdaptiveDepthNetwork/".
  - log_only_super(equal to original ResNet50-FPN but with pretrained ADN weigths) : 
  
  - log_only_base : 
  - log_super_base : 

</br>

- [ ] 
**`04_?(Not yet planned)?/` :**</br>
Developing a new technique to improve performance while reducing the size of the RetinaNet model, achieving either the same performance with increased speed or a slight decrease in speed with performance enhancement.