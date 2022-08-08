# CV Interview Codes
Minimal implementation codes of the modules that are frequently asked in CV interviews.

## Traditinal Algorithms
### Computational Geometry
1. Point in Polygon (PIP)

### Image Processing
1. [Gaussian Filter](image_processing/filter.py)

## Deep Learning
### Backbones
1. ViT
    + [Patch Embedding](backbone/vit/patch_embed.py)
    + [MultiHead Attention (MHA)](backbone/vit/attention.py)
2. Swin Transformer
    + Window Partition
    + Window-based MHA
    + Shifted Window-based MHA

### Object Detection
1. [IoUs(IoU/GIoU/DIoU/CIoU) Loss](object_detection/iou_loss.py)
2. [NMS](object_detection/nms.py)

### Pose Estimation
1. Associative Embedding
    + loss
    + group

### Generative Network
1. Flow-based Generative Model
    + [RealNVP](generative_model/realnvp.py)
