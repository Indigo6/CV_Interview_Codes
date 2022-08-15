import math
import torch

from torch import nn

"""
    https://zhuanlan.zhihu.com/p/270663039
    IOU Loss: 考虑了重叠面积，归一化坐标尺度;
    GIOU Loss: 考虑了重叠面积，基于IOU解决边界框不相交时loss等于0的问题;
    DIOU Loss: 考虑了重叠面积和中心点距离，基于IOU解决GIOU收敛慢的问题;
    CIOU Loss: 考虑了重叠面积、中心点距离、纵横比，基于DIOU提升回归精确度;
    EIOU Loss: 考虑了重叠面积，中心点距离、长宽边长真实差，基于CIOU解决了纵横比的模糊定义，并添加Focal Loss解决BBox回归中的样本不平衡问题。
"""


class IoULoss(nn.Module):
    """IoU loss.
    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.
    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        torch.Tensor: Loss tensor.
    """
    def __init__(self, eps=1e-9) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred_w,pred_h = pred[..., 2] - pred[..., 0], pred[..., 3] - pred[..., 1]
        target_w,target_h = target[..., 2] - target[..., 0], target[..., 3] - target[..., 1]
        pred_area = pred_w * pred_h
        target_area = target_w * target_h

        inter_lt = torch.max(pred[..., :2], target[..., :2])
        inter_rb = torch.min(pred[..., 2:], target[..., 2:])
        inter_wh = (inter_rb - inter_lt).clamp(min=0)
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]

        ious = inter_area / (pred_area + target_area - inter_area + self.eps)
        loss = 1-ious
        return loss


class GIoULoss(nn.Module):
    """GIoU loss.
    Computing the GIoU loss between a set of predicted bboxes and target bboxes.
    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        torch.Tensor: Loss tensor.
    """
    def __init__(self, eps=1e-9) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred_w,pred_h = pred[..., 2] - pred[..., 0], pred[..., 3] - pred[..., 1]
        target_w,target_h = target[..., 2] - target[..., 0], target[..., 3] - target[..., 1]
        pred_area = pred_w * pred_h
        target_area = target_w * target_h
        union_area = pred_area + target_area

        inter_lt = torch.max(pred[..., :2], target[..., :2])
        inter_rb = torch.min(pred[..., 2:], target[..., 2:])
        inter_wh = (inter_rb - inter_lt).clamp(min=0)
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]

        ious = inter_area / (union_area - inter_area + self.eps)
        loss = (1-ious).mean()

        enclosed_lt = torch.min(pred[..., :2], target[..., :2])
        enclosed_rb = torch.max(pred[..., 2:], target[..., 2:])
        enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1] + self.eps

        # GIoU
        gious = ious - (enclose_area - union_area) / enclose_area
        loss = 1 - gious

        return loss


class  DIoULoss(nn.Module):
    """CIoU loss.
    Computing the DIoU loss between a set of predicted bboxes and target bboxes.
    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        torch.Tensor: Loss tensor.
    """
    def __init__(self, eps=1e-9) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred_w,pred_h = pred[..., 2] - pred[..., 0], pred[..., 3] - pred[..., 1]
        target_w,target_h = target[..., 2] - target[..., 0], target[..., 3] - target[..., 1]
        pred_area = pred_w * pred_h
        target_area = target_w * target_h

        inter_lt = torch.max(pred[..., :2], target[..., :2])
        inter_rb = torch.min(pred[..., 2:], target[..., 2:])
        inter_wh = (inter_rb - inter_lt).clamp(min=0)
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]

        ious = inter_area / (pred_area + target_area - inter_area + self.eps)

        center_pred = (pred[..., :2] + pred[..., 2:]) / 2
        center_target = (target[..., :2] + target[..., 2:]) / 2
        dist1 = torch.sum((center_pred - center_target) ** 2, dim=-1)
        enclosed_lt = torch.min(pred[..., :2], target[..., :2])
        enclosed_rb = torch.max(pred[..., 2:], target[..., 2:])
        dist2 = torch.sum((enclosed_rb - enclosed_lt) ** 2, dim=-1) + self.eps

        # DIoU
        dious = ious - dist1 / dist2

        loss = (1-dious).mean()
        return loss


class  CIoULoss(nn.Module):
    """CIoU loss.
    Computing the CIoU loss between a set of predicted bboxes and target bboxes.
    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        torch.Tensor: Loss tensor.
    """
    def __init__(self, eps=1e-9) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred_w,pred_h = pred[..., 2] - pred[..., 0], pred[..., 3] - pred[..., 1]
        target_w,target_h = target[..., 2] - target[..., 0], target[..., 3] - target[..., 1]
        pred_area = pred_w * pred_h
        target_area = target_w * target_h

        inter_lt = torch.max(pred[..., :2], target[..., :2])
        inter_rb = torch.min(pred[..., 2:], target[..., 2:])
        inter_wh = (inter_rb - inter_lt).clamp(min=0)
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]

        ious = inter_area / (pred_area + target_area - inter_area + self.eps)

        center_pred = (pred[..., :2] + pred[..., 2:]) / 2
        center_target = (target[..., :2] + target[..., 2:]) / 2
        dist1 = torch.sum((center_pred - center_target) ** 2, dim=-1)
        enclosed_lt = torch.min(pred[..., :2], target[..., :2])
        enclosed_rb = torch.max(pred[..., 2:], target[..., 2:])
        dist2 = torch.sum((enclosed_rb - enclosed_lt) ** 2, dim=-1) + self.eps

        factor = 4 / math.pi**2
        v = factor * torch.pow(torch.atan(target_w / (target_h+self.eps)) - torch.atan(pred_w / (pred_h+self.eps)), 2)

        with torch.no_grad():
            alpha = (ious > 0.5).float() * v / (1 - ious + v)

        # CIoU
        cious = ious - (dist1 / dist2 + alpha * v)

        loss = 1-cious
        return loss


class  EIoULoss(nn.Module):
    """EIoU loss.
    Computing the EIoU loss between a set of predicted bboxes and target bboxes.
    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        torch.Tensor: Loss tensor.
    """
    def __init__(self, eps=1e-9) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred_w,pred_h = pred[..., 2] - pred[..., 0], pred[..., 3] - pred[..., 1]
        target_w,target_h = target[..., 2] - target[..., 0], target[..., 3] - target[..., 1]
        pred_area = pred_w * pred_h
        target_area = target_w * target_h

        inter_lt = torch.max(pred[..., :2], target[..., :2])
        inter_rb = torch.min(pred[..., 2:], target[..., 2:])
        inter_wh = (inter_rb - inter_lt).clamp(min=0)
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]

        ious = inter_area / (pred_area + target_area - inter_area + self.eps)

        center_pred = (pred[..., :2] + pred[..., 2:]) / 2
        center_target = (target[..., :2] + target[..., 2:]) / 2
        rho_bbox = torch.sum((center_pred - center_target) ** 2, dim=-1)
        enclosed_lt = torch.min(pred[..., :2], target[..., :2])
        enclosed_rb = torch.max(pred[..., 2:], target[..., 2:])
        c_bbox = torch.sum((enclosed_rb - enclosed_lt) ** 2, dim=-1) + self.eps

        rho_w = (pred_w - target_w) ** 2
        c_w = (enclosed_rb[..., 0] - enclosed_lt[..., 0]) ** 2 + self.eps
        rho_h = (pred_h - target_h) ** 2
        c_h = (enclosed_rb[..., 1] - enclosed_lt[..., 1]) ** 2 + self.eps

        # EIoU
        eious = ious - rho_bbox / c_bbox - rho_w/c_w - rho_h/c_h
        loss = 1-eious
        return loss

if __name__ == "__main__":
    pred_bboxes = torch.tensor([[20, 30, 80, 90, 0.7], 
                                [50, 50, 140, 210, 0.6], 
                                [20, 30, 70, 100, 0.8], 
                                [200, 200, 400, 400, 0.6]])
    gt_bboxes = torch.tensor([[40, 30, 100, 90, 0.7], 
                              [50, 50, 140, 210, 0.6], 
                              [80, 120, 170, 200, 0.8], 
                              [250, 250, 350, 350, 0.6]])
    for loss in [IoULoss(), GIoULoss(), CIoULoss(), DIoULoss(), EIoULoss()]:
        print(loss(pred_bboxes[:, :4], gt_bboxes[:, :4]))