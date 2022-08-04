import numpy as np

def nms(bboxes, thresh=0.5):
    """NMS: Non-maximum Suppression
        Args:
            bboxes: [N,4], torch.Tensor
            scores: [N,1], torch.Tensor
    """
    x1 = bboxes[:, 0] # xmin
    y1 = bboxes[:, 1] # ymin
    x2 = bboxes[:, 2] # xmax
    y2 = bboxes[:, 3] # ymax
    scores = bboxes[:, 4] # score
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # 每个boundingbox的面积
    order = scores.argsort()[::-1] # boundingbox的置信度排序
    keep = [] # 用来保存最后留下来的boundingbox
    while True:     
        if len(order)<=1:
            keep.extend(order)
            break
        i = order[0] # 置信度最高的boundingbox的index
        keep.append(i) # 添加本次置信度最高的boundingbox的index
        
        # 当前bbox和剩下bbox之间的交叉区域
        # 选择大于x1,y1和小于x2,y2的区域
        xx1 = np.maximum(x1[i], x1[order[1:]]) #交叉区域的左上角的横坐标
        yy1 = np.maximum(y1[i], y1[order[1:]]) #交叉区域的左上角的纵坐标
        xx2 = np.minimum(x2[i], x2[order[1:]]) #交叉区域右下角的横坐标
        yy2 = np.minimum(y2[i], y2[order[1:]]) #交叉区域右下角的纵坐标
        
        # 当前bbox和其他剩下bbox之间交叉区域的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
        ious = inter / (areas[i] + areas[order[1:]] - inter)
        #保留交集小于一定阈值的boundingbox
        inds = np.where(ious <= thresh)[0]
        order = order[inds + 1]
        
    return keep

def soft_nms(boxes, sigma=0.5, threshold1=0.7, threshold2=0.4, method=1):
    '''
    paper:Improving Object Detection With One Line of Code
    '''
    N = boxes.shape[0]
    pos = 0
    maxscore = 0
    maxpos = 0

    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = boxes[i, 4]

        pos = i + 1
        # 得到评分最高的box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1

        # 交换第i个box和评分最高的box,将评分最高的box放到第i个位置
        boxes[i, 0] = boxes[maxpos, 0]
        boxes[i, 1] = boxes[maxpos, 1]
        boxes[i, 2] = boxes[maxpos, 2]
        boxes[i, 3] = boxes[maxpos, 3]
        boxes[i, 4] = boxes[maxpos, 4]

        boxes[maxpos, 0] = tx1
        boxes[maxpos, 1] = ty1
        boxes[maxpos, 2] = tx2
        boxes[maxpos, 3] = ty2
        boxes[maxpos, 4] = ts

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = boxes[i, 4]

        pos = i + 1
        # softNMS迭代
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    uinon = float((tx2 - tx1 + 1) *
                                  (ty2 - ty1 + 1) + area - iw * ih)
                    iou = iw * ih / uinon  # 计算iou
                    if method == 1:  # 线性更新分数
                        if iou > threshold1:
                            weight = 1 - iou
                        else:
                            weight = 1
                    elif method == 2:  # 高斯权重
                        weight = np.exp(-(iou * iou) / sigma)
                    else:  # 传统 NMS
                        if iou > threshold1:
                            weight = 0
                        else:
                            weight = 1

                    boxes[pos, 4] = weight * boxes[pos, 4]  # 根据和最高分数box的iou来更新分数

                    # 如果box分数太低，舍弃(把他放到最后，同时N-1)
                    if boxes[pos, 4] < threshold2:
                        boxes[pos, 0] = boxes[N - 1, 0]
                        boxes[pos, 1] = boxes[N - 1, 1]
                        boxes[pos, 2] = boxes[N - 1, 2]
                        boxes[pos, 3] = boxes[N - 1, 3]
                        boxes[pos, 4] = boxes[N - 1, 4]
                        N = N - 1  # 注意这里N改变
                        pos = pos - 1

            pos = pos + 1

    keep = [i for i in range(N)]
    return keep


if __name__ == "__main__":
    test_bboxes = np.array([[20, 30, 80, 90, 0.7], 
                            [50, 50, 140, 210, 0.6], 
                            [20, 30, 70, 100, 0.8], 
                            [200, 200, 290, 360, 0.6]])
    print(nms(test_bboxes))
    print(soft_nms(test_bboxes))