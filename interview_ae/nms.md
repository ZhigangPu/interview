nms 输入是predict出来的部分box的坐标以及对应的score. 
算法流程:
1. 按照置信度降序排列(取前两百个)
2. 计算boxes之间的IOU, 得到IOU矩阵, 对角线上是1
3. 取出置信度最高的box开始消除重合度过高的box. 具体做法是把该box对应的IOU矩阵的那一行取出来与nms阈值比较,得到相同大小的逻辑矩阵, 然后把大于阈值的IOU全部置为1, 检查刚刚的逻辑矩阵,如果全部是False,则循环结束. 否则找到小于阈值的最大置信度的box继续. 最后返回留下来的idx的值.
``` python
def compute_nms(boxes, scores, nms_threshold, limit=200):
    """ Perform Non Maximum Suppression algorithm
        to eliminate boxes with high overlap

    Args:
        boxes: tensor (num_boxes, 4)
               of format (xmin, ymin, xmax, ymax)
        scores: tensor (num_boxes,)
        nms_threshold: NMS threshold
        limit: maximum number of boxes to keep

    Returns:
        idx: indices of kept boxes
    """
    if boxes.shape[0] == 0:
        return tf.constant([], dtype=tf.int32)
    selected = [0]  # 从第一个框开始看，看其与所有的框的iou情况, 第一个框是score最高的
    idx = tf.argsort(scores, direction='DESCENDING')  # 按照分类得分降序排列
    idx = idx[:limit]  # 只看前limit个元素
    boxes = tf.gather(boxes, idx)  # 把前limit个元素挑出来

    iou = compute_iou(boxes, boxes)  # 计算IOU

    while True:
        row = iou[selected[-1]]
        next_indices = row <= nms_threshold  # 该行所有iou小于阈值的box
        # iou[:, ~next_indices] = 1.0
        iou = tf.where(
            tf.expand_dims(tf.math.logical_not(next_indices), 0),  # 对于大于thres的改成1, 这样在下一轮就不需要考虑
            tf.ones_like(iou, dtype=tf.float32),
            iou)

        if not tf.math.reduce_any(next_indices):  # 如果next_indice里面都是False，则退出循环, 但是应该存在iou比较小的
            break

        selected.append(tf.argsort(
            tf.dtypes.cast(next_indices, tf.int32), direction='DESCENDING')[0].numpy())  # 添加最大的值到select里面

    return tf.gather(idx, selected)
```

``` python
import numpy as np
def nms(dets, thresh):
    """Pure Python NMS baseline
    Args:
        dets: [N, 5]
        thresh: threshold
    Rtns:
        keep: list, idx of box remained.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # [N,]
    order = scores.argsort()[::-1] # 降序排列 [N,]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]) # [len(order[1:]),]
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]]）
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h # [len(order[1:]),]
        ovr = inter / (areas[i] + areas[order[1:]] - inter) # [len(order[1:]),]
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep
```