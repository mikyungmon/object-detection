import torch
# import torch.utils.tensorboard


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    bat_size = pred_boxes.size(0)
    num_anchor = pred_boxes.size(1)
    cls_num = pred_cls.size(-1)
    grid_size = pred_boxes.size(2)

    # Output tensors
    # 물체인지 아닌지에 대한 정보를 담을 mask 생성
    obj_mask = torch.zeros(bat_size, num_anchor, cls_num, grid_size, dtype=torch.bool)
    noobj_mask = torch.ones(bat_size, num_anchor, cls_num, grid_size, dtype=torch.bool)
    class_mask = torch.zeros(bat_size, num_anchor, cls_num, grid_size, dtype=torch.float)
    iou_scores = torch.zeros(bat_size, num_anchor, cls_num, grid_size, dtype=torch.float)
    tx = torch.zeros(bat_size, num_anchor, cls_num, grid_size, dtype=torch.float)
    ty = torch.zeros(bat_size, num_anchor, cls_num, grid_size, dtype=torch.float)
    tw = torch.zeros(bat_size, num_anchor, cls_num, grid_size, dtype=torch.float)
    th = torch.zeros(bat_size, num_anchor, cls_num, grid_size, dtype=torch.float)
    tcls = torch.zeros(bat_size, num_anchor, cls_num, grid_size, dtype=torch.float)

    # ground truth 상자를 상대적인 위치로 변환
    target_boxes = target[:, 2:6] * grid_size   # target 이 1*1로 생성되고 이를 그리드 크기에 맞게 변화시킴
    gxy = target_boxes[:, :2]   # ground truth 의 x,y
    gwh = target_boxes[:, 2:]   # ground truth 의 w,h

    # 가장 높은 iou 가진 anchor 얻기
    # 만약 ground truth 박스가 4개라고 했을 때 1번 앵커박스랑 iou 한 값 4개,2번이랑 한거 4개 ~~ 해서 총 12개가 나옴
    # 그럼 열방향으로 제일 큰 iou 값을 갖는 인덱스를 뽑음 -> ex.ground truth 박스 1번은 앵커 3번과 모양이 제일 비슷하다 이런 의미
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    _, best_ious_idx = ious.max(0)  # 열방향으로 제일 큰 인덱스 값이 들어감

    # target 값 분리
    b, target_labels = target[:, :2].long().t()   # b는 배치 사이즈, target_label 는 num_anchors?
    gx, gy = gxy.t()   # gx랑 gy를 분리
    gw, gh = gwh.t()   # gw랑 gh를 분리
    gi, gj = gxy.long().t()   # gx,gy를 소숫점말고 정수로 만들기 위해 long 사용(물체의 실제 중심좌표)

    # Set masks
    obj_mask[b, best_ious_idx, gj, gi] = 1  # 물체가 있는 곳을 1로 만들어줌. 만약 best_ious_idx가 3이고 실제 물체좌표가 0,0이면 obj_mask[3,0,0]을 1로?
    noobj_mask[b, best_ious_idx, gj, gi] = 0  # 물체가 있는 곳을 0으로 만들어줌

    # ignore 임계값을 초과하는 경우 noobj 마스크를 0으로 설정 -> 물체가 있다고 판단
    for i, anchor_ious in enumerate(ious.t()):
        # 물체가 없다고 판단했는데 ground truth랑 iou계산했을 때 임계값 넘으면 물체 있다고 판단하는거? 질문
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0 

    # 좌표(ground truth 의 변화량). grid cell 에서의 위치 계산
    # [b, best_ious_idx, gj, gi]을 어떻게 해석해야할지? 가장 높은 iou값을 갖으며 중심좌표가 gj,gi인 앵커박스?
    tx[b, best_ious_idx, gj, gi] = gx - gx.floor()  # .floor()는 소수점 아래는 버린다는 의미 -> 따라서 계산하면 소수점 아래 값만 남음
    ty[b, best_ious_idx, gj, gi] = gy - gy.floor()  # 가장 높은 iou값을 갖으며 중심좌표가 gj,gi인 앵커박스의 실제 앵커박스 중심 y좌표

    # Width and height
    tw[b, best_ious_idx, gj, gi] = torch.log(gw / anchors[best_ious_idx][:, 0] + 1e-16)
    th[b, best_ious_idx, gj, gi] = torch.log(gh / anchors[best_ious_idx][:, 1] + 1e-16)

    # One-hot encoding of label
    tcls[b, best_ious_idx, gj, gi, target_labels] = 1

    # 최고 anchor 에서 label 정확성 및 iou 계산
    # 내가 예측한 class 값과 target 의 라벨값이 같은지 판단
    class_mask[b, best_ious_idx, gj, gi] = (pred_cls[b, best_ious_idx, gj, gi].argmax(-1) == target_labels).float()
    # 내가 예측한 박스와 target 박스의 iou 값을 구함(비슷할수록 iou score 가 높음)
    iou_scores[b, best_ious_idx, gj, gi] = bbox_iou(pred_boxes[b, best_ious_idx, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf


def bbox_wh_iou(wh1, wh2):   # 만약 물체가 정사각형 모양이라고 했을 때 3개의 앵커박스가 있고 그 3개 중 정사각형이 맞다라고 판단하기 위해?
    wh2 = wh2.t()   # wh1과 곱하기 또는 더하기 할 때 계산 편리하게 하려고 전치
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """Returns the IoU of two bounding boxes."""
    if not x1y1x2y2:
        # 중심과 너비에서 정확한 좌표로 변환(좌표값이 xywz로 되어있다면 x1y1 x2y2로 변경)
        # iou 구하려면 좌표형식을 x,y,w,h가 아닌 x1y1,x2y2로 바꿔야하는데 그걸 바꿔주는 것
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # bounding boxes 의 좌표 얻기  # 이미 x1,y1 x2,y2로 되어있는 상황
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # 교차된 박스의 좌표 얻기
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)  # y 좌표는 0,0을(그리드셀 좌표) 기준으로 하기 때문에 더 밑에 위치한게 큰 값
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # 겹치는 부분(교집합)
    # torch.clamp는  min 혹은 max 의 범주에 해당하도록 값 변경해주는 것
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )  # 1을 더하는 이유: 0부터 시작하는 픽셀 좌표를 보완하기 위함이라는 말도 있고, 0으로 나눠지는 것을 막기 위함이라는 말도 있는데 둘 중 뭔지 모름

    # 전체 부분(union area)
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


# 예측한 박스들을 score가 높은 순으로 정렬 후 가장 높은 박스와 iou가 일정 이상인 박스는 동일한 물체를 detect 했다고 판단해 지움
def NMS(prediction, conf_thres, nms_thres):
    """
    conf_thres보다 낮은 신뢰도 점수를 가진 것들을 제거하고 필터링하기 위해 비 최대 억제(nms)수행.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # (cx, cy, w, h) -> (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]

    # prediction는 배치사이즈가 20이면 shape가 (20,10647,85). 즉, 예를 들어 이미지가 100장이고 배치 사이즈가 20이면 for문은 20번 돌게됨
    # 첫 for문을 돌 때 image_pred는 20개중 이미지 1번에 대한 정보. shape은 (10647,85)가 되는 것
    for image_i, image_pred in enumerate(prediction):  
        # 임계값 미만의 신뢰도 점수(confidence) 필터링. 임계값 미만인거는 삭제
        # confidence 란 bounding box 안에 오브젝트가 있을 확률
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]

        # 모든 predict 이 다되면 다음 이미지로 넘어감.
        if not image_pred.size(0):  # image_pred 0번인덱스 크기가 0이 아니면 밑으로 계속. 0이면 다시 for문 돌기?
            continue                # 즉, c값이 임계값보다 높은 박스가 한 이미지에서 0개 이면 20개 이미지 중 두번째 이미지로 간다는 것

        # Object confidence times class confidence
        # confidence * cls확률 -> 해당 박스가 특정 클래스일 확률 -> 물체가 있는데 그게 고양이 일 확률
        # score 는 해당 박스가 특정 클래스일 확률 중 가장 큰 값을 담고 있음
        # 20장의 이미지 중 이미지 1번에서 만약 제외되고 남은 박스들이 10개라면 score 에는 10개의 계산 값들이 들어가 있음
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]  # max(1)은 행에서 제일 큰 값과 그 인덱스 값 반환. [0]은 값 의미

        # 동일한 클래스에 대해 높은-낮은 score 값 순서로 정렬. argsort는 낮은(작은)값 부터 인덱스값 반환하기 때문에 -score 함(내림차순 위해)
        # score 높->낮 순으로 살아남은 박스 정보 정렬
        image_pred = image_pred[(-score).argsort()]
        # 정렬된 살아남은 박스정보 중 class 값들 중에서 제일 큰 class 확률값(class_prds)과 그 인덱스 class_confs
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        # torch.cat dim=1이기 때문에 행으로 cls확률 제일 높은 것과 그 때 인덱스 값을 합침
        # 예를 들어 shape가 (10,5) (10,1) (10,1) cat하면 shape가 (10,7)됨
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)

        # non-maximum suppression(nms) 수행
        keep_boxes = []
        while detections.size(0):
            # unsqueeze() -> 차원 하나 늘림
            # 첫 번째 상자와 살아남은 상자들의 모든 iou를 계산하고 임계값 보다 크면 1 작으면 0을 반환
            # return 예시 : (0,1,1,0 ~~)
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            # 첫 번째 상자에서 클래스 확률 제일 높은 인덱스와 살아남은 모든 상자 중에 인덱스가 같으면 1 다르면 0 반환
            # return 예시 : (0,1,0,0 ~~)
            label_match = detections[0, -1] == detections[:, -1]
            # 낮은 신뢰도 점수와 큰 iou 를 가진 상자의 인덱싱 및 매칭 label?
            # 첫번째 상자와의 iou 가 임계값보다 크고 인덱스 값을 & 연산함(invalid)
            invalid = large_overlap & label_match
            # weights 는 첫번째 상자와의 iou 값이 임계값을 넘고 인덱스 값도 동일한 박스들의 confidence 값을 의미
            weights = detections[invalid, 4:5]
            # confidence에 따라 겹치는 앵커박스들 합치기
            # detections[0, :4]는 합친 박스의 좌표값
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            # 합쳐서 만든 박스를 keep_boxes에 넣음
            keep_boxes += [detections[0]]
            # 이전에 계산된 것을 제외하고 다음 연산 수행
            detections = detections[~invalid]
        # 만일 keep_boxes가 None이 아닌 경우 스택의 모든 keep_box를 output 에 저장
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output

# output 형태(batch_size,pred_boxed_num,7)
# 7은 x,y,w,h, conf, class_conf,class_pred
# pre_boxes_num : 이미지에 있는 앵커박스 갯수

