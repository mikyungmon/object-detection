import torch
import torch.nn as nn
import torch.utils.tensorboard

import utils


class DBL(nn.Module):  # torch에 있는 module이라는걸 사용하기 위해 nn.module작성. Pytorch는 nn.module이라는
                       # class를 제공하여 사용자가 이 위에서 자신이 필요로 하는 model architecture를 구현할 수 있도록 함
    def __init__(self, input_ch, output_ch, kernel_size, strides, padding):
        super(DBL, self).__init__()  # nn.module을 실행시키는데 필요
        self.conv = nn.Conv2d(in_channels=input_ch,
                              out_channels=output_ch,
                              kernel_size=kernel_size,
                              stride=strides,
                              padding=padding,
                              bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=output_ch)  # C from an expected input of size (N, C, H, W)
        self.relu = nn.LeakyReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class ResUnit(nn.Module):
    def __init__(self, input_ch, output_ch, double_out_ch):
        super(ResUnit, self).__init__()  # nn.module을 실행시키는데 필요
        self.darkDBL1 = DBL(input_ch=input_ch, output_ch=output_ch, kernel_size=1, strides=1, padding=0)
        self.darkDBL2 = DBL(input_ch=output_ch, output_ch=double_out_ch, kernel_size=3, strides=1, padding=1)

    def forward(self, inputs):
        x = self.darkDBL1(inputs)
        x = self.darkDBL2(x)
        x = torch.add(inputs, x)
        return x


class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()  # nn.module을 실행시키는데 필요
        self.conv1 = DBL(input_ch=3, output_ch=32, kernel_size=3, strides=1, padding=1)
        self.conv2 = DBL(input_ch=32, output_ch=64, kernel_size=3, strides=2, padding=1)

        # res1
        self.res1 = ResUnit(input_ch=64, output_ch=32, double_out_ch=64)

        self.conv5 = DBL(input_ch=64, output_ch=128, kernel_size=3, strides=2, padding=1)

        # res2
        self.res2 = ResUnit(input_ch=128, output_ch=64, double_out_ch=128)

        self.conv10 = DBL(input_ch=128, output_ch=256, kernel_size=3, strides=2, padding=1)

        # res8
        self.res3 = ResUnit(input_ch=256, output_ch=128, double_out_ch=256)

        self.conv27 = DBL(input_ch=256, output_ch=512, kernel_size=3, strides=2, padding=1)

        # res8
        self.res4 = ResUnit(input_ch=512, output_ch=256, double_out_ch=512)

        self.conv44 = DBL(input_ch=512, output_ch=1024, kernel_size=3, strides=2, padding=1)

        # res4
        self.res5 = ResUnit(input_ch=1024, output_ch=512, double_out_ch=1024)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.conv5(x)

        for i in range(2):
            x = self.res2(x)

        x = self.conv10(x)

        for i in range(8):
            x = self.res3(x)

        y3 = x
        x = self.conv27(x)

        for i in range(8):
            x = self.res4(x)

        y2 = x
        x = self.conv44(x)

        for i in range(4):
            x = self.res5(x)
        y1 = x

        return y1, y2, y3  # y1 : 13*13*1024 , y2: 26*26*512, y3: 52*52*256


class DBL5(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(DBL5, self).__init__()
        self.d_1 = DBL(input_ch=input_ch, output_ch=input_ch, kernel_size=1, strides=1, padding=0)
        self.d_2 = DBL(input_ch=input_ch, output_ch=input_ch, kernel_size=3, strides=1, padding=1)
        self.d_3 = DBL(input_ch=input_ch, output_ch=input_ch, kernel_size=1, strides=1, padding=0)
        self.d_4 = DBL(input_ch=input_ch, output_ch=input_ch, kernel_size=3, strides=1, padding=1)
        self.d_5 = DBL(input_ch=input_ch, output_ch=output_ch, kernel_size=1, strides=1, padding=0)

    def forward(self, inputs):
        x = self.d_1(inputs)
        x = self.d_2(x)
        x = self.d_3(x)
        x = self.d_4(x)
        x = self.d_5(x)
        return x


class Small(nn.Module):
    def __init__(self):
        super(Small, self).__init__()
        self.d5 = DBL5(input_ch=1024, output_ch=512)  # 13*13*512
        self.dbl1 = DBL(input_ch=512, output_ch=1024, kernel_size=3, strides=1, padding=1)  # 13*13*1024
        self.conv = nn.Conv2d(in_channels=1024, out_channels=255, kernel_size=1, stride=1, padding=0)  # 13*13*255

    def forward(self, inputs):
        small = self.d5(inputs)
        s1 = small
        small = self.dbl1(small)
        small = self.conv(small)
        return s1, small  # s1: 13*13*512, small : 13*13*255


class Medium(nn.Module):
    def __init__(self):
        super(Medium, self).__init__()
        self.dbl = DBL(input_ch=512, output_ch=256, kernel_size=1, strides=1, padding=0)  # 13*13*256
        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')  # 크기를 2배로 늘림 # 26*26*256
        self.dbl5 = DBL5(input_ch=768, output_ch=256)  # 26*26*(256+512) in , 26*26*256 out
        self.after_dbl = DBL(input_ch=256, output_ch=512, kernel_size=3, strides=1, padding=1)  # 26*26*512
        self.conv = nn.Conv2d(in_channels=512, out_channels=255, kernel_size=1, stride=1, padding=0)  # 26*26*255

    def forward(self, s1, y2):
        medium = self.dbl(s1)
        medium = self.up_sample(medium)  # 26*26*256
        # print("medium_shape:", medium.shape)
        medium = torch.cat((medium, y2), dim=1)  # dim=1 열로 합치기
        medium = self.dbl5(medium)
        m1 = medium
        medium = self.after_dbl(medium)
        medium = self.conv(medium)

        return m1, medium


class Large(nn.Module):
    def __init__(self):
        super(Large, self).__init__()
        self.dbl = DBL(input_ch=256, output_ch=128, kernel_size=1, strides=1, padding=0)  # in 26*26*256 out 26*26*128
        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')  # 52*52*128
        self.dbl5 = DBL5(input_ch=384, output_ch=128)  # in 52*52*(128+256) out 52*52*128
        self.after_dbl = DBL(input_ch=128, output_ch=256, kernel_size=3, strides=1, padding=1)  # 52*52*256
        self.conv = nn.Conv2d(in_channels=256, out_channels=255, kernel_size=1, stride=1, padding=0)  # 52*52*255

    def forward(self, m1, y3):
        large = self.dbl(m1)
        large = self.up_sample(large)
        large = torch.cat((large, y3), dim=1)
        large = self.dbl5(large)
        large = self.after_dbl(large)
        large = self.conv(large)

        return large


def main():
    inputs = torch.randn(1, 3, 416, 416)
    y1, y2, y3 = Darknet53().forward(inputs)

    # 1번 feature뽑기
    s1, feature1 = Small().forward(y1)
    print("feature1:", feature1.shape)
    # print(y2.shape)

    # 2번 feature뽑기
    m1, feature2 = Medium().forward(s1, y2)
    print("feature2:", feature2.shape)

    # 3번 feature뽑기
    feature3 = Large().forward(m1, y3)
    print("feature3:", feature3.shape)


class YOLODetection(nn.Module):
    def __init__(self, anchors, image_size, num_classes):
        super(YOLODetection, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.image_size = image_size
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.ignore_thres = 0.5
        self.obj_scale = 1
        self.no_obj_scale = 100

    def forward(self, x, targets=None):
        batch_size = x.size(0)
        grid_size = x.size(2)

        # 출력값 형태 변환하기
        prediction = (
            x.view(batch_size, self.num_anchors, self.num_classes + 5, grid_size, grid_size).permute(0, 1, 3, 4, 2)
                .contiguous())  # contiguous()는 tensor에서 바로 옆에 있는 요소가 실제로 메모리상에서 서로 인접한 것

        # outputs
        bx = torch.sigmoid(prediction[..., 0])  # Center x   # 앞의 값은 모두 포함하고 맨 뒤에 인덱스는 0번 인덱스만 포함한다는 뜻
        by = torch.sigmoid(prediction[..., 1])  # Center y
        bw = prediction[..., 2]  # Width
        bh = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Object confidence (objectness)
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Class prediction

        # 각 그리드에 맞춰 offsets 계산하기
        stride = self.image_size / grid_size
        cx = torch.arange(grid_size, dtype=torch.float).repeat(grid_size, 1).view(
            [1, 1, grid_size, grid_size])
        # arange 는 주어진 범위 내 정수를 순서대로 생성 , repeat는 dim=0으로 grid size만큼 dim=1로 1만큼 반복 의미
        cy = torch.arange(grid_size, dtype=torch.float).repeat(grid_size, 1).t().view(
            [1, 1, grid_size, grid_size])
        scaled_anchors = torch.as_tensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors],
                                         dtype=torch.float)
        # scaled_anchors 에는 w와 h값 밖에 없는데 왜 굳이 [:,0:1]이라고 써주는지..? 질문
        anchor_w = scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

        # anchors 에 offset 과 scale 추가
        pred_boxes = torch.zeros_like(prediction[..., :4])
        pred_boxes[..., 0] = bx + cx
        pred_boxes[..., 1] = by + cy
        pred_boxes[..., 2] = torch.exp(bw) * anchor_w
        pred_boxes[..., 3] = torch.exp(bh) * anchor_h

        # x,y,w,h와 conf,cls 합치기
        # stride 곱해서 이미지에서 실제 좌표로 만들어주기
        pred = (pred_boxes.view(batch_size, -1, 4) * stride,  # batch_size 가 의미하는건 무엇인지..? 굳이 여기 있는 이유는?
                pred_conf.view(batch_size, -1, 1),
                pred_cls.view(batch_size, -1, self.num_classes))
        output = torch.cat(pred, -1)

        if targets is None:
            return output, 0

        iou_scores, class_mask, obj_mask, no_obj_mask, tx, ty, tw, th, tcls, tconf = utils.build_targets(
            pred_boxes=pred_boxes, pred_cls=pred_cls,
            target=targets, anchors=scaled_anchors
            , ignore_thres=self.ignore_thres)

        # Loss 구하기(존재하지 않는 object를 무시하도록 mask. conf.loss는 제외)
        loss_x = self.mse_loss(bx[obj_mask], tx[obj_mask])
        loss_y = self.mse_loss(by[obj_mask], ty[obj_mask])
        loss_w = self.mse_loss(bw[obj_mask], tw[obj_mask])
        loss_h = self.mse_loss(bh[obj_mask], th[obj_mask])
        loss_bbox = loss_x + loss_y + loss_w + loss_h

        # bounding box안에 물체가 있는지 없는지에 대한 loss
        # 왜 bce loss썼는지?
        loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
        loss_conf_no_obj = self.bce_loss(pred_conf[no_obj_mask], tconf[no_obj_mask])
        # scale 은 패널티 의미. 물체가 없을 때 있다고 하면 더 크게 패널티를 줌
        loss_conf = self.obj_scale * loss_conf_obj + self.no_obj_scale * loss_conf_no_obj

        # class 예측에 대한 loss
        loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])

        loss_layer = loss_bbox + loss_conf + loss_cls

        return output, loss_layer


class YOLOv3(nn.Module):
    def __init__(self, image_size, num_classes):
        super(YOLOv3, self).__init__()
        anchors = {'a1': [(10, 13), (16, 30), (33, 23)],
                   'a2': [(30, 61), (62, 45), (59, 119)],
                   'a3': [(116, 90), (156, 198), (373, 326)]}

        self.image_size = image_size
        self.num_classes = num_classes
        self.darknet53 = Darknet53()

        self.small = Small()
        self.yolo_layer_1 = YOLODetection(anchors['a3'], self.image_size, self.num_classes)

        self.medium = Medium()
        self.yolo_layer_2 = YOLODetection(anchors['a2'], self.image_size, self.num_classes)

        self.large = Large()
        self.yolo_layer_3 = YOLODetection(anchors['a1'], self.image_size, self.num_classes)

        self.yolo_layer = [self.yolo_layer_1, self.yolo_layer_2, self.yolo_layer_3]

    def forward(self, x, targets=None):
        loss = 0

        y1, y2, y3 = Darknet53().forward(x)

        # 1번 feature 뽑기
        s1, feature1 = self.small(y1)
        output_1, loss_1 = self.yolo_layer_1(feature1)
        print("feature1:", feature1.shape)
        # output1 shape: [1, 507, 85]
        # 507 -> 13*13*3 -> feature1 크기가 13*13인데 한 그리드당 앵커박스가 3개.따라서 507은 feature map1에서의 총 앵커박스 갯수
        print("output1:", output_1.shape)
        loss += loss_1

        # 2번 feature 뽑기
        m1, feature2 = self.medium(s1, y2)
        output_2, loss_2 = self.yolo_layer_2(feature2)
        print("feature2:", feature2.shape)
        # output2 shape: [1, 2028, 85]
        # 2028 -> feature2(26*26*255)에서 총 앵커박스 갯수
        print("output2:", output_2.shape)
        loss += loss_2

        # 3번 feature 뽑기
        feature3 = self.large(m1, y3)
        output_3, loss_3 = self.yolo_layer_3(feature3)
        print("feature3:", feature3.shape)
        # output3 shape: [1, 8112, 85]
        # 8112 -> feature3(52*52*255)에서 총 앵커박스 갯수
        print("output3:", output_3.shape)
        loss += loss_3

        yolo_outputs = [output_1, output_2, output_3]
        yolo_outputs = torch.cat(yolo_outputs, 1).detach().cpu()
        print("yolo_outputs:", yolo_outputs.shape)

        return yolo_outputs if targets is None else (loss, yolo_outputs)
