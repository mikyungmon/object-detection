import torch
import torch.nn as nn
import torch.nn.functional as F


class DBL(nn.Module):  # torch에 있는 module이라는걸 사용하기 위해 nn.module작성. Pytorch는 nn.module이라는 class를 제공하여 사용자가 이 위에서 자신이 필요로 하는 model architecture를 구현할 수 있도록 함
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


main()
