import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
           nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(UpSample, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        )

    def forward(self, x):
        return self.layer(x)


class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()
        # encoder
        self.conv1 = ConvBlock(in_channels=3, out_channels=64)

        self.down1 = DownSample()
        self.conv2 = ConvBlock(in_channels=64, out_channels=128)

        self.down2 = DownSample()
        self.conv3 = ConvBlock(in_channels=128, out_channels=256)

        self.down3 = DownSample()
        self.conv4 = ConvBlock(in_channels=256, out_channels=512)

        self.down4 = DownSample()
        self.conv5 = ConvBlock(in_channels=512, out_channels=1024)

        # decoder
        self.up1 = UpSample(in_channels=1024, out_channels=512)
        self.conv6 = ConvBlock(in_channels=1024, out_channels=512)

        self.up2 = UpSample(in_channels=512, out_channels=256)
        self.conv7 = ConvBlock(in_channels=512, out_channels=256)

        self.up3 = UpSample(in_channels=256, out_channels=128)
        self.conv8 = ConvBlock(in_channels=256, out_channels=128)

        self.up4 = UpSample(in_channels=128, out_channels=64)
        self.conv9 = ConvBlock(in_channels=128, out_channels=64)

        self.conv = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        en_1 = self.conv1(x)
        en_2 = self.conv2(self.down1(en_1))
        en_3 = self.conv3(self.down2(en_2))
        en_4 = self.conv4(self.down3(en_3))
        en_5 = self.conv5(self.down4(en_4))

        de_1 = torch.cat((en_4, self.up1(en_5)), dim=1)
        de_1 = self.conv6(de_1)

        de_2 = torch.cat((en_3, self.up2(de_1)), dim=1)
        de_2 = self.conv7(de_2)

        de_3 = torch.cat((en_2, self.up3(de_2)), dim=1)
        de_3 = self.conv8(de_3)

        de_4 = torch.cat((en_1, self.up4(de_3)), dim=1)
        de_4 = self.conv9(de_4)

        out = self.conv(de_4)

        return out


if __name__ == "__main__":
    import torchvision

    image_size = (128, 128)
    image = torch.randint(high=255, size=(1, 3, 512, 512), dtype=torch.float)
    print(image.shape)

    model = UNet()
    print(model(image).shape)
