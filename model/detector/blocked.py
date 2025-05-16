import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, device, insize, outsize):
        super(ConvBlock, self).__init__()
        self.act =  nn.ReLU().to(device)
        self.conv1 = nn.Conv2d(insize, insize, 3, padding = 1).to(device)
        self.conv2 = nn.Conv2d(insize, insize, 3, padding = 1).to(device)
        self.conv3 = nn.Conv2d(insize, outsize, 3, padding = 1).to(device)
        self.skipconv = nn.Conv2d(insize, outsize, 2, stride=2).to(device)
        self.pool = nn.MaxPool2d(2, 2).to(device)
        self.bn = nn.BatchNorm2d(outsize).to(device)

    def forward(self, x):
        x1 = x.clone()
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.pool(x) + self.skipconv(x1)
        return self.bn(x)

class Head(nn.Module):
    def __init__(self, device, insize, midsize, outsize):
        super(Head, self).__init__()
        self.insize = insize
        self.fc1 = nn.Linear(insize, outsize)
        self.fc2 = nn.Linear(midsize, midsize)
        self.fc3 = nn.Linear(midsize, outsize)
        self.act = nn.LeakyReLU(0.01).to(device)
    def forward(self, x):
        x = x.view(-1, self.insize)
        x = self.act(self.fc1(x))
        # x = self.act(self.fc2(x))
        # x = self.act(self.fc3(x))
        return x

def print1(x):
    # print(x)
    pass


class MultyLayer(nn.Module):
    def __init__(self, device):
        super(MultyLayer, self).__init__()
        self.pulconv = nn.Conv2d(3, 3, 7, stride=2, padding=3).to(device)

        self.cblock1 = ConvBlock(device, 3, 1) # now 128x128
        self.cblock2 = ConvBlock(device, 1, 2) # now 64x64
        self.cblock3 = ConvBlock(device, 2, 4) # now 32x32
        self.cblock4 = ConvBlock(device, 4, 8) # now 16x16
        self.cblock5 = ConvBlock(device, 8, 16) # now 8x8
        self.cblock6 = ConvBlock(device, 16, 32) # now 4x4
        self.cblock7 = ConvBlock(device, 32, 64) # now 2x2
        self.cblock8 = ConvBlock(device, 64, 128) # now 1x1

        self.head = Head(device, 128, 64, 60)

    def forward(self, x):
        # Input: [batch, 3, H, W]
        print1(x.shape)
        x = self.pulconv(x)
        print1(x.shape)
        x = self.cblock1(x)
        print1(x.shape)
        x = self.cblock2(x)
        print1(x.shape)
        x = self.cblock3(x)
        print1(x.shape)
        x = self.cblock4(x)
        print1(x.shape)
        x = self.cblock5(x)
        print1(x.shape)
        x = self.cblock6(x)
        print1(x.shape)
        x = self.cblock7(x)
        print1(x.shape)
        x = self.cblock8(x)
        print1(x.shape)
        x = self.head(x)
        print1(x.shape)
        x = x.view(-1, 60)
        print1(x.shape)
        return x
