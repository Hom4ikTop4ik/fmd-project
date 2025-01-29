import torch.nn as nn
class FaceDetector(nn.Module):
    def __init__(self, device):
        super(FaceDetector, self).__init__()
        self.adpool = nn.AdaptiveAvgPool2d((128, 128)).to(device)
        self.pool = nn.MaxPool2d(2, 2).to(device)
        self.mid_depth = mid_depth = 16
        self.conv1 = nn.Conv2d(3, mid_depth, 3, padding = 1).to(device) 
        self.conv2 = nn.Conv2d(mid_depth, mid_depth, 3, padding = 1).to(device)
        self.conv3 = nn.Conv2d(mid_depth, mid_depth, 3, padding = 1).to(device)
        self.conv4 = nn.Conv2d(mid_depth, mid_depth, 3, padding = 1).to(device)
        self.conv5 = nn.Conv2d(mid_depth, mid_depth, 3, padding = 1).to(device)
        self.conv6 = nn.Conv2d(mid_depth, mid_depth, 3, padding = 1).to(device)
        self.conv7 = nn.Conv2d(mid_depth, mid_depth, 3, padding = 1).to(device)
        self.conv8 = nn.Conv2d(mid_depth, mid_depth, 3, padding = 1).to(device)
        self.conv9 = nn.Conv2d(mid_depth, mid_depth, 3, padding = 1).to(device)
        self.conv10 = nn.Conv2d(mid_depth, mid_depth, 3, padding = 1).to(device)
        self.conv11 = nn.Conv2d(mid_depth, mid_depth, 3, padding = 1).to(device)
        self.conv12 = nn.Conv2d(mid_depth, mid_depth, 3, padding = 1).to(device)
        self.conv13 = nn.Conv2d(mid_depth, mid_depth, 3, padding = 1).to(device)
        self.conv14 = nn.Conv2d(mid_depth, 1, 3, padding = 1).to(device)
        self.act = nn.ReLU().to(device)

    def forward(self, x):
        # Input: [batch, 3, H, W]
        x = self.adpool(x)
        print(x.shape)
        x1 = x.clone()
        x1 = x1.repeat(1, 6, 1, 1)[:, :self.mid_depth, :, :]
        x = self.act(self.conv1(x))
        x = self.pool(self.act(self.conv2(x)) + x1)
        x1 = x.clone()
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x)) + x1 # add skip connections (+x.copy()) if needed
        x1 = x.clone()
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x)) + x1 
        x1 = x.clone()
        x = self.act(self.conv7(x))
        x = self.act(self.conv8(x)) + x1
        x1 = x.clone()
        x = self.act(self.conv9(x))
        x = self.act(self.conv10(x)) + x1
        x1 = x.clone()
        x = self.act(self.conv11(x))
        x = self.act(self.conv12(x)) + x1
        x1 = x.clone()
        x = self.act(self.conv14(x))
        return x

