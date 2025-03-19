import torch.nn as nn
import torch

def center_of_masses(img_tens_batch: torch.Tensor):
    '''
    img_tens_batch (batchcnt, 1, size, size).
    first calculate x_mean, secondly y_mean
    '''
    sizex, sizey = img_tens_batch.shape[2], img_tens_batch.shape[3]
    device = img_tens_batch.device
    sumall = img_tens_batch.sum(3).sum(2)
    
    # find x_mean
    ordered = torch.Tensor([i + 1 for i in range(sizex)]).to(device)
    weighted_sum = (img_tens_batch * ordered).sum(3) / img_tens_batch.sum(3)
    weighted_sum.nan_to_num_(0)
    x_mean = (weighted_sum * img_tens_batch.sum(3)).sum(2) / sumall - 1
    x_mean /= sizex

    # find y_mean
    ordered = torch.Tensor([i + 1 for i in range(sizey)]).to(device)
    img_tens_batch_perm = img_tens_batch.permute(0, 1, 3, 2)
    weighted_sum = (img_tens_batch_perm * ordered).sum(3) / img_tens_batch_perm.sum(3)
    weighted_sum.nan_to_num_(0)
    y_mean = (weighted_sum * img_tens_batch_perm.sum(3)).sum(2) / sumall - 1
    y_mean /= sizey

    return torch.stack([x_mean, y_mean]).squeeze(2).permute(1, 0).nan_to_num(0)


class FaceDetector(nn.Module):
    def __init__(self, device):
        super(FaceDetector, self).__init__()
        self.adpool = nn.AdaptiveAvgPool2d((128, 128)).to(device)
        self.pool = nn.MaxPool2d(2, 2).to(device)
        self.mid_depth = mid_depth = 1
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
        x1 = x.clone()
        x1 = x1.repeat(1, 6, 1, 1)[:, :self.mid_depth, :, :]
        x = self.act(self.conv1(x))
        x = self.pool(self.act(self.conv2(x)) + x1)
        x1 = x.clone()
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x)) + x1 * 0.2
        x1 = x.clone()
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x)) + x1 * 0.2
        x1 = x.clone()
        x = self.act(self.conv7(x))
        x = self.act(self.conv8(x)) + x1 * 0.2
        x1 = x.clone()
        x = self.act(self.conv9(x))
        x = self.act(self.conv10(x)) + x1 * 0.2
        x1 = x.clone()
        x = self.act(self.conv11(x))
        x = self.act(self.conv12(x)) + x1 * 0.2
        x1 = x.clone()
        x = self.act(self.conv14(x)) + x1 * 0.2
        return x
        #return center_of_masses(x)

