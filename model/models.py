import torch
import torch.nn as nn
import modelutils

class EyesBoundDetector(nn.Module):
    def __init__(self, device):
        super(EyesBoundDetector, self).__init__()
        self.last_detector_size = 128 
        self.adpool = nn.AdaptiveAvgPool2d((128, 128)).to(device) 
        self.conv1 = nn.Conv2d(3, 6, 3, padding = 1).to(device)
        self.pool = nn.MaxPool2d(2, 2).to(device)
        self.conv2 = nn.Conv2d(6, 9, 3, padding = 1).to(device)
        self.conv3 = nn.Conv2d(9, 20, 3, padding = 1).to(device)
        self.conv4 = nn.Conv2d(20, self.last_detector_size, 3, padding = 1).to(device)
        fcsize = 256
        self.fc1 = nn.Linear(self.last_detector_size*16*16, fcsize).to(device)
        self.fc_list = nn.ModuleList([nn.Linear(fcsize, fcsize).to(device) for _ in range(3)])
        self.prelast = nn.Linear(fcsize, fcsize).to(device)
        self.fc_last = nn.Linear(fcsize, 3 * 68).to(device)
        self.act = nn.ReLU().to(device)
        self.sigm = nn.Sigmoid().to(device)
    
    def forward(self, x, needshow = False):
        # Input: [batch, 3, H, W]
        x = self.adpool(x)
        x = self.pool(self.act(self.conv1(x)))
        if(needshow):
            xshow = modelutils.show_tensor(x[0:2], landmarks=None, nolandmarks=True)
            xshow.show()
            print(x.shape)
        x = self.pool(self.act(self.conv2(x)))
        x = self.act(self.conv3(x))  
        x = self.pool(self.act(self.conv4(x)))
        x = x.view(-1, self.last_detector_size*16*16)       
        x = self.act(self.fc1(x))
        for i in range(len(self.fc_list)):
            x = self.act(self.fc_list[i](x))
        x = self.act(self.prelast(x))
        x = self.fc_last(x)
        x = x.view(-1, 68, 3)
        return x


class MouthBoundDetector(nn.Module):
    def __init__(self, device):
        super(mouthBoundDetector, self).__init__()
        self.last_detector_size = 128 
        self.adpool = nn.AdaptiveAvgPool2d((128, 128)).to(device) 
        self.conv1 = nn.Conv2d(3, 6, 3, padding = 1).to(device)
        self.pool = nn.MaxPool2d(2, 2).to(device)
        self.conv2 = nn.Conv2d(6, 9, 3, padding = 1).to(device)
        self.conv3 = nn.Conv2d(9, 20, 3, padding = 1).to(device)
        self.conv4 = nn.Conv2d(20, self.last_detector_size, 3, padding = 1).to(device)
        fcsize = 256
        self.fc1 = nn.Linear(self.last_detector_size*16*16, fcsize).to(device)
        self.fc_list = nn.ModuleList([nn.Linear(fcsize, fcsize).to(device) for _ in range(3)])
        self.prelast = nn.Linear(fcsize, fcsize).to(device)
        self.fc_last = nn.Linear(fcsize, 3 * 68).to(device)
        self.act = nn.ReLU().to(device)
        self.sigm = nn.Sigmoid().to(device)
    
    def forward(self, x, needshow = False):
        # Input: [batch, 3, H, W]
        x = self.adpool(x)
        x = self.pool(self.act(self.conv1(x)))
        if(needshow):
            xshow = modelutils.show_tensor(x[0:2], ans_with_mouth_bounds[0], nolandmarks=True)
            xshow.show()
            print(x.shape)
        x = self.pool(self.act(self.conv2(x)))
        x = self.act(self.conv3(x))  
        x = self.pool(self.act(self.conv4(x)))
        x = x.view(-1, self.last_detector_size*16*16)       
        x = self.act(self.fc1(x))
        for i in range(len(self.fc_list)):
            x = self.act(self.fc_list[i](x))
        x = self.act(self.prelast(x))
        x = self.fc_last(x)
        x = x.view(-1, 68, 3)
        return x




def crop_mouth(image_tensor_array, landmarks_tensor_array, mouth_landmarks):
    # Get mouth landmarks

    for image_tensor, landmarks_tensor in zip(image_tensor_array, landmarks_tensor_array):
        x1, y1, x2, y2 = 1, 1, 0, 0
        for i in mouth_landmarks:
            x = landmarks_tensor[i, 0].item()
            y = landmarks_tensor[i, 1].item()
            if x < x1:
                x1 = x
            if x > x2:
                x2 = x
            if y < y1:
                y1 = y
            if y > y2:
                y2 = y
        size = image_tensor.shape[2]
        # make rectangle 1 + b times bigger
        b = 2.0
        x = (x2 + x1) / b
        y = (y2 + y1) / b

        x1 = int(x * size - 100)
        x2 = int(x * size + 100)
        y1 = int(y * size - 64)
        y2 = int(y * size + 64)

        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x1 > size:
            x1 = size
        if y1 > size:
            y1 = size
        print(x1, y1, x2, y2)
        # Crop image
        image_tensor = image_tensor[:, y1:y2, x1:x2]
        # Get bounding box of mouth
    return image_tensor


class MouthPointsDetector(nn.Module):
    def __init__(self, device, mouth_bound_detector, mouth_landmarks):
        super(mouthPointsDetector, self).__init__()
        self.mb_detector = mouth_bound_detector
        self.last_detector_size = 8
        self.mouth_landmarks = mouth_landmarks
        
        self.conv1 = nn.Conv2d(3, 5, 3, padding = 1).to(device)
        self.pool = nn.MaxPool2d(2, 2).to(device)
        self.conv2 = nn.Conv2d(5, 5, 3, padding = 1).to(device)
        self.conv3 = nn.Conv2d(5, 5, 3, padding = 1).to(device)
        self.conv4 = nn.Conv2d(5, 5, 3, padding = 1).to(device)
        self.conv5 = nn.Conv2d(5, 6, 3, padding = 1).to(device)
        self.conv6 = nn.Conv2d(6, self.last_detector_size, 3, padding = 1).to(device)
        fcsize = 612
        
        self.adpool_size = 54, 30
        self.adpool = nn.AdaptiveAvgPool2d(self.adpool_size).to(device) 

        self.fc1 = nn.Linear(self.last_detector_size*self.adpool_size[0]*self.adpool_size[1],
                             fcsize).to(device)
        self.fc_list = nn.ModuleList([nn.Linear(fcsize, fcsize).to(device) for _ in range(12)])
        self.prelast = nn.Linear(fcsize, fcsize).to(device)
        self.fc_last = nn.Linear(fcsize, 3 * 68).to(device)
        self.act = nn.ReLU().to(device)
        self.sigm = nn.Tanh().to(device)
    
    def forward(self, x, needshow = False):
        # Input: [batch, 3, H, W]
        ans_with_mouth_bounds = self.mb_detector(x)
        x = crop_mouth(x, ans_with_mouth_bounds, self.mouth_landmarks)
        
        if(needshow):
            xshow = modelutils.show_tensor(x[0:2], ans_with_mouth_bounds[0], nolandmarks=True)
            xshow.show()
            print(x.shape)

        x = self.act(self.conv1(x))
        
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.pool(self.act(self.conv5(x)))
        x = self.act(self.conv6(x))  
        #print(x.shape)
        
        #x = self.act(self.conv6(x))
        x = self.adpool(x)
        
        x = x.view(-1, self.last_detector_size*self.adpool_size[0]*self.adpool_size[1])       
        x = self.act(self.fc1(x))
        for i in range(len(self.fc_list)):
            x = self.sigm(self.fc_list[i](x))
        x = self.sigm(self.prelast(x))
        x = self.fc_last(x)
        x = x.view(-1, 68, 3)
        return x


class CnnDetector(nn.Module):
    def __init__(self, device):
        super(CnnDetector, self).__init__()
        self.adpool = nn.AdaptiveAvgPool2d((64, 64)).to(device)
        self.glpool = nn.AdaptiveAvgPool2d((1, 1)).to(device)
        self.pool = nn.MaxPool2d(2, 2).to(device)
        self.conv1 = nn.Conv2d(3, 4, 3, padding = 1).to(device) 
        self.conv2 = nn.Conv2d(4, 4, 3, padding = 1).to(device)
        self.conv21 = nn.Conv2d(4, 4, 3, padding = 1).to(device)
        self.conv3 = nn.Conv2d(4, 8, 3, padding = 1).to(device) 
        self.conv31 = nn.Conv2d(8, 8, 3, padding = 1).to(device)
        self.conv4 = nn.Conv2d(8, 16, 3, padding = 1).to(device)
        self.conv41 = nn.Conv2d(16, 16, 3, padding = 1).to(device)
        self.conv42 = nn.Conv2d(16, 16, 3, padding = 1).to(device) 
        self.conv5 = nn.Conv2d(16, 32, 3, padding = 1).to(device)
        self.conv6 = nn.Conv2d(32, 64, 3, padding = 1).to(device)

        self.fc1 = nn.Linear(64, 1).to(device)
        self.fc2 = nn.Linear(256, 1).to(device)
        self.act = nn.ReLU().to(device)
        self.sigm = nn.Sigmoid().to(device)


    def forward(self, x, fully_connected = False):
        # Input: [batch, 3, H, W]
        # x = self.adpool(x)
        x1 = x.clone()
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = self.act(self.conv21(x)) + self.pool(self.pool(x1)).repeat(1, 2, 1, 1)[:, :4, :, :] * 0.1
        x1 = x.clone()
        x = self.pool(self.act(self.conv3(x)))
        x = self.act(self.conv31(x))
        x = self.pool(self.act(self.conv4(x))) + self.pool(self.pool(x1)).repeat(1, 4, 1, 1)[:, :16, :, :] * 0.1
        x1 = x.clone()
        x = self.act(self.conv41(x))
        x = self.act(self.conv42(x))
        x = self.pool(self.act(self.conv5(x))) + self.pool(self.pool(x1)).repeat(1, 4, 1, 1)[:, :32, :, :] * 0.1
        x = self.pool(self.act(self.conv6(x)))
        x = self.glpool(x).view(-1, 64)
        # x = self.act(self.fc1(x))
        x = self.sigm(self.fc1(x))

        return x.view(-1, 1)


# this one works well with BCELoss, window size is 63
class CnnDetector(nn.Module):
    def __init__(self, device):
        super(CnnDetector, self).__init__()
        self.adpool = nn.AdaptiveAvgPool2d((64, 64)).to(device)
        self.glpool = nn.AdaptiveAvgPool2d((1, 1)).to(device)
        self.pool = nn.MaxPool2d(2, 2).to(device)
        self.conv1 = nn.Conv2d(3, 4, 3, padding = 1).to(device) 
        self.conv2 = nn.Conv2d(4, 4, 3, padding = 1).to(device)
        self.conv21 = nn.Conv2d(4, 4, 3, padding = 1).to(device)
        self.conv3 = nn.Conv2d(4, 8, 3, padding = 1).to(device) 
        self.conv31 = nn.Conv2d(8, 8, 3, padding = 1).to(device)
        self.conv4 = nn.Conv2d(8, 16, 3, padding = 1).to(device)
        self.conv41 = nn.Conv2d(16, 16, 3, padding = 1).to(device)
        self.conv42 = nn.Conv2d(16, 16, 3, padding = 1).to(device) 
        self.conv5 = nn.Conv2d(16, 32, 3, padding = 1).to(device)
        self.conv6 = nn.Conv2d(32, 64, 3, padding = 1).to(device)

        self.fc1 = nn.Linear(64, 4).to(device)
        self.fc2 = nn.Linear(4, 1).to(device)
        self.act = nn.ReLU().to(device)
        self.sigm = nn.Sigmoid().to(device)


    def forward(self, x, fully_connected = False):
        # Input: [batch, 3, H, W]
        # x = self.adpool(x)
        x1 = x.clone()
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = self.act(self.conv21(x)) + self.pool(self.pool(x1)).repeat(1, 2, 1, 1)[:, :4, :, :] * 0.1
        x1 = x.clone()
        x = self.pool(self.act(self.conv3(x)))
        x = self.act(self.conv31(x))
        x = self.pool(self.act(self.conv4(x))) + self.pool(self.pool(x1)).repeat(1, 4, 1, 1)[:, :16, :, :] * 0.1
        x1 = x.clone()
        x = self.act(self.conv41(x))
        x = self.act(self.conv42(x))
        x = self.pool(self.act(self.conv5(x))) + self.pool(self.pool(x1)).repeat(1, 4, 1, 1)[:, :32, :, :] * 0.1
        x = self.pool(self.act(self.conv6(x)))
        x = self.glpool(x).view(-1, 64)
        x = self.act(self.fc1(x))
        x = self.sigm(self.fc2(x))

        return x.view(-1, 1)