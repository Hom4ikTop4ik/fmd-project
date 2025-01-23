
import cv2
import torch
from torch import device, nn
import numpy as np

class CnnDetector(nn.Module):
    def __init__(self, device):
        super(CnnDetector, self).__init__()
        self.adpool = nn.AdaptiveAvgPool2d((128, 128)).to(device)
        self.pool = nn.MaxPool2d(2, 2).to(device)
        self.mid_depth = mid_depth = 7
        self.conv1 = nn.Conv2d(3, mid_depth, 7, padding = 3).to(device) 
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
        fcsize = 200
        self.fc1 = nn.Linear(64 * 64, fcsize).to(device)
        self.fc2 = nn.Linear(fcsize, fcsize).to(device)
        self.fc3 = nn.Linear(fcsize, 64 * 64).to(device)
        self.act = nn.ReLU().to(device)

    def forward(self, x, fully_connected = False):
        # Input: [batch, 3, H, W]
        x = self.adpool(x)
        x1 = x.clone()
        x1 = x1.repeat(1, 3, 1, 1)[:, :self.mid_depth, :, :]
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
        x = self.act(self.conv13(x))
        x = self.act(self.conv14(x)) + x1[:, :1, :, :] * 0.3
        # make freeze layers when needed amount of epochs exceed
        if fully_connected:
            x = x.view(-1, 64*64)
            x1 = x.clone()
            x = self.act(self.fc1(x))
            # x = self.act(self.fc2(x))
            x = self.act(self.fc3(x))
            x = (x + x1 * 0.3).view(-1, 64, 64).unsqueeze(1) 
        return x

        
def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    det1 = CnnDetector(device).to(device)
    det2 = CnnDetector(device).to(device)
    det1.load_state_dict(torch.load("eye_area_detector_fc1.pth"))
    det2.load_state_dict(torch.load("eye_area_detector_fc2.pth"))
    det1.eval()
    det2.eval()

    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = torch.from_numpy(frame.astype(np.float32)).to(device) / 255
        frame = frame.permute(2, 0, 1)
        height = frame.shape[1]
        width = frame.shape[2]
        dif = width - height
        frame = frame[:, :height, dif//2:width - dif//2]
        if not ret:
            break
        
        # Detect and visualize eye areas
        frame = (frame - frame.min()) / (frame.max() - frame.min())
        brightmask = (frame.mean(dim = 0) > 0.8).cpu().detach().numpy().astype(np.float32)
        # print(brightmask.shape)
        brightmask = cv2.GaussianBlur(brightmask, (131, 131), sigmaX=30, borderType=cv2.BORDER_REPLICATE)
        frame -= torch.from_numpy(brightmask * 0.7).to(device)
        res1 = det1(frame, fully_connected = False)
        res2 = det2(frame, fully_connected = False)
        result_frame = res1[0]
        
        #result_frame = result_frame[0][1:4]
        #print(result_frame.shape)

        result_frame = result_frame.permute(1, 2, 0).cpu().detach().numpy().squeeze(2)
        # ret, result_frame = cv2.threshold(result_frame, 0.3,1, cv2.THRESH_BINARY)
        # print(type(result_frame), result_frame, result_frame.shape, result_frame.max(), result_frame.min(), result_frame.mean(), result_frame.std())

        result_frame = cv2.resize(result_frame, (512, 512))
        
        # print(brightmask)

        # Display the resulting frame
        cv2.imshow('Eye Area Detection', result_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

test()