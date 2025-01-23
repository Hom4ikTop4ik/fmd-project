
import cv2
import torch
from torch import device, nn
import numpy as np

class CnnDetector(nn.Module):
    def __init__(self, device):
        super(CnnDetector, self).__init__()
        self.adpool = nn.AdaptiveAvgPool2d((128, 128)).to(device)
        self.pool = nn.MaxPool2d(2, 2).to(device)
        self.mid_depth = mid_depth = 2

        self.conv11 = nn.Conv2d(3, mid_depth, 7, padding = 3).to(device) 
        self.conv21 = nn.Conv2d(mid_depth, mid_depth, 3, padding = 1).to(device)
        self.conv31 = nn.Conv2d(mid_depth, mid_depth, 3, padding = 1).to(device)
        self.conv41 = nn.Conv2d(mid_depth, 1, 3, padding = 1).to(device)

        self.conv12 = nn.Conv2d(3, mid_depth, 7, padding = 3).to(device) 
        self.conv22 = nn.Conv2d(mid_depth, mid_depth, 3, padding = 1).to(device)
        self.conv32 = nn.Conv2d(mid_depth, mid_depth, 3, padding = 1).to(device)
        self.conv42 = nn.Conv2d(mid_depth, 1, 3, padding = 1).to(device)

        self.conv13 = nn.Conv2d(3, mid_depth, 7, padding = 3).to(device) 
        self.conv23 = nn.Conv2d(mid_depth, mid_depth, 3, padding = 1).to(device)
        self.conv33 = nn.Conv2d(mid_depth, mid_depth, 3, padding = 1).to(device)
        self.conv43 = nn.Conv2d(mid_depth, 1, 3, padding = 1).to(device)

        self.act = nn.ReLU().to(device)

    def forward(self, x, fully_connected = False):
        # Input: [batch, 3, H, W]
        x = self.adpool(x)
        xf = x.clone()

        xb = xf.repeat(1, 3, 1, 1)[:, :self.mid_depth, :, :]
        x1 = self.act(self.conv11(xf))
        x1 = self.pool(self.act(self.conv21(x1)) + 0.2 * xb)
        xb = x1.clone()
        x1 = self.act(self.conv31(x1))
        x1 = self.act(self.conv41(x1)) + 0.2 * xb[:, :1, :, :]

        xb = xf.repeat(1, 3, 1, 1)[:, :self.mid_depth, :, :]
        x2 = self.act(self.conv12(xf))
        x2 = self.pool(self.act(self.conv22(x2)) + 0.2 * xb)
        xb = x2.clone()
        x2 = self.act(self.conv32(x2))
        x2 = self.act(self.conv42(x2)) + 0.2 * xb[:, :1, :, :]

        xb = xf.repeat(1, 3, 1, 1)[:, :self.mid_depth, :, :]
        x3 = self.act(self.conv13(xf))
        x3 = self.pool(self.act(self.conv23(x3)) + 0.2 * xb)
        xb = x3.clone()
        x3 = self.act(self.conv33(x3))
        x3 = self.act(self.conv43(x3)) + 0.2 * xb[:, :1, :, :]

        # return torch.min(x1, torch.min(x2, x3)) + x1 * x2 * x3
        return (x1 + x2 + x3) / 3
        
def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    det2 = CnnDetector(device).to(device)
    det2.load_state_dict(torch.load("eye_area_detector_good.pth"))
    det2.eval()

    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = torch.from_numpy(frame.astype(np.float32)).to(device) / 255
        # frame = cv2.imread("face.jpeg")
        # frame = torch.from_numpy(frame.astype(np.float32)).to(device) / 255
        frame = frame.permute(2, 0, 1)
        height = frame.shape[1]
        width = frame.shape[2]
        dif = width - height
        frame = frame[:, :height, dif//2:width - dif//2]
        if not ret:
            break
        
        
        # Detect and visualize eye areas
        # frame = (frame - frame.min()) / (frame.max() - frame.min())
        res2 = det2(frame, fully_connected = False)
        print(res2.shape)
        result_frame = res2[0]
        #result_frame = result_frame[0][1:4]
        #print(result_frame.shape)

        result_frame = result_frame.permute(1, 2, 0).cpu().detach().numpy().squeeze(2)
        # ret, result_frame = cv2.threshold(result_frame, 0.3,1, cv2.THRESH_BINARY)
        # print(type(result_frame), result_frame, result_frame.shape, result_frame.max(), result_frame.min(), result_frame.mean(), result_frame.std())

        result_frame = cv2.resize(result_frame, (512, 512))
        
        result_frame = result_frame * 2

        # Display the resulting frame
        cv2.imshow('Eye Area Detection', result_frame)
        # cv2.waitKey(0)
        # Exit on 'q' key press
        # break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

test()