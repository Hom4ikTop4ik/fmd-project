
import cv2
import torch
from torch import device, nn
import numpy as np
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

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    newmodel = FaceDetector(device).to(device)
    newmodel.load_state_dict(torch.load("registry/weights/face_det_works.pth"))
    newmodel.eval()

    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (256, 256))
        
        frame = torch.from_numpy(frame.astype(np.float32)).to(device) / 255
        frame = frame.permute(2, 0, 1)
        height = frame.shape[1]
        width = frame.shape[2]
        dif = width - height
        frame = frame[:, 100:356, 100:356]
        
        if not ret:
            break
        
        # Detect and visualize eye areas
        result_frame = newmodel(frame)
        
        result_frame = result_frame[0]
        result_frame = result_frame.permute(1, 2, 0).cpu().detach().numpy()
        thr, result_frame = cv2.threshold(result_frame, 0.2, 1, cv2.THRESH_BINARY)
        result_frame = cv2.resize(result_frame, (512, 512))

        # Display the resulting frame
        cv2.imshow('Face Area Detection', result_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

test()