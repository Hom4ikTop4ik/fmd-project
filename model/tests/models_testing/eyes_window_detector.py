
import cv2
import torch
from torch import device, nn
import numpy as np

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

def crop_center(imgtensor, size, coordstens, cordx = None, cordy = None, shift = 0):
    x = cordx * torch.ones(coordstens.shape[0])
    y = cordy * torch.ones(coordstens.shape[0])
    left = (x.int() - size)
    top = (y.int() - size)
    right = (x.int() + size)
    bottom = (y.int() + size)
    


    if (torch.min(left) < 0 or torch.min(top) < 0 or 
        torch.max(right) > imgtensor.shape[3] or 
        torch.max(bottom) > imgtensor.shape[2]):
        return None
    
    tenslist = []
    id = 0
    for left_el, top_el, right_el, bottom_el in zip(left, top, right, bottom):
        tenslist.append(imgtensor[id, :, top_el:bottom_el, left_el:right_el])
        id += 1
    return torch.stack(tenslist)


def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    newmodel = CnnDetector(device).to(device)
    newmodel.load_state_dict(torch.load("eyes_window_detector.pth"))
    newmodel.eval()

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
        
        frame = frame.unsqueeze(0)
        wsize = 63
        # Detect and visualize eye areas
        result_frame = torch.zeros(1, frame.shape[2], frame.shape[3]).to(device)
        for x in range(wsize, frame.shape[3]-wsize, 6):
            for y in range(wsize, frame.shape[2]-wsize, 5):
                tens = crop_center(frame, wsize, frame, x, y)
                result_frame[0, y, x] = newmodel(tens)
            print(x)

        result_frame = result_frame.permute(1, 2, 0).cpu().detach().numpy()
        result_frame = cv2.resize(result_frame, (512, 512))

        # Display the resulting frame
        cv2.imshow('Eye Area Detection', result_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

test()