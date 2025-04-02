import cv2
import torch
from torch import nn
import numpy as np

class Level1Detector(nn.Module):
    def __init__(self, device):
        super(Level1Detector, self).__init__()
        self.adpool = nn.AdaptiveAvgPool2d((128, 128)).to(device) 
        self.pool = nn.MaxPool2d(2, 2).to(device)
        self.conv1 = nn.Conv2d(3, 6, 3, padding = 1).to(device)
        self.bn1 = nn.BatchNorm2d(6).to(device)
        self.conv2 = nn.Conv2d(6, 9, 3, padding = 1).to(device)
        self.bn2 = nn.BatchNorm2d(9).to(device)
        self.conv3 = nn.Conv2d(9, 16, 3, padding = 1).to(device)
        self.bn3 = nn.BatchNorm2d(16).to(device)
        self.conv4 = nn.Conv2d(16, 32, 3, padding = 1).to(device)
        self.bn4 = nn.BatchNorm2d(32).to(device)
        self.conv5 = nn.Conv2d(32, 64, 3, padding = 1).to(device)
        self.bn5 = nn.BatchNorm2d(64).to(device)
        self.conv6 = nn.Conv2d(64, 128, 3, padding = 1).to(device)
        self.bn6 = nn.BatchNorm2d(128).to(device)
        self.conv7 = nn.Conv2d(128, 256, 3, padding = 1).to(device)
        self.bn7 = nn.BatchNorm2d(256).to(device)
        self.fcsize = fcsize = 256
        self.fc1 = nn.Linear(256*2*2, fcsize).to(device)
        self.fc2 = nn.Linear(fcsize, 7 * 2).to(device)
        self.act = nn.LeakyReLU(0.01).to(device)
    
    def forward(self, x):
        # Input: [batch, 3, H, W]
        x = self.adpool(x)
        x1 = x.clone()
        x = self.pool(self.act(self.bn1(self.conv1(x))))
        x = self.pool(self.act(self.bn2(self.conv2(x)))) + self.pool(self.pool(x1)).repeat(1, 3, 1, 1) * 0.2
        x1 = x.clone()
        x = self.act(self.bn3(self.conv3(x)))
        x = self.pool(self.act(self.bn4(self.conv4(x)))) + self.pool(x1).repeat(1, 4, 1, 1)[:, :32, :, :] * 0.2
        x = self.pool(self.act(self.bn5(self.conv5(x))))
        x1 = x.clone()
        x = self.pool(self.act(self.bn6(self.conv6(x))))
        x = self.pool(self.act(self.bn7(self.conv7(x)))) + self.pool(self.pool(x1)).repeat(1, 4, 1, 1) * 0.2
        x = x.view(-1, 256*2*2)       
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, 7, 2)
        return x


    
        
def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    det1 = Level1Detector(device).to(device)
    det1.load_state_dict(torch.load("registry/weights/lvl1det_bns.pth"))
    det1.eval()
    
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = torch.from_numpy(frame.astype(np.float32)).to(device) / 255
        imgtens = frame.permute(2, 0, 1)[:, 0:432, 0:432].unsqueeze(0)
        print(imgtens.shape)
        predict = det1(imgtens)
        newimg = (imgtens[0, :, :, :].cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)

        newimg = np.ascontiguousarray(newimg)
        for coord in predict[0]:
            x, y = coord[0], coord[1]
            # print(coord.shape, newimg.shape, predict.shape)
            newimg = cv2.circle(newimg, (int(x * newimg.shape[1]), int(y * newimg.shape[0])), 2, (255, 255, 255), 2)
        
        cv2.imshow("img", newimg)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

test()