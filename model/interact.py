import cv2
import os
import sys
import signal
import torch
import torch.nn as nn
import numpy as np

from data_process import augment_gen, make_filter
from data_process import load
from cascade_detector import Level1Detector


def interactor(signal, frame):
    print('SIGINT was received.')
    cmd = input('Choose action: save/test/exit/look (continue by default): ')
    match cmd:
        case 'save':
            path = weight_save_path
            print(f'saving to {path}')
            torch.save(lvl1det.state_dict(), path)
        case 'test':
            model_test(look=False)
        case 'exit':
            sys.exit(0)
        case 'look':
            model_test(look=True)

def model_test(look=False):
    with torch.no_grad():
        iteration = 0
        loss = 0.0
        for bt_images, bt_coords in augment_gen(dataset, epochs=1, device=device, 
                                                noise=0, part=-0.1, verbose=True):
            
            truth = coordfilter(bt_coords)[:, :, 0:2]
            ans = lvl1det(bt_images)
            loss += criterion(ans, truth)
            iteration += 1
            if look:
                look_predict(bt_images, ans)
        
        print(f'total loss {loss / iteration}')
        print(f'total iterations count {iteration}')

def look_predict(imgtens: torch.Tensor, predict: torch.Tensor):
    predict = predict[0]

    newimg = (imgtens[0,[2,1,0], :, :].cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)

    newimg = np.ascontiguousarray(newimg)
    for coord in predict:
        x, y = coord[0], coord[1]
        newimg = cv2.circle(newimg, (int(x * newimg.shape[1]), int(y * newimg.shape[0])), 2, (255, 255, 255), 2)
    
    cv2.imshow("img", newimg)
    cv2.waitKey(0)

# establishing paths and loading
current_path = os.path.dirname(os.path.abspath(__file__))
registry_path = os.path.join(current_path, 'registry')
weight_save_path = os.path.join(registry_path, 'weights', 'lvl1det.pth')
dataset = load(1000, 20, os.path.join(current_path, registry_path, 'dataset'))

# establishing devices and signal handler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("devise is: ", device)
signal.signal(signal.SIGINT, interactor)


lvl1det = Level1Detector(device).to(device)
lvl1det.load_state_dict(torch.load(weight_save_path))

optimizer = torch.optim.Adam(lvl1det.parameters(), 0.0001, betas=(0.9, 0.95))
criterion = nn.MSELoss().to(device)
coordfilter = make_filter(53, 36, 62, 13, 14, 30, 44)

# learning cycle
iteration = 0
for bt_images, bt_coords in augment_gen(dataset, epochs=2,
                                        device=device, noise=0, part=0.9):
    
    truth = coordfilter(bt_coords)[:, :, 0:2]
    ans = lvl1det(bt_images)
    loss = criterion(ans, truth)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    iteration += 1
    print(f'loss {loss:.5f}, iteration: {iteration}')
