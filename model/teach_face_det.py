import cv2
import os
import sys
import signal
import torch
import torch.nn as nn
import numpy as np

from data_process import augment_gen, make_highlighter, face_coord
from data_process import load
from cascade_detector import FaceDetector


def interactor(signal, frame):
    print('SIGINT was received.')
    cmd = input('Choose action: save/test/exit/look (continue by default): ')
    match cmd:
        case 'save':
            path = weight_save_path
            print(f'saving to {path}')
            torch.save(model.state_dict(), path)
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
            
            truth = highlight_face(bt_coords)
            ans = model(bt_images)
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
        truth = face_coord(bt_coords)
        ans = model(bt_images)
        print(truth.shape, ans.shape)
        loss = criterion(ans, truth)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(iteration % 20 == 2):
            face = bt_images[0].permute(1, 2, 0)[:, :, [2, 1, 0]].cpu().numpy()
            cv2.circle(face, (int(ans[0][0] * face.shape[0]), int(ans[0][1] * face.shape[1])), 5, 1)
            cv2.imshow('face', face)
            cv2.waitKey(3000)
            cv2.destroyAllWindows() 
        cv2.imshow("img", newimg)
        cv2.waitKey(0)

# establishing paths and loading
current_path = os.path.dirname(os.path.abspath(__file__))
registry_path = os.path.join(current_path, 'registry')
weight_save_path = os.path.join(registry_path, 'weights', 'face_det.pth')
dataset = load(2000, 10, os.path.join(current_path, registry_path, 'dataset'))

# establishing devices and signal handler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("devise is: ", device)
signal.signal(signal.SIGINT, interactor)


model = FaceDetector(device).to(device)
# if input('load weigths from selected weight save path? (y/n) ') == 'y':
    # lvl1det.load_state_dict(torch.load(weight_save_path))

optimizer = torch.optim.Adam(model.parameters(), 0.001, betas=(0.9, 0.95))
criterion = nn.MSELoss().to(device)
highlight_face = make_highlighter(64, 20, 15, device)
# learning cycle
iteration = 0
for bt_images, bt_coords in augment_gen(dataset, epochs=10, device=device,
                                        noise=0.1, part=0.9, displace=128, rotate=20):
    
    truth = face_coord(bt_coords)
    ans = model(bt_images)
    print(truth.shape, ans.shape)
    loss = criterion(ans, truth)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if(iteration % 20 == 2):
        face = bt_images[0].permute(1, 2, 0)[:, :, [2, 1, 0]].cpu().numpy()
        cv2.circle(face, (int(ans[0][0] * face.shape[0]), int(ans[0][1] * face.shape[1])), 5, 1)
        cv2.imshow('face', face)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
    
    iteration += 1
    print(f'loss {loss:.5f}, iteration: {iteration}')
