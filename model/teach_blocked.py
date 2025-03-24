import cv2
import os
import sys
import signal
import torch
import torch.nn as nn
import numpy as np

from data_process import augment_gen, make_filter
from data_process import MakerPCA
from data_process import load
from detector import MultyLayer


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
            
            truth = coordfilter(bt_coords)[:, :, 0:2]
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
        x, y, value = coord[0], coord[1], coord[2]
        x_pixel = int(x * newimg.shape[1])
        y_pixel = int(y * newimg.shape[0])
        newimg = cv2.circle(newimg, (x_pixel, y_pixel), 2, (255, 255, 255), 2)
        text = f"{value:.2f}"
        cv2.putText(newimg, text, (x_pixel + 5, y_pixel - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("img", newimg)
    cv2.waitKey(0)

def gen_lim(generator, limit):
    for i, obj in enumerate(generator):
        if i >= limit:
            break
        yield obj

# establishing paths and loading
current_path = os.path.dirname(os.path.abspath(__file__))
registry_path = os.path.join(current_path, 'registry')
weight_save_path = os.path.join(registry_path, 'weights', 'model_bns.pth')
dataset = load(500, 40, os.path.join(current_path, registry_path, 'dataset'), 
               imagesfile='dataset_coords_ext.pt')

# establishing devices and signal handler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("devise is: ", device)
signal.signal(signal.SIGINT, interactor)


model = MultyLayer(device).to(device)
# if input('load weigths from selected weight save path? (y/n) ') == 'y':
#     model.load_state_dict(torch.load(weight_save_path))

optimizer = torch.optim.AdamW(model.parameters(), 0.001, betas=(0.9, 0.99))
criterion = nn.MSELoss().to(device)

mypca = MakerPCA()


mypca.load(os.path.join(current_path,'data_process/pcaweights_ext.pca'))


# learning cycle
iteration = 0
for bt_images, bt_coords in augment_gen(dataset, epochs=10, device=device,
                                        noise=0, part=0.9, displace=80, rotate=20):
    print(bt_coords.shape)
    truth = mypca.compress(bt_coords).to(device)
    ans = model(bt_images)
    loss = criterion(ans, truth)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    
    iteration += 1
    print(f'loss {loss:.5f}, iteration: {iteration}')

    if iteration % 2500 == 0:
        dec_answer = mypca.decompress(ans[:1]).reshape(-1, 3, 72).permute(0, 2, 1)
        dec_truth = mypca.decompress(truth[:1]).reshape(-1, 3, 72).permute(0, 2, 1)
        print('dec ans shape: ', dec_answer.shape)
        print('dec tru shape: ', dec_answer.shape)
        look_predict(bt_images, dec_truth)
        look_predict(bt_images, dec_answer)
        


