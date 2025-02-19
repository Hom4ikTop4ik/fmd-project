import cv2
import os
import sys
import signal
import torch
import torch.nn as nn
import numpy as np
import time

from data_process import augment_gen, make_filter
from data_process import load
from cascade_detector import Level1Detector

noise = 0.05
part = 0.9
displace = 80
rotate = 20

epochs = 10
batch_size = 40
# microset is including 1k images / 40 = 25
# dataset is including >=20k images / 40 = 500
total_iterations = 500

iter_k = 20
iter_l = 20 # similar with iter_k, why not

print(time.time())

def interactor(signal, frame):
    print('SIGINT was received.')
    while (True):
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
weight_save_path = os.path.join(registry_path, 'weights', 'lvl1det_bns.pth')
dataset = load(
    total_iterations, batch_size, 
    os.path.join(current_path, registry_path, 'dataset'), 
    coordsfile="microset_coords.pt", imagesfile="microset_images.pt"
)

# establishing devices and signal handler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("devise is: ", device)
signal.signal(signal.SIGINT, interactor)


lvl1det = Level1Detector(device).to(device)
if input('load weigths from selected weight save path? (y/n) ') == 'y':
    lvl1det.load_state_dict(torch.load(weight_save_path))

optimizer = torch.optim.Adam(lvl1det.parameters(), 0.001, betas=(0.9, 0.95))
criterion = nn.MSELoss().to(device)
coordfilter = make_filter(53, 36, 62, 13, 14, 30, 44)

"""
# learning cycle
iteration = 0
for bt_images, bt_coords in augment_gen(dataset, epochs=10, device=device,
                                        noise=0, part=0.9, displace=80, rotate=20):
    
    truth = coordfilter(bt_coords)[:, :, 0:2]
    ans = lvl1det(bt_images)
    loss = criterion(ans, truth)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    iteration += 1
    print(f'loss {loss:.5f}, iteration: {iteration}')
"""

# new learning cycle
def print_progress_bar(iteration, total, epoch, epochs, iter_per_second, loss, average_loss, median_loss):
    percent = (iteration / total) * 100
    bar_length = 10  # Length of the progress bar
    block = int(round(2*bar_length * percent / 100))
    progress = '▓' * (block//2) + '▒' * (block%2) + '░' * (bar_length - block // 2 - block % 2)
    sys.stdout.write(f'\r[{progress}] {percent:.0f}% (Epoch: {epoch}/{epochs}, Iteration: {iteration}/{total}, Iter/s: {iter_per_second:4.2f}, Loss: {loss:.5f}, Average Loss: {average_loss:.5f}, Median Loss: {median_loss:.5f})')
    sys.stdout.flush()


# Learning cycle
for epoch in range(epochs):
    print(f'Starting Epoch {epoch + 1}/{epochs}')
    print(f'Device: {device}, Noise: {noise}, Part: {part}, Displace: {displace}, Rotate: {rotate}')

    iter_times = []
    iter_loss = []
    average_loss = 1
    median_loss = 1

    for iteration in range(1, total_iterations + 1):
        # Get the batch of images and coordinates
        bt_images, bt_coords = next(augment_gen(dataset, epochs=1, device=device, noise=noise, part=part, displace=displace, rotate=rotate))
        
        # Move tensors to the device
        bt_images = bt_images.to(device)
        bt_coords = bt_coords.to(device)

        # Training code
        truth = coordfilter(bt_coords)[:, :, 0:2] 
        ans = lvl1det(bt_images)
        loss = criterion(ans, truth)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print device information
        # print(f'Model is on device: {next(lvl1det.parameters()).device}')
        # print(f'Input images are on device: {bt_images.device}')

        # Record the time taken for this iteration
        iter_times.append(time.time())
        iter_loss.append(loss)
        l = len(iter_times)

        # Keep only the last k times
        if l > iter_k:
            iter_times.pop(0)
            iter_loss.pop(0)

        # Calculate iterations per second
        if (l > 1):
            iter_per_second = l / (iter_times[-1] - iter_times[0])
        else:
            iter_per_second = 9999.99
        if (l > 0):
            average_loss = sum(iter_loss) / len(iter_loss)
            median_loss = iter_loss[l // 2]
        # print(f"Iterations per second: {iter_per_second:.2f}")
        
        # Update the progress bar
        print_progress_bar(iteration, total_iterations, epoch + 1, epochs, iter_per_second, loss, average_loss, median_loss)
    print("\n")




# Save after every iterations
if (input("Do you wanna save weigths? (y/n) ")[0] == 'y'):
    postfix = input(f"Enter a postfix (enter - save to {weight_save_path}): ")
    if (postfix == ""):
        path = weight_save_path
    else:
        path = os.path.join(registry_path, 'weights', f'lvl1det_bns_{postfix}.pth')
    
    print(f'saving to {path}')
    torch.save(lvl1det.state_dict(), path)
