import cv2
import os
import sys  
import signal
import torch
import torch.nn as nn
import numpy as np
import time

from data_process import make_filter
from data_process import load, noise, displace, rotate
from data_process import USE_CPU_WHATEVER
from cascade_detector import Level1Detector

global dataset, device, lvl1det, optimizer, criterion, coordfilter

def interactor(signal, frame):
    print('SIGINT was received.')
    while (True):
        cmd = input('Choose action: save/exit (continue by default): ')
        match cmd:
            case 'save':
                path = weight_save_path
                print(f'saving to {path}')
                torch.save(lvl1det.state_dict(), path)
            case 'учше':
                sys.exit(0)
            case 'exit':
                sys.exit(0)


def model_test(look=False):
    with torch.no_grad():
        iteration = 0
        loss = 0.0
        for bt_images, bt_coords in iter(dataset):
            
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


def print_progress_bar(iteration, total, epoch, epochs, iter_per_second, loss, average_loss, median_loss):
    percent = (iteration / total) * 100
    bar_length = 10  # Length of the progress bar
    block = int(round(2*bar_length * percent / 100))
    progress = '▓' * (block//2) + '▒' * (block%2) + '░' * (bar_length - block // 2 - block % 2)
    sys.stdout.write(f'\r[{progress}] {percent:3.0f}% (Epoch: {epoch}/{epochs}, Iteration: {iteration}/{total}, Iter/s: {iter_per_second:4.2f}, Loss: {loss:.5f}, Average Loss: {average_loss:.5f}, Median Loss: {median_loss:.5f})')
    sys.stdout.flush()


epochs = 10
batch_size = 40
imgs_count = 20000
total_iterations = imgs_count // batch_size

iter_k = 20

print(time.time())

# paths and loading
current_path = os.path.dirname(os.path.abspath(__file__))
registry_path = os.path.join(current_path, 'registry')
weight_save_path = os.path.join(registry_path, 'weights', 'lvl1det_bns.pth')


signal.signal(signal.SIGINT, interactor)
def main():
    global dataset, device, lvl1det, optimizer, criterion, coordfilter
    
    device = 'cpu'
    if (USE_CPU_WHATEVER == False):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("devise is: ", device)

    print("Start  load dataset with time: {}".format(time.time()))
    dataset = load(
        bsize=batch_size, 
        dataset_path=os.path.join(current_path, registry_path, 'dataset', 'train'),
        device=device
    )
    print("Finish load dataset with time: {}".format(time.time()))

    lvl1det = Level1Detector(device).to(device)

    if input('load weigths from selected weight save path? (y/n) ') in 'yYнН':
        lvl1det.load_state_dict(torch.load(weight_save_path, map_location = device))

    optimizer = torch.optim.Adam(lvl1det.parameters(), 0.001, betas=(0.9, 0.95))
    criterion = nn.MSELoss().to(device)
    coordfilter = make_filter(53, 36, 62, 13, 14, 30, 44)

    a1 = input("train or test?")
    if (a1 != "train"):
        a2 = input("test or look?")
        if (a2 == "test"):
            model_test(look = False)
        else:
            model_test(look = True)
    else:
        # Learning cycle
        pupupu = time.time()
        print(f"Start epochs with time: {pupupu:.2f}")
        for epoch in range(epochs):
            print(f'Starting Epoch {epoch + 1}/{epochs}')
            print(f'Start time: {time.time() - pupupu:.2f} Device: {device}, Noise: {noise}, Displace: {displace}, Rotate: {rotate}')

            iter_times = []
            iter_loss = []
            average_loss = 1
            median_loss = 1
        
            dataloader = iter(dataset)

            for iteration in range(1, total_iterations + 1):
                bt_images, bt_coords = next(dataloader)

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
                    iter_per_second = 0.00
                if (l > 0):
                    average_loss = sum(iter_loss) / len(iter_loss)
                    median_loss = iter_loss[l // 2]
                
                # Update the progress bar
                print_progress_bar(iteration, total_iterations, epoch + 1, epochs, iter_per_second, loss, average_loss, median_loss)
            print("\n")
        
        print(f"End epichs time: {time.time() - pupupu:.2f}")

        # Save after every iterations
        if (input("Do you wanna save weigths? (y/n) ")[0] in 'yYнН'):
            postfix = input(f"Enter a postfix (enter - save to {weight_save_path}): ")
            if (postfix == ""):
                path = weight_save_path
            else:
                path = os.path.join(registry_path, 'weights', f'lvl1det_bns_{postfix}.pth')
            
            print(f'saving to {path}')
            torch.save(lvl1det.state_dict(), path)


if __name__ == "__main__":
    main()
