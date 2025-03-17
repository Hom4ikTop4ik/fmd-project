import cv2
import os
import sys
import signal
import torch
import torch.nn as nn
import numpy as np
import time

# from data_process import augment_gen, make_filter
from data_process.convertor_pca import MakerPCA
# from data_process import load
from detector.blocked import MultyLayer

from data_process import make_filter
from data_process import load, noise, rotate, min_scale, max_scale, epochs, batch_size, imgs_count, total_iterations, iter_k
from data_process import USE_CPU_WHATEVER
# from cascade_detector import Level1Detector

from tb.log_frog import get_writer


def interactor(signal, frame):
    print('SIGINT was received.')
    while (True):
        cmd = input('Choose action: save/test/look/exit (continue by default): ')
        match cmd:
            case 'save':
                path = weight_save_path
                print(f'saving to {path}')
                torch.save(model.state_dict(), path)
            case 'test':
                model_test(look=False)
            case 'look':
                model_test(look=True)

            case 'exit':
                sys.exit(0)
            case 'учше':
                sys.exit(0)
            case 'clear':
                sys.exit(0)
            case 'сдуфк':
                sys.exit(0)

def model_test(look=False):
    with torch.no_grad():
        iteration = 0
        loss = 0.0
        for bt_images, bt_coords in iter(dataloader):
            
            truth = coordfilter(bt_coords)[:, :, 0:2]
            ans = model(bt_images)
            loss += criterion(ans, truth)
            iteration += 1
            if look:
                look_predict(bt_images, ans)
        
        print(f'total loss {loss / iteration}')
        print(f'total iterations count {iteration}')

def look_predict(imgtens: torch.Tensor, predict: torch.Tensor, show_depth = True):
    predict = predict[0]

    newimg = (imgtens[0,[2,1,0], :, :].cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)

    newimg = np.ascontiguousarray(newimg)
    for coord in predict:
        x, y, value = coord[0], coord[1], coord[2]
        x_pixel = int(x * newimg.shape[1])
        y_pixel = int(y * newimg.shape[0])
        newimg = cv2.circle(newimg, (x_pixel, y_pixel), 2, (255, 255, 255), 2)
        if (show_depth):
            text = f"{value:.2f}"
            cv2.putText(newimg, text, (x_pixel + 5, y_pixel - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("img", newimg)
    cv2.waitKey(0)

def gen_lim(generator, percent):
    for i, obj in enumerate(generator):
        if i >= percent * len(generator):
            break
        yield obj

# establishing paths and loading
# dataset = load(500, 40, os.path.join(current_dir, registry_path, 'dataset'))

# # establishing devices and signal handler
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("devise is: ", device)
# signal.signal(signal.SIGINT, interactor)


# model = MultyLayer(device).to(device)
# # if input('load weigths from selected weight save path? (y/n) ') == 'y':
# #     model.load_state_dict(torch.load(weight_save_path))

# optimizer = torch.optim.Adam(model.parameters(), 0.001, betas=(0.9, 0.95))
# criterion = nn.MSELoss().to(device)
# coordfilter = make_filter(53, 36, 62, 13, 14, 30, 44)

# mypca = MakerPCA()
# mypca.load(os.path.join(current_dir,'data_process/pcaweights.pca'))

# writer = get_writer()

# learning cycle
# iteration = 0
# for bt_images, bt_coords in augment_gen(dataset, epochs=10, device=device,
#                                         noise=0, part=0.9, displace=80, rotate=20):
    
#     truth = mypca.compress(bt_coords).to(device)

#     ans = model(bt_images)
#     loss = criterion(ans, truth)
#     optimizer.zero_grad()

#     loss.backward()
#     optimizer.step()
    
#     iteration += 1
#     print(f'loss {loss:.5f}, iteration: {iteration}')

#     if iteration % 1500 == 0:
#         dec_answer = mypca.decompress(ans[:1]).reshape(-1, 3, 68).permute(0, 2, 1)
#         dec_truth = mypca.decompress(truth[:1]).reshape(-1, 3, 68).permute(0, 2, 1)
#         print('dec ans shape: ', dec_answer.shape)
#         print('dec tru shape: ', dec_answer.shape)
#         look_predict(bt_images, dec_truth)
#         look_predict(bt_images, dec_answer)

def print_progress_bar(iteration, total, epoch, epochs, iter_per_second, loss, average_loss, median_loss):
    percent = (iteration / total) * 100
    bar_length = 10  # Length of the progress bar
    block = int(round(2*bar_length * percent / 100))
    progress = '▓' * (block//2) + '▒' * (block%2) + '░' * (bar_length - block // 2 - block % 2)
    sys.stdout.write(f'\r[{progress}] {percent:3.0f}% (Epoch: {epoch}/{epochs}, Iteration: {iteration}/{total}, Iter/s: {iter_per_second:4.2f}, Loss: {loss:.5f}, Average Loss: {average_loss:.5f}, Median Loss: {median_loss:.5f})')
    sys.stdout.flush()

print(time.time())

def main():
    global dataloader, device, model, optimizer, criterion, coordfilter, current_dir, registry_path, weight_save_path
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    registry_path = os.path.join(current_dir, 'registry')
    weight_save_path = os.path.join(registry_path, 'weights', 'model_bns.pth')

    device = 'cpu'
    if (USE_CPU_WHATEVER == False):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("devise is: ", device)

    print("Start  load dataset with time: {}".format(time.time()))
    dataloader, sampler = load(
        bsize=batch_size, 
        dataset_path=os.path.join(current_dir, registry_path, 'dataset', 'train'),
        device=device
    )
    print("Finish load dataset with time: {}".format(time.time()))
    signal.signal(signal.SIGINT, interactor)

    model = MultyLayer(device).to(device)
    if input('load weigths from selected weight save path? (y/n) ')[0] in 'yYнН':
        model.load_state_dict(torch.load(weight_save_path, map_location = device))

    optimizer = torch.optim.Adam(model.parameters(), 0.001, betas=(0.9, 0.95))
    criterion = nn.MSELoss().to(device)
    coordfilter = make_filter(53, 36, 62, 13, 14, 30, 44)

    mypca = MakerPCA()
    mypca.load(os.path.join(current_dir,'data_process/pcaweights.pca'))

    writer = get_writer()

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
            print(f'Start time: {time.time() - pupupu:.2f} Device: {device}, Noise: {noise}, Rotate: {rotate}')

            iter_times = []
            iter_loss = []
            average_loss = 1
            median_loss = 1

            next_seed = epoch + int(time.time())
            sampler.set_seed(next_seed) # new shuffle
            dataloader_iterator = iter(dataloader)

            for iteration in range(1, total_iterations + 1):
                bt_images, bt_coords = next(dataloader_iterator)

                # Move tensors to the device
                bt_images = bt_images.to(device)
                bt_coords = bt_coords.to(device)

                # Training code
                truth = coordfilter(bt_coords)[:, :, 0:2] 
                ans = model(bt_images)
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

        # Save after all iterations
        if (input("Do you wanna save weigths? (y/n) ")[0] in 'yYнН'):
            postfix = input(f"Enter a postfix (enter - save to {weight_save_path}): ")
            if (postfix == ""):
                path = weight_save_path
            else:
                path = os.path.join(registry_path, 'weights', f'lvl1det_bns_{postfix}.pth')
            
            print(f'saving to {path}')
            torch.save(model.state_dict(), path)


if __name__ == "__main__":
    main()
