import cv2
import os
import sys
import signal
import torch
import torch.nn as nn
import numpy as np
import time
from datetime import datetime

from data_process.convertor_pca import MakerPCA, PCA_COUNT, VERSION
from detector.blocked import MultyLayer

from data_process import make_filter, scale_img, show_image_coords
from data_process import load, noise, rotate, min_scale, max_scale, epochs, BATCH_SIZE, total_iterations, iter_k
from data_process import PROGRESS_BAR, USE_CPU_WHATEVER, DA, NET, MODE

import tb.log_frog as log_frog

DOTS = 72
LOG_IMAGES_OR_NO = -1 # 0 -> YES; less than 0 -> NO, because if (smth1 % smth2 == -1) is never
DEBUG = NET

AUG = DA

head_desc1 = [
    ('linear', 512), 
    ('linear', 384), 
    ('linear', 256), 
    ('linear', 128), 
    ('linear', 512), 
    ('linear', 64), 
    ('linear', PCA_COUNT)
]

def interactor(signal, frame):
    print('\n\nSIGINT was received.')
    while DA:
        cmd = input('Choose action: save/exit (continue by default): ')
        match cmd:
            case 'save':
                path = weight_save_path
                print(f'saving to {path}')
                torch.save(model.state_dict(), path)
                writer.close()
            case 'test':
                print("no test, rerun program")
                # model_test(look=NET)
            case 'look':
                print("no look, rerun program")
                # model_test(look=DA)

            case 'exit':
                writer.close()
                sys.exit(0)
            case 'учше':
                writer.close()
                sys.exit(0)
            case 'clear':
                writer.close()
                sys.exit(0)
            case 'сдуфк':
                writer.close()
                sys.exit(0)

def model_test(look=NET):
    with torch.no_grad():
        iteration = 0
        loss = 0.0
        for bt_images, bt_coords in iter(dataloader):
            bt_images = bt_images.to(device)
            bt_coords = bt_coords.to(device)
            
            truth = mypca.compress(bt_coords).to(device)
            ans = model(bt_images)

            current_loss = criterion(ans, truth)
            loss += current_loss
            iteration += 1

            if look:
                for i in range(BATCH_SIZE):
                    ans_i = ans[i:i+1].to("cpu")
                    dec = mypca.decompress(ans_i)
                    PCA2coords = dec
                    look_predict(bt_images[i], PCA2coords)
        
        avg_loss = log_frog.format_number(loss / iteration)
        writer.add_scalar('test_loss', avg_loss, global_step=global_iteration)
        print(f'Test loss: {avg_loss}')
        print(f'Iterations count {iteration}')

def look_predict(imgtens: torch.Tensor, predict: torch.Tensor, show_depth=DA):
    predict = predict[0]
    img = scale_img(imgtens, 2, "bilinear")
    newimg = (img[[0,1,2], :, :].cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
    newimg = np.ascontiguousarray(newimg)
    
    for coord in predict:
        x, y, value = coord[0], coord[1], coord[2]
        x_pixel = int(x * newimg.shape[1])
        y_pixel = int(y * newimg.shape[0])
        cv2.circle(newimg, (x_pixel, y_pixel), 2, (255, 255, 255), 2)
        if show_depth:
            cv2.putText(newimg, f"{value:.4f}", (x_pixel + 5, y_pixel - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Prediction", newimg)
    cv2.waitKey(1000)

def print_progress_bar(start_if_line: str, iteration, total, epoch, epochs, iter_per_second, loss, average_loss, median_loss, timetime):
    percent = (iteration / total) * 100
    bar_length = 10
    block = int(round(2*bar_length * percent / 100))
    progress = '▓' * (block//2) + '▒' * (block%2) + '░' * (bar_length - block // 2 - block % 2)
    sys.stdout.write(f'\r{start_if_line}[{progress}] {percent:3.0f}% (Epoch: {epoch}/{epochs}, Iter: {iteration}/{total}, Iter/s: {iter_per_second:4.2f}, Time: {timetime},Loss: {loss:.4f}, Avg: {average_loss:.4f}, Med: {median_loss:.4f})')
    sys.stdout.flush()


def passer(a, b):
    pass
signal.signal(signal.SIGINT, passer)

current_dir = os.path.dirname(os.path.abspath(__file__))
registry_path = os.path.join(current_dir, 'registry')
if VERSION == "NEW":
    last_weigth_path_part = 'model_bns_GROUPS_3.pth'
elif VERSION == "OLD":
    last_weigth_path_part = 'model_bns_16PCA_60_epochs.pth'
weight_save_path = os.path.join(registry_path, 'weights', last_weigth_path_part)

def main():
    global dataloader, device, model, optimizer, criterion, coordfilter, mypca, writer, global_iteration
    
    log_frog.setup()
    
    device = 'cpu'
    if (USE_CPU_WHATEVER == NET):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    log_subdir = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = os.path.join(current_dir, 'logs', log_subdir)
    os.makedirs(logs_dir, exist_ok=DA)
    writer, log_frog_dir = log_frog.get_writer(logs_dir)
    print(f"Log directory: {log_frog_dir}")


    model = MultyLayer(device, PCA_COUNT, head_desc=head_desc1).to(device)
    if input('load weigths from selected weight save path? (y/n) ') in 'yYнН':
        print(f"Loaded weights from {weight_save_path}")
        model.load_state_dict(torch.load(weight_save_path))
        # model.load_state_dict(torch.load(weight_save_path, map_location = device))

    mypca = MakerPCA()
    # if input('Load PCA weigths from its weight save path? (y/n) ') in 'yYнН':
    if VERSION == "NEW":
        last_PCA_path_part = 'pcaweights_ext_GROUPS_3.pca'
    elif VERSION == "OLD":
        last_PCA_path_part = 'pcaweights_16PCA.pca'
    mypca.load(os.path.join(current_dir, 'data_process', last_PCA_path_part))

    print("Loading DataLoader...")
    start_loading_dataloader = time.time()
    dataloader, sampler = load(
        bsize=BATCH_SIZE,
        dataset_path=os.path.join(current_dir, registry_path, 'dataset', 'train'),
        device=device,
        coords_dir="coords",
        sampler_seed=int(time.time()),
        augments=AUG,
        workers=DA
    )

    signal.signal(signal.SIGINT, interactor) # Now I want stop program and ask user "save/exit?"
    
    # launch workers
    next(iter(dataloader))
    print(f"Finish load DataLoader after {log_frog.format_number(time.time() - start_loading_dataloader)} secs")

    optimizer = torch.optim.Adam(model.parameters(), 0.001, betas=(0.9, 0.95))
    criterion = nn.MSELoss().to(device)
    coordfilter = make_filter(53, 36, 62, 13, 14, 30, 44)

    global_iteration = 0
    start_time = time.time()
    
    match MODE:
        case "test":
            print("Test loss mode")
            model_test(look = NET)
        case "look":
            print("Look mode")
            model_test(look = DA)
        case "train":
            print("Train mode")
            print(f"Device: {device}, Noise: {noise}, Rotate: {rotate}, min_scale: {min_scale}, max_scale: {max_scale}")
            pupupu = time.time()
            print(f"\tStart epochs with time: {pupupu:.2f}")
            # Learning cycle
            for epoch in range(epochs):
                print(f'\t\tStarting Epoch {epoch + 1}/{epochs}, time: {log_frog.format_number(time.time() - pupupu)}')

                iter_times = []
                iter_loss = []
                average_loss = 1
                median_loss = 1

                next_seed = epoch
                next_seed += int(time.time())
                sampler.set_seed(next_seed) # new shuffle
                dataloader_iterator = iter(dataloader)

                epoch_loss = 0.0

                for iteration in range(1, total_iterations + 1):
                    bt_images, bt_coords = next(dataloader_iterator)

                    # Move tensors to the device
                    bt_images = bt_images.to(device)
                    bt_coords = bt_coords.to(device)

                    # Training code
                    truth = mypca.compress(bt_coords).to(device)
                    
                    if DEBUG:
                        decompress_for_debug = mypca.decompress(truth).to(device)
                        show_image_coords(bt_images[0], decompress_for_debug[0])
                        show_image_coords(bt_images[0], bt_coords[0])
                        # show_image_coords(bt_images[1], decompress_for_debug[1])
                        # show_image_coords(bt_images[2], decompress_for_debug[2])
                        # show_image_coords(bt_images[3], decompress_for_debug[3])
                        sys.exit()
                    
                    
                    ans = model(bt_images)
                    loss = criterion(ans, truth)
                    optimizer.zero_grad()
                    
                    loss.backward()
                    optimizer.step()
                
                    epoch_loss += loss
                    formatted_loss = log_frog.format_number(loss)
                    writer.add_scalar('loss', formatted_loss, global_iteration)

                    if global_iteration % log_frog.LOG_IMAGES_EVERY == LOG_IMAGES_OR_NO:
                        with torch.no_grad():
                            print("\n\rLOG FROG \t\t_\r\n")
                            pred_coords = mypca.decompress(ans)#.reshape(-1, 3, DOTS).permute(0, 2, 1)
                            target_coords = mypca.decompress(truth)#.reshape(-1, 3, DOTS).permute(0, 2, 1)
                            log_frog.log_images_to_tensorboard(writer, bt_images, pred_coords, target_coords, global_iteration)
                
                    global_iteration += 1
                    
                    if PROGRESS_BAR:
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
                        
                        print_progress_bar("\t\t", iteration, total_iterations, epoch + 1, epochs, iter_per_second, loss, average_loss, median_loss, log_frog.format_number(time.time() - pupupu))
                    else:
                        print(f'\t\tloss {loss:.5f}, iteration: {iteration}')

                backup_path = weight_save_path + f'{time.time()}'
                print(f'\nbackup in {backup_path}')
                torch.save(model.state_dict(), backup_path)

                print(f"\n\t\tEpoch {epoch+1} completed. Avg loss: {log_frog.format_number(epoch_loss/total_iterations)}\n")

                # # NO, PLEASE, NO. IT WILL START TEST ALL DATASET 💀 (about 20k images!)      
                # if epoch % 2 == 0:
                #     print("Let's start model_test(look=DA)!")
                #     model_test(look=DA)
            #end for epochs
        
            # Save after all iterations
            if (input("\nDo you wanna save weigths? (y/n) ")[0] in 'yYнН'):
                postfix = input(f"Enter a postfix (enter key - save to {weight_save_path}): ")
                if (postfix == ""):
                    path = weight_save_path
                else:
                    path = os.path.join(registry_path, 'weights', f'model_bns_{postfix}.pth')
                
                print(f'saving to {path}')
                torch.save(model.state_dict(), path)

            secs = log_frog.format_number(time.time() - start_time)
            mins = secs / 60
            print(f"Total training time: {mins} minutes  or  {secs} seconds")
    
    writer.close()

if __name__ == "__main__":
    main()
else:
    print(f"\tI'm a worker, and this is a time: {time.time()}")
