import cv2
import os
import sys
import signal
import torch
import torch.nn as nn
import numpy as np
import time
import random

from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter
import locale


from data_process.convertor_pca import MakerPCA
from detector.blocked import MultyLayer

from data_process import make_filter, scale_img
from data_process import load, noise, rotate, min_scale, max_scale, epochs, batch_size, total_iterations, iter_k
from data_process import PROGRESS_BAR, USE_CPU_WHATEVER, DA, MODE

import tb.log_frog as log_frog

DOTS = 72

layer_sizes_list = [
    [128, 64, 40],
    [128, 96, 64, 40],
    [256, 192, 128, 96, 40],
    [256, 192, 96, 64, 40],
    [256, 192, 128, 96, 64, 40]
]

MODELS_CNT = len(layer_sizes_list)

def my_exit():
    [writer.close() for writer in writer_list]
    sys.exit(0)

def interactor(signal, frame):
    print('SIGINT was received.')
    while DA:
        cmd = input('Choose action: save/exit (continue by default): ')
        match cmd:
            case 'save':
                for i in range(MODELS_CNT):
                    print(f"cur model's layers size: {layer_sizes_list[i]}")
                    postfix = input(f"Enter a postfix (enter key - save to {weight_save_path}): ")
                    if (postfix == ""):
                        path = weight_save_path
                    else:
                        path = os.path.join(registry_path, 'weights', f'model_bns_{postfix}.pth')
                    
                    print(f'saving to {path}')
                    torch.save(model_list[i].state_dict(), path)
            case 'test':
                print("no test, rerun program")
                # model_test(look=False)
            case 'look':
                print("no look, rerun program")
                # model_test(look=True)

            case 'exit':
                my_exit()
            case 'учше':
                my_exit()
            case 'clear':
                my_exit()
            case 'сдуфк':
                my_exit()

def model_test(look=False):
    with torch.no_grad():
        iteration = 0
        loss_list = [0.0]*MODELS_CNT
        for bt_images, bt_coords in iter(dataloader):
            iteration += 1
            
            bt_images = bt_images.to(device)
            bt_coords = bt_coords.to(device)
            
            truth = mypca.compress(bt_coords).to(device)
            for j in range(MODELS_CNT):
                ans = model_list[j](bt_images)

                current_loss = criterion(ans, truth)
                loss_list[j] += current_loss

                if look:
                    for i in range(batch_size):
                        ans_i = ans[i:i+1].to("cpu")
                        dec = mypca.decompress(ans_i)
                        PCA2coords = dec.reshape(-1, 3, DOTS).permute(0, 2, 1)
                        look_predict(bt_images[i], PCA2coords, show_depth=False, layers=layer_sizes_list[j])
        
        avg_loss_list = [log_frog.format_number(loss / iteration) for loss in loss_list]
        for i in range(MODELS_CNT):
            writer_list[i].add_scalar('test_avg_loss', avg_loss_list[i], global_step=global_iteration)
        print(f'Test avg_loss_list: {avg_loss_list}')
        print(f'Iterations count {iteration}')

def look_predict(imgtens: torch.Tensor, predict: torch.Tensor, show_depth: bool = True, layers: list = []):
    predict = predict[0]
    img = scale_img(imgtens, 1.5, "bilinear")
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
    
    cv2.imshow(f"Prediction {layers} {str(random.random())}", newimg)
    cv2.waitKey(1000)
    time.sleep(2)

def print_progress_bar(start_if_line: str, iteration, total, epoch, epochs, iter_per_second, loss, average_loss, median_loss):
    percent = (iteration / total) * 100
    bar_length = 10
    block = int(round(2*bar_length * percent / 100))
    progress = '▓' * (block//2) + '▒' * (block%2) + '░' * (bar_length - block // 2 - block % 2)
    sys.stdout.write(f'\r{start_if_line}[{progress}] {percent:3.0f}% (Epoch: {epoch}/{epochs}, Iteration: {iteration}/{total}, Iter/s: {iter_per_second:4.2f}, Loss: {loss:.4f}, Avg: {average_loss:.4f}, Med: {median_loss:.4f})')
    sys.stdout.flush()


def passer(a, b):
    pass
signal.signal(signal.SIGINT, passer)

current_dir = os.path.dirname(os.path.abspath(__file__))
registry_path = os.path.join(current_dir, 'registry')
weight_save_path = os.path.join(registry_path, 'weights', 'model_bns.pth')

def main():
    global dataloader, device, model_list, optimizer_list, criterion, coordfilter, mypca, writer_list, global_iteration
    
    log_frog.setup()
    
    device = 'cpu'
    if (USE_CPU_WHATEVER == False):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    log_subdir = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = os.path.join(current_dir, 'logs', log_subdir)
    writer_list = []
    for layer_sizes in layer_sizes_list:
        logs_dir_model = os.path.join(logs_dir, str(layer_sizes))
        os.makedirs(logs_dir_model, exist_ok=True)
        writer_tmp, log_frog_dir_tmp = log_frog.get_writer(logs_dir_model)
        writer_list.append(writer_tmp)
        print(f"Writer {str(layer_sizes)} — log directory: {log_frog_dir_tmp}")
        
    # os.makedirs(logs_dir, exist_ok=True)
    # writer, log_frog_dir = log_frog.get_writer(logs_dir)
    # print(f"Log directory: {log_frog_dir}")

    print("\nNow you can load none/several/all weights:")
    model_list = []
    for layer_sizes in layer_sizes_list:
        model_list.append(MultyLayer(device, layer_sizes)) 
        print(f"cur layer_sizes: {layer_sizes}")
        
        if input('load weigths from selected weight save path? (y/n) ') in 'yYнН':
            postfix = input(f"Enter a postfix (enter key - save to {weight_save_path}): ")
        
            if (postfix == ""):
                path = weight_save_path
            else:
                path = os.path.join(registry_path, 'weights', f'model_bns_{postfix}.pth')
            print(f"Loaded weights from {path}")
            model_list[-1].load_state_dict(torch.load(path))
    
    # model = MultyLayer(device, layer_sizes).to(device)
    # if input('load weigths from selected weight save path? (y/n) ') in 'yYнН':
    #     postfix = input(f"Enter a postfix (enter key - save to {weight_save_path}): ")
    #     if (postfix == ""):
    #         path = weight_save_path
    #     else:
    #         path = os.path.join(registry_path, 'weights', f'model_bns_{postfix}.pth')
    #     print(f"Loaded weights from {path}")
    #     model.load_state_dict(torch.load(path))


    print("Loading DataLoader...")
    start_loading_dataloader = time.time()
    dataloader, sampler = load(
        bsize=batch_size,
        dataset_path=os.path.join(current_dir, registry_path, 'dataset', 'train'),
        device=device,
        coords_dir="coords",
        # sampler_seed=int(time.time())
    )

    signal.signal(signal.SIGINT, interactor) # I WANT CLOSE PROGRAM
    
    # launch workers
    next(iter(dataloader))
    print(f"Finish load DataLoader after {log_frog.format_number(time.time() - start_loading_dataloader)} secs")

    optimizer_list = [
        torch.optim.Adam(model.parameters(), 0.001, betas=(0.9, 0.95)) 
        for model in model_list
    ]
    
    # optimizer = torch.optim.Adam(model.parameters(), 0.001, betas=(0.9, 0.95))
    criterion = nn.MSELoss().to(device)
    coordfilter = make_filter(53, 36, 62, 13, 14, 30, 44)

    mypca = MakerPCA()
    mypca.load(os.path.join(current_dir, 'data_process', 'pcaweights_ext.pca'))

    global_iteration = 0
    start_time = time.time()
    
    match MODE:
        case "test":
            print("Test loss mode")
            model_test(look = False)
        case "look":
            print("Look mode")
            model_test(look = True)
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
                # next_seed += int(time.time())
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
                    
                    loss_fst = None
                    for i in range(MODELS_CNT):
                        ans = model_list[i](bt_images)
                        loss = criterion(ans, truth)
                        
                        if loss_fst == None:
                            loss_fst = loss

                        optimizer_list[i].zero_grad()
                        
                        loss.backward()
                        optimizer_list[i].step()
                    
                        epoch_loss += loss
                        formatted_loss = log_frog.format_number(loss)
                        writer_list[i].add_scalar('loss', formatted_loss, global_iteration)

                        if global_iteration % log_frog.LOG_IMAGES_EVERY == 0:
                            with torch.no_grad():
                                pred_coords = mypca.decompress(ans).reshape(-1, 3, DOTS).permute(0, 2, 1)
                                target_coords = mypca.decompress(truth).reshape(-1, 3, DOTS).permute(0, 2, 1)
                                log_frog.log_images_to_tensorboard(writer_list[i], bt_images, pred_coords, target_coords, global_iteration)

                    # ans = model(bt_images)
                    # loss = criterion(ans, truth)
                    # optimizer.zero_grad()
                    
                    # loss.backward()
                    # optimizer.step()
                
                    # epoch_loss += loss
                    # formatted_loss = log_frog.format_number(loss)
                    # writer.add_scalar('loss', formatted_loss, global_iteration)

                    # if global_iteration % log_frog.LOG_IMAGES_EVERY == 0:
                    #     with torch.no_grad():
                    #         pred_coords = mypca.decompress(ans).reshape(-1, 3, DOTS).permute(0, 2, 1)
                    #         target_coords = mypca.decompress(truth).reshape(-1, 3, DOTS).permute(0, 2, 1)
                    #         log_frog.log_images_to_tensorboard(writer, bt_images, pred_coords, target_coords, global_iteration)
                
                    global_iteration += 1
                    
                    if PROGRESS_BAR:
                        # Record the time taken for this iteration
                        iter_times.append(time.time())
                        iter_loss.append(loss_fst)
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
                        
                        print_progress_bar("\t\t", iteration, total_iterations, epoch + 1, epochs, iter_per_second, loss, average_loss, median_loss)
                    else:
                        print(f'\r\t\tloss {loss_fst:.5f}, iteration: {iteration}', end = '')
            
                print(f"\n\t\tEpoch {epoch+1} completed. Avg loss: {log_frog.format_number(epoch_loss/total_iterations)}\n")

                # # NO, PLEASE, NO. IT WILL START TEST ALL DATASET 💀 (about 20k images!)      
                # if epoch % 2 == 0:
                #     print("Let's start model_test(look=True)!")
                #     model_test(look=True)
        
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
