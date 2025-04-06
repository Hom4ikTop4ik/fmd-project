import cv2
import os
import sys
import signal
import torch
import torch.nn as nn
import numpy as np
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import locale

from data_process.convertor_pca import MakerPCA
from detector.blocked import MultyLayer

from data_process import make_filter, scale_img
from data_process import load, noise, rotate, min_scale, max_scale, epochs, batch_size, imgs_count, total_iterations, iter_k
from data_process import PROGRESS_BAR, USE_CPU_WHATEVER, DA

np.set_printoptions(suppress=True, precision=4)
torch.set_printoptions(sci_mode=False, precision=4)


LOG_IMAGES_EVERY = 100
LOG_IMAGES_COUNT = 4
locale.setlocale(locale.LC_ALL, '') 

DOTS = 72

def format_number(value):
    if isinstance(value, torch.Tensor):
        value = value.item()
    return float(f"{value:.4f}")

def log_images_to_tensorboard(writer, images, predictions, targets, iteration):
    images = images[:LOG_IMAGES_COUNT]
    predictions = predictions[:LOG_IMAGES_COUNT]
    targets = targets[:LOG_IMAGES_COUNT]
    
    result_images = []
    
    for img, pred, target in zip(images, predictions, targets):
        img_np = img.cpu().numpy().transpose(1, 2, 0) * 255
        img_np = img_np.astype(np.uint8).copy()

        for coord in pred:
            x, y = int(coord[0] * img_np.shape[1]), int(coord[1] * img_np.shape[0])
            cv2.circle(img_np, (x, y), 3, (255, 0, 0), -1)

        for coord in target:
            x, y = int(coord[0] * img_np.shape[1]), int(coord[1] * img_np.shape[0])
            cv2.circle(img_np, (x, y), 2, (0, 255, 0), -1)

        cv2.putText(img_np, f"Iter: {iteration}", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
        result_images.append(img_tensor)
    
    grid = make_grid(result_images, nrow=2)
    writer.add_image('Predictions vs Targets', grid, iteration)

def interactor(signal, frame):
    print('SIGINT was received.')
    while DA:
        cmd = input('Choose action: save/exit (continue by default): ')
        match cmd:
            case 'save':
                path = weight_save_path
                print(f'saving to {path}')
                torch.save(model.state_dict(), path)
                writer.close()
            case 'exit':
                writer.close()
                sys.exit(0)
            case _:
                continue

def model_test(look=False):
    with torch.no_grad():
        iteration = 0
        loss = 0.0
        for bt_images, bt_coords in iter(dataloader):
            bt_images = bt_images.to(device)
            bt_coords = bt_coords.to(device)
            
            truth = mypca.compress(bt_coords).to(device)
            ans = model(bt_images)

            current_loss = criterion(ans, truth)
            loss += format_number(current_loss)
            iteration += 1

            if look:
                for i in range(min(4, batch_size)):
                    ans_i = ans[i:i+1].to("cpu")
                    dec = mypca.decompress(ans_i)
                    PCA2coords = dec.reshape(-1, 3, DOTS).permute(0, 2, 1)
                    look_predict(bt_images[i], PCA2coords)
        
        avg_loss = format_number(loss / iteration)
        writer.add_scalar('test_loss', avg_loss, global_step=global_iteration)
        print(f'Test loss: {avg_loss:.4f}')

def look_predict(imgtens: torch.Tensor, predict: torch.Tensor, show_depth=True):
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

def print_progress_bar(iteration, total, epoch, epochs, iter_per_second, loss, average_loss, median_loss):
    percent = (iteration / total) * 100
    bar_length = 10
    block = int(round(2*bar_length * percent / 100))
    progress = '▓' * (block//2) + '▒' * (block%2) + '░' * (bar_length - block // 2 - block % 2)
    sys.stdout.write(f'\r[{progress}] {percent:3.0f}% (Epoch: {epoch}/{epochs}, Iteration: {iteration}/{total}, Iter/s: {iter_per_second:4.2f}, Loss: {loss:.4f}, Avg: {average_loss:.4f}, Med: {median_loss:.4f})')
    sys.stdout.flush()

current_dir = os.path.dirname(os.path.abspath(__file__))
registry_path = os.path.join(current_dir, 'registry')
weight_save_path = os.path.join(registry_path, 'weights', 'model_bns.pth')

def main():
    global dataloader, device, model, optimizer, criterion, coordfilter, mypca, writer, global_iteration
    
    device = 'cuda' if torch.cuda.is_available() and not USE_CPU_WHATEVER else 'cpu'
    print(f"Using device: {device}")
    log_subdir = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = os.path.join(current_dir, 'logs', log_subdir)
    os.makedirs(logs_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=logs_dir)
    


    print("Loading dataset...")
    dataloader, sampler = load(
        bsize=batch_size,
        dataset_path=os.path.join(current_dir, registry_path, 'dataset', 'train'),
        device=device,
        coords_dir="coords",
        sampler_seed=int(time.time())
    )

    model = MultyLayer(device).to(device)
    if os.path.exists(weight_save_path):
        model.load_state_dict(torch.load(weight_save_path, map_location=device))
        print(f"Loaded weights from {weight_save_path}")

    signal.signal(signal.SIGINT, interactor)

    optimizer = torch.optim.Adam(model.parameters(), 0.001, betas=(0.9, 0.95))
    criterion = nn.MSELoss().to(device)
    coordfilter = make_filter(53, 36, 62, 13, 14, 30, 44)

    mypca = MakerPCA()
    mypca.load(os.path.join(current_dir, 'data_process', 'pcaweights_ext.pca'))

    global_iteration = 0
    start_time = time.time()

    for epoch in range(epochs):
        sampler.set_seed(epoch + int(time.time()))
        dataloader_iterator = iter(dataloader)
        
        epoch_loss = 0.0
        iter_times = []
        iter_losses = []
        
        for iteration in range(1, total_iterations + 1):
            iter_start = time.time()
            bt_images, bt_coords = next(dataloader_iterator)
            
            bt_images = bt_images.to(device)
            bt_coords = bt_coords.to(device)
            truth = mypca.compress(bt_coords).to(device)
            ans = model(bt_images)
            loss = criterion(ans, truth)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            formatted_loss = format_number(loss)
            writer.add_scalar('loss', formatted_loss, global_iteration)
            epoch_loss += formatted_loss

            if global_iteration % LOG_IMAGES_EVERY == 0:
                with torch.no_grad():
                    pred_coords = mypca.decompress(ans).reshape(-1, 3, DOTS).permute(0, 2, 1)
                    target_coords = mypca.decompress(truth).reshape(-1, 3, DOTS).permute(0, 2, 1)
                    log_images_to_tensorboard(writer, bt_images, pred_coords, target_coords, global_iteration)
            
            global_iteration += 1
            
            iter_times.append(time.time() - iter_start)
            iter_losses.append(formatted_loss)
            if len(iter_times) > iter_k:
                iter_times.pop(0)
                iter_losses.pop(0)
            
            iter_speed = len(iter_times) / sum(iter_times) if iter_times else 0
            avg_loss = sum(iter_losses) / len(iter_losses) if iter_losses else 0
            med_loss = sorted(iter_losses)[len(iter_losses)//2] if iter_losses else 0
            
            if PROGRESS_BAR:
                print_progress_bar(iteration, total_iterations, epoch+1, epochs, 
                                iter_speed, formatted_loss, avg_loss, med_loss)
        
        print(f"\nEpoch {epoch+1} completed. Avg loss: {format_number(epoch_loss/total_iterations)}")
        
        if epoch % 2 == 0:
            model_test(look=True)
    
    torch.save(model.state_dict(), weight_save_path)
    print(f"Model saved to {weight_save_path}")
    
    writer.close()
    print(f"Total training time: {format_number((time.time()-start_time)/60)} minutes")

if __name__ == "__main__":
    main()
