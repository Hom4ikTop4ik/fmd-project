import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import numpy as np
import torch
import cv2
import locale


LOG_IMAGES_EVERY = 100
LOG_IMAGES_COUNT = 4


# current_dir = os.path.dirname(os.path.abspath(__file__))
# log_subdir = datetime.now().strftime("%Y%m%d_%H%M%S")
# cur_logs_dir = os.path.join(current_dir, 'logs', log_subdir)
# os.makedirs(cur_logs_dir, exist_ok=True)  
# log_dir_hard_coded = os.path.join(cur_logs_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))

project_root = os.getcwd() 
cur_logs_dir = os.path.join(project_root, 'logs')
log_dir_hard_coded = os.path.join(cur_logs_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(cur_logs_dir, exist_ok=True)  

# return writer and log directory
def get_writer(log_dir : str = log_dir_hard_coded):
    writer = SummaryWriter(log_dir=log_dir)
    return writer, log_dir

def setup():
    np.set_printoptions(suppress=True, precision=8)
    torch.set_printoptions(sci_mode=False, precision=8)
    locale.setlocale(locale.LC_ALL, '')

# return float .4f
# 4 digits after point
def format_number(value):
    if isinstance(value, torch.Tensor):
        value = value.item()
    return float(f"{value:.4f}")

# put images to log
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

# add_scalar(
#   tag : str, 
#   scalar_value : float / str / blobname, 
#   global_step=None : int, 
#   walltime=None : float,           # not use, Optional override default walltime (time.time()) with seconds after epoch of event
#   new_style=False : boolean,       # not use, Whether to use new style (tensor field) or old style (simple_value field). New style could lead to faster data loading.
#   double_precision=False : boolean # not use, I don't know what is it.
# )
# 
# Example:
# writer.add_scalar('Loss/train', cur_loss, iteration)


# add_scalars(
#   main_tag, 
#   tag_scalar_dict, 
#   global_step=None, 
#   walltime=None
# )
#
# Example:
# writer.add_scalars(
#   'run_14h', # main_tag
#   {
#       'xsinx':i*np.sin(i/r),
#       'xcosx':i*np.cos(i/r),
#       'tanx': np.tan(i/r)
#   }, # dict {'tag' : value}
#   i # iteration
# )


# add_image(
#   tag : str, 
#   img_tensor : torch.Tensor / numpy.ndarray / str / blobname, 
#   global_step=None : int, 
#   walltime=None : float, 
#   dataformats='CHW' # CHW, HWC, HW, WH, etc.
# )
# 
# Example:
# writer.add_image('my_image', img, 0)
# writer.add_image('my_image_HWC', img_HWC, 0, dataformats='HWC')


# add_text(
#   tag : str, 
#   text_string : str, 
#   global_step=None : int, 
#   walltime=None : float
# )
# 
# Examples:
# writer.add_text('hihi', 'This is a hihi', 0)
# writer.add_text('haha', 'This is a haha', 10)

