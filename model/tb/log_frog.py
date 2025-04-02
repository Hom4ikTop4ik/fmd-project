import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

your_name = "BDD"

current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.getcwd() 

logs_dir = os.path.join(project_root, 'logs', your_name)
os.makedirs(logs_dir, exist_ok=True)  

log_dir_hard_coded = os.path.join(logs_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))

def get_writer(log_dir : str = log_dir_hard_coded):
    writer = SummaryWriter(log_dir=log_dir)
    return writer

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

