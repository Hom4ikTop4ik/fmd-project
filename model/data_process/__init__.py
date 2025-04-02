from .data_loader import *
from .normalizer import *
from .augments import *
from .utils import *

noise = 0.05
rotate = 45
min_scale = 0.4
max_scale = 1.3

epochs = 50
batch_size = 40
imgs_count = 20000
part = 0.9
total_iterations = int(imgs_count * part) // batch_size

iter_k = 20

DA = True
NET = False
USE_CPU_WHATEVER = NET
PROGRESS_BAR = DA
