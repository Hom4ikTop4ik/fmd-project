from .data_loader import *
from .normalizer import *
from .augments import *
from .utils import *

noise = 0.05
rotate = 45
min_scale = 0.8
max_scale = 1.2

epochs = 50
batch_size = 40
imgs_count = 20000
total_iterations = imgs_count // batch_size

iter_k = 20

MODE = "train"

DA = True
NET = False
USE_CPU_WHATEVER = NET
PROGRESS_BAR = DA
