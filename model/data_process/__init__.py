from .data_loader import *
from .normalizer import *
from .augments import *
from .utils import *
from .convertor_pca import MakerPCA

noise = 0.05
rotate = 30
min_scale = 0.8
max_scale = 1.2
blur_level = 7

epochs = 500
BATCH_SIZE = 64
imgs_count = 31455
total_iterations = imgs_count // BATCH_SIZE

iter_k = 200

MODE = "train"

DA = True
NET = False
POCHTI = "pochti"
USE_CPU_WHATEVER = NET
PROGRESS_BAR = DA
