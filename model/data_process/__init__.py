from .data_loader import *
from .normalizer import *
from .augments import *
from .utils import *
from .convertor_pca import MakerPCA

noise = 0.05
rotate = 45
min_scale = 0.8
max_scale = 1.2
blur_level = 7

epochs = 50
BATCH_SIZE = 64
imgs_count = 20000
total_iterations = imgs_count // BATCH_SIZE

iter_k = 20

MODE = "train"

DA = True
NET = False
USE_CPU_WHATEVER = NET
PROGRESS_BAR = DA
