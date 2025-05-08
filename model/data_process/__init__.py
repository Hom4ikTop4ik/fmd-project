from .data_loader import *
from .normalizer import *
from .augments import *
from .utils import *
from .convertor_pca import MakerPCA

noise = 0.15
rotate = 30
min_scale = 1
max_scale = 1

epochs = 4
batch_size = 60
imgs_count = 20000
part = 0.9
total_iterations = int(imgs_count * part) // batch_size

iter_k = 20

DA = True
NET = False
USE_CPU_WHATEVER = NET
PROGRESS_BAR = DA
