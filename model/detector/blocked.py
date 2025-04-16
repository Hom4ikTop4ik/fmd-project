import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, device, insize, outsize):
        super(ConvBlock, self).__init__()
        self.act = nn.ReLU().to(device)
        #                      in/out_channels*
        #      torch.nn.Conv2d(in_chs, out_chs, kernel_size, stride=1, 
        #                      padding=0, dilation=1, groups=1, bias=True, 
        #                      padding_mode='zeros', device=None, dtype=None)
        self.conv1    = nn.Conv2d(insize, insize, 3, padding = 1).to(device) #ker=3*3, stride=1
        self.conv2    = nn.Conv2d(insize, insize, 3, padding = 1).to(device) 
        self.conv3    = nn.Conv2d(insize, outsize, 3, padding = 1).to(device)
        self.skipconv = nn.Conv2d(insize, outsize, 2, stride=2).to(device) # ker=2*2, stride=2
        
        #     torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, 
        #                        dilation=1, return_indices=False, 
        #                        ceil_mode=False)
        # объединить блок 2*2 в 1*1, то есть сжать в 2 раза, оставив максимум
        # |a‾b|
        # |c_d| -> |e|, где e = max(a,b,c,d)
        self.pool = nn.MaxPool2d(2, 2).to(device) 

    def forward(self, x):
        x1 = x.clone()
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        return self.pool(x) + self.skipconv(x1)

class Head(nn.Module):
    def __init__(self, device, insize, layer_sizes, use_bn=True, dropout_prob=0.0):
        super(Head, self).__init__()
        self.insize = insize
        self.use_bn = use_bn
        self.dropout_prob = dropout_prob

        layers = []
        current_size = insize

        for next_size in layer_sizes:
            layers.append(nn.Linear(current_size, next_size))

            if use_bn:
                layers.append(nn.BatchNorm1d(next_size))

            layers.append(nn.LeakyReLU(0.01).to(device))

            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))

            current_size = next_size

        self.model = nn.Sequential(*layers).to(device) # звёздочка для распаковки списка в кучу аргументов

    def forward(self, x):
        x = x.view(-1, self.insize)
        return self.model(x)

def print1(x):
    # print(x)
    pass

class MultyLayer(nn.Module):
    def __init__(self, device):
        super(MultyLayer, self).__init__()
        self.pulconv = nn.Conv2d(3, 3, 7, stride=2, padding=3).to(device) 
        # out_size = floor((512 - kernel_size + 2*padding)/stride) + 1
        # out_size = (512 - 7 + 6)//2 + 1 = 511//2 + 1 = 256

        self.cblock1 = ConvBlock(device, 3, 6) # 256*256 -> 128x128
        self.bn1 = nn.BatchNorm2d(6).to(device)
        self.cblock2 = ConvBlock(device, 6, 9) # 128*128 -> 64x64
        self.bn2 = nn.BatchNorm2d(9).to(device)
        self.cblock3 = ConvBlock(device, 9, 16) # 64*64 -> 32x32
        self.bn3 = nn.BatchNorm2d(16).to(device)
        self.cblock4 = ConvBlock(device, 16, 32) # 32*32 -> 16x16
        self.bn4 = nn.BatchNorm2d(32).to(device)
        self.cblock5 = ConvBlock(device, 32, 64) # 16*16 -> 8x8
        self.bn5 = nn.BatchNorm2d(64).to(device)
        self.cblock6 = ConvBlock(device, 64, 128) # 8*8 -> 4x4
        self.bn6 = nn.BatchNorm2d(128).to(device)
        self.cblock7 = ConvBlock(device, 128, 256) # 4*4 -> 2*2
        self.bn7 = nn.BatchNorm2d(256).to(device)
        
        self.cblock_final = ConvBlock(device, 256, 256) # 2*2 -> 1*1
        self.bn_final = nn.BatchNorm2d(256).to(device)

        # now tensor size is [40, 256, 1, 1]

        # # input_size = 256, mid_size = 128, output_size=PCA_count=40
        # self.head_old = Head_old(device, 256, 128, 40)
        # # is equals
        # self.head = Head(device, insize=256, layer_sizes=[128, 40], use_bn=False, dropout_prob=0.0) 
        
        self.head = Head(device, insize=256, layer_sizes=[128, 64, 40], use_bn=False, dropout_prob=0.0) 

    def forward(self, x):
        # Input: [batch, 3, H, W]
        print1(x.shape)
        x = self.pulconv(x)
        print1(x.shape)
        x = self.bn1(self.cblock1(x))
        print1(x.shape)
        x = self.bn2(self.cblock2(x))
        print1(x.shape)
        x = self.bn3(self.cblock3(x))
        print1(x.shape)
        x = self.bn4(self.cblock4(x))
        print1(x.shape)
        x = self.bn5(self.cblock5(x))
        print1(x.shape)
        x = self.bn6(self.cblock6(x))
        print1(x.shape)
        x = self.bn7(self.cblock7(x))
        print1(x.shape)
        x = self.bn_final(self.cblock_final(x))
        print1(x.shape)
        x = self.head(x)
        print1(x.shape)
        x = x.view(-1, 40)
        return x
