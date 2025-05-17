import torch
import torch.nn as nn

DA = True
NET = False
POFIG = NET

def print1(x):
    # print(x)
    pass

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

class ConvFeatureExtractor(nn.Module):
    def __init__(self, device, conv_layers_description):
        super(ConvFeatureExtractor, self).__init__()
        layers = []
        for desc in conv_layers_description:
            typee = desc[0].lower()
            if typee in ['conv']:
                _, in_ch, out_ch, k, s, p = desc
                layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p).to(device))
            elif typee in ['convblock']:
                _, in_ch, out_ch = desc
                layers.append(ConvBlock(device, in_ch, out_ch))
            elif typee in ['batch']:
                _, ch = desc
                layers.append(nn.BatchNorm2d(ch).to(device))
            else:
                raise ValueError(f'Unknown layer type: {desc[0]}')
            
        self.model = nn.Sequential(*layers)

    def get_output_size(self, input_size, device):
        """
        Пропускает фиктивное изображение через conv-модель и возвращает выходной размер.
        Для вычисления выходного размера после свёртки и пулинга.
        """
        was_training = self.model.training  # запомним состояние
        self.model.eval()  # выключаем dropout и batchnorm training-mode

        with torch.no_grad():
            x = torch.randn(1, 3, *input_size).to(device)  # Создаём случайный тензор с размерами [1, 3, H, W]
            for layer in self.model:
                x = layer(x)  # Пропускаем через все слои

        if was_training:
            self.model.train()  # восстановим исходное состояние

        return int(torch.prod(torch.tensor(x.shape[1:])))  # Возвращаем размерность выходного тензора (кроме размера батча)
    
    def forward(self, x):
        return self.model(x)

class Head(nn.Module):
    def __init__(self, device, insize, head_description):
        super(Head, self).__init__()
        self.insize = insize

        layers = []
        current_size = insize

        for desc in head_description:
            """
            desc = (type, size) or just (type)
            """
            typee = desc[0].lower()

            if typee in ['linear']:
                next_size = desc[1]
                layers.append(nn.Linear(current_size, next_size))
                current_size = next_size

            elif typee in ['batch']:
                layers.append(nn.BatchNorm1d(current_size))

            elif typee in ['leaky', 'leakyrelu']:
                arg = 0.01 # bazovichok
                if len(desc) > 1:
                    arg = desc[1]
                layers.append(nn.LeakyReLU(arg).to(device))

            elif typee in ['drop', 'dropout']:
                dropout_prob = desc[1]
                layers.append(nn.Dropout(dropout_prob))
                
            else:
                raise ValueError(f'Unknown layer type: {desc[0]}')


        self.model = nn.Sequential(*layers).to(device) # звёздочка для распаковки списка в кучу аргументов

    def forward(self, x):
        x = x.view(-1, self.insize)
        return self.model(x)

class MultyLayer(nn.Module):
    def __init__(self, device, PCA_COUNT, conv_desc = None, head_desc = None, IMG_SIZE = (512, 512)):
        super(MultyLayer, self).__init__()
        
        # base
        self.conv_description = [
            #         in_ch, out_ch, k, s, p
            ('conv',      3, 3,      7, 2, 3),  # pulconv: 512 -> 256
            ('convblock', 3, 6),                # 256 -> 128
            ('batch',     6),
            ('convblock', 6, 9),                # 128 -> 64
            ('batch',     9),
            ('convblock', 9, 16),               # 64 -> 32
            ('batch',     16),
            ('convblock', 16, 32),              # 32 -> 16
            ('batch',     32),
            ('convblock', 32, 64),              # 16 -> 8
            ('batch',     64),
            ('convblock', 64, 128),             # 8 -> 4
            ('batch',     128),
            ('convblock', 128, 256),            # 4 -> 2
            ('batch',     256),
            ('convblock', 256, 256),            # 2 -> 1
            ('batch',     256)
        ]
        if conv_desc:
            self.conv_description = conv_desc

        # base
        self.head_description = [
            ('linear', 128), 
            ('linear', 64), 
            ('linear', PCA_COUNT)
        ]
        if head_desc:
            self.head_description = head_desc

        self.conv_extractor = ConvFeatureExtractor(device, self.conv_description)
        # Получаем выходной размер
        conv_output_size = self.conv_extractor.get_output_size(IMG_SIZE, device)  # Размер входа 512x512
        
        self.head = Head(
            device,
            insize=conv_output_size,
            head_description=self.head_description
        )

    def forward(self, x):
        x = self.conv_extractor(x)  # [B, C, H, W]
        if POFIG:
            # если провалились с размерами слоёв, но хотим чтобы модель не упала
            x = x.view(x.size(0), -1)   # [B, C*H*W]
        x = self.head(x)            # подаём в полносвязную часть
        x = x.view(-1, x.shape[1])  # например, -1, 40
        return x
