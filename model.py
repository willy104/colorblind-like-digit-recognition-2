import torch.nn as nn
import config as cfg


class ConvBlock(nn.Module):
    '''Convolution + LeakyReLU (+ optional MaxPool)'''

    def __init__(self, in_ch, out_ch, ks=3, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=ks, padding=ks // 2, bias=True),
            nn.LeakyReLU(0.01, inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CNN(nn.Module):
    '''
    Structure：
      5 ConvBlocks (3→16→32→64→128→256) (前 4 個 conv  blocks 後皆會有 MaxPooling)
      第 5 個 conv  blocks 後是 AdaptiveAvgPool2d(1,1) 自適應的全域平均池化，把空間維度壓縮成 1x1，讓結果只剩每個 channels 的平均值
      假如不先 Pool 到 1x1，直接 flatten 16x16x256，再接全連接會需要非常多參數，容易過擬合且佔用大量資源
      Pool 到 1x1 後，只剩 256 個值，適合進 FC 層計算
      接著是兩個 FC layers ，全連接層之間有 Dropout 把神經網路內部的中間神經元隨機暫時關掉一部分，用來防止過擬合
    '''

    def __init__(self, num_classes=cfg.NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            # 256×256 → 128×128
            ConvBlock(3, 16, pool=True),
            # 128×128 → 64×64
            ConvBlock(16, 32, pool=True),
            # 64×64 → 32×32
            ConvBlock(32, 64, pool=True),
            # 32×32 → 16×16
            ConvBlock(64, 128, pool=True),
            # 16×16, no additional pool
            ConvBlock(128, 256, pool=False),
        )
        # Global average pooling → (batch, 256, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
