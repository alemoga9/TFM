import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor

class ResidualConvBlock(nn.Module):
    """Bloques convoluciones residuales.
    Args:
        channels (int): número de canales de la imagen de entrada.
    """
    def __init__(self, channels):
        super(ResidualConvBlock, self).__init__()
        self.rc_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        out = self.rc_block(x)
        out = out + x
        return out

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            # input size. (3) x 256 x 256
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, True),

            # state size. (64) x 128 x 128
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            # state size. (128) x 64 x 64
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            # state size. (256) x 32 x 32
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            # state size. (512) x 16 x 16
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512*16*16, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Primera capa convolucional
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4), 
            nn.PReLU()
            )

        # Bloques residuales
        res_blocks = []
        for _ in range(16):
            res_blocks.append(ResidualConvBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Segunda capa convolucional
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.BatchNorm2d(64)
            )

        # Bloque convolución upsampling (x4)
        self.upsampling = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

        # Capa de salida
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

        # Se inicializa los pesos 
        self.initialize_weights()  

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = out1 + out2
        out = self.upsampling(out)
        out = self.conv3(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                m.weight.data *= 0.1
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                m.weight.data *= 0.1

class ContentLoss(nn.Module):
    """Se construye una función de pérdida de contenido basada en la red VGG19.
       El uso de capas de mapeo de características de alto nivel de las últimas 
       capas se centrará más en el contenido de textura de la imagen.
    """

    def __init__(self) -> None:
        super(ContentLoss, self).__init__()

        # Se carga el modelo VGG19 entrenado en el dataset ImageNet.
        vgg19 = models.vgg19(pretrained=True, num_classes=1000).eval()

        # Se extrae la salida de la capa 36 del modelo VGG19 como pérdida de contenido.
        self.feature_extractor = nn.Sequential(*list(vgg19.features.children())[:36])

        # Se congelan los parámetros del modelo.
        for parameters in self.feature_extractor.parameters():
            parameters.requires_grad = False

    def forward(self, sr, hr):
        # Pérdida basada en las diferencias de los mapas de características de 
        # ambas imágenes.
        loss = F.mse_loss(self.feature_extractor(sr), self.feature_extractor(hr))

        return loss