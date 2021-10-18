import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor


class ResidualDenseBlock(nn.Module):
    """capas convolucionales densamente conectadas.
            channels (int): número de canales de la imagen de entrada.
            growths (int): número de canales que aumenta en cada capa de convolución.
    """

    def __init__(self, channels, growths):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels + growths * 0, growths, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(channels + growths * 1, growths, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(channels + growths * 2, growths, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(channels + growths * 3, growths, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(channels + growths * 4, channels, (3, 3), (1, 1), (1, 1))

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

    def forward(self, x):
        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(torch.cat([x, out1], 1)))
        out3 = self.leaky_relu(self.conv3(torch.cat([x, out1, out2], 1)))
        out4 = self.leaky_relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        out5 = self.identity(self.conv5(torch.cat([x, out1, out2, out3, out4], 1)))
        out = out5 * 0.2 + x

        return out


class ResidualInResidualDenseBlock(nn.Module):
    """Bloque de convolución densa residual multicapa.
    Args:
        channels (int): número de canales de la imagen de entrada.
        growths (int): número de canales que aumenta en cada capa de convolución.
    """

    def __init__(self, channels, growths):
        super(ResidualInResidualDenseBlock, self).__init__()

        self.rdb1 = ResidualDenseBlock(channels, growths)
        self.rdb2 = ResidualDenseBlock(channels, growths)
        self.rdb3 = ResidualDenseBlock(channels, growths)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = out * 0.2 + x

        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            # input size. (3) x 256 x 256
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),

            # state size. (64) x 128 x 128
            nn.Conv2d(64, 64, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            # state size. (128) x 64 x 64
            nn.Conv2d(128, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            # state size. (256) x 32 x 32
            nn.Conv2d(256, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            # state size. (512) x 16 x 16
            nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            # state size. (512) x 8 x 8
            nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 100),
            nn.LeakyReLU(0.2, True),
            nn.Linear(100, 1),
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
        self.conv_block1 = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))

        # Red troncal de extracción de características
        trunk = []
        for _ in range(16):
            trunk += [ResidualInResidualDenseBlock(64, 32)]
        self.trunk = nn.Sequential(*trunk)

        # Después de la red de extracción de características, reconecta una capa 
        # de bloques convolucionales.
        self.conv_block2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))

        # Capa convolucional de upsampling
        self.upsampling = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        # Reconecta un bloque de convolución tras la capa de upsampling
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        # Capa de salida
        self.conv_block4 = nn.Conv2d(64, 3, (3, 3), (1, 1), (1, 1))

    
    def forward(self, x):
        out1 = self.conv_block1(x)
        out = self.trunk(out1)
        out2 = self.conv_block2(out)
        out = out1 + out2
        out = self.upsampling(F.interpolate(out, scale_factor=2, mode="nearest"))
        out = self.upsampling(F.interpolate(out, scale_factor=2, mode="nearest"))
        out = self.conv_block3(out)
        out = self.conv_block4(out)

        return out

    def _initialize_weights(self):
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

    def __init__(self):
        super(ContentLoss, self).__init__()

        # Se carga el modelo VGG19 entrenado en el dataset ImageNet.
        vgg19 = models.vgg19(pretrained=True, num_classes=1000).eval()

        # Se extrae la salida de la capa 35 del modelo VGG19 como pérdida de contenido.
        self.feature_extractor = nn.Sequential(*list(vgg19.features.children())[:35])

        # Se congelan los parámetros del modelo.
        for parameters in self.feature_extractor.parameters():
            parameters.requires_grad = False

    def forward(self, sr, hr):
        # Pérdida basada en las diferencias de los mapas de características de 
        # ambas imágenes.
        loss = F.l1_loss(self.feature_extractor(sr), self.feature_extractor(hr))

        return loss