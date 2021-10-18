import os 
import torch
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from sklearn.model_selection import train_test_split
import albumentations as A
from model import *

# ------------------------------------------------------------------------------
# ------------ Configuración general  ------------------------------------------
# ------------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_imagenes = 500     # Número de imaganes empleadas
bs         = 4      # Tamaño del batch
p_epochs   = 8      # Número de epocas de entrenamiento del generador
epochs     = 4      # Número de epocas de entrenamiento adversarial


# ------------------------------------------------------------------------------
# ------------ DataFrame -------------------------------------------------------
# ------------------------------------------------------------------------------
# Carpeta que contiene las imágenes
img_fol= '/content/drive/MyDrive/Colab_Notebooks/carvana/train/' 

# Dataframe de n_imagenes
df = pd.DataFrame(os.listdir(img_fol), columns = ['name']) 
df = df.iloc[:n_imagenes]  
  
# Se divide el dataframe en entrenamiento y validación 
df_train, df_val = train_test_split(df, test_size=.2, random_state=42, shuffle=True, stratify=None) 

# Se resetan los índices
df_train  = df_train.reset_index()  # DataFrame de entrenamiento
df_val    = df_val.reset_index()    # DataFrame de validación


# ------------------------------------------------------------------------------
# ------------ Transformaciones ------------------------------------------------
# ------------------------------------------------------------------------------
transforms = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
        ])


# ------------------------------------------------------------------------------
# ------------ Entrenamiento y validación --------------------------------------
# ------------------------------------------------------------------------------

# Se instancian los modelos y se envían a la gpu si está disponible
generator       = Generator().to(device)      # Generador
discriminator   = Discriminator().to(device)  # Discriminador


# Funciones de pérdida
pixel_criterion       = nn.L1Loss().to(device)
content_criterion     = ContentLoss().to(device)
adversarial_criterion = nn.BCELoss().to(device)   # Adversarial loss.

# Optimizers
optimizer_P = optim.Adam(generator.parameters(), lr=0.0001)     # LR generador durante entrenamiento del generador
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001) # LR discriminador durante entrenamiento adversarial
optimizer_G = optim.Adam(generator.parameters(), lr=0.0001)     # LR generador durante entrenamiento adversarial

# Schedulers
milestones  = [epochs * 0.125, epochs * 0.250, epochs * 0.500, epochs * 0.750]
scheduler_P = CosineAnnealingLR(optimizer_P, p_epochs // 4, 1e-7)               # Generator model scheduler during generator training.
scheduler_D = MultiStepLR(optimizer_D, list(map(int, milestones)), 0.5)         # Discriminator model scheduler during adversarial training.
scheduler_G = MultiStepLR(optimizer_G, list(map(int, milestones)), 0.5)         # Generator model scheduler during adversarial training.
