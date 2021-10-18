import numpy as np
import os 
import cv2
import torch
from torch.utils.data import Dataset
from skimage.transform import resize

class CarDataset(Dataset):
    def __init__(self, df, transforms = None):
        # self.df=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/carvana/train_masks.csv')
        self.df = df 
        self.transforms = transforms
        self.img_fol= '/content/drive/MyDrive/Colab_Notebooks/carvana/train/'
        self.mask_fol = '/content/drive/MyDrive/Colab_Notebooks/carvana/train_masks/' 
       

    def __getitem__(self, idx):
        img_name=self.df['name'][idx]
        img_path=os.path.join(self.img_fol,img_name)    # Ruta completa imagen

        img_HR = cv2.imread(img_path)      # Se carga la imagen
        img_HR= cv2.cvtColor(img_HR, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            img_HR = self.transforms(image=img_HR)['image']

        img_HR = img_HR.transpose(2,0,1)

        # Factor de escala x4 en cada dimensión. Se mantiene el nº de canales
        img_LR = resize(img_HR, (3,img_HR.shape[1]//4,img_HR.shape[2]//4), anti_aliasing=True)

        img_HR = torch.from_numpy(img_HR)
        img_LR = torch.from_numpy(img_LR)
        
        # Se normaliza entre 0 y 1  
        img_HR = ((img_HR + img_HR.min().abs())/(img_HR + img_HR.min().abs()).max())
        img_LR = ((img_LR + img_LR.min().abs())/(img_LR + img_LR.min().abs()).max())
        
        return img_HR, img_LR

    def __len__(self):
        return len(self.df)