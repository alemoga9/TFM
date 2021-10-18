import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim

# Métricas
from torchmetrics.functional import ssim
from torchmetrics.functional import psnr
from torchmetrics.functional import mean_squared_error as mse

def train_generator(epoch, epochs, tloader, generator, pixel_criterion, optimizer_P,device):

    # Se calcula el número de batches de entrenamiento
    batches = len(tloader)

    # Se establece el generador en modo entrenamiento
    generator.train()

    tloader.set_description(f'GENERATOR TRAIN - EPOCH: [{epoch+1}/{epochs}]')

    # Se inicializan a 0 las variables que almacenarán las métricas   
    # obtenidas en cada batch
    epoch_pixel_loss, epoch_psnr, epoch_ssim, epoch_rmse = 0,0,0,0
    for hr, lr in tloader:
        # Los batches de imágenes se envían a la GPU si está disponible
        lr, hr= lr.to(device), hr.to(device)
        
        # Se ponen a cero los gradientes del generador
        generator.zero_grad()

        # Se obtiene la imagen SR
        sr = generator(lr)

        # Calculate the difference between the super-resolution image and the high-resolution image at the pixel level.
        pixel_loss = pixel_criterion(sr, hr)
        epoch_pixel_loss += pixel_loss.item()

        # Back propagation
        pixel_loss.backward()

        # Actualización de los pesos del generador
        optimizer_P.step()

        # Se calcula la PSNR
        psnr_score = psnr(sr,hr).to(device)
        epoch_psnr += psnr_score.item()      

        # Se calcula el SSIM
        ssim_score = ssim(sr, hr).to(device)
        epoch_ssim += ssim_score.item()   

        # Se calcula el RSME
        rmse_score = mse(sr,hr,squared = False).to(device)
        epoch_rmse += rmse_score.item()

        tloader.set_postfix(G_loss=pixel_loss.item(), RMSE=rmse_score.item(), PSNR=psnr_score.item(), SSIM=ssim_score.item())


    return [(epoch_pixel_loss/batches),(epoch_psnr/batches),(epoch_ssim/batches),(epoch_rmse/batches)]


def train_adversarial(epoch, epochs, tloader, generator, optimizer_G, discriminator, optimizer_D, adversarial_criterion, pixel_criterion, content_criterion, device):
    # Se calcula el número de batches de entrenamiento
    batches = len(tloader)

    # Se establece la red adversarial en modo entrenamiento
    discriminator.train()
    generator.train()

    tloader.set_description(f'ADVERSARIAL TRAIN - EPOCH: [{epoch+1}/{epochs}]')

    # Se inicializan a 0 las variables que almacenarán las métricas   
    # obtenidas en cada batch
    epoch_g_loss, epoch_d_loss, epoch_psnr, epoch_ssim, epoch_rmse = 0,0,0,0,0

    for hr, lr in tloader:
        # Los batches de imágenes se envían a la GPU si está disponible
        lr, hr= lr.to(device), hr.to(device)

        # Cantidad de imágenes que forman un batch
        label_size = lr.size(0)

        # Se crean las etiquetas. Se establecen en 1 para el caso real, y en 0 para el caso falso.
        real_label = torch.full([label_size, 1], 1.0, dtype=lr.dtype, device=device)
        fake_label = torch.full([label_size, 1], 0.0, dtype=lr.dtype, device=device)
 
        
        #----------------------------
        #------- Discriminator ------
        #----------------------------
        # Se ponen a cero los gradientes del discriminador
        discriminator.zero_grad()

        # Se obtienen las imágenes SR
        sr = generator(lr)

        # Se calcula la pérdida del discriminador en la imagen HR y SR
        hr_output = discriminator(hr)
        sr_output = discriminator(sr.detach())
        d_loss_hr = adversarial_criterion(torch.sigmoid(hr_output - torch.mean(sr_output)), real_label)
        d_loss_hr.backward()

        hr_output = discriminator(hr)
        sr_output = discriminator(sr.detach())
        d_loss_sr = adversarial_criterion(torch.sigmoid(sr_output - torch.mean(hr_output)), fake_label)
        d_loss_sr.backward()       

        # Pérdida total discriminador
        d_loss = d_loss_hr + d_loss_sr
        epoch_d_loss += d_loss.item()

        # Actualización de pesos del discriminador
        optimizer_D.step()


        #----------------------------
        #--------- Generator --------
        #----------------------------
        # Se ponen a cero los gradientes del discriminador
        generator.zero_grad()

        # Se obtienen las imágenes SR
        sr = generator(lr)
        
        # Se calcula la pérdida
        hr_output = discriminator(hr.detach())
        sr_output = discriminator(sr)

        # Perceptual loss=0.01 * pixel loss + 1.0 * content loss + 0.005 * adversarial loss.
        pixel_loss = 0.01 * pixel_criterion(sr, hr.detach())
        content_loss = 1.0 * content_criterion(sr, hr.detach())
        adversarial_loss = 0.005 * adversarial_criterion(torch.sigmoid(sr_output - torch.mean(hr_output)), real_label)
        
        g_loss = pixel_loss + content_loss + adversarial_loss
        
        # Retropropagación y actualización de pesos del generador
        g_loss.backward()
        optimizer_G.step()
    

        #----------------------------
        #--------- Métricas ---------
        #----------------------------      
        # Se calcula la PSNR
        psnr_score = psnr(sr,hr).to(device)
        epoch_psnr += psnr_score.item()      

        # Se calcula el SSIM
        ssim_score = ssim(sr, hr).to(device)
        epoch_ssim += ssim_score.item()   

        # Se calcula el RSME
        rmse_score = mse(sr,hr,squared = False).to(device)
        epoch_rmse += rmse_score.item() 

        tloader.set_postfix(G_Loss=g_loss.item(), D_Loss=d_loss.item(), RMSE=rmse_score.item(), PSNR=psnr_score.item(), SSIM=ssim_score.item())

    # Se devuelve el valor medio de cada métrica
    return [(epoch_g_loss/batches),(epoch_d_loss/batches),(epoch_psnr/batches),(epoch_ssim/batches),(epoch_rmse/batches)]


def validate(epoch,epochs,vloader,stage,generator,device,bs):
    
    vloader.set_description(f'{stage} VALIDATION')

    # Se calcula el número de batches de validación
    batches = len(vloader)

    # Se establece el generador en modo evaluación
    generator.eval()

    # Se inicializan a 0 las variables que almacenarán las métricas   
    # obtenidas en cada batch
    epoch_psnr, epoch_ssim, epoch_rmse = 0,0,0

    with torch.no_grad():
        for index, (hr, lr) in enumerate(vloader):

            # Los batches de imágenes se envían a la GPU si está disponible
            hr, lr = hr.to(device), lr.to(device)

            # Se obtienen las imágenes SR
            sr = generator(lr)

            #------------
            # Métricas
            #------------

            # Se calcula la PSNR
            psnr_score = psnr(sr,hr).to(device)
            epoch_psnr += psnr_score.item()      

            # Se calcula el SSIM
            ssim_score = ssim(sr, hr).to(device)
            epoch_ssim += ssim_score.item()   

            # Se calcula el RSME
            rmse_score = mse(sr,hr,squared = False).to(device)
            epoch_rmse += rmse_score.item()  

            vloader.set_postfix(RMSE=rmse_score.item(), PSNR=psnr_score.item(), SSIM=ssim_score.item())

            if index == 0:
                  sr_norm = ((sr + sr.min().abs())/(sr + sr.min().abs()).max())

                  # Se realiza una interpolación bicúbica a la imagen LR para tener
                  # las mismas dimensiones que la imagen HR y poder obtener las métricas
                  lr_resize = F.interpolate(lr,(hr.shape[2],hr.shape[3]),mode='bicubic')
                  lr_resize = ((lr_resize + lr_resize.min().abs())/(lr_resize + lr_resize.min().abs()).max())

                  plt.figure(figsize=(15,15))
                  plt.suptitle(f"{stage} VALIDATION [EPOCH: {epoch+1}/{epochs}]",fontsize=16, y=0.92)
                  for i in range(bs):
                      plt.subplot(bs,3,(3*i+1))
                      plt.title(f"HR")
                      plt.imshow(hr[i].cpu().numpy().transpose(1,2,0))
                      plt.subplot(bs,3,(3*i+2))
                      plt.title(f"LR--RMSE: {mse(lr_resize,hr,squared = False):.2f} -- PSNR: {psnr(lr_resize,hr):.2f} -- SSIM: {ssim(lr_resize, hr):.2f}")
                      plt.imshow(lr_resize[i].cpu().numpy().transpose(1,2,0))
                      plt.subplot(bs,3,(3*i+3))
                      plt.title(f"SR--RMSE: {mse(sr,hr,squared = False):.2f} -- PSNR: {psnr(sr,hr):.2f} -- SSIM: {ssim(sr, hr):.2f}")
                      plt.imshow(sr_norm[i].cpu().numpy().transpose(1,2,0))


    
    # Se devuelve el valor medio de cada métrica
    return [(epoch_psnr/batches),(epoch_ssim/batches),(epoch_rmse/batches)]
