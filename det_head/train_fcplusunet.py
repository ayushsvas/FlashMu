# Imports
import utils
import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
sys.path.insert(0, "/user/apaliwa/u12018/")



import time
import torch 
import torchvision
from tqdm import tqdm 
import numpy as np
import torch.nn as nn 
import torch.optim as optim 
import random
import gc


from CNOModule import CNO
# from upernetconvnext import uperconvnext
# import segmentation_models_pytorch as smp
# from FU import CNO
# from Holography.FNO.DCNO_dilated_fixed_nonlinearinterp_lastFNOLayerNotSeparabele import f_c_network
# from DCNO_dilated_fixed_nonlinearinterp_firstFNOLayerNotSeparable import f_c_network
from hope.dfc_old import dfc as f_c_network

from utils import *
from config_dfc_unet import config

import wandb
wandb.login()


init_seed = config["unet"]["training"]["SPLIT_SEED"]

# reproducibility 
torch.manual_seed(init_seed)
random.seed(init_seed)
np.random.seed(init_seed)


# model_pipeline, calling the main()
def model_pipeline(config):
    # tell wandb to get started
    with wandb.init(project = 'holo_inversion_unet', config = config, mode = 'online', name = config["unet"]["architecture"]):
        # access all hyperparamters through wandb.congig, so logging matches execution 
        config = wandb.config
        main(config)


# Training function 
def train_fn(loader, model, fc_net, optimizer, radii_loss, huber_loss, mse_loss, loss_fn, scaler, scheduler1, scheduler2, device):   


    wandb.watch(model, criterion = mse_loss, log = 'all', log_freq = 10000)

    loop = tqdm(loader)

    xy_loss_per_epoch = []
    z_loss_per_epoch = []
    s_loss_per_epoch = []
    pi_loss_per_epoch = []
    loss_per_epoch = []


    for _, (data, target) in enumerate(loop):

        data = data.to(device = device).float()
        
        target = target.to(device = device).float()
        #forward 
        with torch.autocast(device_type = 'cuda', enabled = True):
            if fc_net is not None:
                data = torch.concat((data, get_grid(data.shape, device)), dim = 1)
                data = torch.concat((data[:,0].unsqueeze(1),fc_net(data)), dim = 1) # have to place before continue statment otherwise mismatch with gt

            prediction = torch.sigmoid(model(data))

            loss, xy_loss, z_loss, s_loss, pi_loss = loss_fn(prediction, target)

        xy_loss_per_epoch.append(float(xy_loss))
        z_loss_per_epoch.append(float(z_loss))
        s_loss_per_epoch.append(float(s_loss))
        pi_loss_per_epoch.append(float(pi_loss))
        loss_per_epoch.append(float(loss))

        # del target
        # gc.collect()
        
        #backward
        optimizer.zero_grad()  
        scaler.scale(loss).backward() 
        scaler.step(optimizer) 
        scaler.update()        

        loop.set_postfix(loss=loss.item())

    
    loss_per_epoch = sum(loss_per_epoch)/len(loss_per_epoch)
    xy_loss_per_epoch = float(sum(xy_loss_per_epoch)/len(xy_loss_per_epoch))
    z_loss_per_epoch = float(sum(z_loss_per_epoch)/len(z_loss_per_epoch))
    s_loss_per_epoch = float(sum(s_loss_per_epoch)/len(s_loss_per_epoch))
    pi_loss_per_epoch = float(sum(pi_loss_per_epoch)/len(pi_loss_per_epoch))
    
    scheduler1.step(loss_per_epoch)
    # scheduler2.step()

    return float(loss_per_epoch),  float(xy_loss_per_epoch), float(z_loss_per_epoch), float(s_loss_per_epoch), float(pi_loss_per_epoch)


# Main funtion which calls the training function

def main(config):
    """
    Main function for the training process.
    """

    if config["train"]["dfc"]:
        img_dir =  config["dfc"]["data"]["IMG_DIR"]
        wimg_dir =  config["dfc"]["data"]["MASK_DIR"]
        checkpoint_dir = config["dfc"]["data"]["CHECKPOINT_DIR"]
        save_dir = config["dfc"]["data"]["SAVE_FOLDER"]
        bg_augment = config["dfc"]["training"]["BACKGROUND_AUGMENT"]
        num_samples = config["dfc"]["training"]["NUM_SAMPLES"]
        bg_dir =  config["dfc"]["training"]["BACKGROUND_DIR"]
        lr = config["dfc"]["training"]["LEARNING_RATE"]
        scheduler_count =  config["dfc"]["training"]["SCHEDULER_COUNT"]
        lr_decay_fac = config["dfc"]["training"]["LR_DECAY_FAC"]
        val_frac = config["dfc"]["training"]["VAL_FRAC"]
        split_seed = config["dfc"]["training"]["SPLIT_SEED"]
        batch_size = config["dfc"]["training"]["BATCH_SIZE"]
        num_epochs = config["dfc"]["training"]["NUM_EPOCHS"]
        early_stopping_count = config["dfc"]["training"]["EARLY_STOPPING_COUNT"]
        num_workers = config["dfc"]["training"]["NUM_WORKERS"]
        pin_memory = config["dfc"]["training"]["PIN_MEMORY"]
        device = config["dfc"]["training"]["DEVICE"]
        augment = config["dfc"]["training"]["AUGMENT"]
        mmap_mode = config["dfc"]["training"]["MMAP_MODE"]
        mean = config["dfc"]["training"]["mean"]
        std = config["dfc"]["training"]['std']
        save_freq = config["dfc"]["training"]['SAVE_FREQ']

    if config["train"]["unet"]:
        img_dir =  config["unet"]["data"]["IMG_DIR"]
        wimg_dir = config["unet"]["data"]["WIMG_DIR"]
        mask_dir =  config["unet"]["data"]["MASK_DIR"]
        checkpoint_dir = config["unet"]["data"]["CHECKPOINT_DIR"]
        save_dir = config["unet"]["data"]["SAVE_FOLDER"]
        bg_augment = config["unet"]["training"]["BACKGROUND_AUGMENT"]
        num_samples = config["unet"]["training"]["NUM_SAMPLES"]
        bg_dir =  config["unet"]["training"]["BACKGROUND_DIR"]
        lr = config["unet"]["training"]["LEARNING_RATE"]
        scheduler_count =  config["unet"]["training"]["SCHEDULER_COUNT"]
        lr_decay_fac = config["unet"]["training"]["LR_DECAY_FAC"]
        val_frac = config["unet"]["training"]["VAL_FRAC"]
        split_seed = config["unet"]["training"]["SPLIT_SEED"]
        batch_size = config["unet"]["training"]["BATCH_SIZE"]
        num_epochs = config["unet"]["training"]["NUM_EPOCHS"]
        early_stopping_count = config["unet"]["training"]["EARLY_STOPPING_COUNT"]
        num_workers = config["unet"]["training"]["NUM_WORKERS"]
        pin_memory = config["unet"]["training"]["PIN_MEMORY"]
        device = config["unet"]["training"]["DEVICE"]
        augment = config["unet"]["training"]["AUGMENT"]
        mmap_mode = config["unet"]["training"]["MMAP_MODE"]
        mean = config["unet"]["training"]["mean"]
        std = config["unet"]["training"]['std']
        save_freq = config["unet"]["training"]['SAVE_FREQ']


    img_transform, mask_transform = None, None  # add these to config
    
    TRAIN = bool(config["train"]["dfc"]+config["train"]["unet"])

    # Get data loaders for training and validation sets
    if config["dfc"]["evaluation"]["mean"] is None or config["unet"]["evaluation"]["mean"] is None:
        print("Getting preliminary loaders with mean and standard deviation...")
        train_loader, val_loader = get_loaders(img_dir, wimg_dir, 
        mask_dir, bg_dir, TRAIN, val_frac, batch_size, img_transform, mask_transform, augment=False, background=False, num_worker = num_workers, 
        pin_memory=pin_memory, split_seed=split_seed, mmap_mode = mmap_mode
        ) 
        mean, std = get_mean_std(train_loader)
    
    # Transform/Normalize data
    print(f"mean:{mean}, std:{std}")


    print("Getting normalized loaders...")
    img_transforms = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean, std)])
    train_loader, val_loader = get_loaders(img_dir, wimg_dir, 
                                           mask_dir, bg_dir, TRAIN, val_frac, batch_size, img_transforms, mask_transform,
                                           augment = augment, background=bg_augment, num_worker = num_workers, 
                                           pin_memory = pin_memory, split_seed = split_seed, mmap_mode = mmap_mode)
    

    img, mask = next(iter(val_loader))
    print(f"Input:{img.shape},{img.dtype}")
    print(f"Output:{ mask.shape},{mask.dtype}")

    if config["train"]["unet"] or config["test"]["unet"]:

        in_channels = config["unet"]["build"]["in_channels"]
        in_size = config["unet"]["build"]["in_size"]
        out_channels = config["unet"]["build"]["out_channels"]
        latent_lift_proj_dim = config["unet"]["build"]["latent_lift_proj_dim"]
        n_layers = config["unet"]["build"]["n_layers"]
        n_res = config["unet"]["build"]["n_res"]
        activations = config["unet"]["build"]['activations']
        
        model = CNO(in_dim = in_channels, in_size = in_size, N_layers = n_layers, out_dim = out_channels,
        activation = activations, N_res=n_res, latent_lift_proj_dim=latent_lift_proj_dim).to(device = device)

        # model = uperconvnext(in_channels = in_channels, out_channels = out_channels).to(device = device)
       
        # model = smp.DeepLabV3(encoder_name = 'resnet101' ,in_channels = in_channels, classes = out_channels, activation = None).to(device = device)

        if config["unet"]["training"]["LOAD_MODEL"]:
            load_checkpoint(
            torch.load(
                config["unet"]["data"]["LOAD_CHECKPOINT_DIR"]), 
                    model, None)
            if config["test"]["unet"]:
                for p in model.parameters():
                    p.requires_grad = False
                model.eval()
    else:
        model = None

    if config["train"]["dfc"] or config["test"]["dfc"]:
        #initialize model and localizer
        in_channels = config["dfc"]["fourier_part"]["in_channels"]
        hidden_channels = config["dfc"]["fourier_part"]["hidden_channels"]
        n_modes = config["dfc"]["fourier_part"]["n_modes"]
        skip = config["dfc"]["fourier_part"]["skip"]
        dilate_fourier_kernel_fac = config["dfc"]["fourier_part"]["dilate_fourier_kernel_fac"]
        lifting_channels = config["dfc"]["fourier_part"]["lifting_channels"]
        projection_channels = config["dfc"]["fourier_part"]["projection_channels"]
        n_layers = config["dfc"]["fourier_part"]["n_layers"]
        factorization = config["dfc"]["fourier_part"]["factorization"]
        if factorization == 'dense':
            factorization = None # set to None when dense
        rank = config["dfc"]["fourier_part"]["rank"]
        separable_fourier_layers = config["dfc"]["fourier_part"]["separable_fourier_layers"]

        kernel_size = config["dfc"]["dilated_cnn_part"]["kernel_size"]
        padding = config["dfc"]["dilated_cnn_part"]["padding"]
        dilations = config["dfc"]["dilated_cnn_part"]["dilations"]

        fc_net = f_c_network(in_channels=in_channels, width=hidden_channels, n_modes=(n_modes,n_modes), spectral_dilation_fac=dilate_fourier_kernel_fac, 
                            lifting_channels=lifting_channels, 
                            projection_channels=projection_channels, kernel_size=kernel_size, padding=padding,
                            dilations=dilations, num_layers=n_layers, fno_block_precision='full', skip = skip, factorization = factorization, rank=rank,
                            separable_fourier_layers=separable_fourier_layers).to(device)
        
        if config["dfc"]["training"]["LOAD_MODEL"]:
            load_checkpoint(
            torch.load(
                config["dfc"]["data"]["LOAD_CHECKPOINT_DIR"]), 
                    fc_net, None)
            if config["test"]["dfc"]:
                for p in fc_net.parameters():
                    p.requires_grad = False
                fc_net.eval()
    else:
        fc_net = None
    
    # model = smp.DeepLabV3(encoder_name = 'resnet101', in_channels = 1, classes = 3, activation = None).to(device = DEVICE)
    # model = smp.DeepLabV3(in_channels=1, activation=None).to(device)
    if config["train"]["unet"]:
        optimizer = optim.Adam(model.parameters(), lr = lr)
    if config["train"]["dfc"]:
        optimizer = optim.Adam(fc_net.parameters(), lr = lr)


    # loss_fn = nn.BCEWithLogitsLoss()
    depth_loss = nn.HuberLoss(reduction = 'mean', delta = config["unet"]["training"]["BETA"])
    radii_loss = nn.HuberLoss(reduction = 'mean', delta = config["unet"]["training"]["BETA"])
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    if config["unet"]["build"]["loss"] == 'shao':
        loss_fn = ThreeHeadLoss(config["unet"]["training"]["ALPHA"], config["unet"]["training"]["BETA"])
    if config["unet"]["build"]["loss"] == 'shao+pi_loss':
        loss_fn = ThreeHeadPILoss(config["unet"]["training"]["ALPHA"], config["unet"]["training"]["BETA"], device, 
                                4096,2048,2048)
    scaler = torch.amp.GradScaler()
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = lr_decay_fac, patience = scheduler_count)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1)

    train_loss, val_loss = [], []

    train_xy_loss, val_xy_loss = [],[]

    train_z_loss, val_z_loss = [],[]

    train_s_loss, val_s_loss = [],[]


    start_time = time.time()

    break_count = 0

    for epoch in range(num_epochs):
        train_epoch_loss, train_loss_xy, train_loss_z, train_loss_s, train_loss_pi = train_fn(train_loader, model, fc_net, optimizer, radii_loss, depth_loss, mse_loss, loss_fn, 
                                                                               scaler, scheduler1, scheduler2, device)
        
        # Reporting training loss to wandb
        train_log(loss = train_epoch_loss, epoch = epoch, type = 'total')
        train_log(loss = train_loss_xy, epoch = epoch, type = 'xy')
        train_log(loss = train_loss_z, epoch = epoch, type = 'z')
        train_log(loss = train_loss_s, epoch = epoch, type = 'r')
        train_log(loss = train_loss_pi, epoch = epoch, type = 'pi')
        
        val_epoch_loss, val_loss_xy, val_loss_z, val_loss_s, val_loss_pi = val_accuracy(val_loader, model, fc_net, radii_loss, depth_loss, mse_loss, loss_fn, device = device)

        # Reporting validation loss to wandb
        val_log(loss = val_epoch_loss, epoch = epoch, type = 'total')
        val_log(loss = val_loss_xy, epoch = epoch, type = 'xy')
        val_log(loss = val_loss_z, epoch = epoch, type = 'z')
        val_log(loss = val_loss_s, epoch = epoch, type = 'r')
        val_log(loss = val_loss_pi, epoch = epoch, type = 'pi')
        


        print(f'At epoch {epoch}: training loss = {train_epoch_loss}| validation loss = {val_epoch_loss}')

        if epoch == 0:
            val_prevepoch_loss = val_epoch_loss

        # Some checkpoint and some samples every Nth epoch 
        if val_epoch_loss <= val_prevepoch_loss:            
            # save model
            print('Saving parameters!')
            checkpoint = {"state_dict":model.state_dict(), "optimizer":optimizer.state_dict(),}
            save_checkpoint(checkpoint, checkpoint_dir)
            val_prevepoch_loss = val_epoch_loss
            break_count = 0
        
        else:
            break_count += 1
            if break_count == early_stopping_count:
                print('Model not learning to generalize anymore. Breaking!')
                break

        if epoch%save_freq == 0:
            # print some examples to folder
            print('Saving some images!') 
            save_predictions_as_imgs(img[:batch_size], mask[:batch_size,0], model, fc_net, folder = save_dir, device =device)
       
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        train_xy_loss.append(train_loss_xy)
        val_xy_loss.append(val_loss_xy)
        train_z_loss.append(train_loss_z)
        val_z_loss.append(val_loss_z)
        train_s_loss.append(train_loss_s)
        val_s_loss.append(val_loss_s)
        


    end_time = time.time()
    print(f"Training took {np.round(end_time - start_time, 0).astype(int)} time units long")

    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)
    
    np.save("data/train_val_loss_acc/synthetic_holo_train_loss.npy", train_loss)
    np.save("data/train_val_loss_acc/synthetic_holo_val_loss.npy", val_loss)
    
    np.save("data/train_val_loss_acc/synthetic_holo_train_xy_loss.npy", train_xy_loss)
    np.save("data/train_val_loss_acc/synthetic_holo_train_z_loss.npy", train_z_loss)
    np.save("data/train_val_loss_acc/synthetic_holo_train_s_loss.npy", train_s_loss)
    np.save("data/train_val_loss_acc/synthetic_holo_val_xy_loss.npy", val_xy_loss)
    np.save("data/train_val_loss_acc/synthetic_holo_val_z_loss.npy", val_z_loss)
    np.save("data/train_val_loss_acc/synthetic_holo_val_s_loss.npy", val_s_loss)
    
# if __name__ == "__main__":
#     main()

if __name__ == "__main__":
    model_pipeline(config = config)

