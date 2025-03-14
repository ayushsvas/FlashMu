# Imports
import sys
from utils_augmented import (dice_score,
    load_checkpoint,
    save_checkpoint, 
    get_loaders,  
    val_accuracy,
    save_predictions_as_imgs,
    get_mean_std,
    total_variation_loss,
    train_log,
    val_log,
    get_grid,
    H1Loss,
    _get_gpu_session,
)

import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = _get_gpu_session(os)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import time
import torch 
import torchvision
from tqdm import tqdm 
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
import torchvision.transforms as transforms

from flashmu.csfm.dfc import dfc

import wandb
wandb.login()

from config_dfc_unet import config

# model_pipeline, calling the main()
def model_pipeline(config):
    # tell wandb to get started
    with wandb.init(project = 'holo_inversion', config = config, mode = 'online', name = config["dfc"]["architecture"]):
        # access all hyperparamters through wandb.congig, so logging matches execution 
        config = wandb.config
        main(config)

# Training function 
def train_fn(loader, model, optimizer, custom_loss, huber_loss, mse_loss, mae_loss, scaler, scheduler1, scheduler2, device):   # scaler?

    wandb.watch(model, criterion = mse_loss, log = 'all', log_freq = 10000)

    model.train()

    loop = tqdm(loader)

    loss_per_epoch = []
    dce_per_epoch = []

    for batch_idx, (data, target) in enumerate(loop):
        data = data.to(device = device).float()
        grid = get_grid(data.shape, device = device)
        data = torch.concat((data, grid), dim = 1)
        target_holo = target.to(device = device).float()
        
        #forward 
        with torch.autocast(device_type = 'cuda', enabled = True):
            prediction_holo = model(data)
            # loss = mse_loss(prediction_holo, target_holo) 
            loss = custom_loss(prediction_holo, target_holo)
        
        # Saving losses and metrics
        loss_per_epoch.append(float(loss))
            

        #backward
        optimizer.zero_grad()  # ?
        scaler.scale(loss).backward() # ?
        scaler.step(optimizer) # ?
        scaler.update()        # ?

        loop.set_postfix(loss=loss.item())


    loss_per_epoch = sum(loss_per_epoch)/len(loss_per_epoch)   
    
    return (loss_per_epoch), (dce_per_epoch)

# Main
def main(config):
    """
    Main function for the training process.
    """
    img_dir =  config["dfc"]["data"]["IMG_DIR"]
    mask_dir =  config["dfc"]["data"]["MASK_DIR"]
    bg_augment = config["dfc"]["training"]["BACKGROUND_AUGMENT"]
    num_samples = config["dfc"]["training"]["NUM_SAMPLES"]
    bg_dir =  config["dfc"]["training"]["BACKGROUND_DIR"]
    lr = config["dfc"]["training"]["LEARNING_RATE"]
    val_frac = config["dfc"]["training"]["VAL_FRAC"]
    split_seed = config["dfc"]["training"]["SPLIT_SEED"]
    batch_size = config["dfc"]["training"]["BATCH_SIZE"]
    num_epochs = config["dfc"]["training"]["NUM_EPOCHS"]
    num_workers = config["dfc"]["training"]["NUM_WORKERS"]
    pin_memory = config["dfc"]["training"]["PIN_MEMORY"]
    device = config["dfc"]["training"]["DEVICE"]
    augment = config["dfc"]["training"]["AUGMENT"]
    mmap_mode = config["dfc"]["training"]["MMAP_MODE"]

    
    # Define image and mask transformations

    # img_transform = transforms.Compose([transforms.ToTensor()])
    # mask_transform = transforms.Compose([transforms.ToTensor()])
 
    img_transform = transforms.ToTensor()
    mask_transform = transforms.ToTensor()

    # mean, std = [124.5178], [15.3115] #256x256
    # mean, std = [124.51788330078124], [15.31218147277832] # 128x128
    mean = config["dfc"]["training"]["mean"]
    std = config["dfc"]["training"]["std"]
    # mean, std = [124.52455139160156], [15.454506874084473]
    
    # mean, std = None, None 

    # Get data loaders for training and validation sets
    if mean is None:
        print("Getting preliminary loaders with mean and standard deviation...")

        train_loader, val_loader = get_loaders(
            img_dir, mask_dir, bg_dir, num_samples, config["train"]["dfc"], val_frac, batch_size, img_transform, mask_transform, 
            augment = False, bg_augment=False, num_worker = num_workers, pin_memory = pin_memory, split_seed = split_seed, 
            mmap_mode = mmap_mode)

        mean, std = get_mean_std(train_loader)

    # Calculate mean and standard deviation of the training data
    # mean, std = [124.52430725097656], [15.45513916015625]
    
    print(f"mean:{mean}, std:{std}")

    # Apply normalization transformation to images using the calculated mean and standard deviation
    print("Getting normalized loaders...")
    img_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    # use below if background is concatenated with the hologram 
    # img_transforms = transforms.Normalize(mean, std)

    # Get data loaders with normalized images
    train_loader, val_loader = get_loaders(
        img_dir, mask_dir, bg_dir, num_samples,config["train"]["dfc"], val_frac, batch_size, img_transforms, mask_transform, 
        augment =  augment, bg_augment=bg_augment, num_worker = num_workers, pin_memory = pin_memory, split_seed = split_seed,
        mmap_mode = mmap_mode)

    # Get some samples to see while training
    images, masks = next(iter(val_loader))
    print(f"Input:{images.shape},{images.dtype}")
    print(f"Output:{ masks.shape},{images.dtype}")

    # Send them to wandb for later analysis, also saving locally.
    #wandb
    # input_holos = [wandb.Image(image.permute(1,2,0).cpu().detach().numpy()) for image in images]
    # target_holos = [wandb.Image(mask.permute(1,2,0).cpu().detach().numpy()) for mask in masks]
    # wandb.log({"input_holos": input_holos, "target_holos": target_holos})
    #local
    torchvision.utils.save_image((images[:,0].unsqueeze(1).float()*std[0]+mean[0])/255, config["dfc"]["data"]["SAVE_FOLDER"]+"input_holo.png")
    # torchvision.utils.save_image(torchvision.transforms.functional.invert(masks[:,0].unsqueeze(1)), f"{SAVE_FOLDER}/true_xy.png")
    torchvision.utils.save_image(masks.float()/255, config["dfc"]["data"]["SAVE_FOLDER"]+"target_holo.png")
    

    # Create an instance of the FNO model

    # model = FNO2d(n_modes_width = n_modes, n_modes_height = n_modes, hidden_channels = hidden_channels, in_channels = in_channels, out_channels = out_channels,
    #                lifting_channels = lifting_channels, projection_channels = projection_channels, n_layers = n_layers, skip = skip)
    # model = TFNO(n_modes=(n_modes, n_modes), hidden_channels=hidden_channels, lifting_channels = hidden_channels, projection_channels=projection_channels, 
    #              factorization=factorization, rank=rank,  skip = skip, n_layers = n_layers)
    # model = model.to(DEVICE)
    
    # model = f_c_network(in_channels=in_channels, width=hidden_channels, n_modes=(n_modes,n_modes), lifting_channels=lifting_channels, 
    #                     projection_channels=lifting_channels, kernel_size=5, padding=True,
    #                     dilations=[1,3,9], num_layers=4, fno_block_precision='full', skip = 'conv3', factorization = factorization, rank=rank)
    # model = f_c_network(in_channels=in_channels, width=hidden_channels, n_modes=(n_modes//1,n_modes//1), lifting_channels=lifting_channels, 
    #                     projection_channels=lifting_channels, kernel_size=kernel_size, padding=True,
    #                     dilations=[1,3,9], num_layers=n_layers, fno_block_precision='full', skip = skip, factorization = factorization, rank=rank, seperable_spectral_conv=True, spectral_dilation=2,
    #                     output_upsample_fac=2)
    # model = model.to(DEVICE)
    
    in_channels = config["dfc"]["fourier_part"]["in_channels"]
    hidden_channels = config["dfc"]["fourier_part"]["hidden_channels"]
    n_modes = config["dfc"]["fourier_part"]["n_modes"]
    fourier_interpolation = config["dfc"]["fourier_part"]["fourier_interpolation"]
    bias = config["dfc"]["fourier_part"]["bias"]
    skip = config["dfc"]["fourier_part"]["skip"]
    dilate_fourier_kernel_fac = config["dfc"]["fourier_part"]["dilate_fourier_kernel_fac"]
    lifting_channels = config["dfc"]["fourier_part"]["lifting_channels"]
    projection_channels = config["dfc"]["fourier_part"]["projection_channels"]
    n_layers = config["dfc"]["fourier_part"]["n_layers"]
    decomposition = config["dfc"]["fourier_part"]["factorization"]
    implementation = config["dfc"]["fourier_part"]["implementation"]
    rank = config["dfc"]["fourier_part"]["rank"]
    mem_checkpoint = config["dfc"]["fourier_part"]["mem_checkpoint"]
    separable_fourier_layers = config["dfc"]["fourier_part"]["separable_fourier_layers"]
    batch_norm = config["dfc"]["fourier_part"]["batch_norm"]
    fno_block_precision = config["dfc"]["fourier_part"]["fourier_block_precision"]


    kernel_size = config["dfc"]["dilated_cnn_part"]["kernel_size"]
    padding = config["dfc"]["dilated_cnn_part"]["padding"]
    dilations = config["dfc"]["dilated_cnn_part"]["dilations"]


    model = dfc(in_channels=in_channels, width=hidden_channels, n_modes=(n_modes,n_modes), fourier_interpolate=fourier_interpolation, bias = bias,
            spectral_dilation_fac=dilate_fourier_kernel_fac, decomposition=decomposition, rank = rank, implementation=implementation, 
                        separable_fourier_layers=separable_fourier_layers, mem_checkpoint=mem_checkpoint, skip = skip, batch_norm=batch_norm,
                        fno_block_precision=fno_block_precision, 
                        lifting_channels=lifting_channels, 
                        projection_channels=projection_channels, kernel_size=kernel_size, padding=padding,
                        dilations=dilations, num_layers=n_layers,)
  
    if torch.cuda.device_count() > 1 and config["train"]["data_parallel"]:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    model = model.to(device)


    # Create an Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    

    # Load model from checkpoint if specified
    if config["dfc"]["training"]["LOAD_MODEL"] is True:
        load_checkpoint(torch.load(config["dfc"]["data"]["LOAD_CHECKPOINT_DIR"]), model, optimizer = optimizer,
                        data_parallel=config["train"]["data_parallel"]) #this copy is from 150k without hard.

    # Loss functions 
    bce_loss = nn.BCEWithLogitsLoss()
    depth_loss = nn.HuberLoss(reduction = 'mean', delta = 0.001)
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    custom_loss = H1Loss(d = 2, T = 1)

    # Create a GradScaler for mixed precision training
    scaler = torch.GradScaler()

    # Create a learning rate scheduler
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config["dfc"]["training"]["LR_DECAY_FAC"], 
                                                      patience=config["dfc"]["training"]["SCHEDULER_COUNT"])
    scheduler2 = optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1)

    # Create a start time stamp for training
    start_time = time.time()

    # Initialize lists to store losses and accuracies for plotting
    train_loss, val_loss, train_acc, val_acc = [], [], [], []

    # initialize break count for early stopping
    break_count = 0

    # Starts training and validation
    for epoch in range(num_epochs):

        train_epoch_loss, train_epoch_dce = train_fn(train_loader, model, optimizer, custom_loss, depth_loss, mse_loss, mae_loss, 
                                                     scaler, scheduler1, scheduler2, device)
        
        # Reporting training loss to wandb
        train_log(loss = train_epoch_loss, epoch = epoch)
        
        val_epoch_loss, val_epoch_dce = val_accuracy(val_loader, model, bce_loss, depth_loss, mse_loss, mae_loss, device = device)

        # update scheduler 
        scheduler1.step(val_epoch_loss)
        
        # Reporting validation loss to wandb
        val_log(loss = val_epoch_loss, epoch = epoch)

        print(f'At epoch {epoch}: training loss = {train_epoch_loss}| validation loss = {val_epoch_loss}')

        if epoch == 0:
            val_prevepoch_loss = val_epoch_loss

         # Some checkpoint and some samples every Nth epoch 
        if val_epoch_loss <= val_prevepoch_loss:            
            # save model
            print('Saving parameters!')
            if config["train"]["data_parallel"]:
                checkpoint = {"state_dict":model.module.state_dict(), "optimizer":optimizer.state_dict(),}
            else:    
                checkpoint = {"state_dict":model.state_dict(), "optimizer":optimizer.state_dict(),}
            save_checkpoint(checkpoint, config["dfc"]["data"]["CHECKPOINT_DIR"])
            val_prevepoch_loss = val_epoch_loss
            break_count = 0
        
        else:
            break_count += 1
            if break_count == config["dfc"]["training"]["EARLY_STOPPING_COUNT"]:
                print('Model not leaning to generalize anymore. Breaking!')
                break

        if epoch%2 == 0:
            # print some examples to folder
            print('Saving some images!') 
            save_predictions_as_imgs(images, masks, model, folder = config["dfc"]["data"]["SAVE_FOLDER"], device = device)
        
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        train_acc.append(train_epoch_dce)
        val_acc.append(val_epoch_dce)

    # Create a end time stamp for training
    end_time = time.time()
    print(f"Training took {np.round(end_time - start_time, 0).astype(int)} time units long")

    # Converting lists to numpy arrays 
    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)
    train_acc = np.array(train_acc)
    val_acc = np.array(val_acc)
    
    # Save
    np.save("/home/apaliwal/data/train_val_loss_acc/synthetic_train_loss.npy", train_loss)
    np.save("/home/apaliwal/data/train_val_loss_acc/synthetic_val_loss.npy", val_loss)
    np.save("/home/apaliwal/data/train_val_loss_acc/synthetic_train_dce.npy", train_acc)
    np.save("/home/apaliwal/data/train_val_loss_acc/synthetic_val_dce.npy", val_acc)

    # # Save the model in the exchangeable ONNX format
    # torch.onnx.export(model, torch.concat((images.to(device = DEVICE), get_grid(images.shape, device=DEVICE)), dim = 1), f = "model.onnx")
    # wandb.save("model.onnx")


if __name__ == "__main__":
    model_pipeline(config = config)