import sys
import torch
import torch.nn as nn
import torchvision 
# from datasetStep2 import holodataset 
# from manuel_holodataset import SparseHolodataset
from holodataset import SparseHolodataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm, trange
import wandb
import random

def _get_gpu_session(os):
    if len(sys.argv) == 2:
        return sys.argv[1]
        
    # Check if we're in a tmux session
    if 'TMUX' not in os.environ:
        print("Training failed, no TMUX environment detected and no GPU identified in argv.")
        exit(1)
    
    # Get session name using tmux display-message
    import subprocess
    try:
        result = subprocess.run(['tmux', 'display-message', '-p', '#S'], 
                              capture_output=True, 
                              text=True)
        result = result.stdout.strip()
        
        if "gpu" not in result:
            exit(1)
        
        return result[3]
    
    except FileNotFoundError:
        exit(1)

def train_log(loss, epoch, type = 'total'):
    # Where the magic happens
    if type == 'total':
        wandb.log({"epoch": epoch, "training_loss": loss})
    if type == 'xy':
        wandb.log({"epoch": epoch, "training_loss_xy": loss})
    if type == 'z':
        wandb.log({"epoch": epoch, "training_loss_z": loss})
    if type == 'r':
        wandb.log({"epoch": epoch, "training_loss_r": loss})


def val_log(loss, epoch, type = 'total'):
    # Where the magic happens
    if type == 'total':
        wandb.log({"epoch": epoch, "validation_loss": loss})
    if type == 'xy':
        wandb.log({"epoch": epoch, "validation_loss_xy": loss})
    if type == 'z':
        wandb.log({"epoch": epoch, "validation_loss_z": loss})
    if type == 'r':
        wandb.log({"epoch": epoch, "validation_loss_r": loss})


def save_checkpoint(state, filename = "/data.lmp/apaliwal/checkpoints/checkpoint_8p_150k_2cUNet_aligned.pth.tar"):
    print("==> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer: None):
    print("==> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        for param_group in optimizer.param_groups:
            lr = param_group["lr"]
        print("Loading Optimizer. Current learning rate = ", lr)

def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_loaders(img_dir, wimg_dir, mask_dir, bg_dir, train = True, val_frac = 0.2, batch_size = 20, img_transform = None, mask_transform = None,
                augment = False, background = True, num_worker = 4, pin_memory = True, split_seed = 42, want_msk_img_fmt = False, mmap_mode = None):

    if train is True:
        train_ds = SparseHolodataset(img_dir, wimg_dir, mask_dir, bg_dir, img_transform, mask_transform, want_msk_img_fmt, augment, background, mmap_mode)
    else:
        test_ds = holodataset(image_dir = img_dir, mask_dir = mask_dir, img_transform = None, mask_transform = mask_transform)
        test_loader = DataLoader(test_ds, batch_size = batch_size, shuffle = False, num_workers = num_worker, pin_memory = pin_memory)
        return test_loader

    num_total_samples = train_ds.__len__()
    num_train_samples = (num_total_samples - np.ceil(val_frac*num_total_samples)).astype(int)
    num_val_samples = np.ceil(val_frac*num_total_samples).astype(int)
    train_ds, val_ds = torch.utils.data.random_split(train_ds, (num_train_samples, num_val_samples), generator = torch.Generator().manual_seed(split_seed))
    
    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True, num_workers = num_worker, 
                              pin_memory = pin_memory, generator=torch.Generator().manual_seed(split_seed), worker_init_fn=_seed_worker)
    val_loader = DataLoader(val_ds, batch_size = batch_size, shuffle = False, num_workers = num_worker, 
                            pin_memory = pin_memory, generator=torch.Generator().manual_seed(split_seed), worker_init_fn=_seed_worker)

    return train_loader, val_loader

def dice_score(y_pred, y):
    return (2*(y_pred*y).sum()+0.1)/((y_pred**2+y**2).sum()+0.1)

def z_dice_score(y_pred, y, z_acc, z_count):
    
    for i in range(y.shape[0]):
        z = int(torch.max(y[i,1]))
        z_acc[z] += 2*(y_pred[i,0]*y[i,0]).sum()/(y_pred[i,0]**2+y[i,0]**2).sum()
        z_count[z]+=1
    return z_acc

def s_dice_score(y_pred, y, s_acc, s_count):
    
    for i in range(y.shape[0]):
        s = int(torch.max(y[i,2]))
        s_acc[s] += 2*(y_pred[i,0]*y[i,0]).sum()/(y_pred[i,0]**2+y[i,0]**2).sum()
        s_count[s]+=1
    return s_acc


def val_accuracy(loader, model, fc_net, radii_loss, huber_loss, mse_loss, loss_fn, device = "cuda"):
    loss_per_epoch = []
    xy_loss_per_epoch = []
    z_loss_per_epoch = []
    s_loss_per_epoch = []
    pi_loss_per_epoch = []
    
    model.eval()

    with torch.no_grad():
        loop = tqdm(loader)
        for batch_idx, (data, target) in enumerate(loop):
            data = data.to(device = device).float()
            # print(data.shape)
            target = target.to(device = device).float()
            # print(target.shape)
            if fc_net is not None:
                data = torch.concat((data, get_grid(data.shape, device = device)), dim = 1)
                data = torch.concat((data[:,0].unsqueeze(1),fc_net(data)), dim = 1) # have to place before continue statment otherwise mismatch with gt

            #forward         
            prediction = torch.sigmoid(model(data))
            loss, xy_loss, z_loss, s_loss, pi_loss = loss_fn(prediction, target)
            # loss, xy_loss, z_loss = loss_fn(prediction, target)
            
            xy_loss_per_epoch.append(xy_loss)
            z_loss_per_epoch.append(z_loss)
            s_loss_per_epoch.append(s_loss)
            pi_loss_per_epoch.append(pi_loss)
            loss_per_epoch.append(loss)


    loss_per_epoch = float(sum(loss_per_epoch)/len(loss_per_epoch))
    xy_loss_per_epoch = float(sum(xy_loss_per_epoch)/len(xy_loss_per_epoch))
    z_loss_per_epoch = float(sum(z_loss_per_epoch)/len(z_loss_per_epoch))
    s_loss_per_epoch = float(sum(s_loss_per_epoch)/len(s_loss_per_epoch))
    pi_loss_per_epoch = float(sum(pi_loss_per_epoch)/len(pi_loss_per_epoch))
    

    model.train()
    
    return float(loss_per_epoch), float(xy_loss_per_epoch), float(z_loss_per_epoch), float(s_loss_per_epoch), float(pi_loss_per_epoch)

def z_dependence(loader, model, device):
    model.eval()

    z_acc = torch.zeros(201).to(device)
    z_count = torch.ones(201).to(device)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)

            z_acc = z_dice_score(torch.sigmoid(pred), y, z_acc, z_count)
    
    model.train()
    return z_acc/z_count

def size_dependence(loader, model, device):
    model.eval()

    s_acc = torch.zeros(38).to(device)
    s_count = torch.ones(38).to(device)

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)

            s_acc = s_dice_score(torch.sigmoid(pred), y, s_acc, s_count)
    
    model.train()
    return s_acc/s_count

def save_predictions_as_imgs(
    x, y, model, fc_net, folder = "holo/saved_predictions_overfitting", device = "cuda"
):
    model.eval()
    x = x.to(device = device).float()
    if fc_net is not None:
        grid = get_grid(x.shape, device = device)
        x = torch.concat((x, grid), dim = 1)
        x = torch.concat((x[:,0].unsqueeze(1),fc_net(x)), dim = 1) # have to place before continue statment otherwise mismatch with gt  
    
    with torch.no_grad():
        pred = torch.sigmoid(model(x))
        torchvision.utils.save_image(torchvision.transforms.functional.invert(pred[:,0].unsqueeze(1)), f"{folder}/pred_synthetic_xy.png")
        torchvision.utils.save_image(torchvision.transforms.functional.invert(pred[:,1].unsqueeze(1)), f"{folder}/pred_synthetic_z.png") 
        torchvision.utils.save_image(torchvision.transforms.functional.invert(pred[:,2].unsqueeze(1)), f"{folder}/pred_synthetic_d.png")  
        torchvision.utils.save_image(torchvision.transforms.functional.invert(y.unsqueeze(1)), f"{folder}/y_synthetic.png")
        torchvision.utils.save_image(x, f"{folder}/x_synthetic.png")
    
    model.train()
    
def get_mean_std(loader):
    num_channels, num_channels_squared, num_batches= 0, 0, 0
    for data,_ in tqdm(loader):
        num_channels+= torch.mean(data.float(), [0, 2, 3], dtype = torch.float32)
        num_channels_squared+= torch.mean((data.float())**2,[0, 2, 3], dtype = torch.float32)
        num_batches+=1
    
    mean= num_channels/num_batches
    std= (num_channels_squared/num_batches-mean**2)**0.5
    wandb.log({"Mean": mean, "Std": std})
    return mean, std

def precision(hits, fp):
    return hits/(hits+fp)

def recall(hits, fn):
    return hits/(hits+fn)

def f1(precision, recall):
    return 2*(precision*recall)/(precision+recall)

def total_variation_loss(img, weight):
     bs_img, c_img, h_img, w_img = img.size()
     tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
     tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
     return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)

def get_grid(shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        # print(gridx.shape)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        # print(gridx.shape)
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)

class YxyLoss(nn.Module):
    def __init__(self, alpha):
        super(YxyLoss, self).__init__()
        
        self.alpha = alpha
        
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        
    def forward(self, y_hat, y):
        return ( 
                ((1-self.alpha) * self.mse_loss(y_hat, y)) + 
                (self.alpha * self.tv_loss(y_hat.unsqueeze(1), weight=1)) 
            )
        
class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, y, weight):
        bs_img, c_img, h_img, w_img = y.size()
        tv_h = torch.pow(y[:,:,1:,:]-y[:,:,:-1,:], 2).sum()
        tv_w = torch.pow(y[:,:,:,1:]-y[:,:,:,:-1], 2).sum()
        return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)
    
# manuel 
class ThreeHeadLoss(nn.Module):
    EPS = 0.25
    
    def __init__(self, alpha, beta):
        super(ThreeHeadLoss, self).__init__()
        
        self.xy_loss = YxyLoss(alpha)
        self.z_loss = nn.HuberLoss(delta=beta)
        self.d_loss = nn.HuberLoss(delta=beta)

        
    def _calc_z_d_loss(self, y_hat, y):
        particle_locs = (y[:, 0] >= self.EPS)
        
        z_hat = y_hat[:, 1][particle_locs]*200.0
        z = y[:, 1][particle_locs]
        
        d_hat = y_hat[:, 2][particle_locs]*100.0
        d = y[:, 2][particle_locs]*2.0 
        
        z_lval = self.z_loss(z_hat, z)
        d_lval = self.d_loss(d_hat, d)
        
        return z_lval, d_lval
        
        
    def forward(self, y_hat, y):
        
        xy_lval = self.xy_loss(y_hat[:, 0], y[:, 0])    
        z_lval, d_lval = self._calc_z_d_loss(y_hat, y)
        
        return (1*xy_lval + 4*d_lval + 1*z_lval), xy_lval, z_lval, d_lval
    

# # manuel 
# import torch.nn as nn
# class FraunhoferPILoss(nn.Module):
#     EPS = 2.2204e-16  #This is the same value used in matlab epsilon
#     DR = 1e-6
#     DZ = 1e-3
#     DX = 3e-6 #3e-6
    
#     def __init__(self, device, img_size = 4096, n_samples=2048, rec_size=2048, lamda=355e-9):
#         super(FraunhoferPILoss, self).__init__()
#         self.n_samples = n_samples
#         #I know it's a spelling mistake, I don't want to overwrite the lambda keyword
#         self.k = 2 * (torch.pi / lamda)
#         self.device = device
#         self.rec_size = rec_size
#         self.max_nx = rec_size//2
#         self.ratio = rec_size / img_size
#         self.DX = self.DX / self.ratio
#         # self.DX = 3e-6
        
#         self.loss = nn.MSELoss() #nn.HuberLoss(delta=0.01)
        
#         # self.radius_bins =  [5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 13.5, 16.5, 23.5, 56.5, 83.5]
#         self.radius_bins = [6, 7, 8, 9, 10, 12, 15, 20]

        
#     def generate_samples(self, batch_size):
#         ny_smpld = torch.linspace(-self.max_nx, self.max_nx, self.n_samples, device=self.device).repeat(batch_size,1)
#         nx_smpld = torch.linspace(-self.max_nx, self.max_nx, self.n_samples, device=self.device).repeat(batch_size,1)
                                      
#         # ny_smpld = (torch.rand(batch_size, self.n_samples, device=self.device, requires_grad=True)* (2 * self.max_nx) - self.max_nx)
#         # nx_smpld = (torch.rand(batch_size, self.n_samples, device=self.device, requires_grad=True) * (2 * self.max_nx) - self.max_nx)
    
#         return nx_smpld * self.DX, ny_smpld * self.DX


#     def calc_fh(self, nx, ny, xs, ys, rs, zs):
#         nx = nx.unsqueeze(1)
#         ny = ny.unsqueeze(1)
#         xs = (((xs.float().unsqueeze(2))-(self.rec_size//2))*self.DX)
    
#         ys = (((ys.float().unsqueeze(2))-(self.rec_size//2))*self.DX)
#         rs = (rs.unsqueeze(2).unsqueeze(3) * self.DR)
#         zs = (zs.unsqueeze(2).unsqueeze(3) * self.DZ)
            
#         rho = torch.sqrt((((nx - xs) + self.EPS)**2)[:,:,None,:] + (((ny - ys) + self.EPS)**2)[:,:,:,None])
        
#         # Compute the grid in one go
#         grid = torch.sum((rs / (2j * rho)) * torch.special.bessel_j1(self.k * rs * rho / zs) * torch.exp(self.k * 1j * (rho**2 / (2 * zs))), dim=1)
        
#         grid -= 1
        
#         return torch.square(torch.abs(grid))
    
#     # def peak_local_max(self,input, threshold_abs=1, min_distance=1):
#     #     '''
#     #     Returns a binary map where maxima positions are true.

#     #         Parameters:
#     #             input (pytorch tensor): image-like pytorch tensor of dimension [batch_size, channels, width, height], where each image will be processed individually
#     #             threshold_abs (float): local maxima below will be dropped
#     #             min_distance (int): min distance (in pixels) between two maxima detections.
#     #         Returns
#     #             pytorch tensor of same shape as input
#     #     '''
#     #     max_filtered=nn.functional.max_pool2d(input, kernel_size=2*min_distance+1, stride=1, padding=min_distance)
#     #     maxima = torch.eq(max_filtered, input)
#     #     return maxima * (input >= threshold_abs)
    
#     # ayush 
#     def extract_positions(self, y_xyng):
#         """Extracts the positions, outputting two vectors containing the locations along the x-axis and y-axis.

#         Args:
#             y_xyng (_type_): Yxy mask without the gaussian mask applied.
#         """
#         # table = torch.where(self.peak_local_max(y_xyng.unsqueeze(1), 0.8, 2) == 1)
#         # from Rayleigh limit, one can detect particles with 1536x1536 crop and distances <= 12cm for sizes >=23µm
#         # for particles with size <23µm the first zero of j1 is outside the crop size, hence an incomplete/masked 
#         # signal. We can give an (masked) autoencoding objective to recover the incomplete signal for such particles. 
#         # call it physics-informed autoencoding objective.
#         # table = torch.where(y_xyng.unsqueeze(1)== 1)
#         # binary_map = (y_xyng > 0)*(y_xyng < 23/2)
#         # table = torch.where(binary_map.unsqueeze(1) == 1) #23µm
        
#         # batch_size = y_xyng.shape[0]
#         # xs = []
#         # ys = []
#         # max_particles = 4
        
#         # for i in range(batch_size):
#         #     selection = table[0]==i
#         #     rnd_indices = torch.randint(low = 0,high=len(table[0][selection]), size = (max_particles,))
#         #     nx = table[3][selection][rnd_indices]
#         #     ny = table[2][selection][rnd_indices]
#         #     xs.append(nx)
#         #     ys.append(ny)
#         # xs = torch.stack(xs)
#         # ys = torch.stack(ys)
#         # binary_map = (y_xyng > 0) & (y_xyng < 40 / 2)
#         rnd_idx = torch.randint(0, len(self.radius_bins)-1)
#         binary_map = (y_xyng > self.radius_bins[rnd_idx]/2) & (y_xyng < self.radius_bins[rnd_idx+1]/2)
#         # binary_map = (y_xyng >= 30.0)
#         # binary_map = y_xyng == 1
#         table = torch.nonzero(binary_map, as_tuple=False)  # More efficient alternative to `torch.where` for indices

#         batch_size = y_xyng.shape[0]
#         max_particles = 4

#         # Split the batch indices
#         batch_indices = table[:, 0]
#         y_indices = table[:, 1]
#         x_indices = table[:, 2]

#         # Prepare to select random indices for each batch
#         unique_batches = torch.arange(batch_size, device=y_xyng.device)
#         xs = []
#         ys = []

#         for i in unique_batches:
#             selection = (batch_indices == i).nonzero(as_tuple=False).squeeze(1)  # Indices for current batch
#             if len(selection) >= max_particles:
#                 rnd_indices = torch.randint(low=0, high=len(selection), size=(max_particles,), device=y_xyng.device)
#             else:
#                 rnd_indices = torch.randint(low=0, high=len(selection), size=(max_particles,), device=y_xyng.device) % len(selection)
#             selected = selection[rnd_indices]
#             xs.append(x_indices[selected])
#             ys.append(y_indices[selected])

#         # Stack to tensors
#         xs = torch.stack(xs)
#         ys = torch.stack(ys)

#         return xs, ys

    
#     def forward(self, y_hat, y):
#         nx, ny = self.generate_samples(y.shape[0])
        
#         # xs, ys = self.extract_positions(y[:, 0])
#         xs, ys = self.extract_positions(y[:, 4]) # sending size for autoencoding objective 

#         # print(xs[0])
#         batch_indices = torch.arange(y.size(0)).unsqueeze(1).expand_as(xs)  # shape (b, n)
        
#         rs_hat = (50) * y_hat[:, 2][batch_indices, ys , xs]
#         zs_hat = (200) * y_hat[:, 1][batch_indices, ys , xs]


#         # out_holo = self.calc_fh(nx, ny, xs*4, ys*4, rs_hat, zs_hat) #rs_hat TODO: CHANGE BACK
#         # out_gt_holo = self.calc_fh(nx, ny, xs*4, ys*4, y[:, 2][batch_indices, ys, xs], y[:, 1][batch_indices, ys , xs])

#         out_holo = self.calc_fh(nx, ny, xs*4, ys*4, rs_hat, zs_hat) #rs_hat TODO: CHANGE BACK
#         out_gt_holo = self.calc_fh(nx, ny, xs*4, ys*4, y[:, 2][batch_indices, ys, xs], y[:, 1][batch_indices, ys , xs])
        
#         out_holo -= torch.min(out_holo)
#         out_holo /= torch.max(out_holo)

#         out_gt_holo -= torch.min(out_gt_holo)
#         out_gt_holo /= torch.max(out_gt_holo)

#         return self.loss(out_holo,out_gt_holo) 


import torch.nn as nn
class FraunhoferPILoss(nn.Module):
    EPS = 2.2204e-16  #This is the same value used in matlab epsilon
    DR = 1e-6
    DZ = 1e-3
    DX = 3e-6 #3e-6
    
    def __init__(self, device, img_size = 4096, n_samples=2048, rec_size=2048, lamda=355e-9):
        super(FraunhoferPILoss, self).__init__()
        self.n_samples = n_samples
        #I know it's a spelling mistake, I don't want to overwrite the lambda keyword
        self.k = 2 * (torch.pi / lamda)
        self.device = device
        self.rec_size = rec_size
        self.max_nx = rec_size//2
        self.ratio = rec_size / img_size
        self.DX = self.DX / self.ratio
        # self.DX = 3e-6
        
        self.loss = nn.MSELoss() #nn.HuberLoss(delta=0.01)
        
        # self.radius_bins =  [5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 13.5, 16.5, 23.5, 56.5, 83.5]
        # self.radius_bins = [6, 7, 8, 9, 10, 12, 15, 20]

        
    def generate_samples(self, batch_size):
        ny_smpld = torch.linspace(-self.max_nx, self.max_nx, self.n_samples, device=self.device).repeat(batch_size,1)
        nx_smpld = torch.linspace(-self.max_nx, self.max_nx, self.n_samples, device=self.device).repeat(batch_size,1)
                                      
        # ny_smpld = (torch.rand(batch_size, self.n_samples, device=self.device, requires_grad=True)* (2 * self.max_nx) - self.max_nx)
        # nx_smpld = (torch.rand(batch_size, self.n_samples, device=self.device, requires_grad=True) * (2 * self.max_nx) - self.max_nx)
    
        return nx_smpld * self.DX, ny_smpld * self.DX


    def calc_fh(self, nx, ny, xs, ys, rs, zs):
        nx = nx.unsqueeze(1)
        ny = ny.unsqueeze(1)
        xs = (((xs.float().unsqueeze(2))-(self.rec_size//2))*self.DX)
    
        ys = (((ys.float().unsqueeze(2))-(self.rec_size//2))*self.DX)
        rs = (rs.unsqueeze(2).unsqueeze(3) * self.DR)
        zs = (zs.unsqueeze(2).unsqueeze(3) * self.DZ)
            
        rho = torch.sqrt((((nx - xs) + self.EPS)**2)[:,:,None,:] + (((ny - ys) + self.EPS)**2)[:,:,:,None])
        
        # Compute the grid in one go
        grid = torch.sum((rs / (2j * rho)) * torch.special.bessel_j1(self.k * rs * rho / zs) * torch.exp(self.k * 1j * (rho**2 / (2 * zs))), dim=1)
        
        grid -= 1
        
        return torch.square(torch.abs(grid))
    
    # def peak_local_max(self,input, threshold_abs=1, min_distance=1):
    #     '''
    #     Returns a binary map where maxima positions are true.

    #         Parameters:
    #             input (pytorch tensor): image-like pytorch tensor of dimension [batch_size, channels, width, height], where each image will be processed individually
    #             threshold_abs (float): local maxima below will be dropped
    #             min_distance (int): min distance (in pixels) between two maxima detections.
    #         Returns
    #             pytorch tensor of same shape as input
    #     '''
    #     max_filtered=nn.functional.max_pool2d(input, kernel_size=2*min_distance+1, stride=1, padding=min_distance)
    #     maxima = torch.eq(max_filtered, input)
    #     return maxima * (input >= threshold_abs)
    
    # ayush 
    def extract_positions(self, y_xyng):
        """Extracts the positions, outputting two vectors containing the locations along the x-axis and y-axis.

        Args:
            y_xyng (_type_): Yxy mask without the gaussian mask applied.
        """
        # table = torch.where(self.peak_local_max(y_xyng.unsqueeze(1), 0.8, 2) == 1)
        # from Rayleigh limit, one can detect particles with 1536x1536 crop and distances <= 12cm for sizes >=23µm
        # for particles with size <23µm the first zero of j1 is outside the crop size, hence an incomplete/masked 
        # signal. We can give an (masked) autoencoding objective to recover the incomplete signal for such particles. 
        # call it physics-informed autoencoding objective.
        # table = torch.where(y_xyng.unsqueeze(1)== 1)
        # binary_map = (y_xyng > 0)*(y_xyng < 23/2)
        # table = torch.where(binary_map.unsqueeze(1) == 1) #23µm
        
        # batch_size = y_xyng.shape[0]
        # xs = []
        # ys = []
        # max_particles = 4
        
        # for i in range(batch_size):
        #     selection = table[0]==i
        #     rnd_indices = torch.randint(low = 0,high=len(table[0][selection]), size = (max_particles,))
        #     nx = table[3][selection][rnd_indices]
        #     ny = table[2][selection][rnd_indices]
        #     xs.append(nx)
        #     ys.append(ny)
        # xs = torch.stack(xs)
        # ys = torch.stack(ys)
        binary_map = (y_xyng > 0) & (y_xyng < 23 / 2)
        # rnd_idx = torch.randint(low = 0, high = len(self.radius_bins)-1, size=(1,))
        # binary_map = (y_xyng > self.radius_bins[rnd_idx]/2) & (y_xyng < self.radius_bins[rnd_idx+1]/2)
        # binary_map = (y_xyng >= 30.0)
        # binary_map = y_xyng == 1
        table = torch.nonzero(binary_map, as_tuple=False)  # More efficient alternative to `torch.where` for indices

        batch_size = y_xyng.shape[0]
        max_particles = 4

        # Split the batch indices
        batch_indices = table[:, 0]
        y_indices = table[:, 1]
        x_indices = table[:, 2]

        # Prepare to select random indices for each batch
        unique_batches = torch.arange(batch_size, device=y_xyng.device)
        xs = []
        ys = []
   
        for i in unique_batches:
            selection = (batch_indices == i).nonzero(as_tuple=False).squeeze(1)  # Indices for current batch
            if len(selection) >= max_particles:
                rnd_indices = torch.randint(low=0, high=len(selection), size=(max_particles,), device=y_xyng.device)
            else:
                rnd_indices = torch.randint(low=0, high=len(selection), size=(max_particles,), device=y_xyng.device) % len(selection)
            selected = selection[rnd_indices]
            xs.append(x_indices[selected])
            ys.append(y_indices[selected])

        # Stack to tensors
        xs = torch.stack(xs)
        ys = torch.stack(ys)

        return xs, ys

    
    def forward(self, y_hat, y):
        nx, ny = self.generate_samples(y.shape[0])
        
        # xs, ys = self.extract_positions(y[:, 0])
        xs, ys = self.extract_positions(y[:, 4]) # sending size for autoencoding objective 

        batch_indices = torch.arange(y.size(0)).unsqueeze(1).expand_as(xs)  # shape (b, n)
        
        rs_hat = (50) * y_hat[:, 2][batch_indices, ys , xs]
        zs_hat = (200) * y_hat[:, 1][batch_indices, ys , xs]

        # rs_hat = (1) * y_hat[:, 2][batch_indices, ys , xs]
        # zs_hat = (1) * y_hat[:, 1][batch_indices, ys , xs]
        
        # out_holo = self.calc_fh(nx, ny, xs*4, ys*4, rs_hat, zs_hat) #rs_hat TODO: CHANGE BACK
        # out_gt_holo = self.calc_fh(nx, ny, xs*4, ys*4, y[:, 2][batch_indices, ys, xs], y[:, 1][batch_indices, ys , xs])

        out_holo = self.calc_fh(nx, ny, xs*4, ys*4, rs_hat, zs_hat) #rs_hat TODO: CHANGE BACK
        out_gt_holo = self.calc_fh(nx, ny, xs*4, ys*4, y[:, 2][batch_indices, ys, xs], y[:, 1][batch_indices, ys , xs])
        

        return self.loss(out_holo,out_gt_holo)*(self.n_samples*self.n_samples)/(384*384)




# manuel 
class ThreeHeadPILoss(nn.Module):
    EPS = 0.25
    
    def __init__(self, alpha, beta, device, img_size, n_samples, rec_size):
        super(ThreeHeadPILoss, self).__init__()
        
        self.xy_loss = YxyLoss(alpha)
        self.z_loss = nn.HuberLoss(delta=beta)
        self.d_loss = nn.HuberLoss(delta=beta)
        self.pi_loss = FraunhoferPILoss(device, img_size, n_samples, rec_size)

        
    def _calc_z_d_loss(self, y_hat, y):
        particle_locs = (y[:, 0] >= self.EPS)
        
        z_hat = y_hat[:, 1][particle_locs]*200.0
        z = y[:, 1][particle_locs]
        
        d_hat = y_hat[:, 2][particle_locs]*100.0 # x2 for same level as z
        d = y[:, 2][particle_locs]*2.0
        
        z_lval = self.z_loss(z_hat, z)
        d_lval = self.d_loss(d_hat, d)
        
        return z_lval, d_lval
        
        
    def forward(self, y_hat, y):
        
        xy_lval = self.xy_loss(y_hat[:, 0], y[:, 0])    
        z_lval, d_lval = self._calc_z_d_loss(y_hat, y)
        pi_loss = self.pi_loss(y_hat, y)
        
        return (100*xy_lval + 4*d_lval + 1*z_lval + pi_loss), xy_lval, z_lval, d_lval, pi_loss