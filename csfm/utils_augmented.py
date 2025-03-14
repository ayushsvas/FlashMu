import sys
import math
import torch
import torch.nn.functional as F
import torchvision 
from dataset_original import holodataset
# from dataset_propagate import holodataset 
# from dataset_propagate_every2cm import holodataset 
# from dataset_propagate_every2cm import holodataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb

avgpool = torch.nn.AvgPool2d(4,4)

def train_log(loss, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "training_loss": loss})

def val_log(loss, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "validation_loss": loss})


def save_checkpoint(state, filename = None):
    print("==> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer = None, data_parallel = False):
    print("==> Loading checkpoint")
    if data_parallel:
        model.module.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        for param_group in optimizer.param_groups:
            lr = param_group["lr"]
        print("Loading Optimizer. Current learning rate = ", lr)

def get_loaders(img_dir, mask_dir, background_dir, num_samples, train = True, val_frac = 0.5, batch_size = 20, img_transform = None, mask_transform = None, 
                augment = False, bg_augment = False, num_worker = 4, pin_memory = True, split_seed = 42, mmap_mode = False):

    if train is True:
        train_ds = holodataset(image_dir = img_dir, mask_dir = mask_dir, background_dir=background_dir, num_samples=num_samples, 
                               img_transform = img_transform, mask_transform = mask_transform,
                                augment = augment, bg_augment=bg_augment, mmap_mode=mmap_mode)
    else:
        test_ds = holodataset(image_dir = img_dir, mask_dir = mask_dir, img_transform = None, mask_transform = mask_transform)
        test_loader = DataLoader(test_ds, batch_size = batch_size, shuffle = False, num_workers = num_worker, pin_memory = pin_memory)
        return test_loader

    num_total_samples = train_ds.__len__()
    num_train_samples = (num_total_samples - np.ceil(val_frac*num_total_samples)).astype(int)
    num_val_samples = np.ceil(val_frac*num_total_samples).astype(int)
    train_ds, val_ds = torch.utils.data.random_split(train_ds, (num_train_samples, num_val_samples), generator = torch.Generator().manual_seed(split_seed))
    
    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True, num_workers = num_worker, pin_memory = pin_memory)
    val_loader = DataLoader(val_ds, batch_size = batch_size, shuffle = False, num_workers = num_worker, pin_memory = pin_memory )

    return train_loader, val_loader#, mean_ds, std_ds

def dice_score(y_pred, y):
    return (2*(y_pred*y).sum())/((y_pred**2+y**2).sum())

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


def val_accuracy(loader, model, loss_fn, huber_loss, mse_loss, mae_loss, device = "cuda"):

    alpha = 0.0001

    loss_this_epoch = []
    dce_this_epoch = []
    model.eval()
    # optimizer.eval()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data = data.to(device = device).float()
            grid = get_grid(data.shape, device = device)
            data = torch.concat((data, grid), dim = 1)
            target_holo = target.to(device = device).float()
            # target_holo = target[:,0].unsqueeze(1).to(device = device).float()
            # target_xy = target[:,0]
            # target_z = target[:,1]
            # target_s = target[:,2]
            # target_holo = target[:,3]
            

            #forward 
            prediction_holo = model(data)

            # Calculate losses
            loss = mse_loss(prediction_holo, target_holo) 
            # loss = mse_loss(prediction_holo, F.interpolate(target_holo, size = (768,768), mode = 'bicubic', antialias=True)) 
            
            # Save losses
            loss_this_epoch.append(float(loss))


    
    loss_this_epoch = float(sum(loss_this_epoch)/len(loss_this_epoch))    
    # dce_this_epoch = float(sum(dce_this_epoch)/len(dce_this_epoch))
    
    model.train()

    return (loss_this_epoch),(dce_this_epoch)


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
    x, y, model, folder = "holo/saved_predictions_overfitting", device = "cuda"
):
    model.eval()  
    x = x.to(device = device).float()
    grid = get_grid(x.shape, device = device)
    x = torch.concat((x, grid), dim = 1)
    # pred_holos = []
    with torch.no_grad():
        predictions = model(x)
        
        # # Save to wandb 
        # pred_holos = [wandb.Image((1-torch.sigmoid(pred[0]).unsqueeze(0)).permute(1,2,0).cpu().detach().numpy()) for pred in predictions]
        # wandb.log({"pred_holos": pred_holos})
        
        #Save locally
        # torchvision.utils.save_image(torchvision.transforms.functional.invert(torch.sigmoid(predictions[:,0].unsqueeze(1))), f"{folder}/pred_xy.png")
        torchvision.utils.save_image(predictions/255, folder+"pred_holo.png")
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


#Set fix{x,y,z}_bnd if function is non-periodic in {x,y,z} direction
#x: (*, s)
#y: (*, s)
def central_diff_1d(x, h, fix_x_bnd=False):
    dx = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1))/(2.0*h)

    if fix_x_bnd:
        dx[...,0] = (x[...,1] - x[...,0])/h
        dx[...,-1] = (x[...,-1] - x[...,-2])/h
    
    return dx

#x: (*, s1, s2)
#y: (*, s1, s2)
def central_diff_2d(x, h, fix_x_bnd=False, fix_y_bnd=False):
    if isinstance(h, float):
        h = [h, h]

    dx = (torch.roll(x, -1, dims=-2) - torch.roll(x, 1, dims=-2))/(2.0*h[0])
    dy = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1))/(2.0*h[1])

    if fix_x_bnd:
        dx[...,0,:] = (x[...,1,:] - x[...,0,:])/h[0]
        dx[...,-1,:] = (x[...,-1,:] - x[...,-2,:])/h[0]
    
    if fix_y_bnd:
        dy[...,:,0] = (x[...,:,1] - x[...,:,0])/h[1]
        dy[...,:,-1] = (x[...,:,-1] - x[...,:,-2])/h[1]
        
    return dx, dy

#x: (*, s1, s2, s3)
#y: (*, s1, s2, s3)
def central_diff_3d(x, h, fix_x_bnd=False, fix_y_bnd=False, fix_z_bnd=False):
    if isinstance(h, float):
        h = [h, h, h]

    dx = (torch.roll(x, -1, dims=-3) - torch.roll(x, 1, dims=-3))/(2.0*h[0])
    dy = (torch.roll(x, -1, dims=-2) - torch.roll(x, 1, dims=-2))/(2.0*h[1])
    dz = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1))/(2.0*h[2])

    if fix_x_bnd:
        dx[...,0,:,:] = (x[...,1,:,:] - x[...,0,:,:])/h[0]
        dx[...,-1,:,:] = (x[...,-1,:,:] - x[...,-2,:,:])/h[0]
    
    if fix_y_bnd:
        dy[...,:,0,:] = (x[...,:,1,:] - x[...,:,0,:])/h[1]
        dy[...,:,-1,:] = (x[...,:,-1,:] - x[...,:,-2,:])/h[1]
    
    if fix_z_bnd:
        dz[...,:,:,0] = (x[...,:,:,1] - x[...,:,:,0])/h[2]
        dz[...,:,:,-1] = (x[...,:,:,-1] - x[...,:,:,-2])/h[2]
        
    return dx, dy, dz


#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=1, p=2, L=2*math.pi, reduce_dims=0, reductions='sum'):
        super().__init__()

        self.d = d
        self.p = p

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims
        
        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                assert reductions == 'sum' or reductions == 'mean'
                self.reductions = [reductions]*len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    assert reductions[j] == 'sum' or reductions[j] == 'mean'
                self.reductions = reductions

        if isinstance(L, float):
            self.L = [L]*self.d
        else:
            self.L = L
    
    def uniform_h(self, x):
        h = [0.0]*self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j]/x.size(-j)
        
        return h

    def reduce_all(self, x):
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == 'sum':
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)
        
        return x

    def abs(self, x, y, h=None):
        #Assume uniform mesh
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h]*self.d
        
        const = math.prod(h)**(1.0/self.p)
        diff = const*torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                                              p=self.p, dim=-1, keepdim=False)

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
            
        return diff

    def rel(self, x, y):

        diff = torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                          p=self.p, dim=-1, keepdim=False)
        ynorm = torch.norm(torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False)

        diff = diff/ynorm

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
            
        return diff

    def __call__(self, y_pred, y, **kwargs):
        return self.rel(y_pred, y)


class H1Loss(object):
    def __init__(self, d=1, L=2*math.pi, reduce_dims=0, reductions='sum', fix_x_bnd=False, fix_y_bnd=False, fix_z_bnd=False, T: int = 1):
        super().__init__()

        assert d > 0 and d < 4, "Currently only implemented for 1, 2, and 3-D."

        self.d = d
        self.fix_x_bnd = fix_x_bnd
        self.fix_y_bnd = fix_y_bnd
        self.fix_z_bnd = fix_z_bnd
        self.T = T

        if self.T > 1:
            self.avgpool = torch.nn.AvgPool2d(self.T, self.T)

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims
        
        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                assert reductions == 'sum' or reductions == 'mean'
                self.reductions = [reductions]*len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    assert reductions[j] == 'sum' or reductions[j] == 'mean'
                self.reductions = reductions

        if isinstance(L, float):
            self.L = [L]*self.d
        else:
            self.L = L
    
    def compute_terms(self, x, y, h):
        dict_x = {}
        dict_y = {}

        if self.d == 1:
            dict_x[0] = x
            dict_y[0] = y

            x_x = central_diff_1d(x, h[0], fix_x_bnd=self.fix_x_bnd)
            y_x = central_diff_1d(y, h[0], fix_x_bnd=self.fix_x_bnd)

            dict_x[1] = x_x
            dict_y[1] = y_x
        
        elif self.d == 2:
            dict_x[0] = torch.flatten(x, start_dim=-2)
            dict_y[0] = torch.flatten(y, start_dim=-2)

            if self.T > 1:
                x = self.avgpool(x)
                y = self.avgpool(y)
                

            x_x, x_y = central_diff_2d(x, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd)
            y_x, y_y = central_diff_2d(y, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd)

            dict_x[1] = torch.flatten(x_x, start_dim=-2)
            dict_x[2] = torch.flatten(x_y, start_dim=-2)

            dict_y[1] = torch.flatten(y_x, start_dim=-2)
            dict_y[2] = torch.flatten(y_y, start_dim=-2)
        
        else:
            dict_x[0] = torch.flatten(x, start_dim=-3)
            dict_y[0] = torch.flatten(y, start_dim=-3)

            x_x, x_y, x_z = central_diff_3d(x, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd, fix_z_bnd=self.fix_z_bnd)
            y_x, y_y, y_z = central_diff_3d(y, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd, fix_z_bnd=self.fix_z_bnd)

            dict_x[1] = torch.flatten(x_x, start_dim=-3)
            dict_x[2] = torch.flatten(x_y, start_dim=-3)
            dict_x[3] = torch.flatten(x_z, start_dim=-3)

            dict_y[1] = torch.flatten(y_x, start_dim=-3)
            dict_y[2] = torch.flatten(y_y, start_dim=-3)
            dict_y[3] = torch.flatten(y_z, start_dim=-3)
        
        return dict_x, dict_y

    def uniform_h(self, x):
        h = [0.0]*self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j]/x.size(-j)
        
        return h
    
    def reduce_all(self, x):
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == 'sum':
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)
        
        return x
        
    def abs(self, x, y, h=None):
        #Assume uniform mesh
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h]*self.d
            
        dict_x, dict_y = self.compute_terms(x, y, h)

        const = math.prod(h)
        diff = const*torch.norm(dict_x[0] - dict_y[0], p=2, dim=-1, keepdim=False)**2

        for j in range(1, self.d + 1):
            diff += const*torch.norm(dict_x[j] - dict_y[j], p=2, dim=-1, keepdim=False)**2
        
        diff = diff**0.5

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
            
        return diff
        
    def rel(self, x, y, h=None):
        #Assume uniform mesh
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h]*self.d
        


        dict_x, dict_y = self.compute_terms(x, y, h)

        diff = torch.norm(dict_x[0] - dict_y[0], p=2, dim=-1, keepdim=False)**2
        ynorm = torch.norm(dict_y[0], p=2, dim=-1, keepdim=False)**2


        for j in range(1, self.d + 1):
            diff += torch.norm(dict_x[j] - dict_y[j], p=2, dim=-1, keepdim=False)**2
            ynorm += torch.norm(dict_y[j], p=2, dim=-1, keepdim=False)**2
        
        diff = (diff**0.5)/(ynorm**0.5)

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
            
        return diff


    def __call__(self, y_pred, y, h=None, **kwargs):
        return self.rel(y_pred, y, h=h)
    

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