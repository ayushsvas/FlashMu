import torch
import numpy as np
import random


def load_checkpoint(checkpoint, model, optimizer: None):
    print("==> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        for param_group in optimizer.param_groups:
            lr = param_group["lr"]
        print("Loading Optimizer. Current learning rate = ", lr)


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
