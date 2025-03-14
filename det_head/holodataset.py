import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import random
from mask_preprocess import *

def file_loader(file):
    return np.load(file)

class SparseHolodataset(Dataset):
    
    def __init__(
        self,
        img_pth,
        wimg_pth,
        msk_pth,
        bg_dir,
        img_trsnf = None,
        msk_trsnf = None,
        want_msk_img_fmt = False,
        augment = False,
        background = False,
        mmap_mode = None
    ):
        
        self.mmap_mode = mmap_mode

        # set manually for now
        self.holosize = 4096
        self.crop_size = 1536
        self.crop_stride = 1280
        
        # self.holosize = 4096
        # self.crop_size = 512
        # self.crop_stride = 512
        
        self.msks = msk_pth
        self.want_msk_img_fmt = want_msk_img_fmt
        self.msk_trnsf = msk_trsnf


        if msk_pth is not None:
            with open(msk_pth, 'rb') as f:
                self.msks = pickle.load(f)

            if '_xypx_binned' not in msk_pth:
                self.ds_factor = 2
                self.msks = preprocess_pparas(pparas=self.msks, holonum_holosize_cropsize_cropstride=(len(self.msks['x']), self.holosize, self.crop_size, self.crop_stride), 
                                            ds_factor=self.ds_factor, want_img_format=self.want_msk_img_fmt, save_pth=msk_pth[:-4]+"_".join(map(str, [self.holosize, self.crop_size, self.crop_stride])))
                self.ds_factor = 2
            else:
                self.ds_factor = 1
                # self.ds_factor = 2
        
        # self.msks['x'] = self.msks['x'][32400:]#+self.msks['x'][1350320:]
        # self.msks['y'] = self.msks['y'][32400:]#+self.msks['y'][1350320:]
        # self.msks['z'] = self.msks['z'][32400:]#+self.msks['z'][1350320:]
        # self.msks['r'] = self.msks['r'][32400:]#+self.msks['r'][1350320:]
        # self.msks = preprocess_pparas(pparas=self.msks, holonum_holosize_cropsize_cropstride=(None, 512, 512, None), 
        #                       ds_factor=4, want_img_format=self.want_msk_img_fmt)
        # self.msks = preprocess_pparas(pparas=self.msks, holonum_holosize_cropsize_cropstride=(261500, 1536, 1536, 1536), 
        #               ds_factor=4, want_img_format=self.want_msk_img_fmt)
        
        self.img_trnsf = img_trsnf
        self.wimg_pth = wimg_pth
        
        if wimg_pth is not None:
            self.imgs = np.load(img_pth, mmap_mode=mmap_mode)
            self.wimgs = np.load(wimg_pth, mmap_mode=mmap_mode)
        else:
            self.imgs = np.load(img_pth, mmap_mode=mmap_mode)


        # if wimg_pth is not None:
        #     self.imgs = np.load(img_pth, mmap_mode=mmap_mode)[32400:]
        #     self.wimgs = np.load(wimg_pth, mmap_mode=mmap_mode)[32400:]
        # else:
        #     self.imgs = np.load(img_pth, mmap_mode=mmap_mode)[32400:]



        self.augment = augment
        self.background_aug = background
        if self.background_aug:
            self.background = np.load(bg_dir)
            print(f"Loaded backgrounds of shape: {self.background.shape}.")
    
       
    def __len__(self):
        # return 32400
        return len(self.imgs)
   
   
    def _gauss_kern(self,l=5, sig=1.):
        l_max = (l-1) / 2.
        
        ax = np.linspace(-l_max, l_max, l)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    
        return np.outer(gauss, gauss)
    
    
    
    def _densify_msk(self, shape, msk, kernel_sz=3):
        msk_shape = (5, shape[1]+kernel_sz+1, shape[2]+kernel_sz+1)

        xx, yy, zz, dd = msk

        dense_msk = np.zeros(tuple(msk_shape), dtype=np.float32)
        gk = self._gauss_kern(kernel_sz, 1)

        for x, y, z, d in zip(xx, yy, zz, dd):
            
        
            y_max = kernel_sz + y
            x_max = kernel_sz + x
            dense_msk[0, y:y_max, x:x_max] = gk
            dense_msk[1, y:y_max, x:x_max] = z
            dense_msk[2, y:y_max, x:x_max] = d
            dense_msk[3, y, x] = z
            dense_msk[4, y, x] = d
    
        offset = 1 + kernel_sz//2
        return dense_msk[:, offset:(shape[1]+offset), offset:(shape[2]+offset)]
    
    
    def custom_transform(self, image, mask):
        hflip_rate, vflip_rate, rot90, contrast_rate, brightness_rate = [(random.random()>0.5) for _ in range(5)]
        if hflip_rate:
          image = np.flip(image, axis = 2).copy() # Flips give rise to negative strides in the npy memory block which torch doesn't have. .copy() works.
          mask = np.flip(mask, axis = 2).copy()
        #   print('Horizontal Flip')
        if vflip_rate:
          image = np.flip(image, axis = 1).copy()
          mask = np.flip(mask, axis = 1).copy()

        if rot90:
          k = random.randint(1,3)
          image = np.rot90(image, k,axes=(1,2)).copy()
          mask = np.rot90(mask, k, axes = (1,2)).copy()
        #   print('Vertical Flip')
        # if contrast_rate:
        #   a = random.uniform(9200/12000*1/22.3824-0.01, 9200/12000*1/22.3824-0.01+0.01) # 27 percent variation contrast around 0 mean
        #   image[0] = np.clip(a*(image[0] - 128) + 128, a_min = None, a_max = 255)
        #   print('Contrast')
        if brightness_rate:
          b = random.uniform(-24,24) # 27 percent variation brightness above and below mean value
          image[0] = np.clip(image[0] + b, a_min = None, a_max = 255) 
        #   print('Brightness')
          
        return image, mask

    def background_augment(self, image):
       
        # print("Background!")
        # randomly select a background from the set of background images 
        random_index = np.random.randint(self.background.shape[0])
        
        # a random contrast factor k\in[0.5, 1] 
        #  k = np.random.uniform(0.5,1)
        k = 1.0            
        
        # 128 is the gray level of the synthetic hologram, the standard deviation of background will be scaled and mean will remain 0.
        # have already made them mean 0 while saving, so simply linearly adding them here to the hologram image
        image += (self.background[random_index]*128.0)*k

        # the background should have the plane wave but during forward simulation, they were fed inside the image
        # so give the background its plane wave and clip
        # background += 128
        # background = np.maximum(np.zeros(background.shape), np.minimum(background, np.ones(background.shape)*255))

        # clip the values between 0-255 for the hologram image
        # image = np.maximum(np.zeros(image.shape), np.minimum(image, np.ones(image.shape)*255))
        np.clip(image, 0, 255, out=image)       
    
        return image
    

         
    def __getitem__(self, ix):
        
        if self.mmap_mode:
            img = (self.imgs[ix].copy())[np.newaxis,:,:]
        else:
            img = self.imgs[ix][np.newaxis,:,:]
        
        # if self.background_aug and ix < 32400:
        if self.background_aug:
            img[0] = self.background_augment(img[0])
        

        if self.wimg_pth is not None:
            if self.mmap_mode:
                wimg = (self.wimgs[ix].copy())[np.newaxis,:,:]
            else:
                wimg = (self.wimgs[ix])[np.newaxis,:,:]

        if self.msks is not None:
            if self.want_msk_img_fmt:
                msk = self.msks[ix]
            else:
                msk = self._densify_msk(img.shape,  (np.int32(np.rint(self.msks['x'][ix]/self.ds_factor)), np.int32(np.rint(self.msks['y'][ix]/self.ds_factor)),
                                                      self.msks['z'][ix], self.msks['r'][ix])) 
                
        if self.msks is not None and self.wimg_pth is not None:
            img = np.concatenate((img, wimg), axis = 0)

        if self.msks is None and self.wimg_pth is not None:
            msk = wimg

        if self.augment:
            img, msk = self.custom_transform(img, msk)

        img = torch.from_numpy(img)
        msk = torch.from_numpy(msk)

        if self.img_trnsf is not None:
            img = self.img_trnsf(img)
        if self.msk_trnsf is not None:
            msk = self.msk_trnsf(msk)

        return img, msk
        
        