
import random
import numpy as np 
from torch.utils.data import Dataset

class holodataset(Dataset):
    def __init__(self, image_dir, mask_dir, num_samples, background_dir = None, img_transform = None, mask_transform = None, augment = False, bg_augment = False,
                 mmap_mode = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.background_dir = background_dir  
        self.num_samples = num_samples

        self.mmmap_mode = mmap_mode

        # self.images = np.load(image_dir, mmap_mode=mmap_mode)[60750:]
        # self.masks = np.load(mask_dir, mmap_mode=mmap_mode)[60750:]

        self.images = np.load(image_dir, mmap_mode=mmap_mode)
        self.masks = np.load(mask_dir, mmap_mode=mmap_mode)

        if self.background_dir is not None:
          self.background = np.load(background_dir)

        self.img_transform = img_transform 
        self.mask_transform = mask_transform 
        self.augment = augment
        self.bg_augment = bg_augment
        

    def __len__(self):
        # return self.num_samples
        return len(self.images)
        
    def background_augment(self, image):
       
      # randomly select a background from the set of background images 
      random_index = np.random.randint(self.background.shape[0])
      
      # a random contrast factor k\in[0.5, 1] 
      #  k = np.random.uniform(0.5,1)
      k = 1 # factor to control the intensity of background to be added
      
      # 128 is the gray level of the synthetic hologram, the standard deviation of background will be scaled and mean will remain 0.
      background = (self.background[random_index]*128)*k
      
      # have already made them mean 0 while saving, so simply linearly adding them here to the hologram image
      image = image + background

      # the background should have the plane wave but during forward simulation, they were fed inside the image
      # so give the background its plane wave and clip
      background += 128
      background = np.maximum(np.zeros(background.shape), np.minimum(background, np.ones(background.shape)*255))

      # clip the values between 0-255 for the hologram image
      image = np.maximum(np.zeros(image.shape), np.minimum(image, np.ones(image.shape)*255))       
          
      return image[:, :]
        
    
   ################custom transforms to channel inputs (hologram, background) #############
    def custom_transform(self, image, mask):
        hflip_rate, vflip_rate, rot90, contrast_rate, brightness_rate = [(random.random()>0.5) for _ in range(5)]
      
        if hflip_rate:
          image = np.flip(image, axis = 0).copy() # Flips give rise to negative strides in the npy memory block which torch doesn't have. .copy() works.
          mask = np.flip(mask, axis = 0).copy()  # image is 2 channel but mask is 1 channel
          # print("hflip")

        if vflip_rate:
          image = np.flip(image, axis = 1).copy()
          mask = np.flip(mask, axis = 1).copy()

        if rot90:
          k = random.randint(1,3)
          image = np.rot90(image, k,axes=(0,1)).copy()
          mask = np.rot90(mask, k, axes = (0,1)).copy()
 
          # print("vflip")
        # if contrast_rate:
        #   a = random.uniform(0.760,1.240) # 24 percent variation contrast around 0 mean
        #   image = np.clip(a*(image - np.mean(image)) + np.mean(image), a_min = None, a_max = 255)

        if brightness_rate:
          b = random.uniform(-24,24) # 24 percent variation brightness above and below mean value
          image = np.clip(image + b, a_min = None, a_max = 255) 
          # print("brightness transform")
        return image, mask

    def __getitem__(self, index):
        if self.mmmap_mode == 'r':
          image = self.images[index].copy()
          mask = self.masks[index].copy()  
        else:
          image = self.images[index]
          mask = self.masks[index]
        
        if self.augment:
          if self.bg_augment:
            # if index < 60750:
              image = self.background_augment(image) 

          image, mask = self.custom_transform(image, mask)

          # image = torch.from_numpy(image)
          # mask = torch.from_numpy(mask).unsqueeze(0)
        
        if self.img_transform is not None:
            image = self.img_transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
           

        return image, mask   


