import numpy as np
from tqdm import tqdm, trange 
from patchify import patchify, unpatchify
import pickle

# some camera and physical parameters
dx = 3e-6
dy = dx
dz = 1e-3
dr = 1e-6

def _gauss_kern(l=5, sig=1.):
    l_max = (l-1) / 2.
    
    ax = np.linspace(-l_max, l_max, l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))

    return np.outer(gauss, gauss)


def make_mask_fn(shape, msk, kernel_sz=3, ds_factor = 1, want_img_format = False):
        msk_shape = (3, shape[1]+kernel_sz+1, shape[2]+kernel_sz+1)

        xx, yy, zz, dd = msk

        # check if type xx is a list or an array (This shouldn't be checked again and again, make the dataloader as fast as possible, add this as preprocessing utility)
        if isinstance(xx, list):
             xx = np.array(xx)
             yy = np.array(yy)
             zz = np.array(zz)
             dd = np.array(dd)

        # convert (meter, meter, meter, meter) to (pixel, pixel, mm, um) (This as well)
        # also apply ds_factor
        xx /= (dx*ds_factor)
        yy /= (dy*ds_factor)
        zz /= dz
        dd /= dr

        # convert to xx and yy to int 
        xx = np.int64(np.round(xx))
        yy = np.int64(np.round(yy))

        # Shift xx and yy to origin top-left because 2d arrays start from top-left
        xx += shape[2]//2
        yy += shape[1]//2
        
        # check whether we don't have any coordinates outside the big hologram
        assert (xx >= shape[2]).sum() == 0 or (yy >= shape[1]).sum() or (xx < 0).sum() == 0 or (yy < 0).sum(), print("Given dictionary \
                                                                                                                      has coordinates \
                                                                                                                      which are not in the \
                                                                                                                      big hologram range!")

        dense_msk = np.zeros(tuple(msk_shape), dtype=np.float32)
        gk = _gauss_kern(kernel_sz, 1)

        
        for x, y, z, d in zip(xx, yy, zz, dd):

            # This needs proper fixing. Having to always check the overlay with hologram.  
            x += 1 + kernel_sz//2
            y += 1 + kernel_sz//2
            
            if want_img_format:
                y_max = kernel_sz + y
                x_max = kernel_sz + x

                dense_msk[0, y:y_max, x:x_max] = gk
                dense_msk[1, y:y_max, x:x_max] = z
                dense_msk[2, y:y_max, x:x_max] = d
            else:
                dense_msk[0, y, x] = 1
                dense_msk[1, y, x] = z
                dense_msk[2, y, x] = d
        
        offset = 1 + kernel_sz//2
        return dense_msk[:, offset:(shape[1]+offset), offset:(shape[2]+offset)]
    

def preprocess_pparas(pparas: dict,  holonum_holosize_cropsize_cropstride: tuple, ds_factor: int, want_img_format: bool = False, save_pth: bool = None):
    # store  holonum_holosize_cropsize_cropstride tuple information in the pparas dict when making the hologram crops 
    holonum, holo_size, patch_size, step = holonum_holosize_cropsize_cropstride
    
    if holo_size == patch_size:
        print("Input hologram size is the same as the crop size. For now we assume in such scenarios that the input dictionary\n \
               has coordinates and size in training units (px, px, mm, um)")      
        return pparas
    
    holo_size = holo_size//ds_factor
    patch_size = patch_size//ds_factor
    step = step//ds_factor
    num_crops_per_holo = ((holo_size - patch_size)//step + 1)**2
    print("Number of crops per hologram:", num_crops_per_holo)
    num_patches = num_crops_per_holo*holonum

    # store_masks = np.zeros((num_patches-num_patches+32*num_crops_per_holo, 3, patch_size, patch_size), dtype = np.float16)
    store_masks = np.zeros((num_patches, 3, patch_size, patch_size), dtype = np.float16)
    msk_count = 0

    for i in trange(len(pparas['x'])):
    # for i in trange(len(pparas['x'])-len(pparas['x'])+32):
        mask = make_mask_fn((1, holo_size, holo_size), (pparas['x'][i], pparas['y'][i], pparas['z'][i], pparas['r'][i]), kernel_sz=3, ds_factor=ds_factor, want_img_format=want_img_format)

        
        patched_mask_xy = patchify(mask[0], patch_size=(patch_size), step = step)
        patched_mask_xy = np.reshape(patched_mask_xy, (patched_mask_xy.shape[0]*patched_mask_xy.shape[1], patched_mask_xy.shape[2], patched_mask_xy.shape[3]))


        patched_mask_z = patchify(mask[1], patch_size=(patch_size), step = step)
        patched_mask_z = np.reshape(patched_mask_z, (patched_mask_z.shape[0]*patched_mask_z.shape[1], patched_mask_z.shape[2], patched_mask_z.shape[3]))


        patched_mask_r = patchify(mask[2], patch_size=(patch_size), step = step)
        patched_mask_r = np.reshape(patched_mask_r, (patched_mask_r.shape[0]*patched_mask_r.shape[1], patched_mask_r.shape[2], patched_mask_r.shape[3]))
        
        store_masks[msk_count:msk_count+patched_mask_xy.shape[0],0] = patched_mask_xy
        store_masks[msk_count:msk_count+patched_mask_z.shape[0],1] = patched_mask_z
        store_masks[msk_count:msk_count+patched_mask_r.shape[0],2] = patched_mask_r

        msk_count += patched_mask_xy.shape[0]

    if want_img_format:
        return store_masks
    
    binned_pparas = {}
    x = []
    y = []
    z = []
    r = []
    for i, mask in tqdm(enumerate(store_masks), total = store_masks.shape[0]):

        local_peak_mask = (mask[0] == 1)
        yy, xx = np.where(local_peak_mask == True)
        zz = mask[1][local_peak_mask]
        rr = mask[2][local_peak_mask]
        x.append(xx)
        y.append(yy)
        z.append(zz)
        r.append(rr)

    binned_pparas['x'] = x
    binned_pparas['y'] = y
    binned_pparas['z'] = z
    binned_pparas['r'] = r

    # save if first time 
    if save_pth is not None:
        with open(save_pth+"_xypx_binned.pkl", mode = "wb") as f:
            pickle.dump(binned_pparas, f)

    return binned_pparas


    

