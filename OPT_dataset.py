"""
Dataset class for loading sinograms and transforming them to well-sampled and 
undersampled slices for neural network training. 

input_folder: path to folder of sinogram tifs

crop_size: [Lx Ly] size of random crop to apply to slices - if you want the 
whole slice, use a -ve value

augment: If True, sinograms will permuted to start at a different angle, and 
randomly flipped

number_angles: How many angles you want to use in your sub-sampling
"""

import torch
from torch.utils.data.dataset import Dataset
from os import listdir
from skimage import io
import os
import numpy as np
import astra

class OPT_dataset(Dataset):
    def __init__(self, input_folder, crop_size, augment, number_angles):
        self.input_folder = input_folder
        self.input_files = [f for f in listdir(input_folder) if f.endswith(".tif")]
        self.crop_size = crop_size
        self.augment = augment
        
        # This assumes your fully sampled data-set has 400 equally spaced 
        # projections
        N = 400
        
        # Create arrays of angles for well-sampled and under-sampled
        self.angles = np.linspace(0,2*np.pi,N,False)
        self.under_angle_indices = np.round(np.linspace(0,400,number_angles,False)).astype(int)
        self.under_angles = self.angles[self.under_angle_indices]
        
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        
        # Get file path to tif and load sinogram
        input_name = os.path.join(self.input_folder,self.input_files[idx])
        sino = np.transpose(io.imread(input_name)).astype(np.float64)
        
        # make sure it has 400 projections
        if sino.shape[0] == 800:
            sino = sino[0:800:2,:]

        # apply random starting projection and random flips
        if self.augment:
            start_position = np.random.randint(sino.shape[0])
            sino = np.concatenate((sino[start_position:,:],sino[:start_position,:]))
            if np.random.randint(2):
                sino = np.flipud(sino)
            if np.random.randint(2):
                sino = np.fliplr(sino)
        
        
        # set up bits for ASTRA reconstruction
        vol_geom = astra.create_vol_geom(752, 752)
        rec_id = astra.data2d.create('-vol', vol_geom)
        
        proj_geom = astra.create_proj_geom('parallel', 1.0, sino.shape[1], self.angles)
        sinogram_id = astra.data2d.create('-sino', proj_geom, sino)
        
        under_geom = astra.create_proj_geom('parallel', 1.0, sino.shape[1], self.under_angles)
        undersino_id = astra.data2d.create('-sino', under_geom, sino[self.under_angle_indices,:])
        
        # configure reconstruction settings
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['FilterType'] = 'Hann'
        cfg['FilterD'] = 0.4
        cfg['ReconstructionDataId'] = rec_id
        
        # configure for full sinogram
        cfg['ProjectionDataId'] = sinogram_id
        
        # reconstruct full sinogram
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        
        # grab reconstruction
        label_image = astra.data2d.get(rec_id)
        
        # stop GPU memory leak
        astra.algorithm.delete(alg_id)

        # configure for undersampled sinogram and reconstruct
        cfg['ProjectionDataId'] = undersino_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        input_image = astra.data2d.get(rec_id)
        
        # stop GPU memory leak
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(sinogram_id)
        astra.data2d.delete(undersino_id)
        
        # zero mean unit variance input
        input_max = np.amax(input_image)
        input_min = np.amin(input_image)
        input_image = (input_image-input_min)/(input_max-input_min)
        label_image = (label_image-input_min)/(input_max-input_min)
        
        input_mean = np.mean(input_image)
        
        input_image = input_image - input_mean
        label_image = label_image - input_mean
        
        # apply random crop to slice
        h, w = input_image.shape[:2]
        if self.crop_size[0] > 0:
            new_h,new_w = self.crop_size                
            top = np.random.randint(0,h-new_h)
            left = np.random.randint(0,w-new_w)
            
            input_image = input_image[top:top+new_h, left: left+new_w]
            label_image = label_image[top:top+new_h, left: left+new_w]
        
        # reshape images to Pytorch style [C X Y]
        input_image = input_image.reshape((1,input_image.shape[0],input_image.shape[0]))
        label_image = label_image.reshape((1,label_image.shape[0],label_image.shape[0]))
        
        return(torch.from_numpy(input_image).float(), torch.from_numpy(label_image).float(), input_max, input_min, input_mean)