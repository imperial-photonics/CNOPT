"""
Dataset class for loading sinograms and reconstructing
undersampled slices for neural network speed test. 

input_folder: path to folder of sinogram tifs

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
    def __init__(self, input_folder, number_angles):
        self.input_folder = input_folder
        self.input_files = [f for f in listdir(input_folder) if f.endswith(".tif")]
        
        # This assumes your fully sampled data-set has 400 equally spaced 
        # projections
        
        N = 400
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
        
        # set up bits for ASTRA reconstruction
        vol_geom = astra.create_vol_geom(702, 702)
        under_geom = astra.create_proj_geom('parallel', 1.0, sino.shape[1], self.under_angles)       
        rec_id = astra.data2d.create('-vol', vol_geom)
        
        # configure reconstruction settings
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['FilterType'] = 'Hann'
        cfg['FilterD'] = 0.4
        cfg['ReconstructionDataId'] = rec_id
        
        undersino_id = astra.data2d.create('-sino', under_geom, sino[self.under_angle_indices,:])
        cfg['ProjectionDataId'] = undersino_id
        alg_id = astra.algorithm.create(cfg)
        
        # do filtered back projection
        astra.algorithm.run(alg_id)
        
        # grab reconstruction
        input_image = astra.data2d.get(rec_id)
        
        # stop memory leak
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(undersino_id)
        
        # normalise
        I_max = np.amax(input_image)
        I_min = np.amin(input_image)
        input_image = (input_image-I_min)/(I_max-I_min)
        
        I_mean = np.mean(input_image)
        
        input_image = input_image - I_mean
        
        # reshape for Pytorch
        input_image = input_image.reshape((1,input_image.shape[0],input_image.shape[0]))
        
        return(torch.from_numpy(input_image).float(), I_max, I_min, I_mean)