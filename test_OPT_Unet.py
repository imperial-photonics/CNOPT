"""
Function for testing Unet trained to remove streaks from undersampled OPT reconstructions

net: file name of trained network

test_input: path to folder with sinograms for testing

savename: folder path to save results to

number_angles: number of angles to be used for undersampled sinogram
"""


import torch
from skimage import io
from OPT_dataset import OPT_dataset
from unet_model_original import UNet
import os
import numpy as np

def test_network(net,test_input,savename,number_angles):
    # set up GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load network
    model = UNet(1,1).to(device)
    model.load_state_dict(torch.load(net))
    model.eval()
    
    # set up data set and data loader
    test_dataset = OPT_dataset(input_folder = test_input, crop_size = [-1,-1], augment = False, number_angles = number_angles)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=1, 
                                                  shuffle=False)
    # make save folders
    save_folder = savename
    save_folder1 = save_folder +'/output'
    save_folder2 = save_folder +'/label'
    save_folder3 = save_folder +'/input'
    
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    if not os.path.isdir(save_folder1):
        os.mkdir(save_folder1)
    if not os.path.isdir(save_folder2):
        os.mkdir(save_folder2)
    if not os.path.isdir(save_folder3):
        os.mkdir(save_folder3)
    
    count = 0
    with torch.no_grad():
        # loop over sinograms
        for images, labels, Imax, Imin, Imean in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            out_image = outputs[0,0,:,:].cpu().numpy()
            label_image = labels[0,0,:,:].cpu().numpy()
            in_image = images[0,0,:,:].cpu().numpy()
            
            Imax = np.amax((np.amax(out_image), np.amax(label_image), np.amax(in_image)))
            Imin = np.amin((np.amin(out_image), np.amin(label_image), np.amin(in_image)))
            
            out_image = (65536*(out_image-Imin)/(Imax-Imin)).astype(np.uint16)
            label_image = (65536*(label_image-Imin)/(Imax-Imin)).astype(np.uint16)
            in_image = (65536*(in_image-Imin)/(Imax-Imin)).astype(np.uint16)
    
            io.imsave(os.path.join(save_folder1,str(count)+'.tif'), out_image)
            io.imsave(os.path.join(save_folder2,str(count)+'.tif'), label_image)
            io.imsave(os.path.join(save_folder3,str(count)+'.tif'), in_image)
            
            count = count + 1
            
if __name__ == "__main__":
    
    dirname = os.path.dirname(__file__)
    net = os.path.join(dirname, 'trainedNetworks/Unet_OPT_40_angles')   
    test_input = os.path.join(dirname, 'fishData/sinograms')   
    savename = os.path.join(dirname, 'fishData/40_projections')
    number_angles = 40
    
    test_network(net,test_input,savename,number_angles)