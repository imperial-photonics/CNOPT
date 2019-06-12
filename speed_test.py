"""
Function for testing speed of Unet trained to remove streaks from undersampled 
OPT reconstructions

net: file name of trained network

test_input: path to folder with sinograms for testing

savename: folder path to save results to

number_angles: number of angles to be used for undersampled sinogram
"""

import torch
from OPT_dataset_no_labels import OPT_dataset
from unet_model_original import UNet
import os
import time
import numpy as np
from skimage import io
def speed_test(net,test_input,savename,number_angles):
    
    # start clock for total reconstruction time
    full_start = time.time()
    
    # check if GPU available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # prepare network
    model = UNet(1,1).to(device)
    model.load_state_dict(torch.load(net))
    model.eval()
    
    # prepare data loading
    batch = 5
    test_dataset = OPT_dataset(input_folder = test_input, number_angles = number_angles)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch, 
                                                  shuffle=False)
    # prepare savefolder
    save_folder = savename
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    
    # prepare volume
    number_slices = 2118
    slice_size = 702
    volume = np.zeros((number_slices,1,slice_size,slice_size))
    
    chunks = np.arange(0,number_slices,batch)
    chunks = np.append(chunks,number_slices)
    
    # prepare speed measurement
    speed = np.zeros(chunks.size)
    
    count = 0
    with torch.no_grad():
        
        # make sure GPU is being honest about what it has done
        torch.cuda.synchronize()
        
        # time how long it takes to get streak corrupted slices
        load_data_time = time.time()
        for images, Imax, Imin, Imean in test_loader:
            
            torch.cuda.synchronize()
            print('loading data time')
            print(time.time()-load_data_time)
            
            # time how long it takes to put streak corrupted slices on GPU
            to_gpu_start = time.time()
            images = images.to(device)
            torch.cuda.synchronize()
            print('gpu mount time')
            print(time.time()-to_gpu_start)
            
            # time how long it takes to apply network to slices           
            start = time.time()
            outputs = model(images)
            torch.cuda.synchronize()
            speed[count] = time.time()-start
            print('model time')
            print(time.time()-start)
            
            # time how long it takes to prep re-normalisation  
            prepare_normalisation_time = time.time()
            Imaxs, c, x, y  = np.meshgrid(Imax,1,np.arange(slice_size),np.arange(slice_size))
            Imins, c, x, y  = np.meshgrid(Imin,1,np.arange(slice_size),np.arange(slice_size))
            Imeans, c, x, y  = np.meshgrid(Imean,1,np.arange(slice_size),np.arange(slice_size))
            print('NormPrep')
            print(time.time()-prepare_normalisation_time)
            
            # time how long it takes gather results from GPU
            move_data_time = time.time()
            numpy_outputs = np.multiply(outputs.cpu().numpy()+Imeans,Imaxs-Imins)+Imins
            #numpyOutputs = outputs.cpu().numpy()
            torch.cuda.synchronize()
            print('move_data_time')
            print(time.time()-move_data_time)
            
            put_in_volume = time.time()
            volume[chunks[count]:chunks[count+1],0,:,:] = numpy_outputs[0:(chunks[count+1]-chunks[count]),0,:,:]
            print('intoVolumeTime')
            print(time.time()-put_in_volume)
            count = count + 1
            load_data_time = time.time()
    
    # save volume
    volume = (65536*(volume-np.amin(volume))/(np.amax(volume)-np.amin(volume))).astype(np.uint16)    
    for count in range(number_slices):
        io.imsave(os.path.join(save_folder,str(count)+'.tif'), np.squeeze(volume[count,0,:,:]))
    
    print('reconstruction time')
    print(np.mean(speed))
    print('standard error')
    print(np.std(speed)/np.sqrt(speed.size))  
    print('Total time')  
    print(time.time()-full_start)
    
    return (np.mean(speed), np.std(speed), time.time()-full_start)

if __name__ == "__main__":
    
    dirname = os.path.dirname(__file__)
    net = os.path.join(dirname, 'trainedNetworks/UnetSinoFinal64')   
    test_input = os.path.join(dirname, 'fishData/sinograms')   
    savename = os.path.join(dirname, 'fishData/speedtest')
    number_angles = 64
    
    speed_test(net,test_input,savename,number_angles)