import matplotlib.pyplot as plt

from postprocessImageWarping import getPatientID
from utils.ioUtils import listDirAbsolute, listDirRelative
from utils.mhaLib import readMha
import os
import numpy as np
slices = list(range(0,90,15))

fractions = list(range(5,33,6))
folder = '/data1/DiffuseRTData/DVFD/Evaluation/ManySamples/Samples'
plotFolder = '/home/andreas/code/DiffuseRT/plots/Image'

patients = [1,5,11,15,24]
for patient in patients:
    print(patient)
    conditional,_ = readMha([p for p in listDirAbsolute(folder) if 'cond_0_UCSF_HN_{:02d}'.format(patient) in p][0])
    
    # STD = inter sample variations
    fig,ax = plt.subplots(len(slices),len(fractions),figsize=(10,10))
    for i,fraction in enumerate(fractions):
        fractionName = 'sample_UCSF_HN_{:02d}_serie{:02d}_CBCT'.format(patient,fraction)
        samples = [p for p in listDirAbsolute(folder) if fractionName in p]
        arrays = np.stack([readMha(sample)[0] for sample in samples],0)
        std = np.std(arrays,0)
        print(np.sqrt(np.mean(np.var(arrays[:,:,:,10:-10],0))))

        for j,slice in enumerate(slices):  
            ax[j,i].imshow(std[:,:,slice],cmap='rainbow',vmin=0,vmax=1000)
            ax[j,i].axis('off')
            ax[j,i].set_xticklabels([])
            ax[j,i].set_yticklabels([])
            ax[j,i].set_aspect('equal')
    fig.subplots_adjust(wspace=0, hspace=0)     
    plt.savefig(os.path.join(plotFolder,'std_HN_{:02d}'.format(patient,fraction,slice)), bbox_inches='tight')

    # MSE = inter conditional variations
    fig,ax = plt.subplots(len(slices),len(fractions),figsize=(10,10))
    conditional,_ = readMha([p for p in listDirAbsolute(folder) if 'cond_0_UCSF_HN_{:02d}'.format(patient) in p][0])
    conditionalBackgroundValue = np.percentile(conditional,10)
    for i,fraction in enumerate(fractions):
        
        fractionName = 'sample_UCSF_HN_{:02d}_serie{:02d}_CBCT'.format(patient,fraction)
        samples = [p for p in listDirAbsolute(folder) if fractionName in p]
        arrays = np.stack([readMha(sample)[0] for sample in samples],0)
        # arrays += -np.percentile(arrays,10) + conditionalBackgroundValue
        arrays = arrays-np.expand_dims(conditional,0)
        print(np.sqrt(np.mean(arrays[:,:,:,10:-10]**2)))
        std = np.sqrt(np.mean(arrays**2,0))
        for j,slice in enumerate(slices):  
            ax[j,i].imshow(std[:,:,slice],cmap='rainbow',vmin=0,vmax=1000)
            ax[j,i].axis('off')
            ax[j,i].set_xticklabels([])
            ax[j,i].set_yticklabels([])
            ax[j,i].set_aspect('equal')
    fig.subplots_adjust(wspace=0, hspace=0)     
    plt.savefig(os.path.join(plotFolder,'mse_HN_{:02d}'.format(patient,fraction,slice)), bbox_inches='tight')