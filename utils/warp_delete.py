from utils.sitkLib import saveSITKImage, warpSITK, invertDVF, getSITKImage, getSITKDVF
import glob
import os
import SimpleITK as sitk
import numpy as np
from utils.ioUtils import makeFolderMaybe

path = '/home/luciano/ResearchData/DataSets/DiffuseRTData/ImageDoseTimeEncodedScaleShift/Evaluation/ManySamples/Samples'

DVF_Files = glob.glob(os.path.join(path,'sampleDVF*'))
outputDir = '/home/luciano/ResearchData/DataSets/DiffuseRTData/ImageDoseTimeEncodedScaleShift/Evaluation/ManySamples/Samples/warpedImages'
makeFolderMaybe(outputDir)
for files in DVF_Files:
    print(files)
    fileName = os.path.basename(files)
    patientNumber = fileName.split('_')[3]
    conditionalPath = glob.glob(os.path.join(path, 'cond_0_UCSF_HN_'+patientNumber+'*'))[0]
    # os.path.join(path,'cond_0_'+"_".join(fileName.split('_')[1:-1])+'.mha')
    condImg = sitk.ReadImage(conditionalPath)
    dvfImg = sitk.ReadImage(files)
    new_img = warpSITK(condImg, dvfImg, -1024)
    newImgName = os.path.join(path,os.path.join(path,'sample_'+"_".join(fileName.split('_')[1:])))
    new_img = sitk.Clamp(new_img,lowerBound=-1024,upperBound = 2000)
    array = sitk.GetArrayFromImage(new_img)
    if outputDir!=None:
        newImgName = os.path.join(outputDir,'sample_'+"_".join(fileName.split('_')[1:]))
    saveSITKImage(new_img,newImgName)