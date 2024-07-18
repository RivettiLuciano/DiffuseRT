import pandas as pd
import os
import glob
from improved_diffusion.image_datasets import * 
from monai.transforms.transform import apply_transform

import SimpleITK as sitk
import torch 
import argparse
import yaml
from monai.transforms.inverse_batch_transform import BatchInverseTransform
from monai.transforms.utils import allow_missing_keys_mode

def CreateDir(dir):
  if not os.path.isdir(dir):
    os.makedirs(dir)

def SaveImage(itk_img, path, type = None):
  if type:
    itk_img = sitk.Cast(itk_img, type)
  CreateDir(os.path.dirname(path))
  ifw = sitk.ImageFileWriter()
  ifw.SetFileName(path)
  ifw.SetUseCompression(True)
  ifw.Execute(itk_img)

def GetSitkImage(data_batch, torch_img_arrray, transforms, data_loader, key = 'CT', binary = False):
    batch_inverter = BatchInverseTransform(transforms, data_loader)

    data_dict = {key: torch_img_arrray.unsqueeze(0), 
    key+"_transforms": data_batch.get(key+"_transforms"),
    key+'_meta_dict': data_batch.get(key+"_meta_dict")}

    if data_batch.get(key+"_transforms"):
        with allow_missing_keys_mode(transforms):
            data_dict = batch_inverter(data_dict)[0]
    else:
        data_dict[key+'_meta_dict'] = data_batch.get("CBCT_meta_dict") ## For inference the strucutres doesnt have transformations

    return CreateImage(data_dict, key, binary)

def CreateImage(dic,key='CT',binary = False):
    img_array = dic[key].squeeze()
    sitk_img = sitk.GetImageFromArray(np.transpose(img_array))
    return SetupImage(sitk_img,dic,key)

def SetupImage(img,dic,key):
    affine = dic[key+'_meta_dict']['affine'].squeeze()
    spacing = np.diagonal(affine)[:-1]
    img.SetSpacing(np.abs(spacing))
    origin = affine[:-1,-1]
    direction = np.sign(affine[:-1,:-1]).ravel()
    img.SetOrigin(origin.numpy())
    img.SetDirection(direction.numpy())
    return img

def ScaleIntensity(img, a_min, a_max, b_min, b_max):
    img = (img - a_min) / (a_max - a_min)
    img = img * (b_max - b_min) + b_min
    # if clip:
    #     img = clip(img, b_min, b_max)
    return img


args = argparse.ArgumentParser()
args.add_argument('--configFile', type=str, help='Path to the config file')
fake_args = ['--configFile', 'experiments/configDVF.yaml']
# fake_args = ['--configFile', 'experiments/config.yaml']
configFile = args.parse_args(fake_args)
output = '/home/luciano'

with open(configFile.configFile) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

data = load_data(
    config,
    mode = 'training',
)

iterations = 1
for i in range(iterations):
  sample = next(data)
  dvf = sample['input'][0,...]
  cond = sample['cond'][0,...]
  dvfName = os.path.basename(sample['input_meta_dict']['filename_or_obj'][0])
  condName = os.path.basename(sample['cond_meta_dict']['filename_or_obj'][0])

  dvf = ScaleIntensity(dvf, a_min=-1, a_max=1, b_min=-15, b_max=15)
  cond = ScaleIntensity(cond, a_min=-1, a_max=1, b_min=-1024, b_max=2000)
  dvf_sitk = sitk.GetImageFromArray(np.transpose(dvf))
  cond_sitk = sitk.GetImageFromArray(np.transpose(cond))
  SaveImage(dvf_sitk, os.path.join(output,dvfName))
  SaveImage(cond_sitk, os.path.join(output,condName))

