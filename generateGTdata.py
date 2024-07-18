import argparse


from improved_diffusion.image_datasets import load_data
import os 
from utils.config import readConfigAndAddDefaults
from image_sample import invertTransform
from utils.sitkLib import invertDVF, saveSITKImage
from utils.imageProcessing import getBodyMask
import numpy as np

def postProcessImage(sample,key,dataloader, voxelSpacing, origin, outputPath):
    origin = [0]*3
    dvfSample = invertTransform({key:sample},dataloader)[key]
    dvfSample = invertDVF(dvfSample, origin, voxelSpacing)
    saveSITKImage(dvfSample,outputPath)

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--configFile', type=str, help='Path to the config file')
    fake_args = ['--configFile', 'experiments/DVF/configDVFDose_XstartTimeEncoded.yaml']
    if args.parse_args().configFile != None:
        parse = args.parse_args()
    else:
        parse = args.parse_args(fake_args)

    # configFile = 'experiments/DVF/configDVFPredicted_XstartTimeEncoded.yaml'
    config = readConfigAndAddDefaults(parse.configFile)

    config['timestep_respacing'] = config['timestep_respacing_validation']
    config['batch_size'] = 1
    config['image_size'] = config['sampling_image_size']

    data = load_data(
        config,
        mode = config['validation_mode'],
    )

    key = 'input_0'
    prefix = 'sampleDVF_'
    outputDir = '/home/luciano/ResearchData/DataSets/DiffuseRTData/input_val'
    for d in data:
        metadata = d[key+'_meta_dict']
        fileName = os.path.basename(metadata['filename_or_obj'][0])
        fileName = fileName.replace('_'+fileName.split('_')[3],'')
        outputdir = os.path.join(outputDir, prefix+fileName)
        cond = invertTransform({'cond_0':d['cond_0']},data)['cond_0']
        # mask = getBodyMask(cond)
        postProcessImage(d[key], key, data, metadata['spacing'].numpy()[0], metadata['original_affine'].numpy()[0][:-1,-1],outputdir)
        print(outputdir)
        break


if __name__ == "__main__":
    main()
