"""
Generate a large batch of image samples from a model and save them as a large
numpy array
"""

import argparse
import os
from typing import Any
import yaml
import numpy as np
import torch as th
import torch.distributed as dist
from datasets.UCSFutils import getSeriesNumber
from utils.config import readConfigAndAddDefaults
from utils.mhaLib import writeMha,getVoxelToWorldMatrix
from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.script_util import (
    create_model_and_diffusion,
    takeModelAndDiffusionArguments
)
from utils.sitkLib import saveSITKImage, warpSITK, invertDVF, getSITKImage, getSITKDVF
import SimpleITK as sitk
from monai.transforms import Compose

class ReNormTensor():
    def __init__(self, cond: th.Tensor) -> None:
        self.cond_mean = cond.mean()
        pass
    def __call__(self, x: th.Tensor) -> Any:
        assert x.shape[0] == 1 #does not work with batchsize different from 1
        return x - x.mean() + self.cond_mean

def invertTransform(data, dataIterator):
    tranform = dataIterator.dataSet.transform.transforms
    scaleTransforms = Compose([T for T in tranform if vars(T).get('ScaleIntesity') and bool(set(data.keys()) & set(T.keys))])
    return scaleTransforms.inverse(data)


class ImageSampler:
    def __init__(self, model, diffusion, data, config, outputFolder, batch_size = 1):
        self.config = config

        self.eval_path = logger.getCurrentEvaluationFolder(self.config['logging_path'],outputFolder)
        logger.configure(self.eval_path)
        self.model = model
        self.diffusion = diffusion

        # always batch size of 1 for now
        self.batchSize = batch_size
        self.config['batch_size'] = self.batchSize
        self.voxelSpacing  = np.array([2.,2.,2.])
        self.origin = np.array([0.,0.,0.])

        # could we maybe set this automatically
        self.saveConditional = self.config['save_conditional']
        self.sampleFractions = self.getSampleFractions(self.config)

        self.dataLoader = self.trimDataLoader(data)

    def generateSamples(self,numberOfSamples,overwrite=False):
        logger.log("sampling...")
        for _, data in enumerate(self.dataLoader):
            if self.saveConditionalCheck(data):
                self.saveConditionalImages(data,overwrite)
            for sampleIndex in range(numberOfSamples):
                self.generateSampleOfInstance(data,sampleIndex,overwrite)
        logger.log("sampling complete")

    def generateSampleOfInstance(self,data,sampleIndex,overWrite=False):
        samplePath = self.getSampleFilePath(data,sampleIndex)
        if not os.path.isfile(samplePath) or overWrite:
            model_kwargs = {
                'cond': th.Tensor(data['cond']).to(dist_util.dev())
            }
            if 'fraction_time_key' in data:
                model_kwargs['fractionTimes'] = th.Tensor(data['fraction_time_key']).float().to(dist_util.dev())
            sample_fn = (
                self.diffusion.p_sample_loop if not self.config['use_ddim'] else self.diffusion.ddim_sample_loop
            )

            sample = sample_fn(
                self.model,
                self.getInputShape(),
                clip_denoised=self.config['clip_denoised'],
                denoised_fn = self.getDenoisedFN(data['cond_0']),
                model_kwargs=model_kwargs,
                progress = True
            )
            
            if self.config['predict_dvf']:
                dvf_mm = sample[:,:3].cpu().numpy() * self.voxelSpacing.reshape(1,3,1,1,1)
                sitkDVF = getSITKImage(dvf_mm, self.origin, self.voxelSpacing)
                saveSITKImage(sitkDVF, self.getSampleDVFFilePath(data,sampleIndex))
                conditionalImage = self.rescale({'cond_0':data['cond_0']})['cond_0']
                conditionalImage = getSITKImage(conditionalImage, self.origin, self.voxelSpacing)
                sampleSITK = warpSITK(conditionalImage, sitkDVF, -1024)
                saveSITKImage(sampleSITK, self.getSampleFilePath(data,sampleIndex))
                return
                # sample = sample[:,3:] ### Warped with Bilinear

            self.postProcessSample(sample,sampleIndex,data)
        
    
    def postProcessSample(self,sample,sampleIndex,data):
        sample = self.rescaleSample(sample[0]).cpu().numpy()
        self.saveSample(sample,sampleIndex,data)

    def saveSample(self,sample,sampleIndex,data):
        writeMha(
            self.getSampleFilePath(data,sampleIndex),
            sample,
            voxelToWorldMatrix= self.getVoxelToWorldMatrix()
        )

    def getDenoisedFN(self, cond):
        if self.config.get('normalize_xstart_while_denoising'):
            return ReNormTensor(cond)
        return None

    def getSampleDVFFilePath(self,data,sampleIndex):
        fileID = self.getFileID(data)
        return os.path.join(self.eval_path,f'sampleDVF_{fileID}_{sampleIndex}.mha')

    def getSampleFilePath(self,data,sampleIndex):
        fileID = self.getFileID(data)
        return os.path.join(self.eval_path,f'sample_{fileID}_{sampleIndex}.mha')
        
    def rescaleSampleToConditional(self,sample,conditional):
        # TODO: make it work with batchsize different from 1
        assert self.batchSize ==1
        return (sample-th.mean(sample))/th.std(sample)*th.std(conditional)+th.mean(conditional)

    def rescaleSample(self,image):
        return self.rescale({'input_0':image})['input_0']
    
    # def rescaleConditionalImage(self,image):
    #     return scaleIntensity(image, -1, 1, -1024, 2000)

    # def rescaleConditionalDose(self,dose):
    #     return scaleIntensity(dose, 0, 1, 0, 72)
    
    def saveConditionalImages(self,data,overwrite=False):
        fileID = self.getConditionalFileID(data)
        keys = [key for key in data.keys() if 'meta' not in key and ('cond_' in key in key)]
        for key in keys:
            imagePath = os.path.join(self.eval_path, f'{key}_{fileID}.mha')
            if overwrite or not os.path.exists(imagePath):
                writeMha(
                    imagePath,
                    self.rescale({key:data[key][0]})[key],
                    voxelToWorldMatrix=self.getVoxelToWorldMatrix()
                )

    def rescale(self,imageDict):
        return invertTransform(imageDict,self.dataLoader)

    def saveConditionalCheck(self,data):
        if isinstance(self.saveConditional,bool):
            return self.saveConditional
        elif self.saveConditional.lower() == 'first':
            conditionalFileID = self.getConditionalFileID(data)
            if not hasattr(self,'currentConditionalFileID') or conditionalFileID != self.currentConditionalFileID:
                self.currentConditionalFileID = conditionalFileID
                return True
            else:
                return False
        else:
            raise NotImplementedError
        
    def getFileID(self,data):
        return (os.path.split(data['input_0_meta_dict']['filename_or_obj'][0])[1]).split('.')[0] #only the file name

    def getConditionalFileID(self,data):
        return (os.path.split(data['cond_0_meta_dict']['filename_or_obj'][0])[1]).split('.')[0] #only the file name

    # def getVoxelToWorldMatrix(self, data):
    #     matrix = data['cond_0_meta_dict']['affine'].numpy()[0]
    #     matrix[0:2,:] *= -1
    #     return matrix #getVoxelToWorldMatrix(self.voxelSpacing,self.origin)

    def getVoxelToWorldMatrix(self): ### Refactor to get correct origin an spacing
        return getVoxelToWorldMatrix(self.voxelSpacing,self.origin)

    def getInputShape(self):
        return tuple([self.batchSize , self.config['image_in_channels']] + self.dataLoader.dataSet.getImageSize())

    def trimDataLoader(self,dataLoader):
        filteredData = []
        for instance in dataLoader.dataSet.data:
            if getSeriesNumber(instance['input_0']) in self.sampleFractions:
                filteredData.append(instance)
        dataLoader.dataSet.data = filteredData
        return dataLoader
        
    def getSampleFractions(self,config):
        if config['sample_fractions'] == 'all':
            return list(range(2,37))
        elif isinstance(config['sample_fractions'],int):
            return np.arange(2,37,config['sample_fractions'])
        else:
            return config['sample_fractions']

class DVFSampler(ImageSampler):    
    def postProcessSample(self,sample,sampleIndex,data):
        dvfSample = self.rescaleSample(sample[0]).cpu().numpy()
        dvfSample = invertDVF(dvfSample, self.origin, self.voxelSpacing)

        conditionalImage = self.rescale({'cond_0':data['cond_0']})['cond_0']
        conditionalImage = getSITKImage(conditionalImage, self.origin, self.voxelSpacing)
        imageSample = warpSITK(conditionalImage, dvfSample, -1024)

        saveSITKImage(imageSample,self.getSampleFilePath(data,sampleIndex))
        saveSITKImage(dvfSample,self.getSampleDVFFilePath(data,sampleIndex))

    def getSampleDVFFilePath(self,data,sampleIndex):
        fileID = self.getFileID(data)
        return os.path.join(self.eval_path,f'sampleDVF_{fileID}_{sampleIndex}.mha')
    

    
if __name__ == "__main__":
    configFile = 'experiments/DVF/configDVFPredicted_XstartTimeEncodedMixedCond.yaml'
    config = readConfigAndAddDefaults(configFile)

    config['timestep_respacing'] = config['timestep_respacing_validation']
    config['batch_size'] = 1
    config['image_size'] = config['sampling_image_size']

    model, diffusion = create_model_and_diffusion(**config)

    config['batch_size'] = 1
    data = load_data(
        config,
        mode = config['validation_mode'],
    )

    model.load_state_dict(
            dist_util.load_state_dict(logger.getModelPath(config['logging_path'],config['model_for_sampling']), map_location="cpu")
        )
    os.environ["GPU_NUMBER"] = dist_util.getGPUID()
    model.to(dist_util.dev())
    model.eval()

    sampler = ImageSampler(model, diffusion, data, config, outputFolder = config['validation_experiment_name']).generateSamples(config['num_samples_per_image'],overwrite=False)
    # sampler = DVFSampler(model, diffusion, data, config, outputFolder = config['validation_experiment_name']).generateSamples(config['num_samples_per_image'],overwrite=False)

    # sampler = DVFSampler(model, diffusion, data, config, outputFolder = 'Model50000').generateSamples(config['num_samples_per_image'],overwrite=False)
