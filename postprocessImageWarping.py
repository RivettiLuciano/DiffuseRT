import tempfile
import numpy as np
import regex
from datasets.UCSFutils import getCBCTPath
from preprocessing.plastimatchRegistration import runPlastimatchRegistration
from utils.interpolate import resampleScan, warpScanWithVectorField
from utils.ioUtils import listDirAbsolute, listDirRelative
from utils.mhaLib import *
import os
import pandas as pd
import SimpleITK as sitk
from utils.niftyLib import readNifti
from utils.sitkLib import translationRegistration
from utils.transformations import toNumpy, toTorch

class ImageWarpingPostprocesser:
    def __init__(self,plastimatchConfig,mappingFile,device,setImageOrigin=True):
        self.plastimatchConfig = plastimatchConfig
        self.setImageOrigin = setImageOrigin
        self.conditionalMapping = pd.read_csv(mappingFile,index_col='PatientID')
        self.device  = device

    def postProcessFolder(self,folder):
        samplePaths = self.getSamplePaths(folder)
        conditionalDict = self.getConditionalDict(folder)
        for samplePath in samplePaths:
            conditionalPath = conditionalDict[getPatientID(samplePath)]
            self.postProcessSample(conditionalPath,samplePath)

    def getSamplePaths(self,folder):
        return [os.path.join(folder,p) for p in listDirRelative(folder) if 'sample_UCSF' in p]
    
    def getConditionalDict(self,folder):
        return {getPatientID(p):os.path.join(folder,p) for p in listDirRelative(folder) if 'cond_0_UCSF' in p}

    def postProcessSample(self,conditionalImagePath,sampledImagePath):
        fullResolutionConditionalImagePath = self.getHighResolutionImagePath(sampledImagePath)
        targetDVFPath = os.path.join(tempfile.mkdtemp(),'DVF.mha')
        targetDVFPath = sampledImagePath.replace('sample_UCSF_HN','sampleDVF_UCSF_HN')
        if self.setImageOrigin:
            imageOrigin = self.getImageOrigin(conditionalImagePath,fullResolutionConditionalImagePath)
            self.resaveWithOrigin(conditionalImagePath,imageOrigin)
            self.resaveWithOrigin(sampledImagePath,imageOrigin)

        runPlastimatchRegistration(
            self.plastimatchConfig,
            sampledImagePath,
            conditionalImagePath,
            targetDVFPath,
            None
        )
        ### To get high resolution image
        # planningImage,planningVTW = readNifti(fullResolutionConditionalImagePath)
        # planningImage = toTorch(planningImage).to(self.device).unsqueeze(0)
        # dvf,vtwMatrix = readMha(targetDVFPath)
        # dvf = toTorch(dvf).to(self.device)
        # highResDVFVtwMatrix = getVoxelToWorldMatrix(getVoxelSpacing(planningVTW),getOrigin(vtwMatrix))
        # targetShape = toTorch(np.floor(np.array(dvf.shape[1:])*getVoxelSpacing(vtwMatrix)/getVoxelSpacing(highResDVFVtwMatrix))).to(self.device).int()
        # highResDVF,_ = resampleScan(dvf,vtwMatrix,highResDVFVtwMatrix,targetShape)
        # highResImage,highResVtwMatrix = warpScanWithVectorField(planningImage,planningVTW,highResDVF,highResDVFVtwMatrix,defaultPixelValue=-1024.)
        # writeMha(self.getTargetImagePath(sampledImagePath),toNumpy(highResImage),highResVtwMatrix)

    def getTargetImagePath(self,samplePath):
        return samplePath.replace(
            '_CBCT_','_warpedCBCT_'
        )


    def getHighResolutionImagePath(self,sampleImagePath):
        pid,sid = getPatientID(sampleImagePath),getSerieID(sampleImagePath)
        conditionalSerieID = self.getConditionalSerieID(pid,sid)
        return getCBCTPath(
            getPatientID(sampleImagePath),
            conditionalSerieID,
            '/home/luciano/ResearchData/DataSets/DL-DIR/DataSet/Original'
        )
    

    def getConditionalSerieID(self,pid,sid):
        return int(self.conditionalMapping.loc[pid,'Serie{:02d}'.format(sid)][-2:])

    def getImageOrigin(self,conditionalImagePath,fullResolutionConditionalImagePath):
        fixed_image = sitk.ReadImage(fullResolutionConditionalImagePath,sitk.sitkFloat32)
        moving_image = sitk.ReadImage(conditionalImagePath,sitk.sitkFloat32)
        
        approximateOrigin = \
                np.array(fixed_image.GetOrigin()) + \
                (np.array(fixed_image.GetSize())*np.array(fixed_image.GetSpacing()))/2 - \
                (np.array(moving_image.GetSize())*np.array(moving_image.GetSpacing()))/2        
        moving_image.SetOrigin(approximateOrigin)
        offset = translationRegistration(fixed_image,moving_image)
        return approximateOrigin - offset


    def resaveWithOrigin(self,imagePath,origin):
        image,vtwMatrix = readMha(imagePath)           
        newVtwMatrix =  getVoxelToWorldMatrix(getVoxelSpacing(vtwMatrix),origin)
        writeMha(imagePath,image,newVtwMatrix)

        

def getPatientID(filePath):
    return int(regex.findall(r'UCSF_HN_[0-9][0-9]_serie',filePath)[0][8:-6])

def getSerieID(filePath):
    return int(regex.findall(r'_serie[0-9][0-9]_',filePath)[0][6:-1])


if __name__=="__main__":
    postProcessor = ImageWarpingPostprocesser(
        'bSplinePlastimatch.cfg',
        mappingFile="/home/luciano/ResearchData/DataSets/DL-DIR/DataSet/DeepMovement/Mapping_CBCT.csv",
        device = 'cuda:1',
        setImageOrigin=False
    )
    sampleFolder = '/home/luciano/ResearchData/DataSets/DiffuseRTData/ImageDoseTimeEncodedMixedConditionals4Days/Evaluation/ManySamples100'
    postProcessor.postProcessFolder(sampleFolder)

