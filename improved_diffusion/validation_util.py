import shutil
import tempfile

import yaml
from datasets.UCSFutils import getPath, getPIDfromName
from utils.ioUtils import makeFolderMaybe, readJson, savePickle, loadPickle
from utils.mhaLib import readMha
from utils.niftyLib import readNifti
# from utils.plotLossCurve import plotLossCurve
import os
from PIL import Image
import numpy as np
from image_sample import ImageSampler, DVFSampler
from improved_diffusion.fid_score import calculate_fid_given_paths
from improved_diffusion import dist_util, logger
import torch
import torch.distributed as dist
import pandas as pd
import json
import pickle 
from utils.sitkLib import getDVFJacobian, getSITKImage, warpSITK, saveSITKImage
from utils.imageProcessing import getBodyMask
import SimpleITK as sitk
import glob
from utils.patientStructure import PatientSequence
from monai.transforms import CenterSpatialCrop, SpatialPad
import copy
from scipy.special import rel_entr
import scipy.stats as st
from scipy.stats import norm

class FIDPreprocessor:
    def __init__(self,folder,targetFolder,instances, modality):
        self.sourceFolder = folder
        self.targetFolder = targetFolder
        makeFolderMaybe(targetFolder)
        self.instances = instances
        self.modality = modality
        
    def preprocessPerPatient(self):
        for patient in self.instances:
            self.preprocessPatient(patient,self.getTargetFolder(patient))
        
    def preprocessAll(self):
        for patient in self.instances:
            self.preprocessPatient(patient,self.getTargetFolder(None))
    
    def preprocessPatient(self,patient,saveFolder):
        for series in range(1,37):
            pixelArray = self.get3DImage(patient,series)
            if pixelArray is not None:
                for i in range(pixelArray.shape[-1]):
                    im = Image.fromarray(pixelArray[:,:,i]).convert('RGB')
                    im.save(os.path.join(saveFolder,"{}_serie{:02d}_{}.png".format(patient,series,i)))

    def get3DImage(self,patient,series):
        path = getPath(getPIDfromName(patient),series,self.sourceFolder, self.modality)
        if os.path.exists(path):
            pixelArray = readNifti(path)[0]
            return self.normalizePixelArray(pixelArray)
        else:
            return None
        
    def getTargetFolder(self,patient=None):
        if patient is None:
            return self.targetFolder
        else:
            path= os.path.join(self.targetFolder,patient)
            makeFolderMaybe(path)
            return path

    def normalizePixelArray(self,pixelArray):
        pixelArray = pixelArray[17:-17,17:-17,:]
        return np.clip(np.round((pixelArray+1024)/3024*255).astype(np.int32),0,255)

class FIDSamplePreprocessor(FIDPreprocessor):
    def preprocessPatient(self,patient,saveFolder):
        cond = [f for f in os.listdir(self.sourceFolder) if f'cond_0_UCSF_{patient}' in f]
        if len(cond) > 0:
            print(cond[0])
            conditional = readMha(os.path.join(self.sourceFolder,cond[0]))[0]
            for sampleFile in self.getSamples(patient):
                pixelArray = self.get3DImage(patient,sampleFile)
                pixelArray = (pixelArray-np.mean(pixelArray))/np.std(pixelArray)*np.std(conditional)+np.mean(conditional)
                pixelArray = self.normalizePixelArray(pixelArray)
                if pixelArray is not None:
                    for i in range(pixelArray.shape[-1]):
                        im = Image.fromarray(pixelArray[:,:,i]).convert('RGB')
                        im.save(os.path.join(saveFolder,"{}_{}.png".format(sampleFile[:-4],i)))

    def getSamples(self,patient):
        return [f for f in os.listdir(self.sourceFolder) if f'sample_UCSF_{patient}' in f]

    def get3DImage(self, patient, sampleFile):
        return readMha(os.path.join(self.sourceFolder,sampleFile))[0]

    def normalizePixelArray(self, pixelArray):
        return np.clip(np.round((pixelArray+1024)/3024*255).astype(np.int32),0,255)


class RunValidation:
    def __init__(
        self,
        model,
        diffusion,
        data,
        batch_size,
        config,
        valMetricDict,
        overWrite = False,
    ):
        self.model = model
        self.diffusion = diffusion
        self.config = config
        self.setDefaultFolders()
        self.overWrite = overWrite
        # self.saveCurrentConfig()
        self.patientSplit = self.getPatientSplit()
        self.data = data
        self.batch_size = batch_size
        self.valMetricDict = valMetricDict
        self.segmentations = self.getSegmentationNames()
        
        logger.configure(self.validationFolder)

    def calculateScores(self):
        outputFile = os.path.join(self.validationFolder,'metrics')
        if os.path.isfile(outputFile+'.pkl') and False:
            self.valMetrics = loadPickle(outputFile+'.pkl')
        else:
            self.valMetrics = {}

        if 'FID' in self.valMetricDict["modelMetrics"]:
            self.generateSamples(self.overWrite)
            self.valMetrics['FID'] = self.calculateFID()
        if 'vlb' in self.valMetricDict["modelMetrics"]:
            metrics, dataFrame = self.calculateVLB()
            self.valMetrics.update(metrics)
            self.saveDataFrameDict(dataFrame)
        if len(self.valMetricDict["fractionMetrics"]) > 0:
            self.generatePatientSequenceData(overWrite=self.overWrite, segmentationList = self.segmentations)
            self.valMetrics.update(self.calculateSampleSpecificMetrics())

        savePickle(self.valMetrics, outputFile+'.pkl')
        pd.DataFrame.from_dict([self.valMetrics]).to_csv(outputFile+'.csv',index = False)
        return self.valMetrics

    def getPatientSplit(self):
        return readJson(self.config['data_spliting'])

    def saveDataFrameDict(self, df_dict: dict):
        for key, df in df_dict.items():
            df.to_csv(os.path.join(self.validationFolder,key+'.csv'))

    def generateModelSamples(self, overwrite=False):
        if self.config['data_set'] == 'ImageDataset':
            sampler = ImageSampler( 
            self.model,
            self.diffusion,
            self.data,
            self.config,
            outputFolder = self.sampleFolder)
        elif self.config['data_set'] == 'DVFDataset':
            sampler = DVFSampler( 
            self.model,
            self.diffusion,
            self.data,
            self.config,
            outputFolder = self.sampleFolder)  

        sampler.generateSamples(self.config['num_samples_per_image'],overwrite)
        return sampler

    def generateSamples(self,overwrite=False):

        sampler = self.generateModelSamples(overwrite)
        preprocessor = FIDSamplePreprocessor(
            sampler.eval_path,
            self.sampleFIDFolder,
            self.getPatientList(self.config['FID_sample_patients']),
            modality = self.config['modality']
        )
        preprocessor.preprocessPerPatient()

    def getPatientList(self, patientKey):
        if not isinstance(patientKey,list):
            if patientKey in self.patientSplit:
                return self.patientSplit[patientKey]
            else:
                KeyError('The dataset avaliable are training, validation or test')
        else:
            return patientKey

    def calculateFID(self):
        referenceSliceFolder,sampleSliceFolder = self.setUpFIDFolders()
        fidScore =  calculate_fid_given_paths(
                                    paths = [referenceSliceFolder, sampleSliceFolder],
                                    batch_size = self.batch_size,
                                    device = dist_util.dev(),
                                    dims = 2048,
                                    num_workers = 1
                                  )
        shutil.rmtree(referenceSliceFolder)
        shutil.rmtree(sampleSliceFolder)
        return fidScore
    
    def setUpFIDFolders(self):
        referenceSliceFolder = tempfile.mkdtemp()
        sampleSliceFolder = tempfile.mkdtemp()
        for patient in self.getPatientList(self.config['FID_reference_patients']):
            for path in os.listdir(os.path.join(self.referenceFIDFolder,patient)):
                os.symlink(os.path.join(self.referenceFIDFolder,patient,path), os.path.join(referenceSliceFolder,path))

        for patient in self.getPatientList(self.config['FID_sample_patients']):
            for path in os.listdir(os.path.join(self.sampleFIDFolder,patient)):
                os.symlink(os.path.join(self.sampleFIDFolder,patient,path), os.path.join(sampleSliceFolder,path))
        return referenceSliceFolder,sampleSliceFolder

    def calculateVLB(self):
        metrics = ["vb", "mse", "xstart_mse","total_bpd"]
        df_dict = {key : pd.DataFrame(columns=np.arange(int(self.config['timestep_respacing']))) for key in metrics}
        df_dict['total_bpd'] = pd.DataFrame([],columns=[0])
        num_complete = 0
        for data in self.data:
            cond = {'cond' : torch.Tensor(data['cond'])}
            batch = data['input']
            batch = batch.to(dist_util.dev())
            model_kwargs = {k: v.to(dist_util.dev()) for k, v in cond.items()}
            if 'fraction_time_key' in data:
                model_kwargs['fractionTimes'] = torch.Tensor(data['fraction_time_key']).float().to(dist_util.dev())
            minibatch_metrics = self.diffusion.calc_bpd_loop(
                self.model, batch, clip_denoised = self.config['clip_denoised'], model_kwargs=model_kwargs
            )
            patientPath_batch = data['input_0_meta_dict']['filename_or_obj']
            for key in metrics:
                terms = minibatch_metrics[key] / dist.get_world_size()
                dist.all_reduce(terms)
                for batch_number in range(len(patientPath_batch)):
                    patientID_batch = "_".join(os.path.basename(patientPath_batch[batch_number]).split('_')[1:4])
                    df_dict[key].loc[patientID_batch] = terms[batch_number].detach().cpu().numpy()

            num_complete += dist.get_world_size() * batch.shape[0]
            logger.log(f"done {num_complete} samples: bpd={np.mean(df_dict['total_bpd'].values)}")
        logger.log("evaluation complete")
        return {key: np.mean(df_dict[key].values) for key in metrics}, df_dict

    def setDefaultFolders(self):
        self.validationFolder = logger.getCurrentEvaluationFolder(self.config['logging_path'],self.config['validation_experiment_name'])
        self.sampleFolder = os.path.join(self.validationFolder,'Samples')
        makeFolderMaybe(self.sampleFolder)
        self.sampleFIDFolder = os.path.join(self.validationFolder,'FIDData')
        makeFolderMaybe(self.sampleFIDFolder)
        self.referenceFIDFolder = self.config['FID_reference_processed_path']

    def saveCurrentConfig(self):
        with open(os.path.join(self.validationFolder,'config.json'),'w') as file:
            json.dump(self.config, file, indent=1)

    def generatePatientSequenceData(self, overWrite = False, segmentationList = []):
        outputPath = os.path.join(os.path.dirname(self.sampleFolder),'PatientSequenceMetrics.pkl')
        if not os.path.isfile(outputPath) or overWrite: 
            self.generateModelSamples(overWrite)
            self.patientSequence = getPatientSequence(self.sampleFolder)
            propagateStructures(self.patientSequence, segmentationList, self.config['data_dir'], self.config['sampling_image_size'], overWrite) 
            calculateMetrics(self.patientSequence, self.valMetricDict)
            self.patientSequence.save(os.path.join(os.path.dirname(self.sampleFolder),'PatientSequenceMetrics.pkl'))
        else:
            self.patientSequence = PatientSequence()
            self.patientSequence.load(outputPath)  

    def calculateSampleSpecificMetrics(self):
        sampleMetrics = {}
        realizationPatientSequencePath = os.path.join(os.path.dirname(self.config['logging_path']),'input_val','PatientSequenceMetrics.pkl')
        if not os.path.isfile(realizationPatientSequencePath):
            getMetricPatientSequenceForValGTData(dataDir = self.config['data_dir'],
                                     path = os.path.join(os.path.dirname(self.config['logging_path']),'input_val'), 
                                     segmentationList = self.segmentations,
                                     imgSize= self.config['image_size'],
                                     valMetricDict=self.valMetricDict,
                                     overWrite = False)
        self.realizationPatientSequence = PatientSequence()
        self.realizationPatientSequence.load(realizationPatientSequencePath) 
        if "KL_Divergence" in self.valMetricDict['modelMetrics']:
            sampleMetrics.update(calculateKLDivergence(self.patientSequence,self.realizationPatientSequence))
        if "inDistribution" in self.valMetricDict['modelMetrics']:
            sampleMetrics.update(calculateInDistribution(self.patientSequence,self.realizationPatientSequence, 10))
        return sampleMetrics

    def getSegmentationNames(self):
        segmentationNames = []
        if not self.valMetricDict.get('fractionMetrics') or not self.valMetricDict.get('fractionMetrics').get('segmentationMetrics'):
            return []
        for values in self.valMetricDict['fractionMetrics']['segmentationMetrics'].values():
            segmentationNames.extend(values)
        return list(set(segmentationNames))


def calculateKLDivergence(sampledSequence, realizationSequence, binNumber = 30):
    KLMetrics = {}
    e = 1e-6 ### To avoid divergence of KL Divergence
    for metricName, metricPatient in sampledSequence.metrics.items():
        realizationDist = []
        sampleDist = []
        for patientID, metricValues in metricPatient.items():
            realizationDist.extend(realizationSequence.metrics[metricName][patientID])
            sampleDist.extend(metricValues)
        Min = min(np.min(realizationDist), np.min(sampleDist))
        Max = max(np.max(realizationDist), np.max(sampleDist))
        realizationDist, _ = np.histogram((realizationDist - Min)/(Max-Min), bins=binNumber, range=(0,1), density = True)
        sampleDist, _ = np.histogram((sampleDist - Min)/(Max-Min), bins=binNumber, range=(0,1), density = True)
        KLMetrics['KL_'+metricName] = sum(rel_entr(realizationDist + e, sampleDist + e))
    return KLMetrics

def calculateInDistribution(sampledSequence, realizationSequence, frequency, confidence = 0.95):
    inDistributionMetrics = {}
    maxNumberOfFractions = 40
    metricNames = list(sampledSequence.metrics.keys())
    fractions = ['serie'+str(n).zfill(2) for n in range(2,maxNumberOfFractions)]
    sampleFrequency = sampledSequence.getSampleFractionFrequency()
    fractions.sort()
    for metricName in metricNames:
        inDistributionList = []
        for fractionID in fractions:
            fractionNumber = int(fractionID[5:])
            for patient in sampledSequence:
                if not sampledSequence[patient.patientID].fractions.get(fractionID) or not realizationSequence[patient.patientID].fractions.get(fractionID) or metricName not in sampledSequence[patient.patientID].metrics.keys():
                    continue
                fractionsInSampled = sampledSequence[patient.patientID].keys
                values = sampledSequence[patient.patientID][fractionID].metrics[metricName]
                # minInt, maxInt = st.t.interval(alpha=confidence, df=len(values)-1, loc=np.mean(values), scale=st.sem(values))
                index =  fractionsInSampled.index(fractionID)
                minInt0, maxInt0 = calculate_normal_interval(values,confidence)
                if index == len(fractionsInSampled)-1:
                    minInt1 = minInt0
                    maxInt1 = maxInt0
                else:
                    minInt1, maxInt1 = calculate_normal_interval(sampledSequence[patient.patientID][fractionsInSampled[index+1]].metrics[metricName],confidence)

                realizationSequenceMetric = realizationSequence[patient.patientID].metrics[metricName]
                inDistributionList.extend(getIntersection(realizationSequenceMetric, [minInt0,minInt1], [maxInt0,maxInt1], fractionNumber, sampleFrequency))
            if fractionNumber % frequency == 0:
                inDistributionMetrics['{}_inDist_{}fractions'.format(metricName, int(fractionNumber))] = np.sum(inDistributionList) / len(inDistributionList)
        inDistributionMetrics['{}_inDist_allfractions'.format(metricName)] = np.sum(inDistributionList) / len(inDistributionList)
    return inDistributionMetrics

def calculate_normal_interval(data, confidence_level):
    mean = np.mean(data)
    std_dev = np.std(data)

    z_score = norm.ppf(1 - (1 - confidence_level) / 2)
    margin_of_error = z_score * std_dev

    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error

    return lower_bound, upper_bound

def linearFunction(xTwoPoints,yTwoPoints,x):
    return (yTwoPoints[1] - yTwoPoints[0]) / ((xTwoPoints[1] - xTwoPoints[0]))*(x-xTwoPoints[0]) + yTwoPoints[0]


def getIntersection(realizationSequenceMetric, minInt, maxInt, fractionNumber, sampleFrequency):
    isInList = []

    for i in range(sampleFrequency):
        fractionID = fractionNumber + i
        realizationValue = realizationSequenceMetric.get('serie'+str(fractionID).zfill(2)) 
        if realizationValue == None:
            continue
        isInList.append((realizationValue > linearFunction([fractionNumber, fractionNumber+sampleFrequency],minInt, fractionID) and realizationValue <= linearFunction([fractionNumber, fractionNumber+sampleFrequency],maxInt, fractionID))[0])
    return isInList

def addToDict(dict,keys,fileName,index):
    if len(keys)==index:
        return
    key = keys[index]
    if dict.get(key):
        addToDict(dict[key], keys, fileName, index+1)
    elif index != len(keys) - 1:
        dict[key] = {}
        addToDict(dict[key], keys, fileName, index+1)
    else:
        dict[key] = fileName
        return dict

def getKeysFromFileName(fileName):
    fileNameSplit = fileName.split('_')
    type = fileNameSplit[0]
    serieID = None
    sampleID = None
    if type == 'cond':
        if fileNameSplit[1]=='dose' and fileName.find('seg') == -1:
            type = 'condDose'
        elif fileName.find('CBCT') > -1 and fileName.find('seg') == -1: 
            type = 'condImage'
        elif fileName.find('seg') > -1: 
            type = 'cond'+fileName[fileName.find('seg'):].split('.')[0]
        patientID = "_".join(fileNameSplit[3:5])
    elif type == 'realization':
        patientID = "_".join(fileNameSplit[2:4])
        serieID = fileNameSplit[4]
    else:
        patientID = "_".join(fileNameSplit[2:4])
        if fileName.count('serie')>1:
            serieID = fileNameSplit[5]
        else:
            serieID = fileNameSplit[4]
        sampleID = fileNameSplit[-1].split('.')[0]

    if (type == 'sample' or type == 'realization') and fileName.find('CBCT') > -1:
        type = 'Image'
    elif type == 'sampleDVF' or (type == 'realization' and fileName.find('DVF') > -1):
        type = 'DVF'
    elif (type == 'sample' or type == 'realization') and fileName.find('seg') > -1:
        type = fileName[fileName.find('seg'):].split(".")[0]


    return {
        'patientID': patientID,
        'serieID': serieID,
        'sampleID': sampleID,
        'type': type
    }

def getPatientSequence(path):
    patientSequence = PatientSequence()
    filePaths = glob.glob(os.path.join(path,'**/*.mha'),recursive= True)
    filePaths.sort()
    for filePath in filePaths:
        fileName = os.path.basename(filePath)
        keyDict = getKeysFromFileName(fileName)
        patientSequence.addElementByKey(keyDict, filePath)
    patientSequence.sort()
    return patientSequence

def calculateRMSE(sampleArray,condArray, ignoreSlices):
    return np.sqrt(np.mean((sampleArray[:,:,ignoreSlices:-ignoreSlices]-condArray[:,:,ignoreSlices:-ignoreSlices])**2))

def calculateJacobianMean(sampleArray, condArray, voxelToWorldMatrix, ignoreSlices, cropBodyMask = False):
    spacing = np.diag(voxelToWorldMatrix)[:-1]
    origin = voxelToWorldMatrix[:-1,-1]
    sitkDVF = getSITKImage(sampleArray,origin,spacing)
    if cropBodyMask:
        mask = getBodyMask(condArray) * 1
        maskSITK = getSITKImage(mask,origin,spacing)
        maskSITKWarped = warpSITK(maskSITK, sitkDVF, defaultValue = 0, interpolator = sitk.sitkNearestNeighbor)
        arrayWarped = np.transpose(sitk.GetArrayFromImage(maskSITKWarped))
    else:
        arrayWarped = np.ones(condArray.shape)
    jacobianArray = getDVFJacobian(sitkDVF) 
    jacobianArrayCropped = jacobianArray[:,:,ignoreSlices:-ignoreSlices]
    arrayWarped = arrayWarped[:,:,ignoreSlices:-ignoreSlices]
    return np.mean(np.abs(1-jacobianArrayCropped) * arrayWarped) 

def calculateVolumeOfSegmentation(sampleArray, voxelToWorldMatrix, ignoreSlices):
    spacing = np.diag(voxelToWorldMatrix)[:-1]
    return np.sum(sampleArray[:,:,ignoreSlices:-ignoreSlices]) * np.prod(spacing)

def calculateCenterOfMass(sampleArray, transformation_matrix, ignoreSlices):
    segmentation_mask = sampleArray[:,:,ignoreSlices:-ignoreSlices]
    nonzero_coords = np.transpose(np.nonzero(segmentation_mask))
    nonzero_coords = np.hstack((nonzero_coords, np.ones(len(nonzero_coords)).reshape(len(nonzero_coords),1)))
    transformed_coords = np.dot(nonzero_coords, np.transpose(transformation_matrix))[:,:3]
    return np.mean(transformed_coords, axis=0)

def equalizeSampleZdirection(condSeg, sampleArray):
    _,_,Z = np.where(condSeg*1 > 0)
    sampleArray[:,:,:np.min(Z)] = 0
    sampleArray[:,:,np.max(Z):] = 0
    return sampleArray

def addMetric(MetriDict, sample, condArray, condSegmentation, valMetrics):
    
    if sample.image != None and 'MSE' in valMetrics["fractionMetrics"]["metrics"]:
        sampleArray, voxelToWorldMatrix = readMha(sample.image)
        MetriDict['RMSE'] = calculateRMSE(sampleArray,condArray,10)
    if sample.dvf != None and 'Jacobian' in valMetrics["fractionMetrics"]["metrics"]:
        sampleArray, voxelToWorldMatrix = readMha(sample.dvf)
        MetriDict['Jacobian'] = calculateJacobianMean(sampleArray,condArray,voxelToWorldMatrix,10)
    if len(sample.segmentation)>0:
        for key, values in sample.segmentation.items():
           if key not in condSegmentation:
               continue
           condSeg, condVoxelToWorldMatrix = readMha(condSegmentation[key]) 
           sampleArray, voxelToWorldMatrix = readMha(values) 
           sampleArray = equalizeSampleZdirection(condSeg, sampleArray)
           sampleArray[sampleArray>0] = 1
           condSeg[condSeg>0] = 1
           if key in valMetrics["fractionMetrics"]['segmentationMetrics']['volume']:
                MetriDict['Volume_'+key] = calculateVolumeOfSegmentation(sampleArray, voxelToWorldMatrix,10) - calculateVolumeOfSegmentation(condSeg, condVoxelToWorldMatrix,10)  
           if key in valMetrics["fractionMetrics"]['segmentationMetrics']['centerOfMassShift']:
                MetriDict['CenterOfMassShift_'+key] = np.linalg.norm(calculateCenterOfMass(condSeg, condVoxelToWorldMatrix,10)  - calculateCenterOfMass(sampleArray, voxelToWorldMatrix,10))


def calculateMetrics(patientSequence, valMetrics):
    for patient in patientSequence:
        condArray,_ = readMha(patient.conditional.image)
        for fraction in patient:
            for sample in fraction:
                metric = {}
                addMetric(metric, sample, condArray, patient.conditional.segmentation, valMetrics)
                sample.metrics = metric


def saveWarpedSegmentation(DVFPath, segSITK, outputSegPath): 
    DVFSample = sitk.ReadImage(DVFPath)
    DVFSample.SetOrigin([0]*3) ### Setting the origin of the DVF the same as the segMask
    warpedSeg =  warpSITK(segSITK, DVFSample, defaultValue = 0, interpolator = sitk.sitkNearestNeighbor)   
    saveSITKImage(warpedSeg, outputSegPath)

def updateSegmentationList(path, segmentationList):
    files = os.listdir(path)
    updatedList = copy.deepcopy(segmentationList)
    if 'CTV' in segmentationList:
        updatedList.remove('CTV')
        updatedList.extend([file.split('.')[0] for file in files if file.lower().find('ctv')!=-1])
    if 'PTV' in segmentationList:
        updatedList.remove('PTV')
        updatedList.extend([file.split('.')[0] for file in files if file.lower().find('ptv')!=-1])        
    if 'GTV' in segmentationList:
        updatedList.remove('GTV')
        updatedList.extend([file.split('.')[0] for file in files if file.lower().find('gtv')!=-1])      
    return updatedList


def generateSampledSegmentations(dataDir, path, segmentationList, imageSize, overWrite = False):
    crop = CenterSpatialCrop(roi_size = imageSize)
    pad = SpatialPad(spatial_size = imageSize)
    filePaths = glob.glob(os.path.join(path,'*.mha'))
    for path in filePaths:
        patientSeriesID = "_".join(os.path.basename(path).split('_')[-5:-1])
        patientID = 'HN_'+patientSeriesID.split('_')[2]
        Name = os.path.basename(path)
        if Name.find('DVF') != -1:
            continue
        SegDir = os.path.join(os.path.dirname(path), 'Segmentation')
        # updatedSgmentationList = updateSegmentationList(os.path.join(dataDir,patientID,'CBCT_Structures',patientSeriesID+'_CBCTSTRUC'),segmentationList)
        for segmentationName in segmentationList:
            segPath = os.path.join(dataDir,patientID,'CBCT_Structures',patientSeriesID+'_CBCTSTRUC',segmentationName+'.nii.gz')
            segCondName = Name.replace("CBCT",'seg'+segmentationName+'.')
            if os.path.isfile(os.path.join(SegDir, segCondName)) and not overWrite or not os.path.isfile(segPath):
                continue
            segSITK = getCroppedSITKImage(segPath, pad, crop, segmentation =True)

            saveSITKImage(segSITK, os.path.join(SegDir, segCondName))

def getMetricPatientSequenceForValGTData(dataDir, path, segmentationList, imgSize, valMetricDict, overWrite = False):
    generateSampledSegmentations(dataDir, path, segmentationList, imgSize, overWrite)
    patientSequence = getPatientSequence(path)
    calculateMetrics(patientSequence, valMetricDict)
    patientSequence.save(os.path.join(path,'PatientSequenceMetrics.pkl'))

def getCroppedSITKImage(path, pad, crop, segmentation = False):
    segArray, voxelToWorldMatrix = readMha(path)
    if segmentation:
        segArray[segArray>0] = 1
    segArrayTransformed = pad(segArray[None])
    segArrayTransformed = crop(segArrayTransformed).numpy()
    return getSITKImage(segArrayTransformed, [0]*3, np.diag(voxelToWorldMatrix)[:-1])        


def propagateStructures(patientSequence, segmentationList, dataDir, imageSize, overWrite = False):
    crop = CenterSpatialCrop(roi_size = imageSize)
    pad = SpatialPad(spatial_size = imageSize)
    logger.log("Propagating Structures...")
    for patient in patientSequence:
        condSerie = os.path.basename(patient.conditional.image).split('_')[5]
        condName = os.path.basename(patient.conditional.image)
        if len([sample for sample in patient.samples if sample.dvf != None]) == 0:
            continue
        else:
            SegDir = os.path.join(os.path.dirname(patient.conditional.image), 'Segmentation')
            makeFolderMaybe(SegDir)
        # updatedSgmentationList = updateSegmentationList(os.path.join(dataDir,patient.patientID,'CBCT_Structures','UCSF_{}_{}_CBCTSTRUC'.format(patient.patientID,condSerie)),segmentationList)
        if 'CTV' in segmentationList:
            segmentationList.extend([segName.split('.')[0] for segName in os.listdir(os.path.join(dataDir,patient.patientID,'CBCT_Structures','UCSF_{}_{}_CBCTSTRUC'.format(patient.patientID,condSerie))) if 'ctv' in segName.lower()])
            segmentationList.remove('CTV')

        if segmentationList=='all':
            segmentationList = [segName.split('.')[0] for segName in os.listdir(os.path.join(dataDir,patient.patientID,'CBCT_Structures','UCSF_{}_{}_CBCTSTRUC'.format(patient.patientID,condSerie)))]

        for segmentationName in segmentationList:
            segPath = os.path.join(dataDir,patient.patientID,'CBCT_Structures','UCSF_{}_{}_CBCTSTRUC'.format(patient.patientID,condSerie),segmentationName+'.nii.gz')
            if not os.path.isfile(segPath):
                continue
            segSITK = getCroppedSITKImage(segPath, pad, crop, segmentation = True)
            segCondName = condName.replace("CBCT",'seg'+segmentationName)
            saveSITKImage(segSITK, os.path.join(SegDir, segCondName))
            patient.conditional.segmentation[segmentationName] = os.path.join(SegDir, segCondName)

            for fraction in patient:
                for sample in fraction:
                    dvfName = os.path.basename(sample.dvf)
                    segName = dvfName.replace("CBCT",'seg'+segmentationName+'.').replace('sampleDVF', 'sample').replace("DVF",'seg'+segmentationName+'.') ### Sometimes the type is CBCT when propagated other times is DVF when predicted from the model
                    outputSegPath = os.path.join(SegDir, segName)
                    if os.path.isfile(outputSegPath) and not overWrite:
                        continue
                    saveWarpedSegmentation(sample.dvf, segSITK, outputSegPath)
                    sample.segmentation[segmentationName] = outputSegPath
    logger.log("The propagation is finished...")
    

def calculateDivergence(sampledSequences, realizationSequence, divergence):
    Min = {}
    Max = {}
    for sampledSequence in sampledSequences:        
        for metricName, metricPatient in sampledSequence.metrics.items():
            for patientID, metricValues in metricPatient.items():
                Min[metricName] = min([np.min(realizationSequence.metrics[metricName][patientID]), np.min(metricValues), Min.get(metricName, 10000)])
                Max[metricName] = max([np.max(realizationSequence.metrics[metricName][patientID]), np.max(metricValues), Max.get(metricName, -10000)])

    MetricsList = []
    for sampledSequence in sampledSequences:     
        Metrics = {}   
        for metricName, metricPatient in sampledSequence.metrics.items():
            realizationDist = []
            sampleDist = []
            for patientID, metricValues in metricPatient.items():
                realizationDist.extend(realizationSequence.metrics[metricName][patientID])
                sampleDist.extend(metricValues)
            Min_metric = Min[metricName]
            Max_metric = Max[metricName]
            Metrics['KL_'+metricName] = divergence((realizationDist - Min_metric)/(Max_metric-Min_metric), (sampleDist - Min_metric)/(Max_metric-Min_metric))
        MetricsList.append(Metrics)
    return MetricsList

