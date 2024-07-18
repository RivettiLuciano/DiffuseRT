import copy
from typing import Hashable, Mapping
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os
import glob
from monai.data import Dataset
from monai.data import DataLoader as DataLoaderMonai
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    EnsureTyped,
    SpatialPadd,
    RandSpatialCropd,
    CenterSpatialCropd,
    EnsureChannelFirstd,
    MapTransform
)
import numpy as np
import json
import torch
from datasets.UCSFutils import convertCBCTToDosePath, getPath, getPIDfromName, getSeriesNumber
from utils.transforms import ScaleDoseIntensityRanged, ConcatenateInputTransform, InvertibleScaleIntensityRanged, InvertibleScaleIntensityMeanStdd


DOSE_KEY = 'cond_dose'
DOSE_SCALING_KEY = 'dose_scaling_key'
FRACTION_TIME_KEY = 'fraction_time_key'

CONDITIONAL_MODE_PLANNING = 'planning'
CONDITIONAL_MODE_SINGLE_ALL = 'single_all'
CONDITIONAL_MODE_SEQUENCE = 'sequence'


class BaseDataset(Dataset):
    def __init__(self, config,mode):
        self.config = config
        self.mode = mode
        datalist = self.getDataList()
        self.setKeys(datalist)
        assert self.mode in ['training','testing','validation','validationAndTesting']
        self.transform = self.getDataTransform()
        super().__init__(data=datalist, transform=self.transform )

    def getInstanceList(self):
        with open(self.getConfigVariable('data_spliting'), 'r') as json_file:
            data = json.load(json_file)
        return data[self.mode]
    
    def getConfigVariable(self,variable):
        return self.config[variable]
    
    def getDataList(self):
        raise NotImplementedError

    def getDataTransform(self):
        raise NotImplementedError
    
    def setKeys(self,dataList):
        self.inputKeys = [f for f in dataList[0].keys() if 'input' in f]
        self.conditionalKeys = [f for f in dataList[0].keys() if 'cond' in f]
    

class ImageDataset(BaseDataset):
    def __init__(self, config, mode):
        self.config = config
        self.conditionalMode = self.getConfigVariable('conditional_mode')
        super().__init__(config, mode)

    def getDataList(self):
        dataList = []
        map_file_df = pd.read_csv(self.getConfigVariable('mappingFile'),index_col='PatientID')
        patientList = self.getInstanceList()
        data_dir = self.getConfigVariable('data_dir')
        
        dataListGenerator = self.getPatientDataListGenerator()

        for patient in patientList:
            patientID = getPIDfromName(patient)
            serie_array = np.array(map_file_df.loc[patientID])
            serie_array = serie_array[~pd.isna(serie_array)]
            dataList += dataListGenerator(data_dir,patientID,serie_array)

        if self.getConfigVariable('include_dose'):
            dataList = self.addConditionalDoseList(dataList)

        if self.getConfigVariable('load_fraction_time'):
            dataList = self.addFractionTime(dataList)

            if self.getConfigVariable('mix_fraction_times') > 0:
                dataList = self.mixFractionTimes(dataList,self.getConfigVariable('mix_fraction_times'))

        return dataList


    def getPatientDataListGenerator(self):
        if self.conditionalMode == CONDITIONAL_MODE_PLANNING:
            return self.getSingleConditionalPatientDataList
        elif self.conditionalMode == CONDITIONAL_MODE_SEQUENCE:
            return self.getSequenceConditionalPatientDataList
        elif self.conditionalMode == CONDITIONAL_MODE_SINGLE_ALL:
            return self.getSingleConditionalWithAllImagesPatientDataList
        else:
            raise ValueError(f'Did not recognize the conditional mode {self.conditionalMode}')

    def getSingleConditionalPatientDataList(self,data_dir,patientID,serie_array):
        dataList = []
        for i, serie in enumerate(serie_array):
            if serie != '-':
                serieIndex = i+1
                referenceSerieIndex = int(serie[-2:])
                if serieIndex > referenceSerieIndex:
                    condPath = getPath(patientID,referenceSerieIndex,data_dir, self.config['modality'])
                    imagePath = getPath(patientID,serieIndex,data_dir, self.config['modality'])
                    if os.path.exists(condPath) and os.path.exists(imagePath):
                        dataList.append({
                                "cond_0":condPath,
                                "input_0": imagePath,
                            })
                else:
                    referenceSerieIndex = serieIndex
        return dataList
    
    def getSingleConditionalWithAllImagesPatientDataList(self,data_dir,patientID,serie_array):
        dataList = []
        for i, serie in enumerate(serie_array):
            if serie != '-':
                serieIndex = i+1
                for j,subsequentSerie in enumerate(serie_array[i+1:]):
                    subsequentSerieIndex = serieIndex + j+1
                    if serie == subsequentSerie:
                        condPath = getPath(patientID,serieIndex,data_dir, self.config['modality'])
                        imagePath = getPath(patientID,subsequentSerieIndex,data_dir, self.config['modality'])
                        if os.path.exists(condPath) and os.path.exists(imagePath):
                            dataList.append({
                                    "cond_0":condPath,
                                    "input_0": imagePath,
                                })
        return dataList

    def getSequenceConditionalPatientDataList(self,data_dir,patient,serie_array):
        raise NotImplementedError # to be rechecked before usage
        dataList = []
        condNumber = self.getConfigVariable('conditional_images')
        imageNumber = self.getConfigVariable('image_in_channels')
        samplingInterval = self.getConfigVariable('sampling_interval')
        CBCTpaths = glob.glob(os.path.join(data_dir,patient,'CBCTs','*.nii.gz'))
        CBCTpaths.sort()
        for i in range(len(serie_array)): 
            Dict = {}
            samples = np.arange(i,i + (condNumber + imageNumber) * samplingInterval, samplingInterval)
            samples_dirs = [os.path.join(data_dir,patient,'CBCTs','UCSF_{}_{}_CBCT.nii.gz'.format(patient,'serie'+str(s+1).zfill(2))) for s in samples]

            samples_isdir  = np.array([not os.path.isfile(dir) for dir in samples_dirs])
            if samples_isdir.any(): continue
            if all(samples_isdir): break
            CTseries = serie_array[samples]
            if (CTseries!=CTseries[0]).any(): continue

            for j in range(condNumber):
                Dict['cond_{}'.format(str(j))] = samples_dirs[j]
            for k in range(imageNumber):
                Dict['input_{}'.format(str(k))] = samples_dirs[j+k+1]
            dataList.append(Dict)
        return dataList

    def addConditionalDoseList(self,dataList):
        for dataInstance in dataList:
            dataInstance[DOSE_KEY] = convertCBCTToDosePath(dataInstance['cond_0'])
            if self.getConfigVariable('scale_dose_with_time'):
                dataInstance[DOSE_SCALING_KEY] = float(getSeriesNumber(dataInstance['input_0'])-1) #since we start at image 1
            else:
                dataInstance[DOSE_SCALING_KEY] = 30 # because this would make the dose approx [0,1]
            assert os.path.exists(dataInstance[DOSE_KEY])
        return dataList

    def addFractionTime(self,dataList):
        for dataInstance in dataList:
            dataInstance[FRACTION_TIME_KEY] = float(getSeriesNumber(dataInstance['input_0'])-getSeriesNumber(dataInstance['cond_0']))
        return dataList

    def mixFractionTimes(self,dataList,mixingDistance):
        additionalDataList = []
        for dataInstance in dataList:
            for i in range(-mixingDistance,mixingDistance+1):
                if i !=0 and dataInstance[FRACTION_TIME_KEY]+i >0 and dataInstance[FRACTION_TIME_KEY]+i < 35:
                    newDataInstance = copy.deepcopy(dataInstance)
                    newDataInstance[FRACTION_TIME_KEY] += i
                    additionalDataList.append(newDataInstance)
        return dataList + additionalDataList

    def getDataTransform(self):
        keys = self.inputKeys+self.conditionalKeys
        imageKeys = [key for key in keys if key not in [DOSE_KEY,DOSE_SCALING_KEY,FRACTION_TIME_KEY]]

        return Compose(
                    [   
                        LoadImaged(
                            keys=keys,
                        ),
                        AddChanneld(
                            keys=keys
                        ),
                        SpatialPadd(
                            keys=imageKeys,
                            spatial_size= self.getImageSize(),
                            mode="constant",
                            constant_values=-1024,
                        ),
                        SpatialPadd(
                            keys=[DOSE_KEY],
                            spatial_size= self.getImageSize(),
                            mode="constant",
                            constant_values=0,
                            allow_missing_keys=True
                        ),
                        self.getNormalization(imageKeys),
                        ScaleDoseIntensityRanged(
                            keys=[DOSE_KEY],
                            doseScalingKey = DOSE_SCALING_KEY,
                        ),
                        self.getCropping(),
                        EnsureTyped(
                            keys=keys,
                            data_type = 'tensor'
                        ),
                        ConcatenateInputTransform(
                            keys = ['cond', 'input']
                        )
                    ]
                )

    def getImageSize(self):
        imageSize = self.getConfigVariable('image_size')
        if isinstance(imageSize,int):
            return [imageSize] *3
        elif isinstance(imageSize,list) or isinstance(imageSize,tuple):
            return imageSize
        else:
            raise ValueError('Did not recognize the "image_size" type')

    def getCropping(self):
        keys = self.inputKeys+self.conditionalKeys
        if self.mode == 'training':
            return RandSpatialCropd(
                keys=keys,
                random_size = False,
                roi_size = self.getImageSize()
            )
        else:
            return CenterSpatialCropd(
                keys=keys,
                roi_size = self.getImageSize()
            )

    def getNormalization(self,imageKeys):
        if 'scale_image_with_conditional_mean' in self.config and self.config['scale_image_with_conditional_mean']:
            return InvertibleScaleIntensityMeanStdd(
                keys=imageKeys,
                mean=0, std=1/3,
                allow_missing_keys=True
            )
        else:
            return InvertibleScaleIntensityRanged(
                keys=imageKeys,
                a_min=-1024, a_max=2000, b_min=-1, b_max=1.0, clip=True,allow_missing_keys=True
            )



class DVFDataset(ImageDataset):
    def getDataList(self):
        patientList = self.getInstanceList()
        data_dir = self.getConfigVariable('data_dir')
        dataList = []
        for patient in patientList:
            patientID = getPIDfromName(patient)
            dvfPath = os.path.join(data_dir,patient,'DVFs')
            dvfs = os.listdir(dvfPath)
            dvfs.sort()
            for dvf in dvfs:
                referenceSeries = int(dvf[16:18])
                cbctPath = getPath(patientID,referenceSeries,data_dir, self.config['modality'])
                dataList.append(
                    {
                        'input_0': os.path.join(dvfPath,dvf),
                        'cond_0': cbctPath
                    }
                )

        if self.getConfigVariable('include_dose'):
            dataList = self.addConditionalDoseList(dataList)
        
        if self.getConfigVariable('load_fraction_time'):
            dataList = self.addFractionTime(dataList)

            if self.getConfigVariable('mix_fraction_times') > 0:
                dataList = self.mixFractionTimes(dataList,self.getConfigVariable('mix_fraction_times'))

        return dataList
    
    def getDataTransform(self):
        imageKeys = [key for key in self.conditionalKeys if key not in [DOSE_KEY,DOSE_SCALING_KEY]]
        return Compose(
                    [   
                        LoadImaged(
                            keys=self.inputKeys+self.conditionalKeys,
                        ),
                        EnsureChannelFirstd(
                            keys=self.conditionalKeys,channel_dim='no_channel'
                        ),
                        EnsureChannelFirstd(
                            keys=self.inputKeys,channel_dim=3
                        ),
                        SpatialPadd(
                            keys=imageKeys,
                            spatial_size= self.getImageSize(),
                            mode="constant",
                            constant_values=-1024
                        ),
                        SpatialPadd(
                            keys=self.inputKeys + [DOSE_KEY],
                            spatial_size= self.getImageSize(),
                            mode="constant",
                            constant_values=0,
                            allow_missing_keys=True
                        ),
                        InvertibleScaleIntensityRanged(
                            keys=imageKeys,
                            a_min=-1024, a_max=2000, b_min=-1, b_max=1.0, clip=True,allow_missing_keys=True
                        ),
                        InvertibleScaleIntensityRanged(
                            keys=self.inputKeys,
                            a_min=-15, a_max=15, b_min=-1, b_max=1.0, clip=False,allow_missing_keys=True
                        ),
                        ScaleDoseIntensityRanged(
                            keys=[DOSE_KEY],
                            doseScalingKey=DOSE_SCALING_KEY,
                        ),
                        self.getCropping(),
                        EnsureTyped(
                            keys=self.inputKeys+self.conditionalKeys,
                            data_type = 'tensor'
                        ),
                        ConcatenateInputTransform(
                            keys = ['cond', 'input']
                        )
                    ]
                )


class DataLoader(DataLoaderMonai): ### In the future inherit the class DataLoader from Monai
    def __init__(self, dataSet, batch_size, shuffle, num_workers):
        self.dataSet = dataSet
        super().__init__(dataSet, batch_size=batch_size, shuffle = shuffle, num_workers = num_workers)
        self.iterNumber = 1

    def __next__(self):
        if self._iterator is None:
            self._iterator = self._get_iterator()
        if self.iterNumber * self.batch_size > len(self.dataSet) :
            self._iterator._reset(self)
            self.iterNumber = 1
        self.iterNumber+=1
        return next(self._iterator)

def load_data(config,mode):
    if config['data_set'] == 'ImageDataset':
        dataSet = ImageDataset(config,mode)
    elif config['data_set'] == 'DVFDataset':
        dataSet = DVFDataset(config,mode)
    else:
        raise ValueError(f'Did not recognize {config["data_set"]} as a dataset type')
    if mode in ['validation', 'testing','validationAndTesting']:
        config['shuffle_data'] = False
    return DataLoader(
        dataSet,
        batch_size=config['batch_size'],
        shuffle = config['shuffle_data'],
        num_workers = config['num_workers']
    )


