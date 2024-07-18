from monai.transforms import (
    ScaleIntensityRange,
    ScaleIntensityRanged,
    MapTransform,
    InvertibleTransform,
    Transform
)
from monai.config import KeysCollection
import torch
from typing import Dict, Hashable, Mapping
from monai.config.type_definitions import NdarrayOrTensor
import numpy as np
from monai.utils.enums import TransformBackends

class ConcatenateInputTransform(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.keys = keys
    
    def __call__(self, data):
        if 'cond' in self.keys and 'cond' not in data: ### Move this
            data['cond'] = self.catTensors(data, 'cond')
        if 'input' in self.keys and 'input' not in data:
            data['input'] = self.catTensors(data, 'input')
        return data

    def catTensors(self,data,key):
        tensors = []
        relevantKeys = sorted([subkey for subkey in data.keys() if key in subkey and 'meta' not in subkey])
        for subKey in  relevantKeys:
            tensors.append(torch.Tensor(data[subKey]))
        return torch.cat(tensors,dim=0)

class InvertibleScaleIntensityRanged(MapTransform, InvertibleTransform):
    backend = ScaleIntensityRange.backend
    def __init__(
        self,
        keys: KeysCollection,
        a_min: float,
        a_max: float,
        b_min: float,
        b_max: float,
        clip: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.ScaleIntesity = True
        self.scaler = ScaleIntensityRange(a_min, a_max, b_min, b_max, clip)
        self.inverse_scaler = ScaleIntensityRange(b_min, b_max, a_min, a_max, clip)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.scaler(d[key])
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.inverse_scaler(d[key])
        return d
    

class ScaleDoseIntensityRanged(InvertibleScaleIntensityRanged):
    def __init__(self, keys: KeysCollection, doseScalingKey) -> None:
        super().__init__(keys, 0, 72, 0, 1, clip=False, allow_missing_keys=True)
        self.doseScalingKey = doseScalingKey

    def __call__(self, data):
        data = super().__call__(data)
        for key in self.keys:
            if key in data:
                data[key] = data[self.doseScalingKey]*data[key]
        return data

    def getDoseScalingFactor(self,dosePath):
        return float(dosePath[-14:-12])
    

class InvertibleScaleIntensityMeanStdd(MapTransform, InvertibleTransform):
    backend = ScaleIntensityRange.backend
    def __init__(
        self,
        keys: KeysCollection,
        mean: float,
        std: float,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.ScaleIntesity = True
        self.newMean = mean
        self.newStd = std


    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        meanCond = torch.mean(data['cond_0'])
        stdCond = torch.std(data['cond_0'])
        for key in self.key_iterator(d):
            d[key] = (d[key]-meanCond)/stdCond*self.newStd + self.newMean
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        meanCond = torch.mean(data['cond_0'])
        stdCond = torch.std(data['cond_0'])
        for key in self.key_iterator(d):
            d[key] = (d[key]-self.newMean)/self.newStd*stdCond+meanCond
        return d
    

# class ScaleIntensityMeanStd(Transform):
#     backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
#     def __init__(
#         self,
#         newMean: float,
#         newStd: float,
#     ):
#         self.newMean = newMean
#         self.newStd = newStd

#     def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
#         """
#         Apply the transform to `img`.
#         """
#         self.img_mean = torch.mean(img)
#         self.img_std = torch.std(img)
#         return (img-self.img_mean) / self.img_std * self.newStd + self.newMean

