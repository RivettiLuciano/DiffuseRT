from typing import Dict
from collections import OrderedDict
import numpy as np
import os
import pickle

def addToDictList(dict,key,value):
    if dict.get(key):
        dict[key].append(value)
    else:
        dict[key] = [value]



class Sample:
    def __init__(self, sampleID):
        self.sampleNumber = sampleID
        self.dvf = None
        self.image = None 
        self.segmentation = OrderedDict()
        self.metrics = None
    
    def addSampleByKey(self, key, value):
        if key == 'Image':
            self.image = value
        elif key == 'DVF':
            self.dvf = value
        elif key.find('seg')==0:
            key = key[3:]
            self.segmentation[key] = value
        elif 'metrics' in key:
            self.metrics = value
        else:
            KeyError("Wrong key for samples")
    

class Conditional:
    def __init__(self):
        self.conditionalNumber = None
        self.dose = None
        self.image = None 
        self.segmentation = OrderedDict()

    def addConditionalByKey(self, key, value):
        if key == 'condImage':
            self.image = value
        elif key == 'condDose':
            self.dose = value
        elif key.find('seg')>-1:
            key = key[7:] ### remove the cond
            self.segmentation[key] = value
        else:
            KeyError("Wrong key for conditional")

class Fraction:
    def __init__(self, fractionID, samples = None):
        if samples == None:
            self.samples = OrderedDict()
        else:
            self.samples: dict[str, Sample] = samples    
        self.fractionID = fractionID

    def __setitem__(self, key, value):
        self.samples[key] = value

    def __getitem__(self, key):
        return self.samples[key]

    def __len__(self):
        return len(self.samples)

    def addSample(self, sampleID):
        self[sampleID] = Sample(sampleID)

    def addElementByKey(self,keyDict,fileName):
        if not keyDict.get('sampleID'):
            if self.samples.get('realization') == None:
                self.addSample('realization')
            self.samples['realization'].addSampleByKey(keyDict['type'], fileName)
            return

        sampleID = keyDict['sampleID']
        if sampleID not in self.samples:
            self.addSample(sampleID)

        self.samples[sampleID].addSampleByKey(keyDict['type'], fileName)
    
    def sort(self):
        self.samples = OrderedDict(sorted(self.samples.items()))

    def __iter__(self):
        self.iter = 0
        return self
    
    def __next__(self):
        if self.iter < len(self):
            value = self.samples[self.keys[self.iter]]
            self.iter += 1
            return value
        else:
            raise StopIteration    
         
    @property
    def keys(self):
        return list(self.samples.keys())

    @property
    def metrics(self):
        metric = {}
        for sample in self.samples.values():
            for key, value in sample.metrics.items():
                addToDictList(metric, key, value)
        return metric

class Patient:
    def __init__(self, patientID = None, fractions  = None, conditional  = None):
        self.patientID = patientID
        
        if fractions == None:
            self.fractions = OrderedDict()
        else:
            self.fractions: dict[str, Fraction] = fractions        

        if conditional == None:
            self.conditional = Conditional()
        else:
            self.conditional: Conditional = conditional

    def __setitem__(self, key, value):
        self.fractions[key] = value

    def __getitem__(self, key):
        return self.fractions[key]

    def __len__(self):
        return len(self.fractions)
    
    def addFraction(self, fractionID, samples = None):
        self[fractionID] = Fraction(fractionID, samples)

    def addElementByKey(self,keyDict,fileName):
        if len(keyDict) == 0:
            return
        
        if keyDict['type'].find('cond')>-1:
            self.conditional.addConditionalByKey(keyDict['type'], fileName)
            return

        serieID = keyDict['serieID']
        if serieID not in self.fractions:
            self.addFraction(keyDict['serieID'])

        self.fractions[serieID].addElementByKey(keyDict, fileName)
    
    def sort(self):
        self.fractions = OrderedDict(sorted(self.fractions.items()))
        for fraction in self.fractions.values():
            fraction.sort()

    @property
    def samples(self):
        samples = []
        for fraction in self.fractions.values():
            samples.extend(list(fraction.samples.values()))
        return samples   
         
    def __iter__(self):
        self.iter = 0
        return self
    
    def __next__(self):
        if self.iter < len(self):
            value = self.fractions[self.keys[self.iter]]
            self.iter += 1
            return value
        else:
            raise StopIteration   

    @property
    def keys(self):
        return list(self.fractions.keys())

    @property
    def metrics(self):
        metrics = {}
        for fraction in self:
            self.addMetric(metrics, fraction.metrics, fraction.fractionID)
        return metrics

    def addMetric(self, dict, metrics, key):
        for metric, valueList in metrics.items():
            if dict.get(metric) != None:
                dict[metric][key] = valueList
            else:
                dict[metric] = {key: valueList}


class PatientSequence:
    def __init__(self):
        self.patients: dict[str, Patient] = OrderedDict()

    def __setitem__(self, key, value):
        self.patients[key] = value

    def __getitem__(self, key):
        return self.patients[key]

    def __len__(self):
        return len(self.patients)
    
    def addPatient(self, patientID, fractions = None, conditional = None):
        self[patientID] = Patient(patientID, fractions, conditional)
    
    def addElementByKey(self,keyDict,fileName):
        if len(keyDict) == 0:
            return
        
        patientid = keyDict['patientID']
        if patientid not in self.patients:
            self.addPatient(keyDict['patientID'])

        self.patients[patientid].addElementByKey(keyDict, fileName)
    
    def sort(self):
        self.patients = OrderedDict(sorted(self.patients.items()))
        for patient in self.patients.values():
            patient.sort()

    def __iter__(self):
        self.iter = 0
        return self
    
    def __next__(self):
        if self.iter < len(self):
            value = self.patients[self.keys[self.iter]]
            self.iter += 1
            return value
        else:
            raise StopIteration

    @property
    def keys(self):
        return list(self.patients.keys())
    
    def save(self,path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
        self.patients = loaded.patients

    @property
    def metrics(self):
        metrics = {}
        for patient in self:
            self.addMetric(metrics, patient.metrics, patient.patientID)
        return metrics
    
    def addMetric(self, dict, metrics, key):
        for metric, valueList in metrics.items():
            valueList = list(np.ravel(list(valueList.values())))
            if dict.get(metric) != None:
                dict[metric][key] = valueList
            else:
                dict[metric] = {key: valueList}
    
    def getSampleFractionFrequency(self):
        if len(self)>0:
            patient0 = list(self.patients.values())[0]
            fractionIDs = patient0.keys
            fractionIDsNumber = np.array([int(fID[5:]) for fID in fractionIDs])
            return fractionIDsNumber[1] - fractionIDsNumber[0]
        else:
            ValueError('Empty Dictionary')

        