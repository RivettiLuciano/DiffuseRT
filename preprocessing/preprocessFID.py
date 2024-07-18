
from datasets.UCSFutils import getPath, getPIDfromName
from utils.ioUtils import makeFolderMaybe, readJson
from utils.mhaLib import readMha
from utils.niftyLib import readNifti
from utils.plotLossCurve import plotLossCurve
import os
from PIL import Image
import numpy as np





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
                        im.save(os.path.join(saveFolder,"{}_{}.png".format(sampleFile,i)))

    def getSamples(self,patient):
        return [f for f in os.listdir(self.sourceFolder) if f'sample_UCSF_{patient}' in f]

    def get3DImage(self, patient, sampleFile):
        return readMha(os.path.join(self.sourceFolder,sampleFile))[0]
         

    def normalizePixelArray(self, pixelArray):
        return np.clip(np.round((pixelArray+1024)/3024*255).astype(np.int32),0,255)



if __name__ == "__main__":
    dataSplit = readJson('/home/andreas/code/DiffuseRT/dataSpliting.json')

    preprocessor = FIDPreprocessor(
        '/home/luciano/ResearchData/DataSets/DL-DIR/DataSet/DeepMovement',
        '/home/luciano/ResearchData/DataSets/DiffuseRTData/FIDData/FIDData',
        dataSplit['training']+dataSplit['validation']
    )
    preprocessor.preprocessPerPatient()