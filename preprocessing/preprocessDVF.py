import os
import shutil
import pandas as pd
import numpy as np
from preprocessing.plastimatchRegistration import runPlastimatchRegistration
from utils.ioUtils import makeFolderMaybe
from datasets.UCSFutils import getSerieName,getDVFPath,getPath

def preprocessDVF(dataFolder,mappingFile,plastimatchConfigPath, modality):

    patientIDs = [int(f[-2:]) for f in os.listdir(dataFolder) if os.path.isdir(os.path.join(dataFolder,f))]
    mapping = pd.read_csv(mappingFile,index_col='PatientID',delimiter=',')

    for patientID in patientIDs:
        patientMapping = mapping.loc[patientID]
        movingSeriesIndex = 1 
        movingReferenceSeries = patientMapping[getSerieName(movingSeriesIndex)]
        shutil.rmtree(os.path.join(dataFolder,'HN_{:02d}'.format(patientID),'DVFs'))
        for i in range(2,37):
            referenceSeries = patientMapping[getSerieName(i)]
            if isinstance(referenceSeries,str) and 'serie' in referenceSeries: # to rule out the NaNs
                if movingReferenceSeries == referenceSeries:
                    targetPath = getDVFPath(patientID,movingSeriesIndex,i,dataFolder)
                    makeFolderMaybe(os.path.split(targetPath)[0])
                    if True: #not os.path.exists(targetPath):
                        print(targetPath)
                        runPlastimatchRegistration(
                            plastimatchConfigPath,
                            getPath(patientID,movingSeriesIndex,dataFolder, modality),
                            getPath(patientID,i,dataFolder, modality),
                            targetPath
                        )
                else:
                    movingReferenceSeries = referenceSeries
                    movingSeriesIndex = i





if __name__ == "__main__":
    preprocessDVF(
        '/home/luciano/ResearchData/DataSets/DL-DIR/DataSet/DeepMovement',
        '/home/luciano/ResearchData/DataSets/DL-DIR/DataSet/DeepMovement/Mapping_CBCT.csv',
        '/home/andreas/code/DiffuseRT/preprocessing/bSplinePlastimatch.cfg'
    )