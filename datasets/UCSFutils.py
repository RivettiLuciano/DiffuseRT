

import os

def getPIDfromName(patient):
    return int(patient[-2:])

def getSerieName(index):
    return 'Serie{:02d}'.format(index)

def getDVFPath(patientID,movingSeries,fixedSeries,dataFolder):
    return os.path.join(
        dataFolder,
        'HN_{:02d}'.format(patientID),
        'DVFs',
        'UCSF_HN_{:02d}_serie{:02d}_serie{:02d}_DVF.mha'.format(patientID,movingSeries,fixedSeries)
    )


def getCBCTPath(patientID,series,dataFolder):
    return getPath(patientID,series,dataFolder,'CBCT')


def getPath(patientID,series,dataFolder,modality):
    return os.path.join(
        dataFolder,
        'HN_{:02d}'.format(patientID),
        modality+'s',
        'UCSF_HN_{:02d}_serie{:02d}_{}.nii.gz'.format(patientID,series,modality),
    )

def convertCBCTToDosePath(cbctPath):
    dosePath = cbctPath.replace('CBCTs','Dose')
    dosePath = dosePath.replace('CBCT','Dose')
    return dosePath

def getDosePath(patientID,series,dataFolder):
    return os.path.join(
        dataFolder,
        'HN_{:02d}'.format(patientID),
        'Dose',
        'UCSF_HN_{:02d}_serie{:02d}_dose.nii.gz'.format(patientID,series),
    )

def getSeriesNumber(filePath):
    folder,fileName = os.path.split(filePath)
    _,folder = os.path.split(folder)
    if folder == 'Dose':
        return int(fileName[-14:-12])
    elif folder == 'DVFs':
        return int(fileName[-10:-8])
    elif folder == 'CBCTs':
        return int(fileName[-14:-12])