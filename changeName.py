import os
import glob

path= '/home/luciano/ResearchData/DataSets/DiffuseRTData/input_val'
paths = glob.glob(os.path.join(path,'*.mha'))

for file in paths:
    fileName = os.path.basename(file)
    newFileName = ['realization']
    newFileName.extend(fileName.split('_')[1:])
    newFileName = "_".join(newFileName)
    os.system('mv '+file+' '+file.replace(fileName,newFileName))