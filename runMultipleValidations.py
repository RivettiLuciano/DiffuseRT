import os
from improved_diffusion.validation_util import getMetricPatientSequenceForValGTData



dataDir = "/home/luciano/ResearchData/DataSets/DL-DIR/DataSet/DeepMovement"

configFiles = [
    # '/home/luciano/Codes/DiffuseRT/experiments/DVF/configDVFPredicted_XstartTimeEncodedMixedCondBendingEnergy0.1.yaml',
    '/home/luciano/Codes/DiffuseRT/experiments/Image/configDoseTimeEncodedMixedConditionals4Days.yaml',
    # '/home/luciano/Codes/DiffuseRT/experiments/Image/configDoseTimeEncodedMixedConditionals.yaml',
    '/home/luciano/Codes/DiffuseRT/experiments/DVF/configDVFPredicted_XstartTimeEncodedMixedCond.yaml',
    # '/home/luciano/Codes/DiffuseRT/experiments/Image/configDoseTimeEncodedFast.yaml',
    # '/home/luciano/Codes/DiffuseRT/experiments/DVF/configDVFPredicted_XstartTimeEncodedNoBendingEnergy.yaml',
    # '/home/luciano/Codes/DiffuseRT/experiments/Image/configDoseWithoutTime.yaml',
    # '/home/luciano/Codes/DiffuseRT/experiments/Image/configDoseTimeEncoded.yaml',
    # '/home/luciano/Codes/DiffuseRT/experiments/DVF/configDVFPredicted_XstartTimeEncoded.yaml',
    '/home/luciano/Codes/DiffuseRT/experiments/DVF/configDVFDose_XstartTimeEncoded.yaml',
    # '/home/luciano/Codes/DiffuseRT/experiments/DVF/configDVFPredicted_XstartTimeEncodedMixedCondBendingEnergy0.5.yaml',

]
# getMetricPatientSequenceForValGTData(dataDir = dataDir,
#                                      path = '/home/luciano/ResearchData/DataSets/DiffuseRTData/input_val', 
#                                      segmentationList = ['Body', 'CTV', 'Esophagus', 'SpinalCord'],
#                                      imgSize= (96,96,96),
#                                      overWrite = False)
pythonPath = '/home/andreas/env/bin/python'
file = '/home/luciano/Codes/DiffuseRT/image_validation.py'
for configPath in configFiles:
    os.system(pythonPath+' '+file+' --configFile '+configPath)
