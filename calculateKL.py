from improved_diffusion.validation_util import calculateDivergence
from utils.config import readConfigAndAddDefaults
from improved_diffusion import dist_util, logger
import os
from utils.patientStructure import PatientSequence
import pandas as pd
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance
from utils.plotLossCurve import plotHistogramNormalized

def KLDivergence(P,Q):
    e = 1e-6
    return sum(rel_entr(P,Q+e))

configFiles = [
    '/home/luciano/Codes/DiffuseRT/experiments/Image/configDoseTimeEncodedMixedConditionals4Days.yaml',
    '/home/luciano/Codes/DiffuseRT/experiments/DVF/configDVFPredicted_XstartTimeEncodedMixedCond.yaml',
    '/home/luciano/Codes/DiffuseRT/experiments/DVF/configDVFDose_XstartTimeEncoded.yaml',
]

patientSequences = []
outputFolders = []
for confiFile in configFiles:
    config = readConfigAndAddDefaults(confiFile)
    validationFolder = logger.getCurrentEvaluationFolder(config['logging_path'],config['validation_experiment_name'])
    sampleFolder = os.path.join(validationFolder,'Samples')
    outputPath = os.path.join(os.path.dirname(sampleFolder),'PatientSequenceMetrics.pkl')
    patientsequence = PatientSequence()
    patientsequence.load(outputPath)
    outputFolders.append(validationFolder) 
    patientSequences.append(patientsequence)

realizationPatientSequencePath = os.path.join(os.path.dirname(config['logging_path']),'input_val','PatientSequenceMetrics.pkl')
realizationPatientSequence = PatientSequence()
realizationPatientSequence.load(realizationPatientSequencePath) 
# plotHistogramNormalized(patientSequences, realizationPatientSequence, ['Image', 'Hybrid', 'DVF'])

metricModels = calculateDivergence(patientSequences, realizationPatientSequence, wasserstein_distance)
# # metricModels = calculateDivergence(patientSequences, realizationPatientSequence, KLDivergence)

for metric, outputPath in zip(metricModels, outputFolders):
    pd.DataFrame.from_dict([metric]).to_csv(outputPath+'/wasserstein_distance.csv',index = False)
    # pd.DataFrame.from_dict([metric]).to_csv(outputPath+'KLMetrics.csv',index = False)