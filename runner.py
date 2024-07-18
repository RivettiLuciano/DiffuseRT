
from utils.plotLossCurve import plotLossCurve, plotMSE, plotMetricsFromPatientSequence, plotFractionComparison, plotHistogramFromPatientSequence
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import yaml

#MSE_Samples.csv
# plotLossCurve('/home/luciano/ResearchData/DataSets/DiffuseRTData/ImageDoseTimeEncodedMixedConditionals_lr0.00001')


patientSequenceMetricsName = 'PatientSequenceMetrics.pkl'
# modelNames = ['DVFDoseWithXstartTimeEncoded', 'ImageDoseTimeEncodedScaleShift', 'ImageDoseWithoutTime', 'DVFPredicted_XstartTimeEncoded', 'ImageDoseTimeEncodedMixedConditionals','ImageDoseTimeEncodedFast','DVFPredicted_XstartTimeEncodedNoBendingEnergy', 'DVFPredicted_XstartTimeEncodedMixedConditionals4days', 'ImageDoseTimeEncodedMixedConditionals4Days', 'DVFPredicted_XstartTimeEncodedMixedConditionals4daysBE0.1', 'DVFPredicted_XstartTimeEncodedMixedConditionals4daysBE0.5']
modelNames = ['ImageDoseTimeEncodedMixedConditionals4Days', 'DVFDoseWithXstartTimeEncoded', 'DVFPredicted_XstartTimeEncodedMixedConditionals4days']
pathToPSMs = ['/home/luciano/ResearchData/DataSets/DiffuseRTData/{}/Evaluation/ManySamples100/{}'.format(model, patientSequenceMetricsName) for model in modelNames]
pathRealizations = '/home/luciano/ResearchData/DataSets/DiffuseRTData/input_val/{}'.format(patientSequenceMetricsName)
# plotMetricsFromPatientSequence(pathToPSMs, pathRealizations, 'histogram', mode='patients')
plotHistogramFromPatientSequence(pathToPSMs, pathRealizations, ['Image Model', 'DVF Model', 'Hybrid Model'])
# plotHistogramFromPatientSequence(pathToPSMs, pathRealizations, ['Image Model', 'DVF Model', 'Hybrid Model'])
# plotMetricsFromPatientSequence(pathToPSMs, pathRealizations, 'scatter', mode='fractions')
# metric = 'Volume_Body'
# metric = 'Volume_Esophagus'
# plotFractionComparison(pathToPSMs, pathRealizations, ['HN_24', 'HN_15'], metric, '/home/luciano/Codes/DiffuseRT/plots/{}_plot.png'.format(metric))
# ['HN_24', 'HN_05', 'HN_11', 'HN_01', 'HN_15']
