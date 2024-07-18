import matplotlib.pyplot as plt
import pandas as pd
import os
from utils.ioUtils import getCurrentPlotPath, getPlotsPath, makeFoldersMaybe
import pickle
import numpy as np
from utils.patientStructure import PatientSequence
import scipy.stats
from improved_diffusion.validation_util import calculate_normal_interval
import matplotlib.ticker as ticker

def plotLossCurve(logFolder):
    losses = pd.read_csv(os.path.join(logFolder,'progress.csv'))
    fig,ax = plt.subplots(1,1)
    for label in ['loss','vb','mse']:
        ax.plot(losses['step'][2:],losses[label][2:],label=label)
    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Step [-]')
    ax.set_ylabel('Loss [-]')
    ax.legend()
    ax.grid()
    fig.tight_layout()
    plt.savefig(getCurrentPlotPath())

def plotMSE(path):
    df = pd.read_csv(path,index_col=0)
    seriesNumbers = [int(df.columns[i][5:7]) for i in range(0,len(df.columns),2)]
    fig, ax = plt.subplots()
    for patientID in df.index:
        array = df.loc[patientID].values
        array = array.reshape((len(seriesNumbers),2))
        ax.errorbar(seriesNumbers, array[:,0], yerr=array[:,1], fmt='-o', label = patientID)
    ax.set_title('MSE score')
    plt.legend()
    plt.savefig(os.path.join(getPlotsPath(),'MSE.png'))

def plotMeanMSEfromDict(path):
    mode = 'sample'
    # type = 'scatter'
    type = 'box'
    with open(path, "rb") as input_file:
        mseData = pickle.load(input_file)
    seriesDict = {}
    for patientID in mseData.keys():
        keys = list(mseData[patientID][mode].keys())
        keys.sort()
        for serieID in keys:
            mseSerie = mseData[patientID][mode][serieID]['MSE']
            serieNumber = int(serieID[-2:])
            if serieNumber not in seriesDict:
                seriesDict[serieNumber] = mseSerie
            else:
                seriesDict[serieNumber].extend(mseSerie)

    seriesDictMean = {key: np.mean(values) for key, values in seriesDict.items()}
    seriesDictStd = {key: np.std(values) for key, values in seriesDict.items()}
    fig, ax = plt.subplots()
    ax.set_title('MSE score')
    ax.errorbar(list(seriesDictMean.keys()),list(seriesDictMean.values()),yerr = list(seriesDictStd.values()), fmt = 'o-')
    plt.savefig(os.path.join(getPlotsPath(),'MSE','meanMSE.png'))

def removeNone(keys, sampledMetrics):
    key = []
    metric = []
    for i, element in enumerate(sampledMetrics):
        if isinstance(element, list) and element[0] != None:
            key.append(keys[i])
            metric.append(element)
    return key, metric

def Density(values,Min,Max):
  kde = scipy.stats.gaussian_kde(values)
  xs = np.linspace(Min,Max)
  return xs,kde.pdf(xs)

def plotMetrics(metricFraction, realizationMetricFraction, plotType, outputDir, title, metricName):
    fig, ax = plt.subplots(figsize=(6.9, 4.8))
    sampledMetrics = []
    realizationMetrics = []
    interval = []
    keys = list(set(realizationMetricFraction.keys()) | set(metricFraction.keys())) 
    keys.sort()
    labelDict = {'Volume': 'Volume Difference [$mm^3$]', 'CenterOfMassShift': 'Distance [mm]', 'RMSE': 'RMSE [HU]'}
    label = labelDict.get(metricName.split('_')[0], metricName)
    ### Plot properties
    binNumber = 30
    colorSample = f"C{0}"
    colorRealization = f"C{1}"
    labelSample = 'AI'
    labelRealization = 'real CBCTs'
    for key in keys:
        sampledMetrics.append(metricFraction.get(key))
        realizationMetrics.append(realizationMetricFraction.get(key))  
        if metricFraction.get(key)!=None:
            interval.append(calculate_normal_interval(metricFraction.get(key),0.95))

    if plotType == 'scatter':
        keys = [int(k[5:]) for k in keys]
        interval = np.array(interval)
        minY, maxY = interval[:,0], interval[:,1]
        sampledKeys, sampledMetrics = removeNone(keys, sampledMetrics)
        ax.plot(sampledKeys, np.mean(sampledMetrics,1),'o-', label = labelSample, color = colorSample)
        ax.plot(keys,realizationMetrics,'o-',color = colorRealization, label = labelRealization, markersize=4)
        ax.fill_between(sampledKeys, minY,  maxY, alpha = 0.2)
        ax.set_xlabel('Fractions')
        ax.set_ylabel(label)
    elif plotType == 'histogram':
        sampledMetrics = [item for sublist in sampledMetrics for item in sublist]
        realizationMetrics = [item for sublist in realizationMetrics for item in sublist]
        Min = min(np.min(sampledMetrics), np.min(realizationMetrics))
        Max = max(np.max(sampledMetrics), np.max(realizationMetrics))
        xs, ys = Density(sampledMetrics, Min, Max)
        xr, yr = Density(realizationMetrics, Min, Max)
        ax.hist(sampledMetrics, range=(Min,Max), bins=binNumber, alpha = 0.5, label = labelSample, edgecolor = "black", color = colorSample, density= True)
        plt.plot(xs, ys, c = colorSample, lw=3, label='fit '+labelSample)
        ax.hist(realizationMetrics, range=(Min,Max), bins=binNumber, alpha = 0.5, label = labelRealization, edgecolor = "black", color = colorRealization, density= True)
        plt.plot(xr, yr, c = colorRealization, lw=3, label='fit '+labelRealization)
        ax.set_ylabel('Density')
        ax.set_xlabel(label)
    elif plotType == 'box':
        ax.boxplot(sampledMetrics)

    ax.set_title(title)

    plt.legend()
    plt.savefig(outputDir,bbox_inches='tight')

def plotMetricsMethods(metricFractions, realizationMetricFraction, plotType, outputDir, title, metricName, modelNames):
    fig, ax = plt.subplots(figsize=(6.9, 4.8))
    xLimPlot = {'Volume_Parotid_R' : (-8e3, 4.5e3),
                'Volume_Parotid_L' : (-8.2e3, 2.5e3),
                'Volume_Body' : (-6e5, 1e5),
                'Volume_Esophagus' : (-3e3, 1.5e3),
                'CenterOfMassShift_Parotid_R' : (0, 8),
                'CenterOfMassShift_Parotid_L' : (0, 7.3),
                'CenterOfMassShift_Body' : (0, 5.1),
                'CenterOfMassShift_Esophagus' : (0, 6.1)}
    realizationMetrics = []
    # colorRealization = "red"#f"C{1}"
    colorRealization = f"C{1}"
    # colors = ['black', 'green', 'blue']
    colors = {"Image Model": f"C{0}", "DVF Model": f"C{3}", "Hybrid Model": f"C{2}"}
    for i, metricFraction in enumerate(metricFractions):
        sampledMetrics = []
        interval = []
        keys = list(set(realizationMetricFraction.keys()) | set(metricFraction.keys())) 
        keys.sort()
        labelDict = {'Volume': 'Volume Difference [$mm^3$]', 'CenterOfMassShift': 'Distance [mm]', 'RMSE': 'RMSE [HU]'}
        label = labelDict.get(metricName.split('_')[0], metricName)
        ### Plot properties
        binNumber = 30
        labelSample = modelNames[i]
        colorSample = colors[labelSample]#f"C{i}" if i!=1 else f"C{3}"
        labelRealization = 'real CBCTs'
        for key in keys:
            sampledMetrics.append(metricFraction.get(key))
            if i==0:
                realizationMetrics.append(realizationMetricFraction.get(key))  
            if metricFraction.get(key)!=None:
                interval.append(calculate_normal_interval(metricFraction.get(key),0.95))

        if plotType == 'scatter':
            keys = [int(k[5:]) for k in keys]
            interval = np.array(interval)
            minY, maxY = interval[:,0], interval[:,1]
            sampledKeys, sampledMetrics = removeNone(keys, sampledMetrics)
            ax.plot(sampledKeys, np.mean(sampledMetrics,1),'o-', label = labelSample, color = colorSample)
            if i==0:
                ax.plot(keys,realizationMetrics,'o-',color = colorRealization, label = labelRealization, markersize=4)
            ax.fill_between(sampledKeys, minY,  maxY, alpha = 0.2)
            ax.set_xlabel('Fractions')
            ax.set_ylabel(label)
        elif plotType == 'histogram':
            sampledMetrics = [item for sublist in sampledMetrics for item in sublist]
            if i==0:
                realizationMetrics = [item for sublist in realizationMetrics for item in sublist]
            Min = np.quantile(sampledMetrics,0.0005)
            Max = np.quantile(sampledMetrics,0.9995)

            if i==0:
                xr, yr = Density(realizationMetrics, np.min(realizationMetrics), np.max(realizationMetrics))
                ax.hist(realizationMetrics, range=(np.min(realizationMetrics), np.max(realizationMetrics)), bins=binNumber, alpha = 0.5, label = labelRealization, color = colorRealization, edgecolor = "black", density= True)
                plt.plot(xr, yr, '-',c = colorRealization, lw=3, label='fit '+labelRealization)

            xs, ys = Density(sampledMetrics, Min, Max)
            ax.hist(sampledMetrics, range=(Min,Max), bins=binNumber, alpha = 0.2, label = labelSample, color = colorSample, density= True)
            plt.plot(xs, ys, '-',c = colorSample, lw=3, label='fit '+labelSample)

            ax.set_ylabel('Probability density', fontsize=18)
            ax.set_xlabel(label, fontsize=18)
        elif plotType == 'box':
            ax.boxplot(sampledMetrics)

    # ax.set_title(title)
    ax.set_xlim(xLimPlot[metricName])
    ax.tick_params(axis='both', labelsize=15)
    plt.gca().yaxis.get_offset_text().set_fontsize(15)
    plt.gca().xaxis.get_offset_text().set_fontsize(15)
    
    # ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))  
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-3,3)) # Set scientific notation for y-axis ticks
    plt.ticklabel_format(axis='x', style='sci', scilimits=(-3,3)) # Set scientific notation for y-axis ticks
    plt.legend(fontsize=15)
    plt.savefig(outputDir,bbox_inches='tight')


def plotMetricsFromPatientSequence(path_samples: list, path_realization, plotType, mode = 'fractions'):
    realizationPatientSequence = PatientSequence()
    realizationPatientSequence.load(path_realization)
    for path in path_samples:
        modelName = path.split('/')[-4]
        patientSequence = PatientSequence()
        patientSequence.load(path)
        if mode == 'fractions':
            for patient in patientSequence:
                plotWithPatientSequence(patient, realizationPatientSequence[patient.patientID], modelName, plotType, patient.patientID)
        if mode == 'patients':
            plotWithPatientSequence(patientSequence, realizationPatientSequence, modelName, plotType, "allPatients")

def plotWithPatientSequence(sampledSequence,realizationSequence, modelName, plotType, titleKey):
    for metricName, metricFraction in sampledSequence.metrics.items():
        realizationMetricFraction = realizationSequence.metrics[metricName]
        outputDir = os.path.join(getPlotsPath(),'Metrics',metricName, modelName ,titleKey+'_'+metricName+'_'+plotType+'.png')
        makeFoldersMaybe(os.path.dirname(outputDir))
        plotMetrics(metricFraction, realizationMetricFraction, plotType, outputDir, titleKey, metricName)

def plotHistogramFromPatientSequence(path_samples: list, path_realization, modelNames):
    realizationPatientSequence = PatientSequence()
    realizationPatientSequence.load(path_realization)
    methods = []
    for path in path_samples:
        patientSequence = PatientSequence()
        patientSequence.load(path)   
        methods.append(patientSequence)
    for metricName in methods[0].metrics.keys():
        metricFraction = []
        for method in methods:
            metricFraction.append(method.metrics.get(metricName))
        if any(x is None for x in metricFraction):
            continue
        outputDir = os.path.join(getPlotsPath(),'Metrics',metricName, f'Histogram_{metricName}.png')
        makeFoldersMaybe(os.path.dirname(outputDir))
        plotMetricsMethods(metricFraction, realizationPatientSequence.metrics[metricName], 'histogram',outputDir, metricName, metricName, modelNames)


def plotFractionComparison(pathToPSMs, pathRealizations, patientsNames, metricName, outputpath):
    realizationPatientSequence = PatientSequence()
    realizationPatientSequence.load(pathRealizations)
    sampledMetricsDict = {path.split('/')[-4]: {} for path in pathToPSMs}
    fig, axs = plt.subplots(nrows=len(patientsNames),ncols=len(pathToPSMs),figsize=(5 * len(patientsNames),3 * len(patientsNames)))
    yMinMax = {patientID: {'min':1e10, 'max': -1e10} for patientID in patientsNames}
    xmin,xmax = 1e10,-1e10
    modelDict = {'ImageDoseTimeEncodedMixedConditionals4Days': 'Image model', 'DVFDoseWithXstartTimeEncoded': 'DVF model', 'DVFPredicted_XstartTimeEncodedMixedConditionals4days': 'Hybrid model'}
    for modelNumber, path in enumerate(pathToPSMs):
        modelName = path.split('/')[-4]
        patientSequence = PatientSequence()
        patientSequence.load(path)    
        for patientNumber, patientName in enumerate(patientsNames):
            patient = patientSequence[patientName]
            if patientName not in sampledMetricsDict:
                sampledMetricsDict[modelName][patientName] = {}
            metricFraction =  patient.metrics[metricName]
            sampledMetricsList = []
            realizationMetricsList = []
            interval = []
            realizationMetricFraction = realizationPatientSequence[patientName].metrics[metricName]
            keys = list(set(realizationMetricFraction.keys()) | set(metricFraction.keys())) 
            keys.sort()
            for key in keys:
                sampledMetricsList.append(metricFraction.get(key))
                realizationMetricsList.append(realizationMetricFraction.get(key))  
                if metricFraction.get(key)!=None:
                    interval.append(calculate_normal_interval(metricFraction.get(key),0.95))

            realizationMetricsList = np.ravel(realizationMetricsList)
            keyNumbers = [int(k[5:]) for k in keys]
            interval = np.array(interval)
            sampledKeys, sampledMetricsList = removeNone(keyNumbers, sampledMetricsList)
            sampledMetricsList = np.mean(sampledMetricsList, 1)
            minY, maxY = interval[:,0], interval[:,1]
            axs[patientNumber][modelNumber].plot(sampledKeys, sampledMetricsList, 'o-', color = 'C0', label='AI')
            axs[patientNumber][modelNumber].plot(keyNumbers,realizationMetricsList,'o-',color = 'C1', markersize=4, label = 'real CBCTs')
            axs[patientNumber][modelNumber].fill_between(np.repeat(sampledKeys, 2), np.repeat(minY, 2),  np.repeat(maxY, 2),alpha = 0.2)
            yMinMax[patientName]['max'] = updateMaxMin(max(np.max(maxY),np.max(realizationMetricsList)), yMinMax[patientName]['max'], 'max')
            xmax = updateMaxMin(keyNumbers, xmax, 'max')
            yMinMax[patientName]['min'] = updateMaxMin(min(np.min(minY),np.min(realizationMetricsList)), yMinMax[patientName]['min'], 'min')
            xmin = updateMaxMin(keyNumbers, xmin, 'min')
            # axs[patientNumber][modelNumber].set_title(modelDict[modelName]+'-'+patientName, loc='right', pad = 5)
            axs[patientNumber][modelNumber].set_title(modelDict[modelName], pad = 5)
            


        ### Remove thicks and add the same limits for all the plots 
        for ax in axs.reshape(-1): 
            ax.yaxis.set_major_locator(plt.NullLocator())
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.set_xlim((xmin-1,xmax+1))
        patientIDs = list(yMinMax.keys())
        for i in range(len(patientsNames)):
            axs[i][0].yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
            for j in range(len(pathToPSMs)):
                axs[i][j].set_ylim((yMinMax[patientIDs[i]]['min'],yMinMax[patientIDs[i]]['max']))

        for j in range(len(pathToPSMs)):
            axs[-1][j].xaxis.set_major_locator(plt.MaxNLocator(nbins=5))

        ##### Set Scientific Notation Axis
        for ax in axs.reshape(-1):
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-3, 3))  # Adjust the power limits as needed
            ax.yaxis.set_major_formatter(formatter)

        fig.text(0.5, 0.04, 'Fractions', ha='center')
        fig.text(0.04, 0.5, '$\Delta {}$ [$mm^3$]'.format(metricName.split('_')[0]), va='center', rotation='vertical')

        ##### remove duplicates in the labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(),loc='upper right')

        ##### Save figs
        fig.savefig(outputpath,bbox_inches='tight')


def updateMaxMin(Data,value, modality):
    if modality == 'max' and np.max(Data) > value:
        return np.max(Data)
    if modality == 'min' and np.min(Data) < value:
        return np.min(Data)   
    return value


def plotHistogramNormalized(sampledSequences, realizationSequence, modelNames, binNumber = 30):
    Min = {}
    Max = {}
    labelDict = {'Volume': 'Volume Difference [$mm^3$]', 'CenterOfMassShift': 'Distance [mm]', 'RMSE': 'RMSE [HU]'}
    ### Plot properties
    binNumber = 30
    colorSample = f"C{0}"
    colorRealization = f"C{1}"
    labelSample = 'AI'
    labelRealization = 'real CBCTs'

    for sampledSequence in sampledSequences:        
        for metricName, metricPatient in sampledSequence.metrics.items():
            for patientID, metricValues in metricPatient.items():
                Min[metricName] = min([np.min(realizationSequence.metrics[metricName][patientID]), np.min(metricValues), Min.get(metricName, 10000)])
                Max[metricName] = max([np.max(realizationSequence.metrics[metricName][patientID]), np.max(metricValues), Max.get(metricName, -10000)])

    for modelName, sampledSequence in zip(modelNames, sampledSequences):     
        Metrics = {}   
        for metricName, metricPatient in sampledSequence.metrics.items():
            realizationDist = []
            sampleDist = []
            label = labelDict.get(metricName.split('_')[0], metricName)
            for patientID, metricValues in metricPatient.items():
                realizationDist.extend(realizationSequence.metrics[metricName][patientID])
                sampleDist.extend(metricValues)
            Min_metric = Min[metricName]
            Max_metric = Max[metricName]
            realizationDist = (realizationDist - Min_metric)/(Max_metric-Min_metric)
            sampleDist = (sampleDist - Min_metric)/(Max_metric-Min_metric)
            xs, ys = Density(sampleDist, 0, 1)
            xr, yr = Density(realizationDist, 0, 1)
            fig, ax = plt.subplots(figsize=(6.9, 4.8))
            ax.hist(sampleDist, range=(0,1), bins=binNumber, alpha = 0.5, label = labelSample, edgecolor = "black", color = colorSample, density= True)
            plt.plot(xs, ys, c = colorSample, lw=3, label='fit '+labelSample)
            ax.hist(realizationDist, range=(0,1), bins=binNumber, alpha = 0.5, label = labelRealization, edgecolor = "black", color = colorRealization, density= True)
            plt.plot(xr, yr, c = colorRealization, lw=3, label='fit '+labelRealization)
            ax.set_ylabel('Density')
            ax.set_xlabel(label)
            ax.set_title(metricName)
            plt.legend()
            plt.savefig(f'/home/luciano/Codes/DiffuseRT/plots/NormalizedHistograms/{modelName}_{metricName}.png',bbox_inches='tight')