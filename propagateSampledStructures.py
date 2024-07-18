from improved_diffusion.validation_util import propagateStructures, getPatientSequence


# path = '/home/luciano/ResearchData/DataSets/DiffuseRTData/DVFDoseWithXstartTimeEncoded/Evaluation/ManySamples_128x128x96'
path = '/home/luciano/ResearchData/DataSets/DiffuseRTData/ImageDoseTimeEncodedMixedConditionals4Days/Evaluation/ManySamples100'
dataMainDir = '/home/luciano/ResearchData/DataSets/DL-DIR/DataSet/DeepMovement'
patientSequence = getPatientSequence(path)
# patientSequence.patients = {'HN_11': patientSequence.patients['HN_11']}

propagateStructures(patientSequence, ['Body', 'Parotid_L', 'Parotid_R', 'Esophagus'], dataMainDir, (96,96,96), False) 

