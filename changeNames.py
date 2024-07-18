import glob
import os

path = '/home/luciano/ResearchData/DataSets/DiffuseRTData/DVFDoseWithXstartTimeEncoded/Evaluation/ManySamples_128x128x96'

dirs = glob.glob(os.path.join(path,'sample_*'))

for dir in dirs:
    os.system(f'mv {dir} {dir.replace("sample_","sampleDVF_")}')