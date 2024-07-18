import os,shutil,tempfile,json
import pickle 


def getCodeBasePath():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def getPlotsPath():
    return os.path.join(getCodeBasePath(),'plots')

def getCurrentPlotPath():
    return os.path.join(getPlotsPath(),'currentPlot.png')


def makeFolderMaybe(folderPath):
    if not os.path.isdir(folderPath):
        os.mkdir(folderPath)

def makeFoldersMaybe(folderPath):
    if not os.path.isdir(folderPath):
        os.makedirs(folderPath)

def makeFolderMaybeRecursive(folderPath, maxDepth=2):
    if not os.path.isdir(folderPath):
        if maxDepth > 0:
            makeFolderMaybeRecursive(os.path.split(folderPath)[0], maxDepth - 1)
            makeFolderMaybe(folderPath)
        else:
            raise RecursionError


def createTemporaryCopy(path):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    shutil.copy2(path, tmp.name)
    return tmp.name


def readJson(path):
    with open(path, 'r') as json_file:
        data = json.load(json_file)
    return data


def listDirAbsolute(dirPath):
    return [os.path.join(dirPath, filename) for filename in listDirRelative(dirPath)]

def listDirRelative(dirPath):
    return os.listdir(dirPath)

def savePickle(file, path):
    with open(path, 'wb') as f:
        pickle.dump(file, f)

def loadPickle(path):
    with open(path, 'rb') as f:
        loaded = pickle.load(f)   
    return loaded
