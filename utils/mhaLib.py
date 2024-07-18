import numpy as np
import medpy.io



def readMha(filepath):
    mha= medpy.io.load(filepath)
    return getMHAPixelArray(mha), getMHAVoxelToWorldMatrix(mha)


def getMHAPixelArray(mha):
    array = mha[0]
    dim = len(array.shape)
    if dim == 4:
        array = np.transpose(array, axes=[3, 0, 1, 2])
    return array


def getMHAVoxelToWorldMatrix(mha):
    header = mha[1]
    voxelSpacing = np.array(header.spacing)
    imagePosition = np.array(header.offset)
    imageOrientation = header.direction
    # assert np.allclose(imageOrientation,np.identity(3))
    orientationMatrix = imageOrientation * voxelSpacing
    voxelToWorldMatrix = np.identity(4)
    voxelToWorldMatrix[:3, :3] = orientationMatrix
    voxelToWorldMatrix[:3, 3] = imagePosition
    return voxelToWorldMatrix



def writeMha(filename, pixelArray,voxelToWorldMatrix=np.identity(4)):
    # can also write mhd file
    spacing = tuple(np.abs(getVoxelSpacing(voxelToWorldMatrix)))
    orientation = getOrientation(voxelToWorldMatrix, spacing)
    offset = tuple(getOrigin(voxelToWorldMatrix))
    header = medpy.io.Header(spacing=spacing, offset=offset)
    header.set_direction(orientation)
    array = pixelArray
    if array.shape[0] != 1:
        if len(array.shape) == 4:
            array = np.transpose(array, axes=[1, 2, 3, 0])
        medpy.io.save(array, filename, hdr=header)
    else:
        medpy.io.save(array[0], filename, hdr=header)



def getVoxelSpacing(voxelToWorldMatrix):
    return np.diagonal(voxelToWorldMatrix)[:3]


def getOrientation(voxelToWorldMatrix, spacing):
    return voxelToWorldMatrix[:3, :3] / spacing


def getOrigin(voxelToWorldMatrix):
    return voxelToWorldMatrix[:3, 3]

def getVoxelToWorldMatrix(voxelSpacing=np.array([1.,1.,1.]),origin=np.array([0.,0.,0.])):
    vtwMatrix = np.identity(4)
    for i in range(3):
        vtwMatrix[i, i] = voxelSpacing[i]
    vtwMatrix[:3, 3] = origin
    return vtwMatrix
