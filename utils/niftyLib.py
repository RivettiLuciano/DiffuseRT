import numpy as np
import nibabel

def readNifti(filepath):
    nifti= nibabel.load(filepath)
    return getPixelArray(nifti), getVoxelToWorldMatrix(nifti)


def getPixelArray(nifti):
    array = nifti.get_fdata()
    array = np.squeeze(array)  # get rid of time dimension
    if len(array.shape) == 4:
        array = np.transpose(array, [3, 0, 1, 2])
    return array


def getVoxelToWorldMatrix(nifti):
    if hasattr(nifti,'get_affine'):
        voxelToWorldMatrix = nifti.get_affine()
    else:
        voxelToWorldMatrix = nifti.get_qform()
    voxelToWorldMatrix[0, :] = -voxelToWorldMatrix[0, :]  # inverted convention?
    voxelToWorldMatrix[1, :] = -voxelToWorldMatrix[1, :]
    return voxelToWorldMatrix
