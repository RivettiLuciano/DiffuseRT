import torch.nn.functional as nnf
import numpy as np
import torch
from utils.transformations import *
DEFAULT_DEVICE = 'cuda:0'

def resampleScan(
    pixelArray,
    voxelToWorldMatrix,
    targetVoxelToWorldMatrix,
    targetShape,
    device=DEFAULT_DEVICE,
    defaultPixelValue=0.0,
    mode="bilinear",
):
    assert len(pixelArray.shape) == 4 # we need a channel dimension for this fuction to run
    # more general formulation of an interpolation with the possibility of having non alignment in the corners
    alignCorners = True
    pixelArray = pixelArray.to(device)
    sourceArray = pixelArray.unsqueeze(0).float() - defaultPixelValue
    targetNormalToVoxelMatrix = getNormalToVoxelMatrix(targetShape)
    sourceNormalToVoxelMatrix = getNormalToVoxelMatrix(pixelArray.shape[1:])

    targetNormalToWorldMatrix = np.matmul(
        targetVoxelToWorldMatrix, targetNormalToVoxelMatrix
    )
    sourceNormalToWorldMatrix = np.matmul(
        voxelToWorldMatrix, sourceNormalToVoxelMatrix
    )
    transformationMatrix = np.matmul(
        np.linalg.inv(sourceNormalToWorldMatrix), targetNormalToWorldMatrix
    )
    # Torch thinks things are z,y,x, Stupid, took 2 days to debug...
    transformationMatrix2 = transformationMatrix[[2, 1, 0, 3]]
    transformationMatrix = transformationMatrix2[:, [2, 1, 0, 3]]

    transformationMatrix = toTorch(transformationMatrix).to(device).float()

    targetGrid = nnf.affine_grid(
        transformationMatrix[:3, :].unsqueeze(0),
        [1, 1, *targetShape],
        align_corners=alignCorners,
    )
    newArray = (
        nnf.grid_sample(sourceArray, targetGrid, mode=mode, align_corners=alignCorners)
        + defaultPixelValue
    )
    
    return newArray[0],targetVoxelToWorldMatrix


def warpScanWithVectorField(
    scan, scanVtwMatrix, vectorField, vectorFieldVtwMatrix,defaultPixelValue=0.0
):
    # VECTORFIELD SHOULD BE IN MM
    device = scan.device
    assert len(scan.shape) == 4 and len(vectorField.shape) == 4
    scanShape = scan.shape[1:]
    vfShape = vectorField.shape[1:]

    worldTransformation = (
        getWorldCoordinates(vfShape,getNormalToWorldMatrix(vectorFieldVtwMatrix,vfShape),device)
        + toTorch(vectorField).to(device)
    ).float()

    worldToNormalMatrix = (
        toTorch(np.linalg.inv(getNormalToWorldMatrix(scanVtwMatrix,scanShape))).to(device).float()
    )
    normalTransformation = transformVectorarray(
        worldToNormalMatrix, worldTransformation
    )
    normalTransformation = torch.permute(
        normalTransformation[[2, 1, 0]], (1, 2, 3, 0)
    ).unsqueeze(0)
    newArray = nnf.grid_sample(
        scan.float().unsqueeze(0)- defaultPixelValue,
        normalTransformation,
        align_corners=True,
        mode="bilinear",
    )[0]+defaultPixelValue

    return newArray,vectorFieldVtwMatrix











