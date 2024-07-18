import torch
import numpy as np
import torch.nn.functional as nnf


def transformVectorarray(transformationMatrix, vectorArray):
    return (
        torch.matmul(transformationMatrix[:3, :3], vectorArray.reshape(3, -1))
        + transformationMatrix[:3, 3:4]
    ).reshape(vectorArray.shape)


def getNormalToWorldMatrix(voxelToWorldMatrix,shape):
        return np.matmul(
            voxelToWorldMatrix, getNormalToVoxelMatrix(shape)
        )

def getNormalToVoxelMatrix(shape):
    normalToVoxelMatrix = np.identity(4)
    for i in range(3):
        # DO WE NEED TO DO -1 here? I think so...
        normalToVoxelMatrix[i, i] = (shape[i] - 1) / 2
        normalToVoxelMatrix[i, -1] = (shape[i] - 1) / 2
    return normalToVoxelMatrix

def getWorldCoordinates(shape,normalToWorldMatrix,device):
    return getGrid(normalToWorldMatrix, shape, device)


def getGrid(normalToSystemMatrix, shape, device):
    normalToSystemMatrix = normalToSystemMatrix[[2, 1, 0, 3]]
    normalToSystemMatrix = normalToSystemMatrix[:, [2, 1, 0, 3]]
    normalToSystemMatrix = toTorch(normalToSystemMatrix).to(device).float()

    worldGrid = nnf.affine_grid(
        normalToSystemMatrix[:3, :].unsqueeze(0),
        [1, 1, *shape],
        align_corners=True,
    )[0]
    worldGrid = torch.permute(worldGrid, (3, 0, 1, 2))
    worldGrid = worldGrid[[2, 1, 0]]
    return worldGrid


def toNumpy(arr):
    if isinstance(arr, np.ndarray):
        return arr
    if isinstance(arr, torch.Tensor):
        if arr.is_cuda:
            arr = arr.cpu()
        return arr.numpy()
    else:
        return np.array(arr)


def toTorch(arr):
    if isinstance(arr, np.ndarray):
        return torch.from_numpy(arr)
    if isinstance(arr, torch.Tensor):
        return arr
    return torch.tensor(arr)