from skimage.measure import label
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import numpy as np

def getBodyMask(
    image,
    threshold=-400,
):
    nDim = len(image.shape)
    mask = np.squeeze(image < threshold)
    mask = binary_erosion(mask, iterations=2, border_value=1)
    mask = floodFill(mask)
    mask = binary_dilation(mask, iterations=2)

    mask = binary_erosion(~mask, iterations=1)
    mask = floodFill(mask)
    mask = binary_dilation(mask, iterations=2)
    while len(mask.shape)<nDim:
        mask = np.expand_dims(mask,0)
    return mask


def floodFill(mask, percentage=0.6):
    labels = label(mask)
    binSizes = np.bincount(labels.flat)[1:]
    indices = np.argsort(binSizes)[::-1]
    if len(indices) == 0:
        return mask
    maskTmp = labels == indices[0] + 1
    mask = np.copy(maskTmp)
    if len(indices) < 2:
        return mask
    sizeInitial = binSizes[indices[0]]
    i = 1
    maskTmp = labels == indices[i] + 1
    while binSizes[indices[i]] > percentage * sizeInitial:
        mask[maskTmp] = 1
        if len(indices) <= i + 1:
            return mask
        i += 1
        maskTmp = labels == indices[i] + 1
    return mask


def floodFillMultipleMasks(mask, numberOfMasks=2):
    labels = label(mask)
    binSizes = np.bincount(labels.flat)[1:]
    indices = np.argsort(binSizes)[::-1]
    masks = []
    for i in range(numberOfMasks):
        if i < len(indices):
            masks.append(labels == indices[i] + 1)
        else:
            masks.append(np.zeros_like(labels, dtype=np.bool))
    return np.stack(masks)