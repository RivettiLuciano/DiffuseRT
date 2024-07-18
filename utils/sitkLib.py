import SimpleITK as sitk
import numpy as np
import os

def getSITKImage(img, origin, spacing):
    img = np.squeeze(img)
    img_sitk = sitk.GetImageFromArray(np.transpose(img))
    img_sitk.SetOrigin(origin)
    img_sitk.SetSpacing(spacing)    
    return img_sitk

def getSITKDVF(dvf, origin, spacing):
    dvf_sitk = getSITKImage(dvf, origin, spacing)
    dvf_sitk= sitk.Cast(dvf_sitk,sitk.sitkVectorFloat64)
    dvf_sitkTransform = sitk.DisplacementFieldTransform(dvf_sitk)
    return dvf_sitkTransform

def invertDVF(dvf, origin, spacing):
    dvf_sitk = getSITKImage(dvf, origin, spacing)
    dvf_sitk= sitk.Cast(dvf_sitk,sitk.sitkVectorFloat64)
    dvf_sitkTransform = sitk.DisplacementFieldTransform(dvf_sitk)
    dvf_inverse_sitk = sitk.InvertDisplacementField(dvf_sitkTransform.GetDisplacementField())
    return dvf_inverse_sitk

def createDir(dir):
  if not os.path.isdir(dir):
    os.makedirs(dir)

def warpSITK(img, dvf, defaultValue, interpolator = sitk.sitkBSpline):
   dvf_sitk= sitk.Cast(dvf,sitk.sitkVectorFloat64)
   dvf_sitkTransform = sitk.DisplacementFieldTransform(dvf_sitk)
   return sitk.Resample(img, img, dvf_sitkTransform, interpolator, defaultValue, img.GetPixelID())

def saveSITKImage(itk_img, path, type = None):
  if type:
    itk_img = sitk.Cast(itk_img, type)
  createDir(os.path.dirname(path))
  ifw = sitk.ImageFileWriter()
  ifw.SetFileName(path)
  ifw.SetUseCompression(True)
  ifw.Execute(itk_img)

def getDVFJacobian(sitkDVF):
  JC = sitk.DisplacementFieldJacobianDeterminant(sitkDVF)
  JC_array = np.transpose(sitk.GetArrayFromImage(JC))
  return JC_array

def translationRegistration(fixed_image,moving_image):

  initial_transform = sitk.TranslationTransform(3)

  registration_method = sitk.ImageRegistrationMethod()


  registration_method.SetMetricAsMeanSquares()
  registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
  registration_method.SetMetricSamplingPercentage(0.50)

  registration_method.SetInterpolator(sitk.sitkLinear)

  # Optimizer settings.
  registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=300, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
  registration_method.SetOptimizerScalesFromPhysicalShift()

  registration_method.SetInitialTransform(initial_transform, inPlace=False)
  parameters = registration_method.Execute(fixed_image,moving_image)
  return np.array(parameters.GetParameters())