import argparse

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    takeModelAndDiffusionArguments,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
import os 
import yaml

from improved_diffusion.unet import UNetModel
import torch

args = argparse.ArgumentParser()
dist_util.setup_dist()
args.add_argument('--configFile', type=str, help='Path to the config file')
fake_args = ['--configFile', 'experiments/configDoseXstart.yaml']
configFile = args.parse_args(fake_args)

with open(configFile.configFile) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


os.environ["GPU_NUMBER"] = dist_util.getGPUID()
device = dist_util.dev()

img = torch.rand((2,3,96,96,96)).to(device)
diffusionStep = torch.Tensor([300,350]).to(device)
fractionStep = torch.Tensor([2,3]).to(device)
model = UNetModel(
        in_channels=3,
        model_channels=32,
        out_channels=2,
        num_res_blocks=2,
        attention_resolutions=tuple([8,4,2]),
        dropout=0,
        channel_mult=[1, 2, 4, 8],
        num_classes=None,
        use_checkpoint=False,
        num_heads=8,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        scaleFractionShiftDiffStepEncoded = False,
        scaleAndShifFractionEncoded = False
    ).to(device)

model.eval()
with torch.no_grad():
    model.forward(img,diffusionStep,fractionStep)
# model.forward(img,diffusionStep,fractionStep)


