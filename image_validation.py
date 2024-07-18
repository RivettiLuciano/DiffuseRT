import argparse

from sqlalchemy import Over

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    takeModelAndDiffusionArguments,
    add_dict_to_argparser,
)
from improved_diffusion.validation_util import RunValidation
import os 
from utils.config import readConfigAndAddDefaults
from utils.ioUtils import readJson

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--configFile', type=str, help='Path to the config file')
    # fake_args = ['--configFile', 'experiments/DVF/configDVFPredicted_XstartTimeEncodedMixedCond.yaml']
    # fake_args = ['--configFile', 'experiments/DVF/configDVFDose_XstartTimeEncoded.yaml']
    fake_args = ['--configFile', 'experiments/Image/configDoseTimeEncodedMixedConditionals4Days.yaml']
    if args.parse_args().configFile != None:
        parse = args.parse_args()
    else:
        parse = args.parse_args(fake_args)

    # configFile = 'experiments/DVF/configDVFPredicted_XstartTimeEncoded.yaml'
    config = readConfigAndAddDefaults(parse.configFile)


    # adjust config variables for validation
    config['timestep_respacing'] = config['timestep_respacing_validation']
    config['batch_size'] = 1
    config['image_size'] = config['sampling_image_size']
    model, diffusion = create_model_and_diffusion(**config)

    data = load_data(
        config,
        mode = config['validation_mode'],
    )

    model.load_state_dict(
            dist_util.load_state_dict(logger.getModelPath(config['logging_path'],config['model_for_sampling']), map_location="cpu")
        )
    os.environ["GPU_NUMBER"] = dist_util.getGPUID()
    dist_util.setup_dist()
    model.to(dist_util.dev())
    model.eval()

    valMetricDict = readJson(config['valJsonDir'])
    scores = RunValidation(
        model,
        diffusion,
        data,
        50,
        config,
        overWrite = False,
        valMetricDict=valMetricDict
    ).calculateScores()
    print(scores)

if __name__ == "__main__":
    main()
