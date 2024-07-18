"""
Train a diffusion model on images.
"""

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

from utils.config import readConfigAndAddDefaults

def main():
    dist_util.setup_dist()
    configFile = 'experiments/DVF/configDVFPredicted_XstartTimeEncodedlr0.00001.yaml'
    config = readConfigAndAddDefaults(configFile)

    logger.configure(config['logging_path'])

    logger.log("creating model and diffusion...")

    os.environ["GPU_NUMBER"] = dist_util.getGPUID()
    print(os.environ["GPU_NUMBER"])

    model, diffusion = create_model_and_diffusion(**config)
    # print(model)

    print(dist_util.dev())
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(config['schedule_sampler'], diffusion)

    logger.log("creating data loader...")
    data = load_data(
        config,
        mode = 'training',
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=config['batch_size'],
        microbatch=config['microbatch'],
        lr=config['lr'],
        ema_rate=config['ema_rate'],
        log_interval=config['log_interval'],
        save_interval=config['save_interval'],
        resume_checkpoint=config['resume_checkpoint'],
        use_fp16=config['use_fp16'],
        fp16_scale_growth=config['fp16_scale_growth'],
        schedule_sampler = schedule_sampler,
        weight_decay=config['weight_decay'],
        lr_anneal_steps=config['lr_anneal_steps'],
    ).run_loop()




if __name__ == "__main__":
    main()
