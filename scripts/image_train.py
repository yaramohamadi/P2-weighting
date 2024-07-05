"""
Train a diffusion model on images.
"""
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 

import argparse
import os 
import copy
import matplotlib.pyplot as plt 
import numpy as np

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

import torch as th

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.log_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    if args.clf_guidance == False:
        logger.log("Loading pretrained model...")
        checkpoint = th.load(args.model_path)
        model.load_state_dict(checkpoint, strict = True)

        model.to(dist_util.dev())
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion) 

        logger.log("creating data loader...")
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
        )

        logger.log("training...")
        TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=args.batch_size,
            microbatch=args.microbatch,
            lr=args.lr,
            ema_rate=args.ema_rate,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            resume_checkpoint=args.resume_checkpoint,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.lr_anneal_steps,
            # Samples
            save_samples_dir=args.save_samples_dir,
            checkpoint_dir=args.checkpoint_dir,
            sample = args.sampling,
            use_ddim=args.use_ddim, # If sampling mid-training
            how_many_samples=args.how_many_samples, # For sampling mid training
            image_size=args.image_size,
        ).run_loop()

    # ________________ CLASSIFIER_FREE_GUIDANCE ______________________
    else:
        for g, g_name in {
            # Fixed
            0.7: '0_7', 0.8: '0_8', 0.9: '0_9', 1: '1', 1.1: '1_1', 1.2: '1_2', 1.3: '1_3',
            # Curved
            #'a0-0_8-1_2', 'a2_5-0_8-1_2', 'a5-0_8-1_2', 'a7_5-0_8-1_2', 
            #'a0-0_8-1', 'a2_5-0_8-1', 'a5-0_8-1', 'a7_5-0_8-1'
            #'a-2_5-0_8-1_2', 'a-5-0_8-1_2', 'a-7_5-0_8-1_2',
            #'a-2_5-0_8-1', 'a-5-0_8-1', 'a-7_5-0_8-1',
            }.items():


            logger.log("Loading pretrained model...")
            checkpoint = th.load(args.model_path)
            model.load_state_dict(checkpoint, strict = True)

            model.to(dist_util.dev())

            pretrained_model = copy.deepcopy(model)
            model.to(dist_util.dev())

            # Fixed
            guidance_scale = np.array([g for _ in range(501)]) # Fixed Line

            schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion) 

            # Directory to save checkpoints in
            checkpoint_dir = f"/export/livia/home/vision/Ymohammadi/Code/results_sketch_guide/{g_name}/checkpoints/"
            # Whenever you are saving checkpoints, a batch of images are also sampled, where to produce these images
            save_samples_dir= f"/export/livia/home/vision/Ymohammadi/Code/results_sketch_guide/{g_name}/samples/"

            logger.log("creating data loader...")
            data = load_data(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                image_size=args.image_size,
                class_cond=args.class_cond,
            )

            pretrained_data = load_data(
                data_dir=args.pretrained_dir,
                batch_size=args.batch_size,
                image_size=args.image_size,
                class_cond=args.class_cond,
            )

            logger.log("training...")
            TrainLoop(
                model=model,
                diffusion=diffusion,
                data=data,
                batch_size=args.batch_size,
                microbatch=args.microbatch,
                lr=args.lr,
                ema_rate=args.ema_rate,
                log_interval=args.log_interval,
                save_interval=args.save_interval,
                resume_checkpoint=args.resume_checkpoint,
                use_fp16=args.use_fp16,
                fp16_scale_growth=args.fp16_scale_growth,
                schedule_sampler=schedule_sampler,
                weight_decay=args.weight_decay,
                lr_anneal_steps=args.lr_anneal_steps,
                # Samples
                save_samples_dir=save_samples_dir,
                checkpoint_dir=checkpoint_dir,
                sample = args.sampling,
                use_ddim=args.use_ddim, # If sampling mid-training
                how_many_samples=args.how_many_samples, # For sampling mid training
                image_size=args.image_size,
                # For classifier-free guidanace
                clf_guidance=args.clf_guidance,
                pretrained_model=pretrained_model,
                guidance_scale=guidance_scale,
                pretrained_data=pretrained_data,
            ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        log_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=25,
        save_interval=25,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        # Added
        use_ddim=False, # If sampling mid-training
        how_many_samples=50, # For sampling mid training
        checkpoint_dir='',
        save_samples_dir='',
        model_path='',
        sampling=True,
        # Next 2 for guidance
        clf_guidance=False,
        pretrained_dir=''
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
