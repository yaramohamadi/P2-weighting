

FFHQ_FLGAS='--attention_resolutions 16 --class_cond False --diffusion_steps 1000 --dropout 0.0 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 1 --resblock_updown True --use_fp16 False --use_scale_shift_norm True'
FFHQ_TRAIN_FLAGS='--lr 2e-5 --batch_size 10 --rescale_learned_sigmas True --p2_gamma 1 --p2_k 1'
DDIM='--timestep_respacing ddim50 --use_ddim True'

# SAMPLE
# MODEL_PATH = '--model_path /export/livia/home/vision/Ymohammadi/Code/ffhq_baseline.pt'
# SAMPLE_DIR = '--sample_dir /export/livia/home/vision/Ymohammadi/Code/results/samples/'
# python /export/livia/home/vision/Ymohammadi/Code/P2-weighting/scripts/image_sample.py $FFHQ_FLGAS $DDIM $MODEL_PATH $SAMPLE_DIR 

# TRAIN
LOG_DIR='--log_dir /export/livia/home/vision/Ymohammadi/Code/results_cat10/logs'
DATA_DIR='--data_dir /export/livia/home/vision/Ymohammadi/Dataset/cat10/'
CHKPT_DIR='--checkpoint_dir /export/livia/home/vision/Ymohammadi/Code/results_cat10/checkpoints/'
SAVE_SAMPLES_DIR='--save_samples_dir /export/livia/home/vision/Ymohammadi/Code/results_cat10/samples/'
MODEL_PATH='--model_path /export/livia/home/vision/Ymohammadi/Code/ffhq_baseline.pt'
TRAIN_FLAGS='--sampling True --how_many_samples 5154'
CUDA_VISIBLE_DEVICES="2" python /export/livia/home/vision/Ymohammadi/Code/P2-weighting/scripts/image_train.py $FFHQ_FLGAS $FFHQ_TRAIN_FLAGS $DDIM $TRAIN_FLAGS $MODEL_PATH $LOG_DIR $DATA_DIR $CHKPT_DIR $SAVE_SAMPLES_DIR 

# For classifier-free guidanace
# pretrained_model=args.pretrained_model,
# guidance_scale=args.guidance_scale,
# pretrained_data=args.pretrained_data,