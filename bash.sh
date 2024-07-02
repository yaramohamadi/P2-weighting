
python /export/livia/home/vision/Ymohammadi/Code/P2-weighting/scripts/image_sample.py --attention_resolutions 16 --class_cond False
 --diffusion_steps 1000 --dropout 0.0 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 128 
 --num_res_blocks 1 --num_head_channels 64 --resblock_updown True --use_fp16 False --use_scale_shift_norm True 
 --timestep_respacing ddim50 --use_ddim True --model_path /export/livia/home/vision/Ymohammadi/Code/ffhq_baseline.pt 
 --sample_dir /export/livia/home/vision/Ymohammadi/Code/results/samples/