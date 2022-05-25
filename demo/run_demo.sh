#!/bin/bash
set -e
# set -x

if [ ! -f utils.py ]; then
    echo "Downloading utils.py from the SuperGlue repo."
    echo "We cannot provide this file directly due to its strict licence."
    wget https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/models/utils.py
fi

# Use webcam 0 as input source. 
#input=0
# or use a pre-recorded video given the path.
input=/data/pantianbo/ptb_LoFTR/demo/new.mp4
# Toggle indoor/outdoor model here.
#model_ckpt=../weights/indoor_ds.ckpt
#model_ckpt=../logs/tb_logs/indoor-ds-bs=4/version_2/checkpoints/epoch=3-auc@5=0.000-auc@10=0.000-auc@20=0.000.ckpt
model_ckpt=../weights/outdoor_ds.ckpt
output_dir=/data/pantianbo/ptb_LoFTR/demo/output:_ImagePairs
output_matches_dir=/data/pantianbo/ptb_LoFTR/demo/output:_MatchingPairs

# Optionally assign the GPU ID.
# export CUDA_VISIBLE_DEVICES=0

echo "Running LoFTR demo.."
eval "$(conda shell.bash hook)"
python demo_loftr.py --weight $model_ckpt --input $input --output_dir $output_dir --output_matches_dir $output_matches_dir
# To save the input video and output match visualizations.
# python demo_loftr.py --weight $model_ckpt --input $input --save_video --save_input 

# Running on remote GPU servers with no GUI.
# Save images first.
# python demo_loftr.py --weight $model_ckpt --input $input --no_display --output_dir="./demo_images/"
# Then convert them to a video.
# ffmpeg -framerate 15 -pern_type glob -i '*.png' -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
