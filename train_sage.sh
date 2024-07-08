#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch --config_file 7_gpu.json --main_process_port 25655 --multi_gpu #--mixed_precision "fp16" tutorial_train.py
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch --num_processes 7 --multi_gpu --mixed_precision "fp16" tutorial_train.py \

export date_now=`date "+%Y-%m-%d"`

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file 8_gpu.json --main_process_port 25655 train_sage.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --data_json_file=None \
  --data_root_path=None \
  --mixed_precision="no" \
  --resolution=512 \
  --train_batch_size=32 \
  --dataloader_num_workers=2 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --output_dir="./experiments/fl_${date_now}" \
  --save_steps=5000
  --class_embedding_path='test_data/fl-class_embeddings.pt'
  --latent_path='test_data/fl-Step3-BFR-e4e'
  --data_root_path='/home/dingguanqi/codes/2d/HAT/results/HAT_GAN_Real_SRx4/visualization/flowers'