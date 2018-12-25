ckpt_dir="/home/gaofeng/workspace/ai_2018fall/final_project/struct2depth/checkpoints/"
kitti_data="/home/gaofeng/datasets/processed_data/kitti/"

python train.py \
  --checkpoint_dir $ckpt_dir \
  --data_dir $kitti_data \
  --architecture resnet \
  --imagenet_norm false \
  --batch_size 1 \
  --handle_motion false
