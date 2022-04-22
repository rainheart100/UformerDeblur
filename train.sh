# Uformer16
# all patch_size is 128
python3 ./train.py --arch Uformer --batch_size 64 --val_interval 1 --gpu '0,1,2,3' \
    --train_ps 128 --train_dir ../MIMO-UNet/dataset/GOPRO/train --env 16_0701_1 \
    --val_dir ../MIMO-UNet/dataset/GOPRO/test --embed_dim 16 --warmup

# Uformer32
# python3 ./train.py --arch Uformer --batch_size 32 --gpu '0,1' \
#     --train_ps 128 --train_dir /cache/SIDD/train --env 32_0701_1 \
#     --val_dir /cache/SIDD/val --embed_dim 32 --warmup

    
# UNet
# python3 ./train.py --arch UNet --batch_size 32 --gpu '0,1' \
#     --train_ps 128 --train_dir /cache/SIDD/train --env 32_0701_1 \
#     --val_dir /cache/SIDD/val --embed_dim 32 --warmup

