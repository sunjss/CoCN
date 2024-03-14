cuda_num='0'

python train.py --cuda_num $cuda_num --nbatch 1 --testmode 'test/' --dataset 'CORNELL' \
        --lr 1e-4 --epoch 500 --nTlayer 0 --nlayer 1 --nblock 3 \
        --filter_size 5 --stride 5 --nh 10 --d_model 128 --dropout 0.5