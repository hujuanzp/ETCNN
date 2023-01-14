export CUDA_VISIBLE_DEVICES=0,1


# etcnn train
python main.py --model ETCNN --dir_data ../../dataset-regular --scale 1 --compressed_rate 6.25% --n_GPUs 2 --patch_size 48 --rgb_range 1 --batch_size 16 --n_feats 64 --save ../etcnn/etcnn_6.25% --data_test DIV2K --data_range 1-800/801-810 --decay 200-400-600-800 --res_scale 1 --data_range 1-800 --epoch 1000 --n_resgroups 10 --n_resblocks 20 --chunk_size 144 --n_hashes 4 --chop --ext sep_reset
