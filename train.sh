 srun -p optimal --gres=gpu:1 --quotatype=auto -J bz8 python -u umt_train.py \
                    --dataset visdroneDay --dataset_t visdroneNight --net res101 \
                    --s 2 --epochs 20 --lr 0.0001 --disp_interval 1000 >> train_DayBlack_s2.out