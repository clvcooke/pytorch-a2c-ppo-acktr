kernprof -l main.py --algo ppo --lr 3e-4 --gamma 0.995 --use-gae --tau 0.95 --num-mini-batch 64 --ppo-epoch 10 --num-steps 2048 --env-name skelefactor_montecarlo --save-interval 5 --num-frames 2049
