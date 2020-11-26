# git stash && git pull && python automl/server_full_eval.py

# bash run_automl.sh server_full_eval 8
# bash run_automl.sh server_full_eval_ppi 1
# start server:
# python automl/tune_params.py

# start workers 
# $1 which file to start
# $2 how many workers on each gpu
total_gpus=`nvidia-smi -q -d PIDS |grep "GPU 00" | wc  -l` 
total_gpus=$(($total_gpus - 1))
rm  log/experiments/elliptic/orig_gcn/baseline/*
for i in $(seq 1 $2); do
    for i in $(seq 0 $total_gpus); do
        echo "automl/$1.py --worker --gpu $i"
        python automl/$1.py --worker --gpu $i & 
        sleep .05
    done
done

