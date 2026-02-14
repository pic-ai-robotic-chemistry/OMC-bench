rm -rf msrun_log
mkdir msrun_log
ulimit -n 65535
export ASCEND_RT_VISIBLE_DEVICES=6,7
export PARALLEL_MODE=DATA_PARALLEL
export PYTHONPATH=$(pwd)/sharker:$PYTHONPATH

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_parallel.sh"
echo "=============================================================================================================="

msrun --worker_num=2 --local_worker_num=2 --master_port=8118 --log_dir=msrun_log --join=True --cluster_time_out=300 Train_dist.py


