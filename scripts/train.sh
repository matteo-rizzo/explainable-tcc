#! /bin/bash
#SBATCH -t 2-00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100l:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24    # There are 24 CPU cores on P100 Cedar GPU nodes
#SBATCH --mem=0               # Request the full memory of the node
#SBATCH --account=def-conati

cd ..

#module load python/3.6

source venv/bin/activate

#export PYTHONPATH=$PYTHONPATH:~/projects/def-conati/marizzo/xai/interpretable-tcc
export PYTHONPATH=$PYTHONPATH:~/home/matteo/Projects/interpretable-tcc

python3 train/tcc/train_interpretable_tccnet.py --model_type "att_tccnet" || exit
python3 train/tcc/train_interpretable_tccnet.py --model_type "conf_tccnet" || exit
python3 train/tcc/train_interpretable_tccnet.py --model_type "conf_att_tccnet" || exit
