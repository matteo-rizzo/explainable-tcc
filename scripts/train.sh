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

declare -a models=("att_tccnet" "conf_tccnet" "conf_att_tccnet")
#declare -a data_folders=("tcc_split" "fold_0" "fold_1" "fold_2")
declare -a data_folders=("tcc_split")

for model in "${models[@]}"; do
  for data_folder in "${data_folders[@]}"; do
    python3 train/tcc/train_interpretable_tccnet.py --data_folder "$data_folder" --model_type "$model" || exit
  done
done
