pwd
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:~/home/matteo/Projects/interpretable-tcc

declare -a models=("att_tccnet" "conf_tccnet" "conf_att_tccnet")
declare -a data_folders=("tcc_split" "fold_0" "fold_1" "fold_2")

for model in "${models[@]}"; do
  for data_folder in "${data_folders[@]}"; do
    python3 test/tcc/test_itccnet.py --data_folder "$data_folder" --model_type "$model" || exit
    python3 test/tcc/test_itccnet.py --data_folder "$data_folder" --model_type "$model" --use_training_set "$false" || exit
  done
done
