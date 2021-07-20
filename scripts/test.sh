pwd
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:~/home/matteo/Projects/interpretable-tcc

declare path_to_script=test/tcc/test_itccnet.py

# Values: "att_tccnet" "conf_tccnet" "conf_att_tccnet"
declare -a models=("att_tccnet")

# Values: "tcc_split" "fold_0" "fold_1" "fold_2"
declare -a data_folders=("tcc_split")

# Values: "" "spat" "temp"
declare -a modes=("spat")

for model in "${models[@]}"; do
  for data_folder in "${data_folders[@]}"; do
    for mode in "${modes[@]}"; do
      python3 $path_to_script --data_folder "$data_folder" --model_type "$model" --use_train_set --deactivate "$mode" || exit
      python3 $path_to_script --data_folder "$data_folder" --model_type "$model" --deactivate "$mode" || exit
    done
  done
done
