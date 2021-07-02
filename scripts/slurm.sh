#SBATCH -t 2-00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100l:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --account=def-conati

module load python/3.6

export PYTHONPATH=$PYTHONPATH:~/projects/def-conati/marizzo/xai/interpretable-tcc