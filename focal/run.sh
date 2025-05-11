input_path=$1
output_path=$2
gpu_index=$3

conda activate defake
cd /home/zyw/DeFake_algo/focal

python /home/zyw/DeFake_algo/focal/main.py --input_path ${input_path} --output_path ${output_path} --gpu_index ${gpu_index}