WEIGHT_PATH=/home/zyw/DeFake_algo/FakeShield/weight/fakeshield-v1-22b
IMAGE_PATH=/home/zyw/DeFake_algo/input/$1
QUESTION_PATH=/home/zyw/DeFake_algo/output/fakeshield/$1/eval_questions.jsonl
DTE_FDM_OUTPUT=/home/zyw/DeFake_algo/output/fakeshield/$1/DTE-FDM_output.jsonl
MFLM_OUTPUT=/home/zyw/DeFake_algo/output/fakeshield/$1

conda activate defake
cd /home/zyw/DeFake_algo/FakeShield

pip install -q transformers==4.37.2  > /dev/null 2>&1
CUDA_VISIBLE_DEVICES=$2 \
python /home/zyw/DeFake_algo/FakeShield/scripts/eval_jsonl.py \
    --image-path ${IMAGE_PATH}  \
    --output-path ${QUESTION_PATH} 2>/dev/null

CUDA_VISIBLE_DEVICES=$2 \
python /home/zyw/DeFake_algo/FakeShield/DTE-FDM/llava/eval/model_vqa.py \
    --model-path ${WEIGHT_PATH}/DTE-FDM  \
    --DTG-path ${WEIGHT_PATH}/DTG.pth \
    --question-file ${QUESTION_PATH} \
    --image-folder / \
    --answers-file ${DTE_FDM_OUTPUT} 2>/dev/null

pip install -q transformers==4.28.0  > /dev/null 2>&1
CUDA_VISIBLE_DEVICES=$2 \
python /home/zyw/DeFake_algo/FakeShield/MFLM/test.py \
    --version ${WEIGHT_PATH}/MFLM \
    --DTE-FDM-output ${DTE_FDM_OUTPUT} \
    --MFLM-output ${MFLM_OUTPUT} 2>error.txt #2>/dev/null

