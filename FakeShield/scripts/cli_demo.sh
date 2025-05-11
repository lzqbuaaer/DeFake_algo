WEIGHT_PATH=./weight/fakeshield-v1-22b
IMAGE_PATH=./playground/images/Sp_D_CND_A_sec0056_sec0015_0282.jpg
DTE_FDM_OUTPUT=./playground/DTE-FDM_output.jsonl
MFLM_OUTPUT=./playground/MFLM_output

pip install -q transformers==4.37.2  > /dev/null 2>&1
CUDA_VISIBLE_DEVICES=3 \
python -m llava.serve.cli \
    --model-path  ${WEIGHT_PATH}/DTE-FDM \
    --DTG-path ${WEIGHT_PATH}/DTG.pth \
    --image-path ${IMAGE_PATH} \
    --output-path ${DTE_FDM_OUTPUT}

pip install -q transformers==4.28.0  > /dev/null 2>&1
CUDA_VISIBLE_DEVICES=3 \
python ./MFLM/cli_demo.py \
    --version ${WEIGHT_PATH}/MFLM \
    --DTE-FDM-output ${DTE_FDM_OUTPUT} \
    --MFLM-output ${MFLM_OUTPUT}
