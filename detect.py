import requests
import os
import json
import logging as logger
import multiprocessing
import time
import argparse
import subprocess
import sys
import json
import shutil
import cv2
import numpy as np

# from focal.main import detect_folder
from alloca_gpu import get_max_free_memory_gpu
from utils.deepseek import translate
from traditional_method_detection.ForgeryDetection import Detect

logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s] %(message)s',
                   datefmt='%m-%d %H:%M:%S')
INPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input')
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

def get_detect_list():
    url = "http://122.9.35.116:8080/api/detection/all"
    payload={}
    headers = {
    'jwtToken': 'magicToken'
    }
    # response = requests.request("GET", url, headers=headers, data=payload)
    try:
        response = requests.request("GET", url, headers=headers, data=payload)
    except requests.exceptions.RequestException as e:
        print(f"发生错误: {e}")
        return []
    response = response.json()
    if response['code'] != 0:
        logger.error("获取检测队列失败")
        return []
    else:
        detect_list = response['data']
        if not detect_list:
            logger.info("队列为空")
        return detect_list

def upload_figure(output_path):
    url = "http://122.9.35.116:8080/api/file/upload"
    payload={}
    files=[
        ('files',(os.path.basename(output_path), open(output_path,'rb'),'application/octet-stream'))
    ]
    headers = {
        'jwtToken': 'magicToken'
    }
    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    print(response)
    print(response.text)
    response = response.json()
    if response['code'] != 0:
        logger.error("上传失败")
        return None
    else:
        return response['data']
    
# def build_image_text_dict(jsonl_file):
#     image_text_map = {}
#     with open(jsonl_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             if line.strip():
#                 obj = json.loads(line)
#                 image = obj.get('image')
#                 outputs = obj.get('outputs')
#                 if image is not None and outputs is not None:
#                     image_text_map[image] = outputs
#     return image_text_map
def build_dict_by_first_key(jsonl_file):
    """
    读取 jsonl 文件，将每行 JSON 对象的第一个键的值作为主字典的 key，
    其余键值对组成 value（子字典）返回。

    Args:
        jsonl_file (str): jsonl 文件路径

    Returns:
        dict: { first_key_value: {other_key: other_value, ...}, ... }
    """
    result = {}
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict) or len(obj) == 0:
                continue

            # 获取第一个键
            first_key = next(iter(obj))
            first_val = obj[first_key]

            # 剩余键值对
            rest = {k: v for k, v in obj.items() if k != first_key}

            # 将第一个键的值作为外层 key，其它组成子字典
            result[first_val] = rest

    return result

def download(url, path, basename):
    response = requests.get(url)
    if response.status_code == 200:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, basename), 'wb') as f:
            f.write(response.content)
        return True
    else:
        logger.error('Fail to get figure')
        return False

def detect_focal(input_path, uuid):
    output_path = os.path.join(OUTPUT_PATH, 'focal', uuid)
    os.makedirs(output_path, exist_ok=True)
    gpu_index, free_mem = get_max_free_memory_gpu()
    if free_mem > 7500:
        logger.info(f"focal 分配 GPU {gpu_index}，剩余 {free_mem} MB")
        # p = multiprocessing.Process(target=detect_folder, args=(input_path, output_path, gpu_index))
        result = subprocess.run(
            ['bash', '/home/zyw/DeFake_algo/focal/run.sh', input_path, output_path, str(gpu_index)],
            capture_output=True,
            text=True
        )
        print(result.stdout) 
        print(result.stderr) 
        rates = np.load(os.path.join(output_path, 'rates.npy'))
        return output_path, list(rates)
    else:
        logger.error("focal 未分配到 GPU")
        return None, None
    
def detect_fakeshield(input_path, uuid):
    output_path = os.path.join(OUTPUT_PATH, 'fakeshield', uuid)
    os.makedirs(output_path, exist_ok=True)
    gpu_index, free_mem = get_max_free_memory_gpu()
    if free_mem > 35000:
        logger.info(f"fakeshield 分配 GPU {gpu_index}，剩余 {free_mem} MB")
        subprocess.run(f"bash /home/zyw/DeFake_algo/FakeShield/scripts/test.sh {uuid} {gpu_index}", shell=True, capture_output=True, text=True)
        return output_path
    else:
        logger.error("fakeshield 未分配到 GPU")
        return None

def detect_sift_dbscan(input_path, uuid, eps=100, min_sample=2):
    output_path = os.path.join(OUTPUT_PATH, 'sift_dbscan', uuid)
    os.makedirs(output_path, exist_ok=True)
    detecteds = []
    for file in os.listdir(input_path):
        detect = Detect(os.path.join(input_path, file))
        key_points, descriptors = detect.siftDetector()
        if len(key_points) == 0:
            cv2.imwrite(os.path.join(output_path, 'sift_' + file), detect.image)
            detecteds.append(False)
        else:
            forgery, detected = detect.locateForgery(eps, min_sample)
            if detected:
                cv2.imwrite(os.path.join(output_path, 'sift_' + file), forgery)
            else:
                cv2.imwrite(os.path.join(output_path, 'sift_' + file), detect.image)
            detecteds.append(detected)
    return output_path, detecteds
            

def upload_result(uuid, figure_id, result):
    upload_url = 'http://122.9.35.116:8080/api/detection/result'
    result_list = []
    if 'focal' in result:
        result_list.append({
            "algorithmName": "Contrastive Clustering",
            "description": "Contrastive Clustering 算法检测结果",
            "paramName1": "置信度",
            "paramValue1": result['focal']['rate'],
            "effectImageURL1": result['focal']['upload_url'][0],
            "effectImageExplanation1": "可疑区域标记图(高敏感度)",
            "effectImageURL2": result['focal']['upload_url'][1],
            "effectImageExplanation2": "可疑区域标记图(中敏感度)",
            "effectImageURL3": result['focal']['upload_url'][2],
            "effectImageExplanation3": "可疑区域标记图(低敏感度)",
        })

    if 'fakeshield' in result:
        result_list.append({
            "algorithmName": "DeFake 造假检测大模型",
            "description": result['fakeshield']['analysis'],
            "paramName1": "造假概率",
            "paramValue1": result['fakeshield']['score'],
            "effectImageURL1": result['fakeshield']['upload_url'],
            "effectImageExplanation1": "造假区域标记图（红色区域）",
        })

    if 'sift' in result:
        result_list.append({
            "algorithmName": "DBSCAN 聚类算法",
            "description": result['sift']['output'],
            "paramName1": "造假概率",
            "paramValue1": result['sift']['rate'],
            "effectImageURL1": result['sift']['upload_url'],
            "effectImageExplanation1": "造假区域标记图",
        })

    payload = json.dumps({
        "taskUUID": uuid,
        "figureId": figure_id,
        "listOfResults": result_list
    })
    headers = {
        'jwtToken': 'magicToken',
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", upload_url, headers=headers, data=payload)
    response = response.json()
    print(response)
    if response['code'] == 0:
        logger.info(f'UPLOAD success : uuid={uuid} figure_id={figure_id}')
    else:
        logger.error(f'UPLOAD failure : uuid={uuid} figure_id={figure_id}')

def renew(uuid, finish_num):
    url = "http://122.9.35.116:8080/api/detection"
    payload = json.dumps({
        "uuid": uuid,
        "numOfFinishedFigure": finish_num
    })
    headers = {
        'jwtToken': 'magicToken',
        'Content-Type': 'application/json'
    }
    response = requests.request("PUT", url, headers=headers, data=payload)
    response = response.json()
    if response['code'] == 0:
        logger.info(f'RENEW success : uuid={uuid}')
    else:
        logger.error(f'RENEW failure : uuid={uuid}')

def detect():
    detect_list = get_detect_list()
    for data in detect_list:
        uuid = data['uuid']
        figure_num = data['numOfAllFigure'] 
        id_list = data['figureIdList']
        url_list = data['figureURLList']
        for url in url_list:
            basename = os.path.basename(url)
            input_path = os.path.join(INPUT_PATH, uuid)
            res = download(url, input_path, basename)
            if res:
                logger.info(f'GET figure : uuid={uuid} path={os.path.join(input_path, basename)}')
        input_path = os.path.join(INPUT_PATH, uuid)
        
        # focal detect
        focal_path, focal_rates = detect_focal(input_path, uuid)
        if focal_path != None:
            logger.info(f'OUTPUT figure : uuid={uuid} path={focal_path}')
        else:
            return

        # sift_dbscan detect
        sift_path, sift_ret = detect_sift_dbscan(input_path, uuid)
        logger.info(f'OUTPUT figure : uuid={uuid} path={sift_path}')


        for i in range(figure_num):
            basename = os.path.basename(url_list[i])
            result = {}

            focal_result = {}
            focal_upload_url = []
            for threshold in [50, 60, 65]:   
                focal_upload_url.append(upload_figure(os.path.join(focal_path, 'focal_threshold=' + str(threshold) + '_' + basename)))
            focal_result['upload_url'] = focal_upload_url
            focal_result['rate'] = focal_rates[i]
            result['focal'] = focal_result

            sift_result = {}
            sift_upload_url = upload_figure(os.path.join(sift_path, 'sift_' + basename))
            sift_result['upload_url'] = sift_upload_url
            if sift_ret[i]:
                sift_result['output'] = '检测出造假痕迹，已在图中标出'
                sift_result['rate'] = 0.75
            else:
                sift_result['output'] = '未检测出造假痕迹'
                sift_result['rate'] = 0.25
            result['sift'] = sift_result
            upload_result(uuid, id_list[i], result)

        renew(uuid, figure_num // 3)
        # fakesheild detect
        fakeshield_path = detect_fakeshield(input_path, uuid)
        if fakeshield_path != None:
            logger.info(f'OUTPUT figure : uuid={uuid} path={fakeshield_path}')
            jsonl_path = f'/home/zyw/DeFake_algo/output/fakeshield/{uuid}/DTE-FDM_output.jsonl'
            backup_jsonl_path = f'/home/zyw/DeFake_algo/output/fakeshield/{uuid}/backup_DTE-FDM_output.jsonl'
            shutil.copy(jsonl_path, backup_jsonl_path)
            renew(uuid, figure_num * 3 // 5)
            translate(jsonl_path)
            fakeshield_analysis = build_dict_by_first_key(jsonl_path)
        else:
            return

        for i in range(figure_num):
            basename = os.path.basename(url_list[i])
            result = {}
            fakeshield_result = {}
            fakeshield_upload_url = upload_figure(os.path.join(fakeshield_path, 'fakeshield_' + os.path.splitext(basename)[0] + '.png'))
            fakeshield_result['upload_url'] = fakeshield_upload_url
            fakeshield_result['score'] = f"{float(fakeshield_analysis[input_path + '/' + basename]['score'])/100}"
            fakeshield_result['analysis'] = fakeshield_analysis[input_path + '/' + basename]['outputs']
            result['fakeshield'] = fakeshield_result
            upload_result(uuid, id_list[i], result)
            
        renew(uuid, figure_num)

            
            

if __name__ == '__main__':
    interval = 60
    if interval == -1:
        detect()
    else:
        while True:
            detect()
            time.sleep(interval)


    