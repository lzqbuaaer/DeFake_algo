import argparse
from PIL import Image, ExifTags, ImageChops
import os
import random
import cv2
import re
from datetime import datetime
import numpy as np

from ForgeryDetection import Detect
import double_jpeg_compression
import noise_variance
import copy_move_cfa
import pdb
import json

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.environ['OPENBLAS_NUM_THREADS'] = '4'

# Function to load and resize image
def getImage(path, width, height):
    """
    Function to return an image as a PhotoImage object
    :param path: A string representing the path of the image file
    :param width: The width of the image to resize to
    :param height: The height of the image to resize to
    :return: The image represented as a PhotoImage object
    """
    img = Image.open(path)
    img = img.resize((width, height), Image.ANTIALIAS)
    return img

def sift_dbscan_detect(path):
    # User has not selected an input image
    if path is None:
        print("Please select image")
        return

    detect = Detect(path)
    key_points, descriptors = detect.siftDetector()
    forgery, detected = detect.locateForgery(30, 2)
    ret = {"result" : detected, "url" : ""}

    if not detected:
        print("Original image, no forgery detected")
    else:
        print("Forgery detected, saving the forged image...")
        name = os.path.basename(path)
        date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
        output = os.path.join(output_path, 'copy_move_forgery')
        if not os.path.exists(output):
            os.makedirs(output)
        new_file_name = os.path.join(output, date + '-' + name)
        ret["url"] = new_file_name
        cv2.imwrite(new_file_name, forgery)
        print(f'Forgery saved as {new_file_name}')
    return ret

def metadata_analysis(path):
    if path is None:
        print("Please select image")
        return None

    ret = {"result" : False, "url" : ""}
    img = Image.open(path)
    img_exif = img.getexif()

    if img_exif is None:
        print("No metadata found")
    else:
        print("Metadata details:")
        ret["result"] = True
        for key, val in img_exif.items():
            if key in ExifTags.TAGS:
                print(f'{ExifTags.TAGS[key]} : {val}')
        
        name = os.path.basename(path)
        name, _ = os.path.splitext(name)
        name += ".txt"
        date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
        output = os.path.join(output_path, 'metadata_analysis')
        if not os.path.exists(output):
            os.makedirs(output)
        detail_file = os.path.join(output, date + '-' + name)
        
        with open(detail_file, 'w') as f:
            for key, val in img_exif.items():
                if key in ExifTags.TAGS:
                    f.write(f'{ExifTags.TAGS[key]} : {val}\n')
        print(f"Metadata saved to {detail_file}")
        ret["url"] = detail_file
    return ret

def noise_variance_inconsistency(path):
    if path is None:
        print("Please select image")
        return
    ret = {"result" : False}

    noise_forgery, img = noise_variance.detect(path)

    if noise_forgery:
        ret["result"] = True
        print("Noise variance inconsistency detected")
        name = os.path.basename(path)
        date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
        output = os.path.join(output_path, 'noise_variance_inconsistency')
        if not os.path.exists(output):
            os.makedirs(output)
        detail_file = os.path.join(output, date + '-' + name)
        cv2.imwrite(detail_file, img)
        print(f'Img saved as {detail_file}')
        ret["url"] = detail_file
    else:
        print("No noise variance inconsistency detected")
    return ret

def copy_move_detect(path):
    if path is None:
        print("Please select image")
        return

    ret = {}
    opt = copy_move_cfa.Opt()
    identical_regions_cfa, img = copy_move_cfa.detect(path, opt)
    ret["result"] = identical_regions_cfa

    if identical_regions_cfa:
        print(f"copy_move detected: {identical_regions_cfa}")
        name = os.path.basename(path)
        date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
        output = os.path.join(output_path, 'copy_move_detect')
        if not os.path.exists(output):
            os.makedirs(output)
        detail_file = os.path.join(output, date + '-' + name)
        img.save(detail_file)
        ret["url"] = detail_file
    else:
        print("No copy_move detected")
    return ret

def ela_analysis(path):
    TEMP = 'temp.jpg'
    SCALE = 10

    if path is None:
        print("Please select image")
        return

    original = Image.open(path)
    original.save(TEMP, quality=90)
    temporary = Image.open(TEMP)

    diff = ImageChops.difference(original, temporary)
    d = diff.load()
    WIDTH, HEIGHT = diff.size
    for x in range(WIDTH):
        for y in range(HEIGHT):
            d[x, y] = tuple(k * SCALE for k in d[x, y])
    # diff.show()
    name = os.path.basename(path)
    date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
    output = os.path.join(output_path, 'ela_analysis')
    if not os.path.exists(output):
        os.makedirs(output)
    detail_file = os.path.join(output, date + '-' + name)
    diff.save(detail_file)
    return {"url" : detail_file}

def jpeg_Compression(path):
    if path is None:
        print("Please select image")
        return

    double_compressed, peak = double_jpeg_compression.detect(path)
    ret = {"result" : double_compressed, "peak" : peak}
    if double_compressed:
        print("Double compression detected")
    else:
        print("Single compression detected")
    return ret

def image_decode(path):
    if path is None:
        print("Please select image")
        return
    
    # Encrypted image
    img = cv2.imread(path)
    width = img.shape[0]
    height = img.shape[1]
    
    # img1 and img2 are two blank images
    img1 = np.zeros((width, height, 3), np.uint8)
    img2 = np.zeros((width, height, 3), np.uint8)
    
    for i in range(width):
        for j in range(height):
            for l in range(3):
                v1 = format(img[i][j][l], '08b')
                v2 = v1[:4] + chr(random.randint(0, 1)+48) * 4
                v3 = v1[4:] + chr(random.randint(0, 1)+48) * 4
                
                # Appending data to img1 and img2
                img1[i][j][l] = int(v2, 2)
                img2[i][j][l] = int(v3, 2)

    cv2.imwrite('output.png', img2)
    print("Image extraction complete, saved as output.png")

def string_analysis(path):
    if path is None:
        print("Please select image")
        return
    
    with open(path, "rb") as f:
        n = 0
        b = f.read(16)
        while b:
            s1 = " ".join([f"{i:02x}" for i in b])  # hex string
            s1 = s1[0:23] + " " + s1[23:]
            s2 = "".join([chr(i) if 32 <= i <= 127 else "." for i in b])
            print(f"{n * 16:08x}  {s1:<48}  |{s2}|")
            n += 1
            b = f.read(16)

def process_image(image_path, detections):
    print(f"Processing image: {image_path}")
    outcome = {}

    detection_functions = {
        "sift_dbscan_detect": sift_dbscan_detect,
        "metadata_analysis": metadata_analysis,
        "noise_variance_inconsistency": noise_variance_inconsistency,
        "copy_move_detect": copy_move_detect,
        "ela_analysis": ela_analysis,
        "jpeg_compression": jpeg_Compression,
        "image_decode": image_decode,
        "string_analysis": string_analysis
    }

    for detection in detections:
        if detection in detection_functions:
            ret = detection_functions[detection](image_path)
            outcome[detection] = ret
            print(f"Detection method {detection} finished.")
        else:
            print(f"Detection method {detection} is not valid.")
    
    return outcome

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose image manipulation detection methods")
    parser.add_argument("-input", "--input", help="Path to the image to be processed", required=True)
    parser.add_argument(
        "-d", "--detections", nargs="+", help="List of detection methods to apply", 
        choices=["sift_dbscan_detect", "metadata_analysis", "noise_variance_inconsistency", 
                 "copy_move_detect", "ela_analysis", "jpeg_compression", "image_decode", "string_analysis"],
        required=True
    )
    args = parser.parse_args()
    outcome = process_image(args.input, args.detections)
    json_outcome = json.dumps(outcome, ensure_ascii=False, indent=4)

    print("===============================================")
    print(json_outcome)
