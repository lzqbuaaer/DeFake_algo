import sys
import math
import numpy as np

from PIL import Image
from scipy import signal
from sklearn.cluster import KMeans
import cv2

def estimate_noise(I):
    H, W = I.shape

    M = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]

    sigma = np.sum(np.sum(np.absolute(signal.convolve2d(I, M))))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

    return sigma

def detect(input, blockSize=32):
    im = Image.open(input)
    im = im.convert('1')

    blocks = []

    imgwidth, imgheight = im.size

    # break up image into NxN blocks, N = blockSize
    for i in range(0,imgheight,blockSize):
        for j in range(0,imgwidth,blockSize):
            box = (j, i, j+blockSize, i+blockSize)
            b = im.crop(box)
            a = np.asarray(b).astype(int)
            blocks.append(a)

    variances = []
    for block in blocks:
        variances.append([estimate_noise(block)])

    kmeans = KMeans(n_clusters=2, random_state=0).fit(variances)
    center1, center2 = kmeans.cluster_centers_

    detected = False
    if abs(center1 - center2) > .4:
        detected = True
    if not detected:
        return False, None

    img = cv2.imread(input)
    labels = kmeans.labels_
    for idx, label in enumerate(labels):
        if label == 1: 
            y = (idx // (imgwidth // blockSize)) * blockSize
            x = (idx % (imgwidth // blockSize)) * blockSize
            cv2.rectangle(img, (x, y), (x+blockSize, y+blockSize), (0, 255, 0), 2)

    return detected, img

