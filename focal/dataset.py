import os
import numpy as np

dataset = '/root/lzq/dataset/compound'
res = []
for root, dirs, files in os.walk(dataset):
    # print(root, dirs, files)
    forgery = ['figure.png', 'figure_v1.png', 'figure_v2.png', 'figure_v3.png']
    forgery_gt = 'figure_forgery_gt.png'
    forgery_files = [f for f in forgery if os.path.exists(os.path.join(root, f))]

    if forgery_files:
        for forgery_file in forgery_files:
            res.append((os.path.join(root, forgery_file), os.path.join(root, forgery_gt)))

np.save('flist/dataset.npy', np.array(res))