import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 读取原图和mask
image_path = 'figure_forgery.png'  # 替换为原图路径
mask_path = 'figure_forgery_gt.png'    # 替换为mask图片路径

image = Image.open(image_path).convert('RGBA')
mask = Image.open(mask_path).convert('L')  # 将mask转换为灰度图

# 将图片和mask转换为NumPy数组
image_np = np.array(image, dtype=np.float32)  # 使用更高精度的float32
mask_np = np.array(mask)

# 设置更透明的红色 (RGBA)，减小alpha值使得红色更透明
red_color_with_alpha = [255, 0, 0, 50]  # 红色 (R:255, G:0, B:0, A:50)

# 创建一个新的输出图像
output_image = image_np.copy()

# 对mask区域应用更透明的红色
for i in range(3):  # 对RGB通道进行修改
    output_image[:, :, i] = np.where(mask_np == 255, 
                                      np.clip((output_image[:, :, i] * (255 - red_color_with_alpha[3]) + red_color_with_alpha[i] * red_color_with_alpha[3]) / 255, 0, 255),
                                      output_image[:, :, i])

# 将输出图像转换回uint8类型
output_image = np.clip(output_image, 0, 255).astype(np.uint8)

# 使用matplotlib展示结果
# plt.imshow(output_image)
# plt.axis('off')  # 不显示坐标轴
# plt.show()
Image.fromarray(output_image).save('output_image.png')
