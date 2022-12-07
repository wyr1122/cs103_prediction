# ----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# ----------------------------------------------------#
import shutil
import time

import cv2
import numpy as np
from PIL import Image

import os
from unet import Unet
from cnt import classes_nums

if __name__ == "__main__":
    # -------------------------------------------------------------------------#
    #   如果想要修改对应种类的颜色，到__init__函数里修改self.colors即可
    # -------------------------------------------------------------------------#
    unet = Unet()
    # -------------------------------------------------------------------------#
    #   count               指定了是否进行目标的像素点计数（即面积）与比例计算
    #   name_classes        区分的种类，和json_to_dataset里面的一样，用于打印种类和数量
    #
    #   count、name_classes仅在mode='predict'时有效
    # -------------------------------------------------------------------------#
    count = True
    # name_classes    = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    name_classes = ["background", "water"]
    # ----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    # ----------------------------------------------------------------------------------------------------------#
    video_path = 0
    video_save_path = ""
    video_fps = 25.0
    # ----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #   
    #   test_interval和fps_image_path仅在mode='fps'有效
    # ----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path = "img/street.jpg"
    # -------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #   
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    # -------------------------------------------------------------------------#
    origin_path = "input\\"
    dir_save_path = "result\\"
    temp_path = "temp\\"
    temp_save_path = "temp_save\\"
    # -------------------------------------------------------------------------#
    #   simplify            使用Simplify onnx
    #   onnx_save_path      指定了onnx的保存路径
    # -------------------------------------------------------------------------#
    simplify = True
    onnx_save_path = "model_data/models.onnx"

    from tqdm import tqdm
    import process
    import shutil

for filename in os.listdir(origin_path):
    name = origin_path + filename  ###图片名字
    img_h, img_w = process.cut(name, temp_path)

    img_names = os.listdir(temp_path)
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path = os.path.join(temp_path, img_name)
            image = Image.open(image_path)
            r_image = unet.detect_image(image, True)
            if not os.path.exists(temp_save_path):
                os.makedirs(temp_save_path)
            r_image.save(os.path.join(temp_save_path, img_name), quality=95)

    shutil.rmtree(temp_path)
    if not os.path.exists(dir_save_path):
        os.makedirs(dir_save_path)
    process.Mosaic(temp_save_path, dir_save_path, 512, 0, img_h, img_w)
    shutil.rmtree(temp_save_path)

    print('-' * 63)
    print("|%25s | %15s | %15s|" % ("Key", "Value", "Ratio"))
    print('-' * 63)
    total_points_num = classes_nums.sum()
    for i in range(2):
        if classes_nums[i] > 0:
            ratio = classes_nums[i] / total_points_num * 100
            print("|%25s | %15s | %14.2f%%|" % (str(name_classes[i]), str(classes_nums[i]), ratio))
            print('-' * 63)
    # print("classes_nums:", classes_nums)
