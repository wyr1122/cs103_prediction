import os
import cv2
from PIL import Image
import numpy as np


def cut(img_name, img_result_dir):
    crop_height = 512  ###***********滑窗裁剪的图片高度***************
    crop_width = 512  ###***********滑窗裁剪的图片宽度***********
    stride = 512  ###***********滑窗裁剪的步长***********
    overlap = int(crop_width - stride)  ###前一张裁剪图和当前裁剪图的重叠区域
    # img_predict_dir = 'D:\\CS103\\VV_imgs\\'  ###***********要进行测试的图片所在的文件夹***********
    # img_result_dir = 'D:\\CS103\\VV_imgs_div\\'  ###***********测试结果保存位置***********
    if not os.path.exists(img_result_dir):  # 文件夹不存在，则创建
        os.mkdir(img_result_dir)
    num = 1

    loaded_image = cv2.imread(img_name)  ##cv2形式读取图片格式为BGR 后面要进行image格式转换，否则测试叠掩的三通道合成图会出问题

    img_h = loaded_image.shape[0]  ##cv2模式读取图片的高用img.shape[0] image.open()读取图片的高度显示用Img.height,宽度用img.width
    img_w = loaded_image.shape[1]  ##cv2模式读取图片的宽用img.shape[1]
    row = int((img_h - crop_height) / stride + 1)  ##高可以切为row块
    column = int((img_w - crop_width) / stride + 1)  ##宽可以切为cloumn块
    res_row = img_h - (crop_height + stride * (row - 1))  ##高度滑窗切分完row块后还剩余的部分
    res_column = img_w - (crop_width + stride * (column - 1))  ##宽度度滑窗切分完row块后还剩余的部分
    if (img_h - crop_height) % stride > 0:  ##判断剩余高度是否还可以继续进行滑窗切分
        row = row + 1
    if (img_w - crop_width) % stride > 0:  ##判断剩余宽度是否还可以继续进行滑窗切分
        column = column + 1
    counter = 1  ##起始从第一块开始

    for i in range(row):
        for j in range(column):
            if i == row - 1:  ##判断高度是否切到最后一块
                H_start = img_h - crop_height  ##如果是，最后一块的起始高度就是图片的高度往前数小图片要切分的高度crop_height
                H_end = img_h  ##结束高度就是图片的高的数字
            else:
                H_start = i * stride  ##如果不是切到最后一块，切分小图的起始高点就是第i块（第i次切分）*步长
                H_end = H_start + crop_height  ##结束高点就是 在开始的基础上直接加上crop_height
            if j == column - 1:
                W_start = img_w - crop_width
                W_end = img_w
            else:
                W_start = j * stride
                W_end = W_start + crop_width

            img_chip = loaded_image[H_start:H_end, W_start:W_end]  ##切块的小图片 nudarray格式
            image = Image.fromarray(cv2.cvtColor(img_chip, cv2.COLOR_BGR2RGB))
            s = '%04d' % num  # 04表示0001,0002等命名排序
            image.save(img_result_dir + str(s) + '.jpg')  # **********注意图片格式********************#
            num = num + 1
    return img_h, img_w


def get_data(path_name):
    img_names = os.listdir(path_name)
    img_names.sort(key=lambda x: int(x[:-4]))
    T1 = []
    num = len(img_names)
    for i in range(num):
        img_name = img_names[i]
        img = cv2.imread(path_name + img_name, cv2.IMREAD_UNCHANGED)
        T1.append(img)
    T1 = np.array(T1)
    return T1


def Mosaic(Path, Save_Path, CropSize, RepetitionSize, height, width):
    # height = 6344
    # width = 10704
    rows = height // CropSize
    cols = width // CropSize
    rem_rows = height % CropSize != 0
    rem_cols = width % CropSize != 0
    result = np.zeros((height, width, 3), float)
    # result[:, :, 0] = np.ones([height, width]) * 255
    imgs = get_data(Path)
    # imgs = cv2.imread(Path)
    imgs = np.squeeze(imgs)
    n = 0
    for i in range(int((height - RepetitionSize) / (CropSize - RepetitionSize))):
        for j in range(int((width - RepetitionSize) / (CropSize - RepetitionSize))):
            result[int(i * (CropSize - RepetitionSize)): int(i * (CropSize - RepetitionSize)) + CropSize,
            int(j * (CropSize - RepetitionSize)): int(j * (CropSize - RepetitionSize)) + CropSize] = imgs[n]
            n = n + 1
        if rem_cols:
            n = n + 1
    # 拼接最后一列
    n = cols
    for i in range(int((height - RepetitionSize) / (CropSize - RepetitionSize))):
        result[int(i * (CropSize - RepetitionSize)): int(i * (CropSize - RepetitionSize)) + CropSize,
        (width - CropSize): width] = imgs[n]
        n = n + cols + 1
    # 拼接最后一行
    n = rows * (cols + 1)
    for j in range(int((width - RepetitionSize) / (CropSize - RepetitionSize))):
        result[(height - CropSize): height,
        int(j * (CropSize - RepetitionSize)): int(j * (CropSize - RepetitionSize)) + CropSize] = imgs[n]
        n = n + 1
    # 拼接最后一个小块
    result[(height - CropSize): height,
    (width - CropSize): width] = imgs[n]
    result1 = result.astype(np.uint8)
    image = Image.fromarray(cv2.cvtColor(result1, cv2.COLOR_BGR2RGB))
    image.save(Save_Path + 'result.jpg')
    # m = (rows + rem_rows) * (cols + rem_cols)
    # for n in range(m):
    #     path = os.path.join(Path, os.listdir(Path)[0])
    #     os.remove(path)
