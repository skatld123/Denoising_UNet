import os
import cv2
import copy
import torch
import skimage
import numpy as np
from torchvision import transforms
from random import randint
from PIL import Image

# 추가할 noise의 종류를 선택
# mode = "saltpepper"
# mode = "chromatic"
mode = "gauss"

# Taget Img Dir 설정 
imgDir = '/root/Denoising_UNet/data_noise/data_label/'

# Noise Img가 저장될 Dir 설정 
# dstDir_salt = '/root/Denoising_UNet/data_noise/data_salt'
# dstDir_chro = '/root/Denoising_UNet/data_noise/data_chromatic'
dstDir_gauss = '/root/Denoising_UNet/data_noise/data_gauss'
# dstDir_chro = '/root/Denoising_UNet/data_noise/datasets_denosing/test'

# SaltPepper 노이즈 강도 (1~5)
noise_intensity_salt = 1
# Chromatic 노이즈 강도 (0.01~0.2)
noise_intensity_chro = 0.2
# 가우시안 노이즈의 variance, 높을수록 Noise의 강도가 커지며 기본은 0.01
noise_intensity_gauss = 0.02

def SaltPepper(img, intensity=1):
    # Getting the dimensions of the image
    if img.ndim > 2:  # color
        height, width, _ = img.shape
    else:  # gray scale
        height, width = img.shape
 
    result = copy.deepcopy(img)
 
    # Randomly pick some pixels in the image
    # Pick a random number between height*width/80 and height*width/10
    # 노이즈 강도 조절을 위한 변수 (intensity) 추가 
    number_of_pixels = intensity*randint(int(height * width / 100), int(height * width / 10))
    print(number_of_pixels)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = randint(0, height - 1)
 
        # Pick a random x coordinate
        x_coord = randint(0, width - 1)
 
        if result.ndim > 2:
            result[y_coord][x_coord] = [randint(0, 255), randint(0, 255), randint(0, 255)]
        else:
            # Color that pixel to white
            result[y_coord][x_coord] = 255
 
    # Randomly pick some pixels in image
    # Pick a random number between height*width/80 and height*width/10
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = randint(0, height - 1)
 
        # Pick a random x coordinate
        x_coord = randint(0, width - 1)
 
        if result.ndim > 2:
            result[y_coord][x_coord] = [randint(0, 255), randint(0, 255), randint(0, 255)]
        else:
            # Color that pixel to white
            result[y_coord][x_coord] = 0
 
    return result

def normalization(object, mean=0.5, std=0.5):
    object = (object - mean) / std
    return object

def add_noise(img, intensity):
    noise = torch.rand(img.size())*intensity
    noisy_img = img + noise
    return noisy_img

# https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv

if mode == 'saltpepper':
    imgList = os.listdir(imgDir)
    imgList.sort()
    if imgList:
        for i, img_name in enumerate(imgList):
            print(img_name)
            imgPath = os.path.join(imgDir, img_name)
            img = cv2.imread(imgPath)
            result = SaltPepper(img, noise_intensity_salt)
            name = "salt_person_%03d.jpg" % (i+1)
            resultPath = os.path.join(dstDir_salt, name)
            cv2.imwrite(resultPath, result)
            
elif mode == 'chromatic':
    imgList = os.listdir(imgDir)
    imgList.sort()
    if imgList:
        for i, img_name in enumerate(imgList):
            imgPath = os.path.join(imgDir, img_name)
            img = Image.open(imgPath)
            tensor = transforms.ToTensor()
            tensor_img = tensor(img)

            # add chromatic noise
            img_noisy = add_noise(tensor_img, noise_intensity_chro)
            transform = transforms.ToPILImage()
            result = transform(img_noisy)
            name = "chroma_person_%03d.jpg" % (i+1)
            resultPath = os.path.join(dstDir_chro, name)
            result.save(resultPath)

elif mode == "gauss":
    imgList = os.listdir(imgDir)
    imgList.sort()
    if imgList:
        for i, img_name in enumerate(imgList):
            imgPath = os.path.join(imgDir, img_name)
            img = skimage.io.imread(imgPath)/255.0
            
            noisy = skimage.util.random_noise(img, mode="gaussian", mean=0, var=noise_intensity_gauss)
        
            name = "gauss_person_%03d.jpg" % (i+1)
            resultPath = os.path.join(dstDir_gauss, name)
            skimage.io.imsave(resultPath, noisy)
            
