import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T

# label 이미지 디렉토리 지정 (person_0001.png가 들어있는 DIR)
labelDirPath = "/root/Denoising_UNet/data_noise/data_label"
# noise 이미지 디렉토리 지정 (gauss_person_0001.png가 들어있는 DIR)
noiseDirPath = "/root/Denoising_UNet/data_noise/data_gauss"

# 훈련 데이터 디렉토리 지정 (실제 학습에 사용하기 위해 이미지를 나눌 DIR)
dir_train = "/root/Denoising_UNet/datasets_denosing/train"
dir_val = "/root/Denoising_UNet/datasets_denosing/val"
dir_test = "/root/Denoising_UNet/datasets_denosing/test"

# 중요! 1_add_noise.py 과정을 통해 노이즈가 추가된 이미지의 이름을 적어준다.
# ex) gauss_person, salt_person, chroma_person
noise_image_name = 'gauss_person'

# label, input(noise) 이미지 개수 출력
labelList = os.listdir(labelDirPath)
noiseList = os.listdir(noiseDirPath)
lst_train = os.listdir(dir_train)


# 넘파이 배열로 변환 후 저장
def png2numpy(dir_data):

    lst_data = os.listdir(dir_data)

    if lst_data:
        for i, filename in enumerate(lst_data):
            name, ext = os.path.splitext(filename)
            if ext == ".jpg" or ext == ".png":
                img_path = os.path.join(dir_data, filename)
                img = Image.open(img_path)

                # 이미지 리사이징을 하고 512X512 저장하기
                transform = T.Resize((512, 512))
                resize_img = transform(img)

                np_path = os.path.join(dir_data, name + ".npy")
                np_img = np.asarray(resize_img)
                np.save(np_path, np_img)
    else:
        print(dir_data + "의 경로에 numpy로 변환할 이미지가 없습니다.")


def splitDataset(dir_data):
    lst_train = os.listdir(dir_data)
    lst_train.sort()

    lst_label = [f for f in lst_train if f.startswith('label')]
    lst_input = [f for f in lst_train if f.startswith('input')]
    
    lst_label.sort()
    lst_input.sort()
    
    cnt_val = int(len(lst_label) * 0.2) 
    cnt_test = int(len(lst_label) * 0.1) 

    print("train : %d, val : %d, test : %d" % ((len(lst_train)-(cnt_val + cnt_test)), cnt_val, cnt_test))

    lst_val_label = lst_label[-cnt_val:]
    lst_val_input = lst_input[-cnt_val:]
    lst_val = lst_val_label + lst_val_input
    
    lst_test_label = lst_label[-(cnt_val + cnt_test) : -cnt_val]
    lst_test_input = lst_input[-(cnt_val + cnt_test) : -cnt_val]
    lst_test = lst_test_label + lst_test_input
    
    del lst_label[-(cnt_val + cnt_test):]
    del lst_input[-(cnt_val + cnt_test):]
    
    lst_train = lst_label + lst_input
    
    lst_train.sort() 
    
    print(lst_train)
    print(lst_val)
    print(lst_test)

    return lst_train, lst_val, lst_test


def changeDataName(lst_train):
    # 데이터 이름 바꾸기
    if lst_train:
        lst_label = [f for f in lst_train if f.startswith('person')]
        print(lst_label)
        lst_input = [f for f in lst_train if f.startswith(noise_image_name)]
        print(lst_input)
        
        if lst_label:
            lst_label.sort()
            lst_input.sort()

            if len(lst_label) == len(lst_input):
                for i, filename in enumerate(lst_label):
                    file = os.path.join(dir_train, filename)
                    nf_name = "label_%03d.png" % (i + 1)
                    newfile = os.path.join(dir_train, nf_name)
                    os.rename(file, newfile)

                for i, filename in enumerate(lst_input):
                    file = os.path.join(dir_train, filename)
                    nf_name = "input_%03d.png" % (i + 1)
                    newfile = os.path.join(dir_train, nf_name)
                    os.rename(file, newfile)
            else:
                print("라벨와 노이즈 이미지의 개수가 다릅니다.")
        else:
            print("이미 이름이 input과 label로 바뀐 데이터 입니다.")
            return

    else:
        print("트레인 데이터가 없습니다.")


# 라벨 파일을 Dataset Train으로 이동
if not lst_train:
    if labelList:
        for filename in labelList:
            file = os.path.join(labelDirPath, filename)
            copyfile = os.path.join(dir_train, filename)
            shutil.copy(file, copyfile)
        print("label 데이터를 이동했습니다")
    else : 
        print("label 데이터가 없습니다.")
    # 노이즈 데이터를 Dataset Train으로 이동
    if noiseList:
        for filename in noiseList:
            file = os.path.join(noiseDirPath, filename)
            copyfile = os.path.join(dir_train, filename)
            shutil.copy(file, copyfile)
        print("input(노이즈) 데이터를 이동했습니다")
    else : 
        print("input(노이즈) 데이터가 없습니다.")

lst_train = os.listdir(dir_train)

changeDataName(lst_train)

lst_train, lst_val, lst_test = splitDataset(dir_train)

# train에서 7:2:1 비율로 val, test Dir로 분배
if lst_train:
    for filename in lst_val:
        file = os.path.join(dir_train, filename)
        shutil.move(file, dir_val)
    print("train -> val로 총 %04d 데이터를 이동했습니다" % len(lst_val))

    for filename in lst_test:
        file = os.path.join(dir_train, filename)
        shutil.move(file, dir_test)
    print("train -> test로 총 %04d 데이터를 이동했습니다" % len(lst_test))

png2numpy(dir_train)
png2numpy(dir_val)
png2numpy(dir_test)

