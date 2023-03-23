import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T

## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        lst_label_0 = [f for f in lst_data if f.startswith('label')]
        lst_input_0 = [f for f in lst_data if f.startswith('input')]

         # 넘파이 배열로 변환 후 저장
        if lst_data :
            for i, filename in enumerate(lst_label_0):
                name, ext = os.path.splitext(filename)
                if ext == ".jpg" or ext == ".png" :
                    img_path = os.path.join(data_dir, filename)
                    img = Image.open(img_path)

                    # 이미지 리사이징을 하고 512X512 저장하기
                    transform = T.Resize((512, 512))
                    resize_img = transform(img)

                    np_path = os.path.join(data_dir, name + ".npy")
                    np_img = np.asarray(resize_img)
                    np.save(np_path, np_img)

            for i, filename in enumerate(lst_input_0):
                name, ext = os.path.splitext(filename)
                if ext == ".jpg" or ext == ".png" :
                    img = Image.open(os.path.join(data_dir, filename))
                    np_img = np.asarray(img)

                    # 이미지 리사이징을 하고 512X512 저장하기
                    transform = T.Resize((512, 512))
                    resize_img = transform(img)

                    np_path = os.path.join(data_dir, name + ".npy")
                    np_img = np.asarray(resize_img)
                    np.save(np_path, np_img)

        lst_label = [f for f in lst_data if f.startswith('label') and f.endswith('.npy')]
        lst_input = [f for f in lst_data if f.startswith('input') and f.endswith('.npy')]

        # lst_label = [f for f in lst_data if f.startswith('label')]
        # lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        label = label/255.0
        input = input/255.0

        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)

        return data


## 트렌스폼 구현하기
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
    
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']
        # 라벨도 normalization 진행
        input = (input - self.mean) / self.std
        label = (label - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data

