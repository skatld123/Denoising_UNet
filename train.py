## 라이브러리 추가하기
import argparse
import os
import logging
import matplotlib
import matplotlib.pyplot as plt
# 이거 안하니까 멈추는 에러가 발생... 벡엔드 설정해주는 코드
matplotlib.use('Agg')
import numpy as np
import torch
import torch.nn as nn
from dataset import *
from model import UNet
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.transforms.functional import rgb_to_grayscale, to_pil_image
from util import *

## Parser 생성하기
parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=200, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="./datasets", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

# 디노이즈로 모드 변경했음
parser.add_argument("--mode", default="denoise", type=str, dest="mode")
# parser.add_argument("--mode", default="test", type=str, dest="mode")
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

parser.add_argument("--dn_data_dir", default="./datasets_denosing", type=str, dest="dn_data_dir")
args = parser.parse_args()

## 트레이닝 파라메터 설정하기
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

mode = args.mode
train_continue = args.train_continue

dn_data_dir = args.dn_data_dir
# dn_noise_dir = args.dn_noise_dir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)
print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("result dir: %s" % result_dir)
print("mode: %s" % mode)

print("denoising data dir: %s" % dn_data_dir)
# print("noising data dir: %s" % dn_noise_dir)

## 디렉토리 생성하기
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

## 네트워크 학습하기
if mode == 'train':
    print("트레인이다")
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

    # 그밖에 부수적인 variables 설정하기
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)

elif mode == 'denoise':
    print("Load Denoising Dataset")
    transform = transforms.Compose([ToTensor()])
    
    dataset_train = Dataset(data_dir=os.path.join(dn_data_dir, 'train'), transform=transform)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    
    dataset_val = Dataset(data_dir=os.path.join(dn_data_dir, 'val'), transform=transform)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 그밖에 부수적인 variables 설정하기
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)

# TEST MODE
else:
    transform = transforms.Compose([ToTensor()])
    
    dataset_test = Dataset(data_dir=os.path.join(dn_data_dir, 'test'), transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)
    print(len(dataset_test.lst_input))
    print(len(dataset_test.lst_label))
    # 그밖에 부수적인 variables 설정하기
    num_data_test = len(dataset_test)

    num_batch_test = np.ceil(num_data_test / batch_size)

## 네트워크 생성하기
net = UNet().to(device)

## 손실함수 정의하기
fn_loss = nn.MSELoss().to(device)

## Optimizer 설정하기
optim = torch.optim.Adam(net.parameters(), lr=lr)

## 그밖에 부수적인 functions 설정하기
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

## Tensorboard 를 사용하기 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

## 네트워크 학습시키기
st_epoch = 0

process_cnt = 0

# Loss를 저장할 로그 파일 생성
logging.basicConfig(filename = './trainlog.log', level=logging.INFO)

# 노이즈 추가 함수
def add_noise(img):
    noise = torch.randn(img.size()) * 0.1
    noisy_img = img + noise
    return noisy_img

# TRAIN MODE
if mode == 'train' or mode == 'denoise':
    if train_continue == "on":
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()
        loss_arr = []
        for batch, data in enumerate(loader_train, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            input, output = net(input)

            # backward pass
            optim.zero_grad()

            # MSE Loss를 사용
            loss = fn_loss(output, label)
            loss.backward()

            optim.step()

            # 손실함수 계산
            loss_arr += [loss.item()]

            if process_cnt % 60 == 0:
                # 노이즈 이미지 그리기
                input_batch = fn_tonumpy(input)
                plt.subplot(131)
                plt.imshow(input_batch[0])
                plt.title('input')

                # 정답 그리기 
                label_batch = fn_tonumpy(label)
                plt.subplot(132)
                plt.imshow(label_batch[0])
                plt.title('label')

                # 결과 이미지 그림그리기
                output_batch = fn_tonumpy(output)
                plt.subplot(133)
                plt.imshow(output_batch[0])
                plt.title('result')

                plt.savefig('/root/Denoising_UNet/process_denoising/process' + str(process_cnt) + '.png')
                
            process_cnt = process_cnt + 1

            train_loss_info = ("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))
            print(train_loss_info)
            # 로그에 남기기
            logging.info(train_loss_info)
            
            # Tensorboard 저장하기
            label = fn_tonumpy(label)
            input = fn_tonumpy(input)
            output = fn_tonumpy(output)

            writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        with torch.no_grad():
            net.eval()
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):
                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)

                input, output = net(input)

                # 손실함수 계산하기
                loss = fn_loss(output, label)

                loss_arr += [loss.item()]

                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

                # Tensorboard 저장하기
                label = fn_tonumpy(label)
                input = fn_tonumpy(input)
                output = fn_tonumpy(output)

                writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

        if epoch % 70 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    writer_train.close()
    writer_val.close()

# TEST MODE
else:
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    with torch.no_grad():
        net.eval()
        loss_arr = []

        for batch, data in enumerate(loader_test, 1):
            # forward pass  
            label = data['label'].to(device)
            input = data['input'].to(device)
            print(label.dtype)
            input, output = net(input)

            # 손실함수 계산하기
            loss = fn_loss(output, label)

            loss_arr += [loss.item()]

            print("TEST: BATCH %04d / %04d | LOSS %.4f" %
                (batch, num_batch_test, np.mean(loss_arr)))

            # Tensorboard 저장하기
            logging.basicConfig(filename='./outputForm', level=logging.INFO)
            
            # label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
            print(label.dtype)
            label = fn_tonumpy(label)
            
            logging.info("input")
            logging.info(input)
            # input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            print(input.dtype)
            input = fn_tonumpy(input)

            logging.info("input")
            logging.info(input)
            
            # output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))
            # output = output.to(dtype=torch.half)
            # output_norm = transforms.Normalize()
            output -= output.min()
            output /= output.max()

            output = fn_tonumpy(output)
            
            logging.info("output")
            logging.info(output)
            
            # 학습 과정 중 이미지 시각화를 위한 png 저장
            for j in range(label.shape[0]):

                id = num_batch_test * (batch - 1) + j

                # 정답 그리기
                img_label = label[j]
                plt.imshow(img_label)
                plt.title('label')
                plt.savefig(
                    '/root/Denoising_UNet/dn_test_result/label_%03d.png' % (id))

                # 노이즈 이미지 그리기
                img_input = input[j]
                plt.imshow(img_input)
                plt.title('input')
                plt.savefig(
                    '/root/Denoising_UNet/dn_test_result/input_%03d.png' % (id))

                # 결과 이미지 그림그리기
                result = output[j]
                plt.imshow(result)
                plt.title('result')
                plt.savefig(
                    '/root/Denoising_UNet/dn_test_result/result_%03d.png' % (id))

                plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')

                np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())

    print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
          (batch, num_batch_test, np.mean(loss_arr)))


