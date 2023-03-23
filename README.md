# Denoising U-Net
- U-Net을 이용하여 이미지의 노이즈를 제거하는 모델

## INSTALL 
- 도커 이미지 혹은 git clone을 사용하여 다운한다.
- Docker hub : https://hub.docker.com/r/skatld802/denoising-unet
``` console
    $ docker pull skatld802/denoising-unet
```

## Modify Image Name
---
- 이미지 이름 라벨링을 위해 0_edit_img_name.py 를 사용하여 이미지의 이름을 변경한다.
    ``` python
    # 이름 변경할 이미지 디렉토리 선택
    targetDirPath = "/root/Denoising_UNet/data_noise/data_label"
    # 변경할 이미지 이름 설정
    data_name = "person"
    ```
- 해당 과정을 거치면 무작위였던 이미지의 이름이 person_0001.png의 방식으로 순차적으로 이름이 변경된다.
  

## Add Noise to image (Data Augmentation)
---
- 이미지에 노이즈를 추가하여 노이즈 데이터를 생성하는 과정이다. 
- 1_add_noise.py 를 사용하여 노이즈를 label 이미지에 추가한다.

    ``` python
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
    ```
- 해당 과정을 거치면 노이즈 데이터가 gauss_person_0001.png의 방식으로 순차적으로 이름이 변경된다.  

## Setting Train Dataset
---
- Model의 학습시킬 수 있도록 데이터 형식을 맞춰주기 위해 라벨, 입력(노이즈)이미지들을 npy 파일로 변환한다.<br/>
- 사실 이 데이터 변환 과정은 매우 번거롭고 불필요해서 추후, 수정해야될 부분이다.

    ### 2_data_setting_noise.py 순서  

    1. labelDirPath/noiseDirPath의 데이터를 dir_train에 모두 이동한다.
        - 이 때, 데이터가 없을 시 오류 발생
        <br/><br/>
    2. changeDataName을 통해 dir_train에 있는 이미지의 이름을 person -> input과 gauss_person -> label로 변경한다.
        - 오류가 발생할 경우 코드를 확인해 볼 것
        <br/><br/>
    3. dir_train에 있는 데이터를 train, val, test의 list로 반환하여 저장한다.
        ``` python
        lst_train, lst_val, lst_test = splitDataset(dir_train)
        ```
    4. splitDataset을 통해 반환된 이미지 이름의 리스트인 lst_train, lst_val, lst_test을 사용하여 7:2:1 비율로 dir_train, dir_val, dir_test로 이미지를 옮긴다.  
    <br/>
    5. png -> numpy 변환을 한다. 해당 과정에서 오류가 발생할 수 있다.
        ``` python
        png2numpy(dir_train)
        png2numpy(dir_val)
        png2numpy(dir_test)
        ```

## Train Denoising U-Net
---
- **train.py**를 사용하여 학습을 진행한다.
    ```console
    <!-- train.py 포맷 -->
    train.py [-h] [--lr LR] [--batch_size BATCH_SIZE] [--num_epoch NUM_EPOCH] [--data_dir DATA_DIR] [--ckpt_dir CKPT_DIR] [--log_dir LOG_DIR] [--result_dir RESULT_DIR] [--mode MODE]
                [--train_continue TRAIN_CONTINUE] [--dn_data_dir DN_DATA_DIR]

    <!-- 학습 진행 시 -->
    train.py --num_epoch 100 --dn_data_dir ./datasets_denosing --mode denoise

    <!-- 테스트 진행 시 -->
    train.py --dn_data_dir ./datasets_denosing --mode test --ckpt_dir ./checkpoint/model_epoch200.pth
    ```

- process_denoising에 일정한 batch마다 이미지가 저장되어 학습 과정을 이미지로 확인할 수 있음.

## PSNR, SSIM 측정
- eval_psnr_ssim 폴더의 ipynb를 따라가자.
  
  
## Reference
- Paper : https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28
- UNET : https://www.youtube.com/@hanyoseob
- PSNR, SSIM : https://m.blog.naver.com/mincheol9166/221771426327
