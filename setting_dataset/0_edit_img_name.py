import os

# 이름 변경할 이미지 디렉토리 선택
targetDirPath = "/root/Denoising_UNet/data_noise/data_label"
# 변경할 이미지 이름 설정
data_name = "person"


targetList = os.listdir(targetDirPath)
print(targetList)

for i, filename in enumerate(targetList):
    full_filename = os.path.join(targetDirPath, filename)
    ext = os.path.splitext(full_filename)[-1]
    if ext == ".jpg" or ext == ".png" :
        nf_name = data_name + ("_%03d.png" % (i + 1))
        newfile = os.path.join(targetDirPath, nf_name)
        os.rename(full_filename, newfile)

targetList.sort()
print(targetList)