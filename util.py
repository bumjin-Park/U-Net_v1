import os
import random
import math
import shutil

data_root = "/root/test/dataset/"
dir_1 = "/root/test/dataset/images"

dir1_filename = os.listdir(dir_1)
train = random.sample(dir1_filename, round((len(dir1_filename)*7)/10))

test = [x for x in dir1_filename if x not in train]
val = random.sample(test, round((len(test)*3)/10))

test = [x for x in test if x not in val]

print(len(train)+len(test)+len(val)) # 개수 확인

# 폴더 생성
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

createFolder(data_root + "Train")
createFolder(data_root + "Train/" + 'images')
createFolder(data_root + "Train/" + 'masks')
createFolder(data_root + "Val")
createFolder(data_root + "Val/" + 'images')
createFolder(data_root + "Val/" + 'masks')
createFolder(data_root + "Test")
createFolder(data_root + "Test/" + 'images')
createFolder(data_root + "Test/" + 'masks')

for i in train:
    shutil.copyfile(os.path.join(data_root, 'images', str(i)), os.path.join(data_root, 'Train', 'images', str(i)))
    shutil.copyfile(os.path.join(data_root, 'masks', str(i)), os.path.join(data_root, 'Train', 'masks', str(i)))

for i in test:
    shutil.copyfile(os.path.join(data_root, 'images', str(i)), os.path.join(data_root, 'Test', 'images', str(i)))
    shutil.copyfile(os.path.join(data_root, 'masks', str(i)), os.path.join(data_root, 'Test', 'masks', str(i)))

for i in val:
    shutil.copyfile(os.path.join(data_root, 'images', str(i)), os.path.join(data_root, 'Val', 'images', str(i)))
    shutil.copyfile(os.path.join(data_root, 'masks', str(i)), os.path.join(data_root, 'Val', 'masks', str(i)))

