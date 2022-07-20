import os
import torch.utils.data as data
from PIL import Image
from utils import transforms as tr
import cv2


'''
   tif文件列表获取
'''
def get_data(data_dir):
    A = [i for i in os.listdir(data_dir + 'temporal1/') if i.endswith('.tif')]
    B = [i for i in os.listdir(data_dir + 'temporal2/') if i.endswith('.tif')]
    A.sort()
    B.sort()
    label = [i for i in os.listdir(data_dir+'gt')  if i.endswith('.tif') ]
    label.sort()
    assert A==B
    label = [ i for i in label if i in A]
    assert A==label
    return A,  label

'''
Load all training and validation data paths
'''
def full_path_loader(data_dir):

    A, label = get_data(data_dir)
    train_data, train_label  = A[:int(0.8*len(A))], label[:int(0.8*len(A))]
    # valid_data = [i for i in os.listdir(data_dir + 'val/A/') if not
    # i.startswith('.')]
    # valid_data.sort()
    valid_data = A[int(0.8*len(A)):int(0.9*len(A))]
    train_label_paths = []
    val_label_paths = []
    # for img in train_data:
    #     train_label_paths.append(data_dir + 'train/OUT/' + img)
    # for img in valid_data:
    #     val_label_paths.append(data_dir + 'val/OUT/' + img)
    for img in train_data:
        train_label_paths.append(data_dir+'gt/'+img)

    for img in valid_data:
        val_label_paths.append(data_dir+'gt/'+img)

    train_data_path = []
    val_data_path = []

    # for img in train_data:
    #     train_data_path.append([data_dir + 'train/', img])
    # for img in valid_data:
    #     val_data_path.append([data_dir + 'val/', img])

    for img in train_data:
        train_data_path.append([data_dir,img])

    for img in valid_data:
        val_data_path.append([data_dir, img])

    train_dataset = {}
    val_dataset = {}
    for cp in range(len(train_data)):
        train_dataset[cp] = {'image': train_data_path[cp],
                         'label': train_label_paths[cp]}
    for cp in range(len(valid_data)):
        val_dataset[cp] = {'image': val_data_path[cp],
                         'label': val_label_paths[cp]}

    # 调试返回值
    return train_dataset, val_dataset

'''
Load all testing data paths
'''

def full_test_loader(data_dir):
    A, label = get_data(data_dir)
    test_data = A[int(0.9*len(A)):]

    test_label_paths = []
    for img in test_data:
        test_label_paths.append(data_dir+'gt/' + img)

    test_data_path = []
    for img in test_data:
        test_data_path.append([data_dir, img])

    test_dataset = {}
    for cp in range(len(test_data)):
        test_dataset[cp] = {'image': test_data_path[cp],
                           'label': test_label_paths[cp]}

    return test_dataset

def tif_loader(img_path, label_path, aug):
    dir = img_path[0]
    name = img_path[1]

    # img1 = Image.open(dir + 'A/' + name)
    # img2 = Image.open(dir + 'B/' + name)
    img1 =  cv2.imread(dir+'temporal1/'+name, cv2.IMREAD_UNCHANGED)
    img1 = cv2.cvtColor(img1,cv2.COLOR_BAYER_BG2GRAY)

    img2 = cv2.imread(dir+'temporal2/'+name, cv2.IMREAD_UNCHANGED)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BAYER_BG2GRAY)
    ## TIF 文件读取

    label = Image.open(label_path)
    sample = {'image': (img1, img2), 'label': label}

    if aug:
        sample = tr.train_transforms(sample)
    else:
        sample = tr.test_transforms(sample)

    return sample['image'][0], sample['image'][1], sample['label']


class CDDloader(data.Dataset):

    def __init__(self, full_load, aug=False):

        self.full_load = full_load
        self.loader = tif_loader
        self.aug = aug

    def __getitem__(self, index):

        img_path, label_path = self.full_load[index]['image'], self.full_load[index]['label']

        return self.loader(img_path,
                           label_path,
                           self.aug)

    def __len__(self):
        return len(self.full_load)
