import os

import numpy as np
import torch.utils.data as data
from PIL import Image
from utils import transforms as tr
import cv2
from osgeo import gdal
import scipy as N
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
    from osgeo import gdal
    res = [ i for i in A if gdal.Open(data_dir+'temporal1/'+i) is not None]
    return res,  res

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

    TIF1 = gdal.Open(dir+'temporal1/'+name)
    img1 = TIF1.ReadAsArray().transpose(1,2,0)
    img_rgb = np.array(img1,dtype=np.uint8)
    r = img_rgb[:, :, 0]
    g = img_rgb[:, :, 1]
    b = img_rgb[:, :, 2]
    img1 = np.dstack((r,g,b))
    img1 = Image.fromarray(img1)
    TIF2 = gdal.Open(dir + 'temporal2/' + name)
    img2 = TIF2.ReadAsArray().transpose(1, 2, 0)
    img_rgb = np.array(img2, dtype=np.uint8)
    r = img_rgb[:, :, 0]
    g = img_rgb[:, :, 1]
    b = img_rgb[:, :, 2]
    img2 = np.dstack((r, g, b))
    img2 = Image.fromarray(img2)


    #label = Image.open(label_path)
    label = gdal.Open(label_path)
    label =label.ReadAsArray()
    label = np.array(label, dtype=np.uint8)
    label[label==1]=255
    label = Image.fromarray(label)

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



#
# '''
# Load all training and validation data paths
# '''
# def full_path_loader(data_dir):
#
#     train_data = [i for i in os.listdir(data_dir + 'train/A/') if
#     i.startswith('.tif')]
#     train_data.sort()
#
#     valid_data = [i for i in os.listdir(data_dir + 'val/A/') if
#     i.startswith('.tif')]
#     valid_data.sort()
#
#     train_label_paths = []
#     val_label_paths = []
#     for img in train_data:
#         train_label_paths.append(data_dir + 'train/OUT/' + img)
#     for img in valid_data:
#         val_label_paths.append(data_dir + 'val/OUT/' + img)
#
#
#     train_data_path = []
#     val_data_path = []
#
#     for img in train_data:
#         train_data_path.append([data_dir + 'train/', img])
#     for img in valid_data:
#         val_data_path.append([data_dir + 'val/', img])
#
#     train_dataset = {}
#     val_dataset = {}
#     for cp in range(len(train_data)):
#         train_dataset[cp] = {'image': train_data_path[cp],
#                          'label': train_label_paths[cp]}
#     for cp in range(len(valid_data)):
#         val_dataset[cp] = {'image': val_data_path[cp],
#                          'label': val_label_paths[cp]}
#
#
#     return train_dataset, val_dataset
#
#
# '''
# Load all testing data paths
# '''
# def full_test_loader(data_dir):
#
#     test_data = [i for i in os.listdir(data_dir + 'test/A/') if not
#                     i.startswith('.')]
#     test_data.sort()
#
#     test_label_paths = []
#     for img in test_data:
#         test_label_paths.append(data_dir + 'test/OUT/' + img)
#
#     test_data_path = []
#     for img in test_data:
#         test_data_path.append([data_dir + 'test/', img])
#
#     test_dataset = {}
#     for cp in range(len(test_data)):
#         test_dataset[cp] = {'image': test_data_path[cp],
#                            'label': test_label_paths[cp]}
#
#     return test_dataset
#
# # def cdd_loader(img_path, label_path, aug):
# #     dir = img_path[0]
# #     name = img_path[1]
# #
# #     img1 = Image.open(dir + 'A/' + name)
# #     img2 = Image.open(dir + 'B/' + name)
# #     label = Image.open(label_path)
# #     sample = {'image': (img1, img2), 'label': label}
# #
# #     if aug:
# #         sample = tr.train_transforms(sample)
# #     else:
# #         sample = tr.test_transforms(sample)
# #
# #     return sample['image'][0], sample['image'][1], sample['label']
#
#
# def tif_loader(img_path, label_path, aug):
#     dir = img_path[0]
#     name = img_path[1]
#     img1 = cv2.imread(dir + 'A/' + name, cv2.IMREAD_UNCHANGED)
#     img1 = cv2.cvtColor(img1, cv2.COLOR_BAYER_BG2GRAY)
#     img2 = cv2.imread(dir + 'B/'+name, cv2.IMREAD_UNCHANGED)
#     img2 = cv2.cvtColor(img2, cv2.COLOR_BAYER_BG2GRAY)
#     label = cv2.imread(label_path,cv2.IMREAD_UNCHANGED)
#     label = label[0]
#     sample = {'image': (img1, img2), 'label': label}
#
#     if aug:
#         sample = tr.train_transforms(sample)
#     else:
#         sample = tr.test_transforms(sample)
#
#     return sample['image'][0], sample['image'][1], sample['label']
#
#
# class CDDloader(data.Dataset):
#
#     def __init__(self, full_load, aug=False):
#
#         self.full_load = full_load
#         #self.loader = cdd_loader
#         self.loader = tif_loader
#         self.aug = aug
#
#     def __getitem__(self, index):
#
#         img_path, label_path = self.full_load[index]['image'], self.full_load[index]['label']
#
#         return self.loader(img_path,
#                            label_path,
#                            self.aug)
#
#     def __len__(self):
#         return len(self.full_load)