import numpy as np
import random
import cv2
import os


class DataLoaderKITTI(object):
    def __init__(self, left_path, right_path, gt_path, batch_size, patch_size=(256, 512), max_disp=129):
        self.left_path = left_path
        self.right_path = right_path
        self.gt_path = gt_path
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.max_disp = max_disp

    def generator(self, is_training=True):
        left_data = os.listdir(self.left_path)
        right_data = os.listdir(self.right_path)
        labels = os.listdir(self.gt_path)
        left_data = [im for im in left_data if im.find('_10.png') > -1]
        right_data = [im for im in left_data if im.find('_10.png') > -1]
        right_data = [im for im in left_data if im.find('_10.png') > -1]
        print(f'{len(left_data)} images found!')
        left_data.sort(key=str.lower)
        right_data.sort(key=str.lower)
        labels.sort(key=str.lower)

        train_left = left_data[:160]
        train_right = right_data[:160]
        train_labels = labels[:160]

        val_left = left_data[160:]
        val_right = right_data[160:]
        val_labels = labels[160:]

        index = [i for i in range(160)]
        random.shuffle(index)
        shuffled_labels = []
        shuffled_left_data = []
        shuffled_right_data = []

        for i in index:
            shuffled_left_data.append(train_left[i])
            shuffled_right_data.append(train_right[i])
            shuffled_labels.append(train_labels[i])
        if is_training:
            for j in range(160 // self.batch_size):
                left, right, label = self.load_batch(shuffled_left_data[j * self.batch_size: (j + 1) * self.batch_size],
                                                     shuffled_right_data[
                                                     j * self.batch_size: (j + 1) * self.batch_size],
                                                     shuffled_labels[j * self.batch_size: (j + 1) * self.batch_size],
                                                     is_training)
                left = np.array(left)
                right = np.array(right)
                label = np.array(label)
                yield left, right, label
        else:
            for j in range(40 // self.batch_size):
                left, right, label = self.load_batch(val_left[j * self.batch_size: (j + 1) * self.batch_size],
                                                     val_right[j * self.batch_size: (j + 1) * self.batch_size],
                                                     val_labels[j * self.batch_size: (j + 1) * self.batch_size],
                                                     is_training)
                left = np.array(left)
                right = np.array(right)
                label = np.array(label)
                yield left, right, label

    def load_batch(self, left, right, labels, is_training):
        batch_left = []
        batch_right = []
        batch_label = []
        for x, y, z in zip(left, right, labels):
            if is_training:
                crop_x = random.randint(0, 368 - self.patch_size[0])
                crop_y = random.randint(0, 1224 - self.patch_size[1])
            else:
                crop_x = (368 - self.patch_size[0]) // 2
                crop_y = (1224 - self.patch_size[1]) // 2

            x = cv2.imread(self.left_path + x)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = x[crop_x: crop_x + self.patch_size[0], crop_y: crop_y + self.patch_size[1], :]
            x = self.mean_std(x)
            batch_left.append(x)

            y = cv2.imread(self.right_path + y)
            y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
            y = y[crop_x: crop_x + self.patch_size[0], crop_y: crop_y + self.patch_size[1], :]
            y = self.mean_std(y)
            batch_right.append(y)

            z = cv2.imread(self.gt_path + z)
            z = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY)
            z = z[crop_x: crop_x + self.patch_size[0], crop_y: crop_y + self.patch_size[1]]
            z[z > (self.max_disp-1)] = self.max_disp - 1
            batch_label.append(z)
        return batch_left, batch_right, batch_label

    @staticmethod
    def mean_std(inputs):
        inputs = np.float32(inputs) / 255.
        inputs[:, :, 0] -= 0.485
        inputs[:, :, 0] /= 0.229
        inputs[:, :, 1] -= 0.456
        inputs[:, :, 1] /= 0.224
        inputs[:, :, 2] -= 0.406
        inputs[:, :, 2] /= 0.225
        return inputs
