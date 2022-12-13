import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from tf_model import Model
import cv2
import matplotlib.pyplot as plt


#set para
parser = argparse.ArgumentParser(description='StereoMatching')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--batch', type=int ,default=1,
                    help='batch_size')	
parser.add_argument('--datapath', default='./data/', help='datapath')
parser.add_argument('--loadmodel', default='./ckpt/final.ckpt',
                    help='load model')
parser.add_argument('--leftimg', default='left/000018_10.png',
                    help='left image')
parser.add_argument('--rightimg', default='right/000018_10.png',
                    help='right image')

args = parser.parse_args()

print('Called with args:')
print(args)


def mean_std(inputs):
    inputs = np.float32(inputs) / 255.
    inputs[:, :, 0] -= 0.485
    inputs[:, :, 0] /= 0.229
    inputs[:, :, 1] -= 0.456
    inputs[:, :, 1] /= 0.224
    inputs[:, :, 2] -= 0.406
    inputs[:, :, 2] /= 0.225
    return inputs


def main():

    height = 368
    weight = 1224
    left_img = args.datapath+args.leftimg
    right_img = args.datapath+args.leftimg


    with tf.compat.v1.Session() as sess:

        img_L = cv2.cvtColor(cv2.imread(left_img), cv2.COLOR_BGR2RGB)
        img_L = cv2.resize(img_L, (weight, height))
        img_R = cv2.cvtColor(cv2.imread(right_img), cv2.COLOR_BGR2RGB)
        img_R = cv2.resize(img_R, (weight, height))		

        img_L = mean_std(img_L)
        img_L = np.expand_dims(img_L, axis=0)
        img_R = mean_std(img_R)
        img_R = np.expand_dims(img_R, axis=0)
		
        net = Model(sess, height=height, weight=weight, batch_size=args.batch, 
                    max_disp=args.maxdisp, lr=0.001, cnn_3d_type='resnet_3d')
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, args.loadmodel)
		
        pred = net.predict(img_L, img_R)
        pred = np.squeeze(pred, axis=0)
        item = (pred * 255 / pred.max()).astype(np.uint8)
        pred_rainbow = cv2.applyColorMap(item, cv2.COLORMAP_RAINBOW)
        cv2.imwrite('tf_prediction.png', pred_rainbow)


if __name__ == '__main__':
    main()
