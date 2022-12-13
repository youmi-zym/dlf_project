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
parser.add_argument('--loadmodel', default='./ckpt/PSMNet.ckpt-50',
                    help='load model')
parser.add_argument('--leftimg', default='left/0006.png',
                    help='left image')
parser.add_argument('--rightimg', default='right/0006.png',
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

    height = 544
    weight = 960
    left_img = args.datapath+args.leftimg
    right_img = args.datapath+args.leftimg


    with tf.Session() as sess:

        img_L = cv2.cvtColor(cv2.imread(left_img), cv2.COLOR_BGR2RGB)
        img_L = cv2.resize(img_L, (weight, height))
        img_R = cv2.cvtColor(cv2.imread(right_img), cv2.COLOR_BGR2RGB)
        img_R = cv2.resize(img_R, (weight, height))		

        img_L = mean_std(img_L)
        img_L = np.expand_dims(img_L, axis=0)
        img_R = mean_std(img_R)
        img_R = np.expand_dims(img_R, axis=0)
		
        net = Model(sess, height=height, weight=weight, batch_size=args.batch, max_disp=args.maxdisp)
        saver = tf.train.Saver()
        saver.restore(sess, args.loadmodel)
		
        pred = net.predict(img_L, img_R)
        pred = np.squeeze(pred,axis=0)

        plt.imsave('pred_plt.png', pred)


if __name__ == '__main__':
    main()
