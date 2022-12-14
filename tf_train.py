import time
import os
from dataloader import DataLoaderKITTI
from tf_model import Model
import tensorflow as tf


def main():
    root = '/home/yzhang/data/KITTI-2015/training/'
    left_img = root+'image_2/'
    right_img = root+'image_3/'
    disp_img = root+'disp_occ_0/'

    bat_size = 8
    # 128, 192
    maxdisp = 128
    epochs = 300
    dg = DataLoaderKITTI(left_img, right_img, disp_img, bat_size)

    if not os.path.exists('./results/'):
        os.mkdir('./results/')
    with tf.compat.v1.Session() as sess:
        Net = Model(sess, height=256, weight=512, batch_size=bat_size,
                    max_disp=maxdisp, lr=0.001, cnn_3d_type='resnet_3d')
        saver = tf.compat.v1.train.Saver()
        for epoch in range(1, epochs + 1):
            total_train_loss = 0

            for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(dg.generator(is_training=True)):
                start_time = time.time()
                train_loss = Net.train(imgL_crop, imgR_crop, disp_crop_L)
                print('Iter %d training loss = %.3f , time = %.2f' % (batch_idx, train_loss, time.time() - start_time))
                total_train_loss += train_loss
            avg_loss = total_train_loss / (160 // bat_size)
            print('epoch %d avg training loss = %.3f' % (epoch, avg_loss))
            if epoch % 30 == 0:
                saver.save(sess, './results/net.ckpt', global_step=epoch)

            total_train_loss = 0
            for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(dg.generator(is_training=False)):
                start_time = time.time()
                pred, train_loss = Net.test(imgL_crop, imgR_crop, disp_crop_L)
                print('Iter %d testing loss = %.3f , time = %.2f' % (batch_idx, train_loss, time.time() - start_time))
                total_train_loss += train_loss

            avg_loss = total_train_loss / (40 // bat_size)
            print('epoch %d avg testing loss = %.3f' % (epoch, avg_loss))
        saver.save(sess, './results/final.ckpt')


if __name__ == '__main__':
    main()
