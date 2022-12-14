import time
import torch
from torch.optim.lr_scheduler import MultiStepLR
import os
from dataloader import DataLoaderKITTI
from torch_model import TorchModel
import numpy as np
import matplotlib.pyplot as plt
import cv2


def main():
    root = '/home/yzhang/data/KITTI-2015/training/'
    left_img = root+'image_2/'
    right_img = root+'image_3/'
    disp_img = root+'disp_occ_0/'

    device = 'cuda:0'
    bat_size = 1
    max_disp = 192
    epochs = 300
    dg = DataLoaderKITTI(left_img, right_img, disp_img, bat_size)
    Net = TorchModel(max_disp=max_disp).to(device)

    loadmodel = './results/torch_final.ckpt'
    ckpt = torch.load(loadmodel, map_location='cpu')
    Net.load_state_dict(ckpt['state_dict'])
    Net = Net.to(device)
    epoch = 0

    if not os.path.exists('./test/'):
        os.mkdir('./test/')
    with torch.no_grad():
        total_train_loss = 0
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(dg.generator(is_training=False)):
            imgL_crop = torch.from_numpy(imgL_crop).to(device).permute(0, 3, 1, 2)
            imgR_crop = torch.from_numpy(imgR_crop).to(device).permute(0, 3, 1, 2)
            disp_crop_L = torch.from_numpy(disp_crop_L).to(device).unsqueeze(dim=1)
            start_time = time.time()
            pred, train_loss = Net.batch_test(imgL_crop, imgR_crop, disp_crop_L)
            print('Iter %d testing loss = %.3f , time = %.2f' % (batch_idx, train_loss, time.time() - start_time))
            total_train_loss += train_loss

            avg_loss = total_train_loss / (40 // bat_size)
            print('\nEpoch %d avg testing loss = %.3f\n' % (epoch, avg_loss))

            disp = pred[0][0, 0].cpu().numpy()
            item = (disp * 255 / disp.max()).astype(np.uint8)
            pred_rainbow = cv2.applyColorMap(item, cv2.COLORMAP_RAINBOW)
            cv2.imwrite('torch_prediction.png', pred_rainbow)
            # plt.imsave(f'./test/{batch_idx}.png', disp)

    print('Done!')


if __name__ == '__main__':
    main()
