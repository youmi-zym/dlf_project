import time
import torch
from torch.optim.lr_scheduler import MultiStepLR
import os
from dataloader import DataLoaderKITTI
from torch_model import TorchModel


def main():
    root = '/home/yzhang/data/KITTI-2015/training/'
    left_img = root+'image_2/'
    right_img = root+'image_3/'
    disp_img = root+'disp_occ_0/'

    device = 'cuda:0'
    bat_size = 4
    max_disp = 192
    epochs = 300
    dg = DataLoaderKITTI(left_img, right_img, disp_img, bat_size, max_disp=max_disp)

    if not os.path.exists('./results/'):
        os.mkdir('./results/')
    with torch.enable_grad():
        Net = TorchModel(max_disp=max_disp).to(device)
        optimizer = torch.optim.Adam(Net.parameters(), lr=0.001)
        scheduler = MultiStepLR(optimizer, milestones=[200, 300], gamma=0.1)
        for epoch in range(1, epochs + 1):
            total_train_loss = 0
            optimizer.zero_grad()

            for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(dg.generator(is_training=True)):
                imgL_crop = torch.from_numpy(imgL_crop).to(device).permute(0, 3, 1, 2)
                imgR_crop = torch.from_numpy(imgR_crop).to(device).permute(0, 3, 1, 2)
                disp_crop_L = torch.from_numpy(disp_crop_L).to(device).unsqueeze(dim=1)
                start_time = time.time()
                train_loss = Net.batch_train(imgL_crop, imgR_crop, disp_crop_L)

                train_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                print('Iter %d training loss = %.3f , time = %.2f' % (batch_idx, train_loss, time.time() - start_time))
                total_train_loss += train_loss
            avg_loss = total_train_loss / (160 // bat_size)
            print('Epoch %d avg training loss = %.3f' % (epoch, avg_loss))

            if epoch % 30 == 0:
                torch.save(
                    {
                        'epoch': epoch,
                        'state_dict': Net.state_dict(),
                    },
                    f'./results/net-{epoch}.ckpt'
                )

            scheduler.step()

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

        print('Saving the final model to ./results/torch_final.ckpt!!!')
        torch.save(
            {
                'epoch': epoch,
                'state_dict': Net.state_dict(),
            },
            f'./results/torch_final.ckpt'
        )

        print('Done!')


if __name__ == '__main__':
    main()
