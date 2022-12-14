import argparse
import torch
import numpy as np
from torch_model import TorchModel
import cv2
import matplotlib.pyplot as plt


#set para
parser = argparse.ArgumentParser(description='StereoMatching')
parser.add_argument('--batch', type=int ,default=1,
                    help='batch_size')	
parser.add_argument('--datapath', default='./data/', help='datapath')
parser.add_argument('--loadmodel', default='./ckpt/torch_final.ckpt',
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
    max_disp = 192
    left_img = args.datapath+args.leftimg
    right_img = args.datapath+args.leftimg
    device = 'cuda:0'


    with torch.no_grad():

        img_L = cv2.cvtColor(cv2.imread(left_img), cv2.COLOR_BGR2RGB)
        img_L = cv2.resize(img_L, (weight, height))
        img_R = cv2.cvtColor(cv2.imread(right_img), cv2.COLOR_BGR2RGB)
        img_R = cv2.resize(img_R, (weight, height))		

        img_L = mean_std(img_L)
        img_R = mean_std(img_R)
		
        net = TorchModel(max_disp=max_disp)
        ckpt = torch.load(args.loadmodel, map_location='cpu')
        net.load_state_dict(ckpt['state_dict'])
        net = net.to(device)
        net.eval()

		
        imgL_t = torch.from_numpy(img_L).to(device).unsqueeze(dim=0).permute(0, 3, 1, 2)
        imgR_t = torch.from_numpy(img_R).to(device).unsqueeze(dim=0).permute(0, 3, 1, 2)
        pred = net.batch_predict(imgL_t, imgR_t)
        pred = pred[0, 0].cpu().numpy()

        item = (pred * 255 / pred.max()).astype(np.uint8)
        pred_rainbow = cv2.applyColorMap(item, cv2.COLORMAP_RAINBOW)
        cv2.imwrite('torch_prediction.png', pred_rainbow)


if __name__ == '__main__':
    main()
