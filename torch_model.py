import torch
import torch.nn as nn
import torch.nn.functional as F


from torch_utils import SOFTARGMIN, cat_fms, Aggregation, Feature


class TorchModel(nn.Module):
    def __init__(self, max_disp=192):
        super(TorchModel, self).__init__()
        self.max_disp = max_disp

        self.feature = Feature()
        self.aggregation = Aggregation(in_planes=64)
        self.prediction = SOFTARGMIN()

    def forward(self, imL, imR):
        featL, featR = self.feature(imL, imR)
        raw_cost = cat_fms(featL, featR, max_disp=self.max_disp//4)
        costs = self.aggregation(raw_cost, to_full=True)
        disps = [self.prediction(c, max_disp=self.max_disp) for c in costs]

        return disps

    def batch_train(self, imL, imR, dispL):
        disps = self(imL, imR)
        mask = (dispL > 0) & (dispL < self.max_disp)
        mask.detach_()
        ws = [1.0, 0.7, 0.5]
        loss = 0.0
        for (disp, w) in zip(disps, ws):
            loss += w*F.smooth_l1_loss(disp[mask], dispL[mask], size_average=True)

        return loss

    def batch_test(self, imL, imR, dispL):
        disps = self(imL, imR)
        mask = (dispL > 0) & (dispL < self.max_disp)
        mask.detach_()
        ws = [1.0, 0.7, 0.5]
        loss = 0.0
        for (disp, w) in zip(disps, ws):
            loss += w*F.smooth_l1_loss(disp[mask], dispL[mask], size_average=True)

        return disps, loss

    def batch_predict(self, imL, imR):
        disps = self(imL, imR)

        return disps[0]


if __name__ == '__main__':
    print('Start testing...')

    net = TorchModel()

    print(net)

    print('Done!')
