import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
from glob import glob
from align import AlignPoints
from src import model
from src import util
from src.body import Body
import torch.nn as nn
import time


class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 4)
        self.drop = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def prepare_posreg_multiperson(corr, subset):
    multi_person = []
    for persion_idx in range(len(subset)):
        pos_idx_tmp = [int(j) for j in subset[persion_idx]]
        pos_idx_tmp = pos_idx_tmp[:18]
        multi_person.append(corr[pos_idx_tmp])
    pos = np.array(multi_person)[:, :18, :3]
    return pos


def reg_infer(pos):
    conf = pos[:, :, -1:]
    pos = aligner.align_points(pos)
    pos = (pos - 129) / 255
    pos = np.concatenate([pos, conf], -1)
    pos = pos.reshape([pos.shape[0], -1])
    pos = torch.Tensor(pos).cuda()
    out = model(pos)
    _, classidx = torch.max(out, 1)
    return classidx


model = NeuralNet(18 * 3).cuda()
model.load_state_dict(torch.load('./49_model.ckpt'))
body_estimation = Body('model/body_pose_model.pth')
aligner = AlignPoints()

print(f"Torch device: {torch.cuda.get_device_name()}")

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('')
# cap.set(3, 400)
# cap.set(4, 300)
j = 0

ims_path = '.\\test_data\\test'
ims_path_lst = glob(ims_path + '\\*.jpg')
# while True:
for im_path in ims_path_lst:
    j += 1
    try:
        oriImg = cv2.imread(im_path)
        st = time.time()
        # oriImg = cv2.resize(oriImg,(240,120),interpolation=cv2.INTER_CUBIC)
        corrs, subset = body_estimation(oriImg)
        subset_vis = subset.copy()
        oriImg = util.draw_bodypose(oriImg, corrs, subset_vis)
        corrs = prepare_posreg_multiperson(corrs, subset)
        preds = reg_infer(corrs)
        preds = preds.cpu().numpy()

        for i in range(corrs.shape[0]):
            xmin, ymin = np.min(corrs[i, :, 0]), np.min(corrs[i, :, 1])
            if preds[i] == 0:
                cv2.putText(oriImg, 'zuo', (int(xmin - 10), int(ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            [255, 0, 255], 1)
            if preds[i] == 1:
                cv2.putText(oriImg, 'tang', (int(xmin - 10), int(ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            [255, 0, 255], 1)
            if preds[i] == 2:
                cv2.putText(oriImg, 'zhan', (int(xmin - 10), int(ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            [255, 0, 255], 1)
            if preds[i] == 3:
                cv2.putText(oriImg, 'zuo_di', (int(xmin - 10), int(ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            [255, 0, 255], 1)
            # print(i, preds)
            # print(time.time()-st)
        cv2.imwrite('.\\test_data\\re\\' + str(j).zfill(4) + '.jpg', oriImg)
        # cv2.imshow('demo', oriImg)
        # cv2.waitKey()
    except:
        continue

cap.release()
cv2.destroyAllWindows()
