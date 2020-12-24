import cv2
import numpy as np
import math
import time
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib
import torch
from torchvision import transforms
import torch.nn.functional as F
from src import util
from src.model import bodypose_model


class Body(object):
    def __init__(self,model_path):
        self.model = bodypose_model()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        model_dict = util.transfer(self.model, torch.load(model_path))
        self.model.load_state_dict (model_dict)
        self.model.eval ()


    def __call__(self, oriImg):
        boxsize = 368
        stride = 8
        padValue = 128
        scale = 1.0 * boxsize / oriImg.shape[0]

        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, stride, padValue)
        im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 -0.5
        im =np.ascontiguousarray(im)

        data = torch.from_numpy(im).float()

        if torch.cuda.is_available():
            data = data.cuda()
        with torch.no_grad():
            start = time.time()
            _, heatmap= self.model(data)


        heatmap = F.interpolate(heatmap, (heatmap.shape[2]*stride, heatmap.shape[3]*stride),mode='bicubic',align_corners=False)
        heatmap = heatmap[:,:,:imageToTest_padded.shape[0] - pad[2],:imageToTest_padded.shape[1] - pad[3]]
        heatmap = F.interpolate(heatmap, (oriImg.shape[0],oriImg. shape [1]), mode='bicubic',align_corners=False)
        heatmap= torch.squeeze(heatmap)
        heatmap = heatmap.permute( (1, 2, 0))


        heatmap_view = heatmap.view(-1, heatmap.shape[2])
        heatmap_max, heatmap_max_index = torch.max (heatmap_view, 0)


        peaks_x = heatmap_max_index//heatmap.shape[1]
        peaks_y = heatmap_max_index%heatmap.shape[1]


        out = torch.stack( (peaks_y.float(),peaks_x.float(),heatmap_max))
        out = torch.transpose(out,0,1)
        fps = 1/(time.time()-start)
        out = out.cpu ().numpy()
        return out, fps


def draw_bodypose(canvas, candidate, thre=0.2):
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]
    limbSeq = np.array(limbSeq)
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for i in range(18):
        if candidate[i,2]>thre:
            cv2.circle(canvas, (int(candidate[i,0]), int(candidate[i,1])), 4, colors[i], thickness=-1)

    for i in range(17):
        if candidate[limbSeq[i,0]-1,2]>thre and candidate[limbSeq[i,1]-1,2]>thre:
            cur_canvas = canvas.copy()
            y1 =  candidate[limbSeq[i,0]-1,0]
            y2 =  candidate[limbSeq[i,1]-1,0]
            x1 =  candidate[limbSeq[i,0]-1,1]
            x2 =  candidate[limbSeq[i,1]-1,1]
            length = ((x1-x2)**2+(y1-y2)**2)**0.5
            angle = math.degrees(math.atan2(x1-x2,y1-y2))
            polygon = cv2.ellipse2Poly((int((y1+y2)/2), int((x1+x2)/2)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    return canvas


if __name__ == "__main__":
    import time

    body_estimation = Body('../model/body_pose_model.pth')

    test_image = 'D:\myFile\data\pose\\0712_pos_reg_myself\im\sit1\\000000.jpg'
    oriImg = cv2.imread(test_image)  # B,G,R order
    st = time.time()
    candidate, subset = body_estimation(oriImg)
    print(time.time()-st)
    # canvas = util.draw_bodypose(oriImg, candidate, subset)
    canvas = draw_bodypose(oriImg, candidate)
    plt.imshow(canvas[:, :, [2, 1, 0]])
    plt.show()
