import torch
import cv2
import numpy as np
from numpy.linalg import inv
import matplotlib.cm as cm
import torch.cuda.profiler as profiler
import pyprof
import imageio
import random
import pdb
import torch.nn as nn

from src.utils.plotting import make_matching_figure
from src.loftr import LoFTR, default_cfg
#from torch.profiler import profile, record_function, ProfilerActivity

def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(image_name)
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return

def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()

# img0_pth = "assets/00293.jpg/img.png"
# img1_pth = "assets/00293.jpg/warped_img.png"
# Homo_gt = np.loadtxt("assets/00293.jpg/gt_homo.txt").reshape(3,3)
# print(Homo_gt[0][0])
img0_pth = "assets/00194.jpg"
img1_pth = "assets/00195.jpg"
image_pair = [img0_pth, img1_pth]
pyprof.init()
# The default config uses dual-softmax.
# The outdoor and indoor models share the same config.
# You can change the default values like thr and coarse_match_type.
matcher1 = LoFTR(config=default_cfg)
matcher1.load_state_dict(torch.load("weights/outdoor_ds.ckpt")['state_dict'])
matcher1 = matcher1.eval().cuda()
matcher2 = LoFTR(config=default_cfg)
matcher2.load_state_dict(torch.load("logs/tb_logs/indoor-ds-bs=4/version_48/checkpoints/last.ckpt")['state_dict'])
matcher2 = matcher2.eval().cuda()
img0_raw = cv2.imread(image_pair[0], cv2.IMREAD_GRAYSCALE)
img1_raw = cv2.imread(image_pair[1], cv2.IMREAD_GRAYSCALE)
#pdb.set_trace()
img0_raw = cv2.resize(img0_raw, (640, 480))
img1_raw = cv2.resize(img1_raw, (640, 480))

img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
batch1 = {'image0': img0, 'image1': img1}
batch2 = {'image0': img0, 'image1': img1}
with torch.no_grad():
    paths = [("LoFTR", "loftr_coarse", "layers", "0", "attention")]
    path = ('LoFTR', 'loftr_coarse', 'layers', '0', 'attention')
    matcher1(batch1)
    matcher2(batch2)
    mkpts0 = batch1['mkpts0_f'].cpu().numpy()
    mkpts1 = batch1['mkpts1_f'].cpu().numpy()
    mkpts0_f = batch2['mkpts0_f'].cpu().numpy()
    mkpts1_f = batch2['mkpts1_f'].cpu().numpy()

M, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
img0_warp = cv2.warpPerspective(img0_raw,M,(640, 480))
criterion = nn.MSELoss()
img0_warp = torch.tensor(img0_warp, dtype=float)
img1_raw = torch.tensor(img1_raw, dtype=float)
loss_img = criterion(img0_warp, img1_raw).numpy()
print('loss:',loss_img)

M_f, mask_f = cv2.findHomography(mkpts0_f, mkpts1_f, cv2.RANSAC, 5.0)
img0_warp_f = cv2.warpPerspective(img0_raw,M_f,(640, 480))
img0_warp_f = torch.tensor(img0_warp_f, dtype=float)
loss_img_f= criterion(img0_warp_f, img1_raw).numpy()
print('Fine-tuned loss:',loss_img_f)

image_list = (img0_warp.numpy(), img1_raw.numpy())
gif_name = 'ptb_test_result/homo.gif'
create_gif(image_list,gif_name)

image_list_f = (img0_warp_f.numpy(), img1_raw.numpy())
gif_name_f = 'ptb_test_result/homo_f.gif'
create_gif(image_list_f,gif_name_f)

image_list_O = (img0_raw, img1_raw.numpy())
gif_name_O = 'ptb_test_result/Oringinal.gif'
create_gif(image_list_O,gif_name_O)