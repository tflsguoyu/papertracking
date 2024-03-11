import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt
import torch 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def warp(img, flow, device):
    h, w = img.shape[:2]
    img = torch.tensor(img).to(device).permute(2,0,1)[None, ...].float()
    flow = torch.tensor(flow).to(device).permute(2,0,1)[None, ...]

    # mesh grid 
    X = torch.arange(0, w)
    Y = torch.arange(0, h)
    X, Y = torch.meshgrid(X, Y, indexing='xy')
    grid = torch.stack((X, Y), 2).float().to(device)
    grid = grid.permute(2,0,1)[None, ...]
    grid += flow

    # xx = torch.arange(0, w).view(1,-1).repeat(h,1)
    # yy = torch.arange(0, h).view(-1,1).repeat(1,w)
    # xx = xx.view(1,1,h,w)
    # yy = yy.view(1,1,h,w)
    # grid = torch.cat((xx, yy), 1).float().to(device) + flow

    # scale grid to [-1,1] 
    grid[:,0,:,:] = 2 * grid[:,0,:,:] / max(w-1, 1) - 1
    grid[:,1,:,:] = 2 * grid[:,1,:,:] / max(h-1, 1) - 1

    grid = grid.permute(0,2,3,1)        
    img = torch.nn.functional.grid_sample(img, grid, align_corners = True, padding_mode = 'zeros')

    return img[0].permute(1,2,0).cpu().numpy().astype(np.uint8)

def warp_image_dense(im, flow):
    H, W = im.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(W)
    flow[:,:,1] += np.arange(H)[:,np.newaxis]
    return cv2.remap(im, flow, None, interpolation=cv2.INTER_LINEAR, borderValue=(0,0,0))

im = cv2.imread('C:/Users/guoyu/data_new/paper4/cam032/color/00033.jpg')
flow = scipy.io.loadmat('C:/Users/guoyu/data_new/paper4/cam032/ours/00033_denseflow.mat')
flow = flow['flow']

flow_img = cv2.imread('C:/Users/guoyu/data_new/paper4/cam032/ours/00033_denseflow.jpg')

im2 = cv2.imread('C:/Users/guoyu/data_new/paper4/cam032/color/00034.jpg')

# im_warp = warp_image_dense(im, flow)
im_warp = warp(im, flow, device)

plt.figure()
plt.subplot(131)
plt.imshow(im[...,::-1])
plt.grid(color='g')

plt.subplot(132)
plt.imshow(im_warp[...,::-1])
plt.grid(color='g')

plt.subplot(133)
plt.imshow(im2[...,::-1])
plt.grid(color='g')

plt.show()