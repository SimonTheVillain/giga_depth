import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class MaskLoss(nn.Module):
    def __init__(self, type="mask"):
        super(MaskLoss, self).__init__()
        self.type = type
        if type == "mask" or type == "automask":
            self.loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        else:
            print("loss not implemented yet")

    def forward(self, sigma, x, x_gt, mask_gt):
        if self.type == "mask":
            return torch.mean(self.loss(sigma, mask_gt))
        if self.type == "automask":
            mask = mask_gt.clone()
            mask[torch.abs(x - x_gt) > (1 / 1216)] = 0.0 # invalidate every pixel we are more than one pixel away
            loss = self.loss(sigma, mask)
            loss[mask == 0.0] *= 10.0
            return torch.mean(loss)

class StereoLoss(nn.Module):

    def __init__(self, ch_in=1, width=1216, baseline_proj=0.0634, baseline_stereo=0.07501):
        super(StereoLoss, self).__init__()

        self.baseline_proj = baseline_proj
        self.baseline_right = baseline_stereo
        self.xtl = XTLoss(width, ch_in)#use width as max disparity

    '''
    Takes left and right images + normalized x_position on the sensor
    '''
    def forward(self, left_img, right_img, x_pos, show_debug=False):
        left_img = torch.nn.functional.interpolate(left_img, (left_img.shape[2] // 2, left_img.shape[3] // 2), mode="bilinear")
        right_img = torch.nn.functional.interpolate(right_img, (left_img.shape[2], left_img.shape[3]), mode="bilinear")
        device = left_img.device
        x_pos = x_pos * x_pos.shape[3]
        x0 = torch.arange(0, x_pos.shape[3], device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        disp = x_pos - x0
        disp *= -1.0 # pattern positions seems to be shifted left with increasing disparity

        return self.xtl(left_img, right_img, disp, show_debug)

class XTLoss(nn.Module):
    '''
    Args:
        left_img right_img: N * C * H * W,
        dispmap : N * H * W
    '''

    def __init__(self, max_disp, ch_in=3):
        super(XTLoss, self).__init__()
        self.max_disp = max_disp
        self.theta = torch.Tensor(
            [[1, 0, 0],
             [0, 1, 0]]
        )
        self.inplanes = ch_in
        self.outplanes = ch_in

    def forward(self, left_img, right_img, dispmap, show_debug=False):

        n, c, h, w = left_img.shape

        # pdb.set_trace()
        theta = self.theta.repeat(left_img.size()[0], 1, 1)

        # grid = F.affine_grid(theta, left_img.size())
        grid = F.affine_grid(theta, left_img.size(), align_corners=True)  # enable old behaviour
        # print(grid)
        grid = grid.cuda()
        # print(grid)
        # print(dispmap.shape) #simon: they tend to go towards 0
        # print(torch.max(dispmap))
        # print(torch.mean(dispmap))
        # print(torch.min(dispmap))
        dispmap_norm = dispmap * 2 / w  # times 2 because grid_sample normalizes between -1 and 1!
        # dispmap_norm = dispmap_norm.cuda() # why cuda? it already is on a cuda device!
        # pdb.set_trace()
        # print(dispmap_norm.shape)
        dispmap_norm = dispmap_norm.squeeze(1).unsqueeze(3)
        # print(dispmap_norm.shape)
        dispmap_norm = torch.cat((dispmap_norm, torch.zeros(dispmap_norm.size()).cuda()), dim=3)
        # print(dispmap_norm.shape)
        grid -= dispmap_norm

        # recon_img = F.grid_sample(right_img, grid)
        recon_img = F.grid_sample(right_img, grid, align_corners=True)  # enable old behaviour

        if show_debug:
            disp_pred_left = dispmap[0, 0, :, :].clone()
            disp_pred_left -= disp_pred_left.min()
            disp_pred_left /= disp_pred_left.max() + 0.1
            cv2.imshow("dispmap", disp_pred_left.clone().detach().cpu().numpy())

            cv2.imshow("recon_img", recon_img[0, 0, :, :].clone().detach().cpu().numpy())
            cv2.imshow("left_img", left_img[0, 0, :, :].clone().detach().cpu().numpy())
            cv2.imshow("right_img", right_img[0, 0, :, :].clone().detach().cpu().numpy())
            cv2.imshow("img_diff", np.abs(left_img[0, 0, :, :].clone().detach().cpu().numpy() -
                                          recon_img[0, 0, :, :].clone().detach().cpu().numpy()))
            # check if the sign is right
            recon_img2 = F.grid_sample(right_img, grid - 0.5,
                                       align_corners=True)  # enable old behaviour (seems about right)
            cv2.imshow("recon_img2", recon_img2[0, 0, :, :].clone().detach().cpu().numpy())
            cv2.waitKey(10)
            cv2.waitKey()
        # pdb.set_trace()

        #TODO: remove the next few lines of debug!!!
        if torch.any(torch.isnan(left_img)):
            print("left image has nan")

        if torch.any(torch.isnan(right_img)):
            print("left image has nan")

        if torch.any(torch.isnan(dispmap_norm)):
            print("dispmap has nan")

        if torch.any(torch.isnan(recon_img)):
            cv2.imshow("nanimage", recon_img[0,0,:,:].detach().cpu().numpy())
            cv2.imshow("nanimage2", recon_img[1,0,:,:].detach().cpu().numpy())
            cv2.waitKey()
            print("shit, there is a nan")

        recon_img_LCN, _, _ = self.LCN(recon_img, 9)


        if torch.any(torch.isnan(recon_img_LCN)):
            print("shit, there is a nan")

        left_img_LCN, _, left_std_local = self.LCN(left_img, 9)

        # pdb.set_trace()
        losses = torch.abs(((left_img_LCN - recon_img_LCN) * left_std_local))

        # pdb.set_trace()
        #print(f"before{losses.shape}")
        losses = self.ASW(left_img, losses, 12, 2)  # adaptive support window
        #print(f"after{losses.shape}")
        if True:
            outlier_loss = torch.zeros_like(dispmap)
            outlier_loss[dispmap < 0] = -dispmap[dispmap < 0]
            outlier_loss[dispmap > self.max_disp] = dispmap[dispmap > self.max_disp] - self.max_disp
            losses += outlier_loss.mean()

        return losses

    def LCN(self, img, kSize):
        '''
            Args:
                img : N * C * H * W
                kSize : 9 * 9
        '''
        eps = 0.001
        w = torch.ones((self.outplanes, self.inplanes, kSize, kSize)).cuda() / (kSize * kSize)
        mean_local = F.conv2d(input=img, weight=w, padding=kSize // 2)

        mean_square_local = F.conv2d(input=img ** 2, weight=w, padding=kSize // 2)
        # std_local = (mean_square_local - mean_local ** 2) * (kSize ** 2) / (kSize ** 2 - 1)
        std_local = torch.sqrt(torch.clamp(mean_square_local - mean_local ** 2, min=eps**2))  # fix by simon!!!!! (why kSize ** 2 - >1< ???)

        #print(torch.min(mean_square_local - mean_local ** 2))
        #print(torch.min(torch.clamp(mean_square_local - mean_local ** 2, min=eps**2)))
        if torch.any(torch.isnan(mean_square_local)):
            print("shit")
        if torch.any(torch.isnan(mean_local)):
            print("shit")
        if torch.any(torch.isnan(std_local)):
            print("shit")
        epsilon = 1e-6

        return (img - mean_local) / (std_local + epsilon), mean_local, std_local

    def ASW(self, img, Cost, kSize, sigma_omega):

        # pdb.set_trace()
        weightGraph = torch.zeros(img.shape, requires_grad=False).cuda()
        CostASW = torch.zeros(Cost.shape, dtype=torch.float, requires_grad=True).cuda()

        pad_len = kSize // 2
        img = F.pad(img, [pad_len] * 4)
        Cost = F.pad(Cost, [pad_len] * 4)
        n, c, h, w = img.shape
        # pdb.set_trace()

        for i in range(kSize):
            for j in range(kSize):
                tempGraph = torch.abs(
                    img[:, :, pad_len: h - pad_len, pad_len: w - pad_len] - img[:, :, i:i + h - pad_len * 2,
                                                                            j:j + w - pad_len * 2])
                tempGraph = torch.exp(-tempGraph / sigma_omega)
                weightGraph += tempGraph
                CostASW += tempGraph * Cost[:, :, i:i + h - pad_len * 2, j:j + w - pad_len * 2]

        CostASW = CostASW / weightGraph

        return CostASW.mean((1, 2, 3))# mean so that we have one value for each sample