import torch
from torch import nn, optim 
import os 
from utils import * 
import numpy as np 
import scipy.io
import matplotlib.pyplot as plt 
import math
import time

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

expand_x = 1; n_3 = 500; 
expand = 1
shrink = 5
gamma = 0.000000001

data_all = ["data/EO1H0420342007273110PF_stacked_"]
c_all = ["3"]

max_iter = 4000
lr_real = 0.003

# --------- CREATE RESULT DIR ----------
os.makedirs("results", exist_ok=True)

class Y_net(nn.Module): 
    def __init__(self,n_1,n_2,n_3):
        super(Y_net, self).__init__()
        self.A_hat = nn.Parameter(torch.Tensor(n_3*expand,n_1,n_2//shrink))
        self.B_hat = nn.Parameter(torch.Tensor(n_3*expand,n_2//shrink,n_2))
        
        self.net = nn.Sequential(
            permute_change(1,2,0),
            nn.Linear(int(n_3*expand),int(n_3*expand),bias = False),
            nn.LeakyReLU(),
            nn.Linear(int(n_3*expand),n_3,bias = False)
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.A_hat.size(0))
        self.A_hat.data.uniform_(-stdv, stdv)
        self.B_hat.data.uniform_(-stdv, stdv)
                                    
    def forward(self):
        x = torch.matmul(self.A_hat,self.B_hat)
        return self.net(x)


for data in data_all:
    for c in c_all:

        F_norm = nn.MSELoss()
        
        file_name = data+c+'.mat'
        mat = scipy.io.loadmat(file_name)

        # --------- LOAD ----------
        X_np = mat["Nhsi"]
        X = torch.from_numpy(X_np).type(dtype).cuda()

        # --------- TRY GT ----------
        if "Ohsi" in mat:
            X_gt = torch.from_numpy(mat["Ohsi"]).type(dtype).cuda()
        elif "GT" in mat:
            X_gt = torch.from_numpy(mat["GT"]).type(dtype).cuda()
        else:
            X_gt = X.clone()  # fallback

        X_obs = X.clone()

        model = Y_net(X.shape[0],X.shape[1],X.shape[2]).type(dtype)
        
        mask = torch.ones(X.shape).type(dtype)
        mask[X == 0] = 0 
        X[mask == 0] = 0
        
        params = [x for x in model.parameters()]
        
        s = sum([np.prod(list(p.size())) for p in params]) 
        print('Number of params: %d' % s)

        optimizier = optim.Adam(params, lr=lr_real, weight_decay=10e-8) 
        
        show = [43,23,12]

        # --------- SELECT 4 PRINT POINTS ----------
        print_iters = [0, 1000, 2000, 3000]

        for iter in range(max_iter):

            X_Out_real = model()

            loss = F_norm(X_Out_real*mask,X*mask)

            X_Out_real[mask == 1] = X[mask == 1]

            i = 0
            for p in params:
                i += 1
                if i == 1:
                    loss += gamma*torch.norm(p[:,1:,:]-p[:,:-1,:],1)
                if i == 4:
                    loss += gamma*torch.norm(p[1:,:]-p[:-1,:],1)
                if i == 2:
                    loss += gamma*torch.norm(p[:,:,1:]-p[:,:,:-1],1)

            optimizier.zero_grad()
            loss.backward(retain_graph=True)
            optimizier.step()

            # --------- CONTROLLED PRINTING ----------
            if iter in print_iters:

                print(f"\n===== ITERATION {iter} =====")

                X_rec = X_Out_real.detach().cpu().numpy()
                X_gt_np = X_gt.cpu().numpy()

                psnr = peak_signal_noise_ratio(X_gt_np, X_rec, data_range=1)
                ssim = structural_similarity(X_gt_np, X_rec, channel_axis=-1, data_range=1)

                print("PSNR:", psnr)
                print("SSIM:", ssim)

                # RGB visualization
                gt_rgb = np.clip(np.stack((
                    X_gt_np[:,:,show[0]],
                    X_gt_np[:,:,show[1]],
                    X_gt_np[:,:,show[2]]),2),0,1)

                obs_rgb = np.clip(np.stack((
                    X_obs[:,:,show[0]].cpu().numpy(),
                    X_obs[:,:,show[1]].cpu().numpy(),
                    X_obs[:,:,show[2]].cpu().numpy()),2),0,1)

                rec_rgb = np.clip(np.stack((
                    X_rec[:,:,show[0]],
                    X_rec[:,:,show[1]],
                    X_rec[:,:,show[2]]),2),0,1)

                plt.figure(figsize=(12,4))

                plt.subplot(1,3,1)
                plt.imshow(gt_rgb)
                plt.title("GT")

                plt.subplot(1,3,2)
                plt.imshow(obs_rgb)
                plt.title("CR (Observed)")

                plt.subplot(1,3,3)
                plt.imshow(rec_rgb)
                plt.title("Recon")

                plt.savefig(f"results/iter_{iter}.png")
                plt.close()

        # --------- FINAL OUTPUT ----------
        print("\n===== FINAL RESULT =====")

        X_rec = X_Out_real.detach().cpu().numpy()
        X_gt_np = X_gt.cpu().numpy()

        # --------- SAVE MAT FILE (NEW) ----------
        scipy.io.savemat(
            "results/final_recon.mat",
            {
                "recon": X_rec.astype(np.float32),
                "gt": X_gt_np.astype(np.float32)
            }
        )

        psnr = peak_signal_noise_ratio(X_gt_np, X_rec, data_range=1)
        ssim = structural_similarity(X_gt_np, X_rec, channel_axis=-1, data_range=1)

        print("Final PSNR:", psnr)
        print("Final SSIM:", ssim)

        # Save checkpoint
        torch.save(model.state_dict(), "results/model_final.pth")

        # Final image
        gt_rgb = np.clip(np.stack((
            X_gt_np[:,:,show[0]],
            X_gt_np[:,:,show[1]],
            X_gt_np[:,:,show[2]]),2),0,1)

        obs_rgb = np.clip(np.stack((
            X_obs[:,:,show[0]].cpu().numpy(),
            X_obs[:,:,show[1]].cpu().numpy(),
            X_obs[:,:,show[2]].cpu().numpy()),2),0,1)

        rec_rgb = np.clip(np.stack((
            X_rec[:,:,show[0]],
            X_rec[:,:,show[1]],
            X_rec[:,:,show[2]]),2),0,1)

        plt.figure(figsize=(12,4))

        plt.subplot(1,3,1)
        plt.imshow(gt_rgb)
        plt.title("GT")

        plt.subplot(1,3,2)
        plt.imshow(obs_rgb)
        plt.title("CR (Observed)")

        plt.subplot(1,3,3)
        plt.imshow(rec_rgb)
        plt.title("Recon")

        plt.savefig("results/final.png")
        plt.close()