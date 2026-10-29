import torch
import gpytorch
import numpy as np
from tqdm import tqdm
import random
import time


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood):
        super().__init__(x_train, y_train, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BaseGP:
    def __init__(self, x_train, y_train, lr=0.1, seed=47, device=None):
            
        torch.seed()
        np.random.seed(None)

        self.device = device if device is not None else torch.device("cpu")

        # 确保 y 是一维
        if y_train.dim() > 1 and y_train.shape[-1] == 1:
            y_train = y_train.view(-1)
        self.x_train = x_train.to(self.device).float()
        self.y_train = y_train.to(self.device).float()

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = ExactGPModel(self.x_train, self.y_train, self.likelihood).to(self.device)

        # 核函数长度尺度
        self.model.covar_module.base_kernel.lengthscale = torch.tensor(
            np.random.uniform(0.1, 5.0), dtype=torch.float32
        )
        # 核函数输出尺度
        self.model.covar_module.outputscale = torch.tensor(
            np.random.uniform(0.5, 2.0), dtype=torch.float32
        )
        # 均值函数
        self.model.mean_module.constant = torch.tensor(
            np.random.uniform(-1.0, 1.0), dtype=torch.float32
        )
        # 噪声
        self.likelihood.noise_covar.noise = torch.tensor(
            np.random.uniform(1e-4, 1e-2), dtype=torch.float32
        )

        # optimizer should include likelihood params as well
        params = list(self.model.parameters()) + list(self.likelihood.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        self.train_losses = []

    def train_model(self, epochs=200, noise_std = 0.01):
        self.model.train()
        self.likelihood.train()
        self.train_losses = []

        for i in tqdm(range(epochs)):
            self.optimizer.zero_grad()
            x_batch = self.x_train
            # if noise_std > 0:
            #     x_batch = x_batch + noise_std * torch.randn_like(x_batch) #进入噪声
            output = self.model(self.x_train)
            noisy_y = self.y_train + noise_std * torch.randn_like(self.y_train)
            loss = -self.mll(output, noisy_y)
            loss.backward()
            self.optimizer.step()
            self.train_losses.append(loss.item())

    def predict(self, x_test, return_std=True):
        """
        x_test : torch.Tensor (n_test, d) -- should already be on the same scale as training (and on self.device)
        返回 (mean, std) 都是 numpy arrays (shape (n_test,))
        """
        self.model.eval()
        self.likelihood.eval()

        x_test = x_test.to(self.device).float()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(x_test))
            mean = pred.mean.detach().cpu().numpy().ravel()
            std = pred.stddev.detach().cpu().numpy().ravel() if return_std else None
        return mean, std

    def to(self, device):
        self.device = device
        self.x_train = self.x_train.to(device)
        self.y_train = self.y_train.to(device)
        self.model.to(device)
        self.likelihood.to(device)
        return self
