import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from kan1 import KAN
# from kan import *

# ----------------------------
# KAN + GP 模型
# ----------------------------

class KAN_GP_Model(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2)
        )     
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)


    def forward(self, x):
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# ----------------------------
# 总封装
# ----------------------------
feature_extractor = KAN([3,3,2])

class KANGP:
    def __init__(self, x_train, y_train, lr=0.01, seed=47, device = None):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = device if device is not None else torch.device("cpu")
        
        self.x_train, self.y_train = x_train, y_train
        self.feature_extractor = feature_extractor

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = KAN_GP_Model(x_train, y_train, self.likelihood, self.feature_extractor)

        self.optimizer = torch.optim.Adam([
            {"params": self.model.feature_extractor.parameters(), "weight_decay": 1e-5},
            {"params": self.model.covar_module.parameters()},
            {"params": self.model.mean_module.parameters()},
            {"params": self.model.likelihood.parameters()},
        ], lr=lr)

        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        self.train_losses = []

    def train_model(self, epochs=1200):
        self.model.train()
        self.likelihood.train()
        self.train_losses = []

        for _ in tqdm(range(epochs)):
            self.optimizer.zero_grad()
            output = self.model(self.x_train)
            loss = -self.mll(output, self.y_train)
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
    
