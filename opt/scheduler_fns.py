import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler

from loss.loss_fns import get_loss, pre_process
from nn.nn_fns import forward_pass


def get_scheduler(opt, args, model, train_loader, device, R):
    if args.sch_name == "cos_grad":  # update later... i know its messy
        sch = CosineSimilarityGradientLR(
            opt=opt,
            model=model,
            train_loader=train_loader,
            device=device,
            args=args,
            R=R,
        )
    else:
        sch = SCHS[args.sch_name](opt, step_size=args.sch_sz, gamma=args.sch_gamma, thres=args.sch_thres)
    return sch


class CosineSimilarityGradientLR(_LRScheduler):
    """
    1. Take model
    2. Sample n batches
    3. Compute gradients for each batch
    4. Compute pair-wise cosine similarity: sigma_g = 2/(n(n-1)) * sum_{i = h} (cos(g_i, g_j))
    5. if sigma_g is large, decrease lr (slowly to learn high precision). if small, increase lr (escape minima)
    """

    def __init__(self, opt, args, model, train_loader, device, R):
        self.args = args
        self.model = model
        self.R = R
        self.train_loader = train_loader
        self.ds_idxS = [idx for idx in range(len(self.train_loader))]
        self.device = device

        self.n_batches = args.sch_batches  # 10
        self.sigma_thres = args.sch_sigma_thres  # 0.9
        self.thres = args.sch_thres  # 1e-6
        self.inc = args.sch_gamma  # 0.1
        self.step_size = args.sch_sz  # 1k

        self.did_update = 0
        self.didnt_update = 0

        total_dim = sum(p.numel() for p in self.model.parameters())
        self.gradients = torch.zeros(self.n_batches, total_dim, device=device)
        self.loss_fn = get_loss(args.loss)

        super().__init__(opt, last_epoch=-1)

    def collect_gradients(self):
        self.model.train()
        self.optimizer.zero_grad()

        for i in range(self.n_batches):
            x, y = next(self.train_loader[np.random.choice(self.ds_idxS)])
            x, y = x.to(self.device), y.to(self.device)

            y_hat = forward_pass(args=self.args, model=self.model, x=x, R=self.R)
            y, y_hat = pre_process(y, y_hat, x, self.args.loss)
            loss = self.loss_fn(y_hat, y)

            grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=False, retain_graph=False)

            grad_flat = torch.cat([g.view(-1) for g in grads])
            self.gradients[i] = F.normalize(grad_flat, p=2, dim=0)
            self.optimizer.zero_grad()

        return self.gradients

    def get_lr(self):
        # no change, keep current LR
        if (self.last_epoch + 1) % self.step_size != 0:
            return [pg["lr"] for pg in self.optimizer.param_groups]

        print("Collecting gradients and updating lr....")
        G = self.collect_gradients()
        G_matr = G @ G.T

        cos_ele = torch.tril(G_matr, diagonal=-1).sum()

        sigma_g = 2 / (self.n_batches * (self.n_batches - 1)) * cos_ele

        # if sigma_g is large, decrease lr
        # if sigma_g is small, increase lr

        factor = self.inc if sigma_g > self.sigma_thres else 1 / self.inc

        if factor < 1:
            print(f"sigma_g: {sigma_g:.3f} | bigger than threshold {self.sigma_thres} | multiplying lr by {factor:.1e}")
            self.did_update += 1
        else:
            print(
                f"sigma_g: {sigma_g:.3f} | smaller than threshold {self.sigma_thres} | multiplying lr by {factor:.1e}"
            )
            self.didnt_update += 1

        return [max(g["lr"] * factor, self.thres) for g in self.optimizer.param_groups]

    def step(self, epoch=None):
        return super().step(epoch)


class StepThresholdLR(_LRScheduler):
    def __init__(self, optimizer, step_size, gamma, thres, **_):
        super().__init__(optimizer, last_epoch=-1)
        self.step_size, self.gamma, self.thres = step_size, gamma, thres

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        coef = self.last_epoch // self.step_size
        out = [max(lr * self.gamma**coef, self.thres) for lr in self.base_lrs]
        return out


class ConstantLR(_LRScheduler):
    def __init__(self, optimizer, **_):
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        return self.base_lrs


SCHS = {
    "step": StepThresholdLR,
    "cons": ConstantLR,
    "cos_grad": CosineSimilarityGradientLR,
}
