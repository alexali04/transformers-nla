import torch
from torch.optim import AdamW


class GrokFast(AdamW):
    def __init__(self, params, lr, wd, kwargs):
        """
        Applies EMA over update vectors in AdamW

        u_t = eta * (m^_t / sqrt(v^_t) + eps)
        ema_t = alpha * ema_t-1 + (1 - alpha) * u_t
        theta = theta + lambda * ema_t
        """
        super().__init__(params, lr=lr, weight_decay=wd)

        self.alpha = kwargs["alpha"]
        self.lamb = kwargs["lambda"]
        self._inited = False

    def _init_state(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                # --- AdamW lazy init, copied from torch/optim/adamw.py ---
                state["step"] = torch.zeros([], dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["mu"] = torch.zeros_like(p, memory_format=torch.preserve_format, device=p.device)

        self._inited = True

    @torch.no_grad()
    def step(self, closure=None):
        if not self._inited:
            self._init_state()

        for group in self.param_groups:
            alph, lambd = self.alpha, self.lamb

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                g = p.grad
                mu = state["mu"]
                mu.mul_(alph).add_(g, alpha=(1 - alph))
                g_hat = g.add(mu, alpha=lambd)
                p.grad.copy_(g_hat)

        return super().step(closure)


class AdamW_default(AdamW):
    def __init__(self, params, lr, wd, _):
        super().__init__(params, lr=lr, weight_decay=wd, eps=1e-16)


def get_optimizer(optimizer, params, lr, wd, args):
    opt = _OPT[optimizer](params, lr, wd, vars(args))
    return opt


_OPT = {
    "adamw": AdamW_default,
    "grokfast": GrokFast,
}
