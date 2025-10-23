import torch
from torch.autograd import Function
from torch.func import vjp
from torch.linalg import solve_triangular as sotri


class GP(Function):
    @staticmethod
    def forward(ctx, A, rhs, vtol):
        out, logdet, quad, A, all_rhs, all_soln = logdet_quad_fwd_solve(A, rhs, vtol)
        ctx.save_for_backward(A, all_rhs, all_soln)
        return out, logdet, quad

    @staticmethod
    def backward(ctx, *grads):
        A, all_rhs, all_soln = ctx.saved_tensors
        out = logdet_quad_bwd(grads, A, all_rhs, all_soln)
        return out


class GPNet(Function):
    @staticmethod
    def forward(ctx, A, rhs, net, vtol):
        out, logdet, quad, A, all_rhs, all_soln = logdet_quad_fwd(A, rhs, net, vtol)
        ctx.save_for_backward(A, all_rhs, all_soln)
        return out, logdet, quad

    @staticmethod
    def backward(ctx, *grads):
        A, all_rhs, all_soln = ctx.saved_tensors
        out = logdet_quad_bwd(grads, A, all_rhs, all_soln)
        return out


def gp_net_logdet_quad(K, y, net, vtol, case):
    if case == "net":
        out = GPNet.apply(K, y, net, vtol)
    elif case == "dense":
        out = GP.apply(K, y, vtol)
    elif case == "actual":
        out = logdet_quad(K, y)
    else:
        raise ValueError(f"{case=} is not valid (net or dense)")
    return out


def logdet_quad(A, rhs):
    L = torch.linalg.cholesky(A)
    logdet = 2.0 * torch.sum(torch.log(torch.diag(L)))
    all_soln = sotri(L.T, sotri(L, rhs, upper=False), upper=True)
    quad = torch.sum(rhs * all_soln[:, [0]])
    out = logdet + quad
    return out


def logdet_quad_fwd_solve(A, rhs, vtol):
    num_samples = round(1 / vtol**2.0)
    probes = torch.randn(A.shape[1], num_samples, dtype=A.dtype, device=A.device)
    coef = 1.0 / probes.shape[-1]
    all_rhs = torch.concatenate((rhs, probes), dim=-1)
    logdet = 0.0
    L = torch.linalg.cholesky(A)
    all_soln = sotri(L.T, sotri(L, all_rhs, upper=False), upper=True)
    all_rhs = torch.concatenate((-all_soln[:, [0]], coef * probes), dim=-1)
    quad = torch.sum(rhs * all_soln[:, [0]])
    out = logdet + quad
    return out, logdet, quad, A, all_rhs, all_soln


def logdet_quad_fwd(A, rhs, net, vtol):
    num_samples = round(1 / vtol**2.0)
    probes = torch.randn(A.shape[1], num_samples, dtype=A.dtype, device=A.device)
    coef = 1.0 / probes.shape[-1]
    all_rhs = torch.concatenate((rhs, probes), dim=-1)
    logdet = 0.0
    all_soln = net(A[None, :, :])[0] @ all_rhs
    all_rhs = torch.concatenate((-all_soln[:, [0]], coef * probes), dim=-1)
    quad = torch.sum(rhs * all_soln[:, [0]])
    out = logdet + quad
    return out, logdet, quad, A, all_rhs, all_soln


def logdet_quad_bwd(grads, A, all_rhs, all_soln):
    def fun(theta):
        return theta @ all_soln

    dA = vjp_derivs(fun=fun, primals=A, duals=grads[0] * all_rhs)
    out = dA + tuple([None] * 3)
    return out


def vjp_derivs(fun, primals, duals):
    _, vjpfun = vjp(fun, primals)
    output = vjpfun(duals)
    return output


def rbf(x1, x2, ls):
    # x1: [N1]
    # x2: [N2]
    ker = x2[:, None] - x2[None, :]
    ker = torch.exp(-((ker / ls) ** 2.0))
    return ker  # [N1,N2]


def ker_fn(x, params):
    # x: [N]
    ls, log_sigma = params
    sigma = torch.exp(torch.clamp(log_sigma, -3.0))
    ker = rbf(x1=x, x2=x, ls=ls)
    ker = ker + sigma * torch.eye(x.shape[-1], device=x.device)
    return ker


def gp_predict(x_test, x_train, y_train, params):
    ls, log_sigma = params
    sigma = torch.exp(torch.clamp(log_sigma, -3.0))
    K_11 = rbf(x1=x_train, x2=x_train, ls=ls)
    K_11 = K_11 + sigma * torch.eye(x_train.shape[-1], device=x_train.device)
    K_21 = rbf(x1=x_test, x2=x_train, ls=ls)
    K_22 = rbf(x1=x_test, x2=x_test, ls=ls) + 1e-2 * torch.eye(x_test.shape[-1], device=x_test.device)

    L = torch.linalg.cholesky(K_11)
    K_11_y = sotri(L.T, sotri(L, y_train, upper=False), upper=True)
    mu = K_21 @ K_11_y
    K_11_K = sotri(L.T, sotri(L, K_21.T, upper=False), upper=True)
    cov = K_22 - K_21 @ K_11_K
    return mu[:, 0], cov
