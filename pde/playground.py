import torch
from cprint import c_print
from codetiming import Timer
import time
import logging

from pde.NeuralPDE import NeuralPDE
from pde.cartesian_grid.cart_grid_setup import grid_setup

from pdes.PDEs import Poisson, LearnedFunc, NNFunc
from config import Config
from pde.loss import MSELoss
from utils import show_grid, setup_logging

setup_logging()
def fit_model():

    cfg = Config()
    pde_fn = NNFunc(cfg, device=cfg.device)

    optim = torch.optim.Adam(pde_fn.parameters(), lr=0.01)

    us_base = torch.load('us.pt', weights_only=True) # torch.full((N_us_grad,), 0., device=cfg.DEVICE)
    loss_fn = MSELoss(us_base)

    pde_adj = NeuralPDE(pde_fn, loss_fn, cfg)

    st = time.time()
    for i in range(200):
        print()
        c_print(f'Iteration {i}, time: {time.time() - st:.2g}s', color="bright_cyan")
        st = time.time()
        with Timer(text="Forward time: {:.3f}", logger=logging.info):
            pde_adj.forward_solve()
        with Timer(text="Adjoint time: {:.3f}", logger=logging.info):
            loss = pde_adj.adjoint_solve()
        pde_adj.backward()
        optim.step()


        loss = loss.item()
        print(f'{loss = :.3g}')
        # for n, p in pde_fn.named_parameters():
        #     print(f'p = {p.data.cpu()}')
            # print(f'grad = {p.grad.data.cpu()}')
        optim.zero_grad()

    us, _, _ = pde_adj.us_grid.get_us_mask()
    show_grid(us, "us")

    torch.save(pde_fn, "model.pt")

def true_pde():
    cfg = Config()
    pde_fn = Poisson(cfg, device=cfg.device)
    pde_grid = grid_setup(cfg)
    pde_adj = NeuralPDE(pde_fn, pde_grid, None, cfg)

    pde_adj.forward_solve()

    us, grad_mask, _ = pde_adj.us_grid.get_us_mask()
    show_grid(us, "us")
    us = us[grad_mask]
    torch.save(us, 'us.pt')


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(linewidth=150, precision=3)

    true_pde()
    exit("Done")
    fit_model()

