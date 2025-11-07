import torch
from cprint import c_print
import numpy as np
from collections import defaultdict
import mup

from pde.config import Config
from pde.NeuralPDE_Graph import NeuralPDEGraph
from pde.pdes.PDEs import Fluid, FluidLearned, NNFunc
from pde.utils import setup_logging
from pde.loss import DummyLoss, MSELoss2, MSELossNorm
from pde.run.generate_graph import mesh_graph, load_graph

def true_pde():
    cfg = Config()
    # U_graph, triangles = load_graph(cfg)
    U_graph, triangles = mesh_graph(cfg)
    # U_graph, triangles = mesh_heat(cfg)

    Us_all, _ = U_graph.get_all_us_Xs()
    pde_fn = Fluid(cfg, device=cfg.device)
    # pde_fn = HeatLearned(cfg, device=cfg.DEVICE)

    # Us_target = U_graph.pde_mask.float()
    # Us_target = torch.repeat_interleave(Us_target, U_graph.N_comp, dim=0)
    loss_fn = DummyLoss()  # MaskLoss(Us_target)

    pde_adj = NeuralPDEGraph(pde_fn, U_graph, cfg, loss_fn, triangles)

    pde_adj.forward_solve()
    pde_adj.plot_interp(title="Initial solution")

    # Us_all, updt_mask, _ = U_graph.get_us_mask()
    # Us = Us_all[updt_mask]
    #
    # with open("./Us_solution.pth", "wb") as f:
    #     torch.save(Us, f)
    # print(f'{Us.shape = }, {Us_all.shape = }')


def test_adjoint():
    cfg = Config()
    # U_graph, triangles = load_graph(cfg)
    U_graph, triangles = mesh_graph(cfg)
    # U_graph, triangles = mesh_heat(cfg)

    Us_true = torch.load("../Us_solution.pth", weights_only=True)
    U_graph.set_grid(Us_true)
    loss_fn = MSELossNorm(Us_true)

    pde_fn = NNFunc(cfg, device=cfg.device)
    pde_adj = NeuralPDEGraph(pde_fn, U_graph, cfg, loss_fn, triangles)

    # optim = torch.optim.SGD(pde_fn.parameters(), lr=0.01, momentum=0.9)
    optim = mup.MuAdamW(pde_fn.mlp.parameters(), lr=0.02, betas=(0.9, 0.99), weight_decay=1e-4)
    optim_other = torch.optim.Adam(pde_fn.other_params.parameters(), lr=0.005)#, betas=(0.95, 0.95))

    pred_loss_hist = []
    for i in range(2001):
        U_graph.set_grid(Us_true.clone())

        converged = pde_adj.forward_solve()
        loss = pde_adj.adjoint_solve()
        pde_adj.backward()

        # print(f'{loss = :.5g}')


        torch.nn.utils.clip_grad_value_(pde_fn.parameters(), clip_value=0.5)
        if i % 25 == 0:
            c_print(f'{i}/400 loss: {loss.detach().cpu().item():.3g}'  # , {loss.detach().cpu().item():.2g}'
                    , color="bright_green")

        if i == 1000 or i == 1500:
            for pg in optim.param_groups:
                pg['lr'] *= 0.5

        optim.step(), optim_other.step()
        optim.zero_grad(), optim_other.zero_grad()

        if i % 100 == 0:
            U_graph.set_grid(Us_true * 0)
            pde_adj.forward_solve()
            Us_pred = U_graph.get_all_us_Xs()[0]
            pred_loss = loss_fn(Us_pred, requires_grad=False)
            pred_loss_hist.append(pred_loss.detach().cpu().item())
            print(f'{pred_loss = }')


    U_graph.set_grid(Us_true)
    pde_adj.plot_interp(title="True solution")
    pde_adj.forward_solve()
    pde_adj.plot_interp(title="Predicted solution")

    print(pred_loss_hist)


def test():
    cfg = Config()
    # U_graph, triangles = load_graph(cfg)
    U_graph, triangles = mesh_graph(cfg)
    # U_graph, triangles = mesh_heat(cfg, max_degree=1, grad_neigh=9)

    Us_true = torch.load("../Us_solution.pth", weights_only=True)
    U_graph.set_grid(Us_true)
    loss_fn = MSELossNorm(Us_true)

    pde_fn = NNFunc(cfg, device=cfg.device)
    pde_adj = NeuralPDEGraph(pde_fn, U_graph, cfg, loss_fn, triangles)

    # optim = torch.optim.SGD(pde_fn.parameters(), lr=0.01, momentum=0.9)
    optim = mup.MuAdamW(pde_fn.mlp.parameters(), lr=0.02, betas=(0.9, 0.99), weight_decay=1e-4)
    optim_other = torch.optim.Adam(pde_fn.other_params.parameters(), lr=0.005)#, betas=(0.95, 0.95))

    resid_factor = 0 # 0.0025
    pde_adj.plot_interp(title=["Exact Velocity x", "Exact Velocity y", "Exact Pressure"])

    pred_loss_hist = []
    for i in range(2001):
        final_loss = 0
        # Adjoint gradient
        init_loss, final_loss, resid = pde_adj.single_step()
        # optim.zero_grad(), optim_other.zero_grad()

        # Residual gradient
        residuals = pde_adj.pde_calc.residuals()
        loss = resid_factor*(residuals**2).mean()
        loss.backward()

        # final_loss += loss
        torch.nn.utils.clip_grad_value_(pde_fn.parameters(), clip_value=0.5)
        if i % 50 == 0:
            c_print(f'{i}/2000 loss: {final_loss.detach().cpu().item():.3g}' # , {loss.detach().cpu().item():.2g}'
                    , color="bright_green")
            # for n, p in pde_fn.other_params.named_parameters():
            #     print(f'{n = }, {p.cpu().detach() }')
                # print(n, f' parameter: {p.cpu().detach()}', "gradient:", p.grad.cpu())

        optim.step(), optim_other.step()
        optim.zero_grad(), optim_other.zero_grad()

        if i == 1000 or i == 1500:
            resid_factor *= 2
            for pg in optim.param_groups:
                pg['lr'] *= 0.5

        if i % 100 == 0:
            U_graph.set_grid(Us_true * 0)
            pde_adj.forward_solve()
            Us_pred = U_graph.get_all_us_Xs()[0]
            pred_loss = loss_fn(Us_pred, requires_grad=False)
            pred_loss_hist.append(pred_loss.detach().cpu().item())
            print(f'{pred_loss = }')

    U_graph.set_grid(Us_true * 0)
    pde_adj.forward_solve()
    pde_adj.plot_interp(title=["Velocity x", "Velocity y", "Pressure"])
    pass
    pass
    pass

    print(pred_loss_hist)


def test2():
    cfg = Config()
    # U_graph, triangles = load_graph(cfg)
    U_graph, triangles = mesh_graph(cfg)
    # U_graph, triangles = mesh_heat(cfg, max_degree=1, grad_neigh=9)

    Us_true = torch.load("../Us_solution.pth", weights_only=True)
    U_graph.set_grid(Us_true.clone())
    loss_fn = MSELoss2(Us_true)

    pde_fn = NNFunc(cfg, device=cfg.device)
    pde_adj = NeuralPDEGraph(pde_fn, U_graph, cfg, loss_fn, triangles)

    # optim = torch.optim.SGD(pde_fn.parameters(), lr=0.01, momentum=0.9)
    optim = mup.MuAdamW(pde_fn.mlp.parameters(), lr=0.02, betas=(0.9, 0.99), weight_decay=1e-4)
    optim_other = torch.optim.Adam(pde_fn.other_params.parameters(), lr=0.005)#, betas=(0.95, 0.95))

    for i in range(1001):

        residuals = pde_adj.pde_calc.residuals()
        loss = (residuals**2).mean()
        loss.backward()

        if i % 50 == 0:
            c_print(f'{i}/1000 loss: {loss.detach().cpu().item():.3g}', color="bright_green")

        optim.step()#, optim_other.step()
        optim.zero_grad(), optim_other.zero_grad()

    # residuals = residuals.view(-1, 3).detach()
    # pde_adj.plot_interp(residuals, title="Updated solution")
    pde_adj.plot_interp(title="Initial solution")
    pde_adj.forward_solve()
    pde_adj.plot_interp(title="Predicted solution")


if __name__ == "__main__":
    setup_logging(debug=2)
    torch.set_printoptions(linewidth=120)
    torch.manual_seed(1)

    true_pde()
    # test()
    # test_adjoint()

    # test2()

    # optim_pde()
    # plot_grads()

