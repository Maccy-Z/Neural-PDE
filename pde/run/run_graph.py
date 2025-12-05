import torch
from cprint import c_print
import mup
import time
import os

from pde.config import Config
from pde.NeuralPDE_Graph import NeuralPDEGraph
from pde.graph_grid.U_graph import UValues, UGraph
from pde.pdes.PDEs import Fluid, FluidLearned, NNFunc
from pde.utils import setup_logging, ARTEFACT_DIR
from pde.loss import DummyLoss, MSELoss2, MSELossNorm
from pde.run.generate_graph import mesh_graph, load_graph
from pde.run.batching import GraphDataset, GraphSample

class MetricTracker:
    tracking_dict: dict[str, list[torch.Tensor]]
    def __init__(self, cfg: Config):
        self.tracking_dict = {}

    def add_metric(self, new_vals: dict[str, torch.Tensor]):
        for key in new_vals.keys():
            if key not in self.tracking_dict:
                self.tracking_dict[key] = []

        for key, val in new_vals.items():
                self.tracking_dict[key].append(val)

    def get_mean_metrics(self, keys: list[str]) -> dict[str, torch.Tensor]:
        """ Return average metrics, and reset. """
        return_dict = {}
        for key in keys:
            if key not in self.tracking_dict:
                raise ValueError(f"Key {key} not found in tracking_dict")
            all_metrics = torch.stack(self.tracking_dict[key])
            mean_metric = all_metrics.mean()
            return_dict[key] = mean_metric
            # Reset
            self.tracking_dict[key] = []

        return return_dict

    def get_metrics(self, key: str) -> torch.Tensor:
        """ Return metrics, and reset. """
        if key not in self.tracking_dict:
            raise ValueError(f"Key {key} not found in tracking_dict")
        all_metrics = torch.stack(self.tracking_dict[key])
        # Reset
        self.tracking_dict[key] = []

        return all_metrics

def setup(cfg: Config):
    # U_graph, _ = mesh_graph(cfg)
    save_files = os.listdir(ARTEFACT_DIR / "dataset")
    save_files = sorted([f for f in save_files if f.endswith(".pth")])
    graphs, Us_values = [], []
    for f in save_files:
        save_dict = torch.load(ARTEFACT_DIR / "dataset" / f, weights_only=False)
        Us_true = save_dict["Us_values"]
        U_graph = save_dict["U_graph"]
        graphs.append(U_graph)
        Us_values.append(Us_true)

    loss_fn = MSELossNorm()

    dataset = GraphDataset(graphs, Us_values, N_steps=cfg.fwd_cfg.N_iter, device=cfg.device)
    norm_mean, norm_std = dataset.get_norm_stats()

    pde_fn = NNFunc(cfg, norm_mean=norm_mean, norm_std=norm_std, device=cfg.device)

    # optim = torch.optim.SGD(pde_fn.parameters(), lr=0.01, momentum=0.9)
    optim = mup.MuAdamW(pde_fn.mlp.parameters(), lr=cfg.mup_lr, betas=cfg.mup_betas, weight_decay=1e-4)
    optim_other = torch.optim.Adam(pde_fn.other_params.parameters(), lr=cfg.scalar_lr)  # , betas=(0.95, 0.95))

    return dataset, pde_fn, loss_fn, optim, optim_other

def true_pde():
    """ Generate true solution using known PDE. """
    cfg = Config()
    U_graph, Us_values = mesh_graph(cfg)
    pde_fn = Fluid(cfg, device=cfg.device)
    # pde_fn = HeatLearned(cfg, device=cfg.DEVICE)

    loss_fn = DummyLoss()

    pde_adj = NeuralPDEGraph(pde_fn, cfg, loss_fn)

    pde_adj.forward_solve(U_graph, Us_values)
    U_graph.plot_interp(Us_values, title="Initial solution")

    # Us = Us_values.Us

    # Save the solution and graph
    save_dict = {"Us_values": Us_values, "U_graph": U_graph}
    with open(ARTEFACT_DIR / "Us_solution.pth", "wb") as f:
        torch.save(save_dict, f)
    return None

#
# def train_adjoint():
#     """ Train PDE using adjoint method."""
#     cfg = Config()
#     U_graph, triangles = mesh_graph(cfg)
#
#     Us_true = torch.load("../Us_solution.pth", weights_only=True)
#     U_graph.set_grid(Us_true)
#     U_values = U_graph.U_values
#     loss_fn = MSELossNorm(Us_true)
#
#     pde_fn = NNFunc(cfg, device=cfg.device)
#     pde_adj = NeuralPDEGraph(pde_fn, U_graph, U_values, cfg, loss_fn=loss_fn)
#
#     # optim = torch.optim.SGD(pde_fn.parameters(), lr=0.01, momentum=0.9)
#     optim = mup.MuAdamW(pde_fn.mlp.parameters(), lr=0.02, betas=(0.9, 0.99), weight_decay=1e-4)
#     optim_other = torch.optim.Adam(pde_fn.other_params.parameters(), lr=0.005)#, betas=(0.95, 0.95))
#
#     pred_loss_hist = []
#     t = time.time()
#     for i in range(2001):
#         U_graph.set_grid(Us_true.clone())
#
#         converged = pde_adj.forward_solve()
#         loss = pde_adj.adjoint_solve()
#         pde_adj.backward()
#
#         torch.nn.utils.clip_grad_value_(pde_fn.parameters(), clip_value=0.5)
#         if i % 25 == 0:
#             c_print(f'{i}/400 loss: {loss.detach().cpu().item():.3g}'  # , {loss.detach().cpu().item():.2g}'
#                     , color="bright_green")
#
#         if i == 1000 or i == 1500:
#             for pg in optim.param_groups:
#                 pg['lr'] *= 0.5
#
#         optim.step(), optim_other.step()
#         optim.zero_grad(), optim_other.zero_grad()
#
#         if i % 100 == 0:
#             U_graph.set_grid(Us_true * 0)
#             pde_adj.forward_solve()
#             Us_pred = U_graph.get_all_us_Xs()[0]
#             pred_loss = loss_fn(Us_pred, requires_grad=False)
#             pred_loss_hist.append(pred_loss.detach().cpu().item())
#             print(f'{pred_loss = }')
#
#
#     U_graph.set_grid(Us_true)
#     pde_adj.plot_interp(title="True solution")
#     pde_adj.forward_solve()
#     pde_adj.plot_interp(title="Predicted solution")
#
#     print(pred_loss_hist)
#
# def train_resid():
#     """ Train model using residual loss only. """
#     cfg = Config()
#     U_graph, triangles = mesh_graph(cfg)
#     # U_graph, triangles = mesh_heat(cfg, max_degree=1, grad_neigh=9)
#
#     Us_true = torch.load("../Us_solution.pth", weights_only=True)
#     U_graph.set_grid(Us_true.clone())
#     loss_fn = MSELoss2(Us_true)
#
#     pde_fn, optim, optim_other = init_setup(cfg)
#     pde_adj = NeuralPDEGraph(pde_fn, U_graph, cfg, loss_fn, triangles)
#
#     for i in range(1001):
#
#         residuals = pde_adj.pde_calc.residuals()
#         loss = (residuals**2).mean()
#         loss.backward()
#
#         if i % 50 == 0:
#             c_print(f'{i}/1000 loss: {loss.detach().cpu().item():.3g}', color="bright_green")
#
#         optim.step()#, optim_other.step()
#         optim.zero_grad(), optim_other.zero_grad()
#
#     # residuals = residuals.view(-1, 3).detach()
#     # pde_adj.plot_interp(residuals, title="Updated solution")
#     pde_adj.plot_interp(title="Initial solution")
#     pde_adj.forward_solve()
#     pde_adj.plot_interp(title="Predicted solution")

class Trainer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        cfg = Config()
        self.metric_tracker = MetricTracker(cfg)

        ds, self.pde_fn, self.loss_fn, self.optim, self.optim_other = setup(cfg)
        self.ds_train, self.ds_valid = ds, ds
        self.pde_adj = NeuralPDEGraph(self.pde_fn, cfg, self.loss_fn)

        U_g_plot, Us_plot = self.ds_valid.samples[0].U_graph, self.ds_valid.samples[0].Us_true
        U_g_plot.plot_interp(Us_plot, title=["Exact Velocity x", "Exact Velocity y", "Exact Pressure"])

        self.cfg = cfg

    def train_model(self):
        cfg = self.cfg

        st = time.time()
        ds_train_iter = iter(self.ds_train)
        for i in range(self.cfg.N_steps):
            # LR schedule
            if i == 1000 or i == 1500:
                for pg in self.optim.param_groups:
                    pg['lr'] *= 0.5

            sample = next(ds_train_iter)
            self.optim.zero_grad(), self.optim_other.zero_grad()

            U_graph, Us_true, Us_step = sample.get_Us_sample(i)

            # Adjoint gradient
            init_loss, final_loss, resid = self.pde_adj.single_step(U_graph, Us_step, Us_true)

            # torch.nn.utils.clip_grad_value_(self.pde_fn.parameters(), clip_value=0.5)
            torch.nn.utils.clip_grad_norm_(self.pde_fn.parameters(), max_norm=cfg.clip_norm)
            self.optim.step(), self.optim_other.step()

            self.metric_tracker.add_metric({"loss": final_loss})

            if i % cfg.N_print == 0:
                dt = time.time() - st
                st = time.time()
                avg_loss = self.metric_tracker.get_mean_metrics(["loss"])["loss"].item()
                c_print(f'{i}/2000 loss: {avg_loss:.3g}, T = {dt:.3g}', color="bright_green")
                # c_print(f'{Us_step.Us.mean():.4g}, {Us_true.Us.mean():.4g}', color="bright_blue")

            if i % cfg.N_valid == 0:
                self._valid_step()

    @torch.no_grad()
    def _valid_step(self):
        """ Run validation. Initialise with zero field and update last prediction. """

        valid_losses = []
        for sample in self.ds_valid.samples:
            U_graph, Us_true = sample.U_graph, sample.Us_true
            Us_test = U_graph.get_zero_U_values(Us_true)
            self.pde_adj.forward_solve(U_graph, Us_test)

            sample.update_Us_last(Us_test)
            valid_loss = self.loss_fn(Us_test, Us_true, requires_grad=False)
            valid_losses.append(valid_loss)

        valid_loss_mean = torch.stack(valid_losses).mean()
        self.metric_tracker.add_metric({"valid_loss": valid_loss_mean})
        print(f'{valid_loss_mean = }')

    def plot_final_results(self):
        U_g_plot, Us_plot = self.ds_valid.samples[0].U_graph, self.ds_valid.samples[0].Us_true
        Us_test = U_g_plot.new_Us(torch.zeros_like(Us_plot.Us))
        self.pde_adj.forward_solve(U_g_plot, Us_test)
        U_g_plot.plot_interp(Us_test)

        print(self.metric_tracker.get_metrics("valid_loss"))


def train_new():
    """ Train PDE using exact newton gradient + residuals. """
    cfg = Config()

    ds, pde_fn, loss_fn, optim, optim_other = setup(cfg)
    pde_adj = NeuralPDEGraph(pde_fn, cfg, loss_fn)

    U_g_plot, Us_plot = ds.samples[0].U_graph, ds.samples[0].Us_true
    U_g_plot.plot_interp(Us_plot, title=["Exact Velocity x", "Exact Velocity y", "Exact Pressure"])

    metric_tracker = MetricTracker(cfg)
    pred_loss_hist = []
    st = time.time()
    batch = iter(ds)
    for i in range(2001):
        # LR schedule
        if i == 1000 or i == 1500:
            for pg in optim.param_groups:
                pg['lr'] *= 0.5

        sample = next(batch)
        optim.zero_grad(), optim_other.zero_grad()

        U_graph, Us_true, Us_step = sample.get_Us_sample(i)

        # Adjoint gradient
        init_loss, final_loss, resid = pde_adj.single_step(U_graph, Us_step, Us_true)

        # # Residual gradient
        # residuals = pde_adj.pde_calc.residuals()
        # loss = resid_factor*(residuals**2).mean()
        # loss.backward()

        torch.nn.utils.clip_grad_value_(pde_fn.parameters(), clip_value=0.5)
        torch.nn.utils.clip_grad_norm_(pde_fn.parameters(), max_norm=1)
        optim.step(), optim_other.step()

        metric_tracker.add_metric({"loss": final_loss})
        # Printing
        if i % 50 == 0:
            dt = time.time() - st
            st = time.time()
            avg_loss = metric_tracker.get_metrics(["loss"])["loss"].item()
            c_print(f'{i}/2000 loss: {avg_loss:.3g}, T = {dt:.3g}', color="bright_green")
            # c_print(f'{Us_step.Us.mean():.4g}, {Us_true.Us.mean():.4g}', color="bright_blue")

        if i % 100 == 0:
            Us_test = U_graph.get_zero_U_values(Us_true)
            pde_adj.forward_solve(U_graph, Us_test)

            sample.update_Us_last(Us_test)
            pred_loss = loss_fn(Us_test, Us_true, requires_grad=False)
            pred_loss_hist.append(pred_loss.detach().cpu().item())
            print(f'{pred_loss = }')

    U_g_plot, Us_plot = ds.samples[0].U_graph, ds.samples[0].Us_true
    Us_test = U_g_plot.new_Us(torch.zeros_like(Us_plot.Us))
    pde_adj.forward_solve(U_g_plot, Us_test)
    U_g_plot.plot_interp(Us_test)

    print(pred_loss_hist)



if __name__ == "__main__":
    setup_logging(debug=4)
    torch.set_printoptions(linewidth=120, precision=7)
    torch.manual_seed(1)
    # torch.autograd.set_detect_anomaly(True)
    # torch.use_deterministic_algorithms(True)

    # true_pde()
    # train_new()
    # test_adjoint()


    trainer = Trainer()
    trainer.train_model()
    trainer.plot_final_results()
