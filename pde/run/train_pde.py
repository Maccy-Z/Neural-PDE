import torch
from cprint import c_print
import mup
import time
import os

from pde.config import Config
from pde.NeuralPDE_Graph import NeuralPDEGraph
from pde.graph_grid.U_graph import UValues, UGraph
from pde.pdes.PDEs import Fluid, NNFunc
from pde.utils import setup_logging, ARTEFACT_DIR
from pde.loss import DummyLoss, MSELossNorm
from pde.run.batching import GraphDataset, GraphSample
from pde.schedulers import CosineAnnealingWarmupScheduler
from pde.run.run_utils import MetricTracker


def setup(cfg: Config):
    # Load save dataset
    ds_dir = ARTEFACT_DIR / "dataset_fvm"
    save_files = os.listdir(ds_dir)
    save_files = sorted([f for f in save_files if f.endswith(".pth")])
    graphs, Us_values = [], []
    for f in save_files:
        save_dict = torch.load(ds_dir / f, weights_only=False)
        Us_true = save_dict["Us_values"]
        Us_true.to(cfg.device)
        U_graph = save_dict["U_graph"]
        graphs.append(U_graph)
        Us_values.append(Us_true)

    dataset = GraphDataset(graphs, Us_values, N_steps=cfg.fwd_cfg.N_iter, cfg=cfg)
    norm_mean, norm_std = dataset.get_norm_stats()

    loss_fn = MSELossNorm(norm_std[0])
    pde_fn = NNFunc(cfg, norm_mean=norm_mean, norm_std=norm_std, device=cfg.device)

    # optim = torch.optim.SGD(pde_fn.parameters(), lr=0.01, momentum=0.9)
    optim = mup.MuAdamW(pde_fn.mlp.parameters(), lr=cfg.mup_lr, betas=cfg.mup_betas, weight_decay=1e-4)
    optim_other = torch.optim.Adam(pde_fn.other_params.parameters(), lr=cfg.scalar_lr)  # , betas=(0.95, 0.95))

    return dataset, pde_fn, loss_fn, optim, optim_other


def true_pde():
    """ Generate true solution using known PDE. """
    from generate_graph import load_ds_graph
    cfg = Config()
    # U_graph, Us_values = mesh_graph(cfg)
    U_graph, Us_values = load_ds_graph(cfg)
    U_graph.init_lin_solver(cfg)
    pde_fn = Fluid(cfg, device=cfg.device)

    loss_fn = DummyLoss()

    pde_adj = NeuralPDEGraph(pde_fn, cfg, loss_fn)

    Us_history, _ = pde_adj.forward_solve(U_graph, Us_values)
    U_graph.plot_interp(Us_values, title="Final solution")

    for i, Us in enumerate(Us_history):
        U_graph.plot_interp(Us, title=f'Solution step {i}')
    # print(Us_history)

    return


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

        # Create cosine annealing schedulers with warmup
        scheduler = CosineAnnealingWarmupScheduler(
            self.optim,
            warmup_steps=cfg.warmup_steps,
            max_steps=cfg.N_steps,
            min_lr_ratio=cfg.min_lr_ratio
        )
        scheduler_other = CosineAnnealingWarmupScheduler(
            self.optim_other,
            warmup_steps=cfg.warmup_steps,
            max_steps=cfg.N_steps,
            min_lr_ratio=cfg.min_lr_ratio
        )

        st = time.time()
        ds_train_iter = iter(self.ds_train)
        for i in range(self.cfg.N_steps):
            self.pde_fn.train()
            sample = next(ds_train_iter)
            self.optim.zero_grad(), self.optim_other.zero_grad()

            U_graph, Us_true, Us_step = sample.get_Us_sample(i)

            # Adjoint gradient
            init_loss, final_loss, resid = self.pde_adj.single_step(U_graph, Us_step, Us_true)

            torch.nn.utils.clip_grad_norm_(self.pde_fn.parameters(), max_norm=cfg.clip_norm)
            self.optim.step(), self.optim_other.step()

            # Step the learning rate schedulers
            scheduler.step()
            scheduler_other.step()

            self.metric_tracker.add_metric({"loss": final_loss})

            if i % cfg.N_print == 0:
                dt = time.time() - st
                st = time.time()
                avg_loss = self.metric_tracker.get_mean_metrics(["loss"])["loss"].item()
                c_print(f'{i}/{cfg.N_steps} loss: {avg_loss:.3g}, T = {dt:.3g}', color="bright_green")
                # c_print(f'{Us_step.Us.mean():.4g}, {Us_true.Us.mean():.4g}', color="bright_blue")

            if i % cfg.N_valid == 0:
                self._valid_step()

    @torch.no_grad()
    def _valid_step(self):
        """ Run validation. Initialise with zero field and update last prediction. """
        self.pde_fn.eval()

        valid_losses = []
        for sample in self.ds_valid.samples:
            U_graph, Us_true = sample.U_graph, sample.Us_true
            Us_test = U_graph.smooth_grid_like(Us_true)
            Us_history, convergence = self.pde_adj.forward_solve(U_graph, Us_test)

            valid_loss = self.loss_fn(Us_test, Us_true, requires_grad=False)
            valid_losses.append(valid_loss)
            # Update saved states
            sample.update_Us_all(Us_history)

        valid_loss_mean = torch.stack(valid_losses).mean()
        self.metric_tracker.add_metric({"valid_loss": valid_loss_mean})
        print(f'{valid_loss_mean = }')

    def plot_final_results(self, plot_history=True):
        c_print("Validation loss history: ", color="bright_magenta")
        c_print(self.metric_tracker.get_metrics("valid_loss"), color="bright_magenta")

        U_g_plot, Us_plot = self.ds_valid.samples[0].U_graph, self.ds_valid.samples[0].Us_true

        Us_test = U_g_plot.smooth_grid_like(Us_plot)
        Us_history, _ = self.pde_adj.forward_solve(U_g_plot, Us_test)

        if plot_history:
            for i, Us in enumerate(Us_history):
                U_g_plot.plot_interp(Us, title=f'Solution step {i}')
        else:
            U_g_plot.plot_interp(Us_test)


if __name__ == "__main__":
    setup_logging(debug=3)
    torch.set_printoptions(linewidth=120, precision=7)
    torch.manual_seed(0)
    # torch.autograd.set_detect_anomaly(True)
    # torch.use_deterministic_algorithms(True)

    trainer = Trainer()
    trainer.train_model()
    trainer.plot_final_results()
