import torch

from pde.graph_grid.U_graph import UValues, UGraph
from pde.config import Config


class GraphSample:
    U_graph: UGraph
    Us_true: UValues
    Us_saved: list[UValues]
    """ A single graph. Consists of the UGraph object, true Us and various predicted Us. """
    def __init__(self, U_graph: UGraph, Us_true: UValues, N_steps: int, cfg: Config):
        self.U_graph = U_graph
        self.Us_true = Us_true

        # Initialise saved Us as zeros
        self.Us_saved = [U_graph.smooth_grid_like(Us_true) for _ in range(N_steps + 1)]

        # Initialise solver
        self.U_graph.init_lin_solver(cfg)

    def update_Us_last(self, Us_pred: UValues):
        """ Update the last saved Us. """
        self.Us_saved[-1] = Us_pred

    def update_Us_all(self, Us_preds: list[UValues]):
        """ Update all saved Us. """
        self.Us_saved = Us_preds

    # def get_Us_sample(self, i) -> tuple[UGraph, UValues, UValues]:
    #     """ Return a single sample to train on.
    #     """
    #     # TODO: implement sampling strategy
    #
    #     r = torch.rand(1).item()
    #     if r < 0.5 == 0:
    #         Us_step = self.U_graph.smooth_grid_like(self.Us_true)
    #     elif r < 0.95 == 0:
    #         n_saved = len(self.Us_saved)
    #         j = torch.randint(0, n_saved, (1,))[0].item()
    #         Us_step = self.Us_saved[j]
    #     else:
    #         Us_step = self.Us_true
    #
    #     return self.U_graph, self.Us_true, Us_step

    def get_Us_sample(self, i) -> tuple[UGraph, UValues, UValues]:
        """ Return a single sample to train on.
        """
        # TODO: implement sampling strategy
        if i % 51 == 0:
            Us_step = self.U_graph.smooth_grid_like(self.Us_true)
        elif i % 11 == 0:
            Us_step = self.Us_true
        else:
            Us_step = self.Us_saved[-1]

        return self.U_graph, self.Us_true, Us_step


class GraphDataset:
    samples: list[GraphSample]
    """ Class to handle batching of UGraphs for PDE solving.  """

    def __init__(self, graphs: list[UGraph], Us_trues: list[UValues], N_steps: int, cfg: Config):
        self.device = cfg.device

        samples = []
        for G, Us in zip(graphs, Us_trues, strict=True):
            samples.append(GraphSample(G, Us, N_steps, cfg))
        self.samples = samples

    def get_norm_stats(self):
        """ Return normalisation stats for dataset """
        # Construct with symmetry in mind. Each variable has same mean/std across derivatives / components.
        avg_regions = [ [(0, 0), (0, 1)],  # V
                        [(0,), (2,)],         # p
                        [(1, 1, 2, 2), (0, 1, 0, 1)],  # d_V
                        [(1, 2), (2, 2)],  # d_p
                        [(3, 3, 4, 4, 5, 5), (0, 1, 0, 1, 0, 1)],  # d2_V
                        [(3, 4, 5), (2, 2, 2)]  # d2_p
                         ]

        batch_means, batch_stds = [], []
        for s in self.samples:
            Us_true = s.Us_true
            U_dUs = s.U_graph.get_Us_dUs(Us_true)[0]

            norm_mean, norm_std = torch.zeros((6, 3), device=self.device), torch.zeros((6, 3), device=self.device)
            for i, region in enumerate(avg_regions):
                std, mean = torch.std_mean(U_dUs[:, region[0], region[1]])
                norm_std[region[0], region[1]] = std
                norm_mean[region[0], region[1]] = mean

            batch_stds.append(norm_std)
            batch_means.append(norm_mean)

        batch_mean = torch.stack(batch_means, dim=0).mean(dim=0)
        batch_std = torch.stack(batch_stds, dim=0).mean(dim=0)
        return batch_mean, batch_std

    def __iter__(self):
        while True:
            for sample in self.samples:
                yield sample

