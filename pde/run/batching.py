from pde.graph_grid.U_graph import UValues, UGraph

class GraphSample:
    U_graph: UGraph
    Us_true: UValues
    Us_saved: list[UValues]
    """ A single graph. Consists of the UGraph object, true Us and various predicted Us. """
    def __init__(self, U_graph: UGraph, Us_true: UValues, N_steps: int):
        self.U_graph = U_graph
        self.Us_true = Us_true

        # Initialise saved Us as zeros
        self.Us_saved = [U_graph.get_zero_U_values(Us_true) for _ in range(N_steps)]


    def update_Us_last(self, Us_pred: UValues):
        """ Update the last saved Us. """
        self.Us_saved[-1] = Us_pred

    def update_Us_all(self, Us_preds: list[UValues]):
        """ Update all saved Us. """
        assert len(Us_preds) == len(self.Us_saved), "Number of predicted Us must match number of saved Us."
        self.Us_saved = Us_preds

    def get_Us_sample(self, i) -> tuple[UGraph, UValues, UValues]:
        """ Return a single sample to train on.
        """
        # TODO: implement sampling strategy
        if i % 51 == 0:
            Us_step = self.U_graph.get_zero_U_values(self.Us_true)  # U_graph.set_grid(torch.zeros_like(Us_true), U_values)
        elif i % 11 == 0:
            Us_step = self.Us_true  # U_graph.set_grid(Us_true, U_values)
        else:
            Us_step = self.Us_saved[-1]  # U_graph.set_grid(Us_init, U_values)

        return self.U_graph, self.Us_true, Us_step

class GraphBatch:
    samples: list[GraphSample]
    """ Class to handle batching of UGraphs for PDE solving.  """

    def __init__(self, graphs: list[UGraph], Us_trues: list[UValues], N_steps: int):
        samples = []
        for g, u in zip(graphs, Us_trues, strict=True):
            samples.append(GraphSample(g, u, N_steps))
        self.samples = samples

    # def __iter__(self):
    #     return self

    def __iter__(self):
        while True:
            for sample in self.samples:
                yield sample
