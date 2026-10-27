"""Load result pickles written before the code moved into the ``pinn_jc`` package.

Those pickles store whole ``nn.Module`` objects under their old import paths
(``src.model.neural_network.Neural_Net`` and ``utils.SIN``). This loader maps the old
paths onto the current classes. Prefer the migrated format produced by
``scripts/migrate_pickles.py``; this exists so old files stay readable.
"""

import pickle
from pathlib import Path

LEGACY_CLASSES = {
    ("src.model.neural_network", "Neural_Net"): ("pinn_jc.models.mlp", "Neural_Net"),
    ("utils", "SIN"): ("pinn_jc.models.activations", "SIN"),
}


class LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        module, name = LEGACY_CLASSES.get((module, name), (module, name))
        return super().find_class(module, name)


def load_legacy_pickle(path):
    """Only use on trusted files: unpickling can execute arbitrary code."""
    with Path(path).open("rb") as f:
        return LegacyUnpickler(f).load()
