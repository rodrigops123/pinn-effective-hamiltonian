import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from utils import SIN
import qutip

ATOM_DIM = 2
FIELD_DIM = 2
JC_DIM = ATOM_DIM * FIELD_DIM

a = qutip.tensor(qutip.destroy(ATOM_DIM), qutip.qeye(FIELD_DIM))
sm = qutip.tensor(qutip.qeye(ATOM_DIM), qutip.destroy(FIELD_DIM))
SEED = 42

model_train_params = {
    "units": [10, 10],
    "activation": SIN(),
}
LEARNING_RATE = 0.001

#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"  # FORCE CPU