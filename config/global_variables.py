import qutip

ATOM_DIM = 2
FIELD_DIM = 2
JC_DIM = ATOM_DIM * FIELD_DIM

a = qutip.tensor(qutip.destroy(ATOM_DIM), qutip.qeye(FIELD_DIM))
sm = qutip.tensor(qutip.qeye(ATOM_DIM), qutip.destroy(FIELD_DIM))
SEED = 42