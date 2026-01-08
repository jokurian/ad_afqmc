import numpy as np
from pyscf import gto, scf

from ad_afqmc import afqmc

mol =  gto.M(atom ="""
    O        0.0000000000      0.0000000000      0.0000000000
    H        0.9562300000      0.0000000000      0.0000000000
    H       -0.2353791634      0.9268076728      0.0000000000
    """,
    basis = 'cc-pvdz',
    verbose = 3)

def test_rhf():
    mf = scf.RHF(mol)
    mf.kernel()

    af = afqmc.AFQMC(mf)
    af.free_projection = True
    af.dt = 0.1
    af.n_prop_steps = 10  # number of dt long steps in a qr block
    af.n_qr_blocks = 1  # number of qr blocks
    af.n_blocks = 3   # number of propagation and measurement blocks
    af.n_ene_blocks= 5  # number of trajectories
    af.no_beta_0_fp = True
    af.n_walkers = 10
    af.ene0 = mf.e_tot
    af.seed = 5
    e, err, sign = af.kernel()
    
    assert abs(e[-1]+7.624795057120907416e+01) < 1e-6
    assert abs(err[-1]-3.039101773566795991e-02) < 1e-8
    assert abs(sign[-1]-np.sign(sign[-1])*9.528328689942754393e-01) < 1e-8

def test_uhf():
    mf = scf.UHF(mol)
    mf.kernel()
    
    af = afqmc.AFQMC(mf)
    af.free_projection = True
    af.dt = 0.1
    af.n_prop_steps = 10  #  number of dt long steps in a qr block
    af.n_qr_blocks = 1  # number of qr blocks
    af.n_blocks = 3  # number of propagation and measurement blocks
    af.n_ene_blocks = 5  # number of trajectories
    af.n_walkers = 10
    af.walker_type = "uhf"
    af.ene0 = mf.e_tot
    af.seed = 5
    e, err, sign = af.kernel()

    assert abs(e[-1]+7.624795058603612574e+01) < 1e-6
    assert abs(err[-1]-3.039109093569811529e-02) < 1e-8
    assert abs(sign[-1]-np.sign(sign[-1])*9.528325187692681109e-01) < 1e-8

if __name__ == "__main__":
    test_rhf()
    test_uhf()
