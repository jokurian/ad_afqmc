import numpy as np
from pyscf import gto,scf, grad, cc,df
from ad_afqmc import pyscf_interface, run_afqmc, grad_utils

from ad_afqmc import config
config.afqmc_config["use_gpu"] = False
config.afqmc_config["use_mpi"] = True

rs = [1.05835]
basis=  "sto6g" #   "631+g"

def geom(r,atom="H"):
    atomstring = f"""
    {atom} 0.0 0.0 0.0
    {atom} 0.0 0.0 {r}
    """
    return atomstring

for r in rs:
  atomstring = geom(r,atom="H")
  mol = gto.M(atom=atomstring, basis=basis, verbose=3, unit="Angstrom")
  mf = df.density_fit(scf.UHF(mol))

  mf.kernel()

  options = {
      "n_eql": 5,
      "n_ene_blocks": 4,
      "n_sr_blocks": 50,
      "n_blocks": 10,
      "n_walkers": 25,
      "do_sr": True,
      "orbital_rotation": True,
      "walker_type": "uhf",
      "trial": "uhf",
      "seed": 101,
      "ad_mode": "nuc_grad",
}                          

  grad_utils.prep_afqmc_nuc_grad(mf)
  run_afqmc.run_afqmc(options=options,nproc=4)
  grad_utils.calculate_nuc_gradients()#(uhf=False)
  print(f"r = {r}")
  mf_grad = mf.Gradients()
  mf_grad.kernel()

  mc = cc.CCSD(mf)
  mc.kernel()
  g = mc.nuc_grad_method().kernel()



