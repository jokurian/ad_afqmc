from functools import partial
from functools import reduce

import numpy as np
from pyscf import fci, gto,ci, scf, grad, cc,df,lib #,ao2mo
import pyscf
from ad_afqmc import pyscf_interface, run_afqmc, linalg_utils , grad_utils
from scipy.linalg import fractional_matrix_power
import jax.numpy as jnp
print = partial(print, flush=True)
import numpy


rs = [1.8]
#rs = [2.0]
basis=  "sto6g" #   "631+g"

for r in rs:
  atom_symbols = np.array(["H","H"])#,"H","H"])
  coords     = np.array([[0,0,0],[0,0,r]])#,[0,0,2*r],[0,0,3*r]])

  atomstring = list(zip(atom_symbols, coords))
  mol = gto.M(atom=atomstring, basis=basis, verbose=3, unit="bohr")
  mf = df.density_fit(scf.RHF(mol))

  mf.kernel()

  options = {
      "n_eql": 10,
      "n_ene_blocks": 8,
      "n_sr_blocks": 50,
      "n_blocks": 20,
      "n_walkers": 100,
      "do_sr": True,
      "orbital_rotation": False,
      "walker_type": "rhf",
      "trial": "rhf",
#      "seed": 101,
      "ad_mode": "reverse",
      "do_grad":True,
}                          

  grad_utils.FD_integrals(mf)
  grad_utils.write_integrals_lowdins(mf)

  run_afqmc.run_afqmc(options=options,nproc=4)
  grad_utils.calculate_nuc_gradients(uhf=False)
  print(f"r = {r}")
  mf_grad = mf.Gradients()
  mf_grad.kernel()

  mc = cc.CCSD(mf)
  mc.kernel()
  g = mc.nuc_grad_method().kernel()



