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


rs = [1.8,2.0,2.2,2.4]
#rs = [2.0]
basis=  "sto6g" #   "631+g"

for r in rs:
  atom_symbols = np.array(["H","H"])#,"H","H"])
  coords     = np.array([[0,0,0],[0,0,r]])#,[0,0,2*r],[0,0,3*r]])

  atomstring = list(zip(atom_symbols, coords))
  mol = gto.M(atom=atomstring, basis=basis, verbose=3, unit="bohr")
  mf = df.density_fit(scf.RHF(mol))

  mf.kernel()

  dR = 0.1
  dE = dR
  e_afqmc = [0., 0., 0., 0., 0., 0.]
  e_err_afqmc = [0., 0., 0., 0., 0., 0.]
  e_rhf = [0., 0., 0., 0., 0., 0.]
  options = {
      "n_eql": 10,
      "n_ene_blocks": 1,
      "n_sr_blocks": 50,
      "n_blocks": 400,
      "n_walkers": 100,
      "do_sr": True,
      "orbital_rotation": False,
      "walker_type": "rhf",
      "trial": "rhf",
#      "seed": 101,
      "ad_mode": None,
      "do_grad":False,
  }
  for i, pm in enumerate([dR*-3.0, dR*-2.0, dR*-1.0, dR*1.0, dR*2.0, dR*3.0]):
    coords_copy = coords.copy()
    coords_copy[0][2] += pm
    atomstring2 = zip(atom_symbols, coords_copy)
    mol2 = gto.M(atom=atomstring2, basis=basis, verbose=3, unit="bohr")
    mf_p = df.density_fit(scf.RHF(mol2))#scf.RHF(mol2)
    mf_p.kernel()
    e_rhf[i] = mf_p.e_tot
    grad_utils.write_integrals_lowdins(mf_p)
    e_afqmc[i],e_err_afqmc[i] = run_afqmc.run_afqmc(options = options)
    print(f"Finished AFQMC / RHF calculation {i}\n")

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
  print("AD AFQMC Calculation")
  grad_utils.FD_integrals(mf)
  grad_utils.write_integrals_lowdins(mf)

  run_afqmc.run_afqmc(options=options)
  grad_utils.calculate_nuc_gradients(uhf=False)

  print(f"R = {r},dR = {dR}")
  #print(f'\ne_rhf: {e_rhf}')
  #print(f'e_afqmc: {e_afqmc}')
  #print(f'e_err_afqmc: {e_err_afqmc}\n')

  obs_afqmc_rhf_3p = (-e_afqmc[2] + e_afqmc[3]) / 2. / dE
  obs_afqmc_rhf_3p_err = (e_err_afqmc[2]**2 + e_err_afqmc[3]**2)**0.5 / 2. / dE
  obs_afqmc_rhf_5p = (e_afqmc[1] - 8 * (e_afqmc[2] - e_afqmc[3]) - e_afqmc[4]) / 12. / dE
  obs_afqmc_rhf_5p_err = (e_err_afqmc[1]**2 + 64 * (e_err_afqmc[2]**2 + e_err_afqmc[3]**2) + e_err_afqmc[4]**2)**0.5 / 12. / dE
  obs_afqmc_rhf_7p = (-e_afqmc[0] + 9 * (e_afqmc[1] - e_afqmc[4]) - 45 * (e_afqmc[2] - e_afqmc[3]) + e_afqmc[5]) / 60. / dE
  obs_afqmc_rhf_7p_err = (e_err_afqmc[0]**2 + 9**2 * (e_err_afqmc[1]**2 + e_err_afqmc[4]**2) + 45**2 * (e_err_afqmc[2]**2 + e_err_afqmc[3]**2) + e_err_afqmc[5]**2)**0.5 / 60. / dE

  print(f'FD obs_rhf_3p = {(-e_rhf[2] + e_rhf[3]) / 2. / dE}')
  print(f'FD obs_rhf_5p = {(e_rhf[1] - 8 * (e_rhf[2] - e_rhf[3]) - e_rhf[4]) / 12. / dE}')
  print(f'FD obs_rhf_7p = {(-e_rhf[0] + 9 * (e_rhf[1] - e_rhf[4]) - 45 * (e_rhf[2] - e_rhf[3]) + e_rhf[5]) / 60. / dE}')
  print(f'FD obs_afqmc_rhf_3p = {obs_afqmc_rhf_3p} +/- {obs_afqmc_rhf_3p_err}')
  print(f'FD obs_afqmc_rhf_5p = {obs_afqmc_rhf_5p} +/- {obs_afqmc_rhf_5p_err}')
  print(f'FD obs_afqmc_rhf_7p = {obs_afqmc_rhf_7p} +/- {obs_afqmc_rhf_7p_err}')


