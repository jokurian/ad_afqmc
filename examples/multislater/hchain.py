from functools import partial

import numpy as np
from pyscf import fci, gto, scf,mcscf
import os
from pyscf.shciscf import shci
from ad_afqmc import pyscf_interface, run_afqmc

print = partial(print, flush=True)
dice_binary = "/projects/joku8258/software/alpine_software/Dice_Dice/Dice/bin/Dice"

r = 1.6#2.0
nH = 4
atomstring = ""
for i in range(nH):
    atomstring += "H 0 0 %g\n" % (i * r)
mol = gto.M(atom=atomstring, basis="sto-6g", verbose=3, unit="bohr")
mf = scf.RHF(mol)
mf.kernel()


norb_frozen = 0
print("\nPreparing Dice calculation")
# dummy shciscf object for specifying options
#mc = shci.SHCISCF(mf, mol.nao, mol.nelec)
mc = shci.SHCISCF(mf, 4, 4)
mc.mo_coeff = mf.mo_coeff
mc.fcisolver.sweep_iter = [ 0 ]
mc.fcisolver.sweep_epsilon = [ 1e-2 ]
mc.fcisolver.davidsonTol = 5.e-5
mc.fcisolver.dE = 1.e-6
#mc.fcisolver.initialStates = [[0,1]]
mc.fcisolver.maxiter = 6
mc.fcisolver.nPTiter = 0
mc.fcisolver.DoRDM = False
shci.dryrun(mc, mc.mo_coeff)
command = "mv input.dat dice.dat"
os.system(command)
with open("dice.dat", "a") as fh:
  fh.write("writebestdeterminants 1000")

# run dice calculation
print("Starting Dice calculation")
command = f"{dice_binary} dice.dat > dice.out; rm -f shci.e"
os.system(command)
print("Finished Dice calculation\n")

numCore = int(mc.ncore)

# ad afqmc
pyscf_interface.prep_afqmc(mf,chol_cut=1e-5)
options = {
    "n_eql": 2,
    "n_ene_blocks": 1,
    "n_sr_blocks": 50,
    "n_blocks": 50,
    "n_walkers": 10,
    "seed": 98,
    "ad_mode":None,#"reverse",
    "walker_type": "rhf",
    "trial":"multislater",
    "orbital_rotation":False,
    "do_sr":True,
    "numCore":numCore,
}

run_afqmc.run_afqmc(options=options, nproc=4)

