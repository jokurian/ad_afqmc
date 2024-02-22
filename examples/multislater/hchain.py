from functools import partial

import numpy as np
from pyscf import fci, gto, scf,mcscf
import os
from pyscf.shciscf import shci
from ad_afqmc import pyscf_interface, run_afqmc

print = partial(print, flush=True)
dice_binary = "/projects/joku8258/software/alpine_software/Dice_Dice/Dice/bin/Dice"

r = 1.6#2.0
nH = 6
atomstring = ""
for i in range(nH):
    atomstring += "H 0 0 %g\n" % (i * r)
mol = gto.M(atom=atomstring, basis="sto-6g", verbose=3, unit="bohr")
mf = scf.RHF(mol)
mf.kernel()

umf = scf.UHF(mol)
umf.kernel()
mo1 = umf.stability(external=True)[0]
umf = umf.newton().run(mo1, umf.mo_occ)

#h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)

## fci
#cisolver = fci.FCI(mf)
#fci_ene, fci_vec = cisolver.kernel()
##print(f"fci_ene: {fci_ene}", flush=True)
##dm1 = cisolver.make_rdm1(fci_vec, mol.nao, mol.nelec)
##print(f"1e ene: {np.trace(np.dot(dm1, h1))}")
#
#_ = pyscf_interface.fci_to_noci(cisolver,ndets=5)



print("\nPreparing Dice calculation")
# dummy shciscf object for specifying options
#mc = shci.SHCISCF(mf, mol.nao, mol.nelec)
mc = shci.SHCISCF(mf, 2, 2)
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
  fh.write("writebestdeterminants 10")

# run dice calculation
print("Starting Dice calculation")
command = f"{dice_binary} dice.dat > dice.out; rm -f shci.e"
os.system(command)
print("Finished Dice calculation\n")
#
##import pdb;pdb.set_trace()
#from ad_afqmc import pyscf_interface
#pyscf_interface.hci_to_noci((1,1),norbT=mol.nao,nelecT=mol.nelec)
##pyscf_interface.hci_to_noci(mol.nelec,norbT=mol.nao,nelecT=mol.nelec)

#mc = mcscf.CASSCF(mf,mol.nao,mol.nelec)


numCore = int(mc.ncore)
#import pdb;pdb.set_trace()
# ad afqmc
pyscf_interface.prep_afqmc(mf)
options = {
    "n_eql": 2,
    "n_ene_blocks": 1,
    "n_sr_blocks": 50,
    "n_blocks": 50,
    "n_walkers": 10,
    "seed": 98,
    "ad_mode":"forward",#"reverse",
    "walker_type": "rhf",
    "trial":"multislater",
    "orbital_rotation":False,
    "do_sr":False,
    "numCore":numCore,
}
# serial run
#driver.run_afqmc(options=options, mpi_prefix='')
# mpi run
run_afqmc.run_afqmc(options=options, nproc=1)

