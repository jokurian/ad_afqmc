from functools import partial
from functools import reduce

import numpy as np
from pyscf import fci, gto,ci, scf, grad, cc,df,lib #,ao2mo
import pyscf
from ad_afqmc import pyscf_interface, run_afqmc, linalg_utils
from scipy.linalg import fractional_matrix_power
import jax.numpy as jnp
print = partial(print, flush=True)
def ao2mo(chol,C): #Convert the 2e integrals from AO to MO basis 
    chol2 = np.zeros((chol.shape[0], C.shape[0], C.shape[0]))
    for i in range(chol.shape[0]):
        chol2[i] = np.dot(C.T,np.dot(chol[i], C))
    return chol2

r = 2.2

basis=     "sto-6g"
atom_symbols = np.array(["N","N"])#,"H","H"])
coords     = np.array([[0,0,0],[0,0,r]])#,[0,0,2*r],[0,0,3*r]])
atomstring = zip(atom_symbols, coords)
mol = gto.M(atom=atomstring, basis=basis, verbose=3, unit="bohr")
#mol.symmetry = True
#mol.verbose = 5
mf = df.density_fit(scf.RHF(mol),auxbasis="weigend")
#mf = scf.addons.remove_linear_dep_(mf)
mf.kernel()
#mf.analyze()
mf_can  = scf.RHF(mol).kernel()
#import pdb;pdb.set_trace() 
       
mf_grad = mf.Gradients()  #Reference gradient
mf_grad.kernel()
mc = cc.CCSD(mf)
mc.kernel()
g = mc.nuc_grad_method().kernel()
         
basis0     = jnp.array(fractional_matrix_power(mf.get_ovlp(), -0.5),dtype="float64")
dm0 = basis0.T @ mf.get_ovlp() @ mf.make_rdm1()@ mf.get_ovlp() @ basis0
h1 = jnp.array(basis0.T @ mf.get_hcore() @ basis0, dtype="float64")  #mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
norb = h1.shape[0]

h0=mf.energy_nuc()
nelec = mol.nelectron//2



df0 = df.incore.cholesky_eri(mol,auxbasis="weigend")#,aosym='s1')
df0 = lib.unpack_tril(df0)
#chol0 = get_chol(df0,mol.nao)
chol1 = ao2mo(df0,basis0) #pyscf_interface.ao2mo_chol_copy(df0,mf.mo_coeff).reshape(df0.shape[0],mol.nao,mol.nao)
print(chol1.shape)


h1_ao = mf.get_hcore()
chol1_ao = df0.copy()


#  pyscf_interface.prep_afqmc(mf,chol_cut=chol_cut)
options = {
    "n_eql": 5,
    "n_ene_blocks": 1,
    "n_sr_blocks": 50,
    "n_blocks": 30,
    "n_walkers": 40,
    "do_sr": False, 
    "orbital_rotation": False,
    "walker_type": "rhf",
    "seed": 101,
    "ad_mode": "reverse",
    "do_grad": True,
}

dr = 0.0001
dm0 = basis0.T @ mf.get_ovlp() @ mf.make_rdm1()@ mf.get_ovlp() @ basis0 #2* jnp.eye(norb, nelec).dot(jnp.eye(norb, nelec).T)
coords2 = coords.copy()
coords2[0][2] += dr
atomstring2 = zip(atom_symbols, coords2)
mol2 = gto.M(atom=atomstring2, basis=basis, verbose=3, unit="bohr")
mf2 = df.density_fit(scf.RHF(mol2),auxbasis="weigend")
#mf2 = scf.addons.remove_linear_dep_(mf2)
mf2.kernel()
h12_ao = mf2.get_hcore()

basis_p = jnp.array(fractional_matrix_power(mf2.get_ovlp(), -0.5),dtype="float64")       
h12_mo = jnp.array(basis_p.T @ mf2.get_hcore() @ basis_p, dtype="float64") #mf2.mo_coeff.T @ mf2.get_hcore() @ mf2.mo_coeff
#h12_mo = mf.mo_coeff.T @ mf2.get_hcore() @ mf.mo_coeff


df2 = df.incore.cholesky_eri(mol2,auxbasis="weigend")#,aosym='s1')
df2 = lib.unpack_tril(df2)
#df2 = get_chol(df2,mol.nao)
chol2_ao = df2.copy()
#chol2_mo = ao2mo(df2,mf.mo_coeff)
chol2_mo = ao2mo(df2,basis_p)  #pyscf_interface.ao2mo_chol_copy(df2,mf2.mo_coeff).reshape(df0.shape[0],mol.nao,mol.nao)
print(get_hfE(h12_mo,chol2_mo,mf2.energy_nuc(),dm0))


coords3 = coords.copy()
coords3[0][2] -=dr
atomstring3 = zip(atom_symbols, coords3)
mol3 = gto.M(atom=atomstring3, basis=basis, verbose=3, unit="bohr")
#mol3.symmetry = True
mf3 = df.density_fit(scf.RHF(mol3),auxbasis="weigend")
mf3.kernel()
#mf3 = scf.addons.remove_linear_dep_(mf3)
h13_ao = mf3.get_hcore()

basis_m = jnp.array(fractional_matrix_power(mf3.get_ovlp(), -0.5),dtype="float64")
h13_mo = jnp.array(basis_m.T @ mf3.get_hcore() @ basis_m , dtype="float64")
#h13_mo = mf3.mo_coeff.T @ mf3.get_hcore() @ mf3.mo_coeff
#h13_mo = mf.mo_coeff.T @ mf3.get_hcore() @ mf.mo_coeff


df3 = df.incore.cholesky_eri(mol3,auxbasis="weigend")#,aosym='s1')
df3 = lib.unpack_tril(df3)
chol3_mo = ao2mo(df3,basis_m)  #pyscf_interface.ao2mo_chol_copy(df3,mf3.mo_coeff).reshape(df0.shape[0],mol.nao,mol.nao)
chol3_ao = df3.copy()
print(get_hfE(h13_mo,chol3_mo,mf3.energy_nuc(),dm0))
print("Gradient using FD of energy=", (mf2.energy_tot()-mf3.energy_tot())/(2.*dr))


h1_der = (h12_mo - h13_mo)/(2.*dr)
#h1_der = (h12_ao - h13_ao)/(2.*dr)
h2_der = (chol2_mo - chol3_mo)/(2.*dr)
#h2_der = (chol2_ao - chol3_ao)/(2.*dr)
h0_der = (mf2.energy_nuc() - mf3.energy_nuc())/(2.*dr)
print(jnp.max(h2_der))

h1_der_array = np.zeros((mol.natm,3,mol.nao,mol.nao))
dI_pyscf = np.zeros((mol.natm,3,df0.shape[0],mol.nao,mol.nao))
h1_der_array[0][2] = h1_der
h1_der_array[1][2] = -h1_der
dI_pyscf[0][2] = h2_der
dI_pyscf[1][2] = -h2_der

h0_der = pyscf.grad.rhf.grad_nuc(mol)
np.savez("Integral_der.npz",array1 = h1_der_array,array2=dI_pyscf,array3=h0_der,dm=dm0)

 
#pyscf_interface.prep_afqmc(mf, chol_cut=chol_cut)#, integrals=integrals)
h1e = h1.copy()
chol = chol1.copy()
nbasis = h1e.shape[-1]
nelec = mol.nelec
enuc = mol.energy_nuc()
chol = chol.reshape((-1,nbasis,nbasis))
v0 = 0.5 * np.einsum("nik,njk->ij",chol,chol,optimize = "optimal")
h1e_mod = h1e - v0
chol = chol.reshape((chol.shape[0],-1))
dm0 = basis0.T @ mf.get_ovlp() @ mf.make_rdm1()@ mf.get_ovlp() @ basis0

q = np.linalg.qr(dm0)[0]

np.savez("rhf.npz", mo_coeff=q)
pyscf_interface.write_dqmc(h1e,h1e_mod,chol,sum(nelec),nbasis,enuc,ms=mol.spin,filename="FCIDUMP_chol")#,mo_coeffs=q)#[basis0,basis0])
print("nchol:",chol.shape[0])
print("nelec:",sum(nelec))


run_afqmc.run_afqmc(options=options,nproc=5)
print(f"r = {r}") 
mf_grad = mf.Gradients()
mf_grad.kernel()

mc = cc.CCSD(mf)
mc.kernel()
g = mc.nuc_grad_method().kernel()
  

 
