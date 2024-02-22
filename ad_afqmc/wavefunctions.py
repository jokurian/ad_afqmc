import os

import numpy as np

os.environ[
    "XLA_FLAGS"
] = "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"
from dataclasses import dataclass
from functools import partial
from typing import Tuple

# os.environ['JAX_DISABLE_JIT'] = 'True'
import jax.numpy as jnp
from jax import jit, lax, vmap
from jax import jvp,vjp
from ad_afqmc import linalg_utils

print = partial(print, flush=True)
MAX_EXCITATION = 4

# we assume afqmc is performed in the rhf orbital basis
@dataclass
class rhf:
    norb: int
    nelec: int  # this is the number of electrons of each spin, so nelec = total_nelec // 2
    n_opt_iter: int = 30

    @partial(jit, static_argnums=0)
    def calc_overlap(self, walker, wave_data=None):
        return jnp.linalg.det(walker[: walker.shape[1], :]) ** 2

    @partial(jit, static_argnums=0)
    def calc_overlap_vmap(self, walkers, wave_data=None):
        return vmap(self.calc_overlap, in_axes=(0, None))(walkers, wave_data)

    @partial(jit, static_argnums=0)
    def calc_green(self, walker, wave_data=None):
        return (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T

    @partial(jit, static_argnums=0)
    def calc_green_vmap(self, walkers, wave_data=None):
        return vmap(self.calc_green, in_axes=(0, None))(walkers, wave_data)

    @partial(jit, static_argnums=0)
    def calc_1rdm(self, walker, wave_data=None):  # shouldnt be here
        rdm1 = (
            walker.dot(jnp.linalg.inv(walker.T.conj().dot(walker))).dot(walker.T.conj())
        ).T
        return rdm1

    @partial(jit, static_argnums=0)
    def calc_1rdm_vmap(self, walkers, wave_data=None):
        return vmap(self.calc_1rdm, in_axes=(0, None))(walkers, wave_data)

    @partial(jit, static_argnums=0)
    def calc_force_bias(self, walker, rot_chol, wave_data=None):
        green_walker = self.calc_green(walker, wave_data)
        fb = 2.0 * jnp.einsum("gij,ij->g", rot_chol, green_walker, optimize="optimal")
        return fb

    @partial(jit, static_argnums=0)
    def calc_force_bias_vmap(self, walkers, ham, wave_data=None):
        return vmap(self.calc_force_bias, in_axes=(0, None, None))(
            walkers, ham["rot_chol"], wave_data
        )

    @partial(jit, static_argnums=0)
    def calc_energy(self, h0, rot_h1, rot_chol, walker, wave_data=None):
        ene0 = h0
        green_walker = self.calc_green(walker, wave_data)
        ene1 = 2.0 * jnp.sum(green_walker * rot_h1)
        f = jnp.einsum("gij,jk->gik", rot_chol, green_walker.T, optimize="optimal")
        c = vmap(jnp.trace)(f)
        exc = jnp.sum(vmap(lambda x: x * x.T)(f))
        ene2 = 2.0 * jnp.sum(c * c) - exc
        return ene2 + ene1 + ene0

    @partial(jit, static_argnums=0)
    def calc_energy_vmap(self, ham, walkers, wave_data=None):
        return vmap(self.calc_energy, in_axes=(None, None, None, 0, None))(
            ham["h0"], ham["rot_h1"], ham["rot_chol"], walkers, wave_data
        )

    def get_rdm1(self, wave_data):
        rdm1 = 2 * np.eye(self.norb, self.nelec).dot(np.eye(self.norb, self.nelec).T)
        return rdm1

    @partial(jit, static_argnums=0)
    def optimize_orbs(self, ham_data, wave_data=None):
        h1 = ham_data["h1"]
        h2 = ham_data["chol"]
        h2 = h2.reshape((h2.shape[0], h1.shape[0], h1.shape[0]))
        nelec = self.nelec
        h1 = (h1 + h1.T) / 2.0

        def scanned_fun(carry, x):
            dm = carry
            f = jnp.einsum("gij,ik->gjk", h2, dm)
            c = vmap(jnp.trace)(f)
            vj = jnp.einsum("g,gij->ij", c, h2)
            vk = jnp.einsum("glj,gjk->lk", f, h2)
            vhf = vj - 0.5 * vk
            fock = h1 + vhf
            mo_energy, mo_coeff = linalg_utils._eigh(fock)
            idx = jnp.argmax(abs(mo_coeff.real), axis=0)
            mo_coeff = jnp.where(
                mo_coeff[idx, jnp.arange(len(mo_energy))].real < 0, -mo_coeff, mo_coeff
            )
            e_idx = jnp.argsort(mo_energy)
            nmo = mo_energy.size
            mo_occ = jnp.zeros(nmo)
            nocc = nelec
            mo_occ = mo_occ.at[e_idx[:nocc]].set(2)
            mocc = mo_coeff[:, jnp.nonzero(mo_occ, size=nocc)[0]]
            dm = (mocc * mo_occ[jnp.nonzero(mo_occ, size=nocc)[0]]).dot(mocc.T)
            return dm, mo_coeff

        norb = h1.shape[0]
        dm0 = 2 * jnp.eye(norb, nelec).dot(jnp.eye(norb, nelec).T)
        _, mo_coeff = lax.scan(scanned_fun, dm0, None, length=self.n_opt_iter)

        return mo_coeff[-1]

    def __hash__(self):
        return hash(
            (
                self.norb,
                self.nelec,
                self.n_opt_iter,
            )
        )


@dataclass
class uhf:
    norb: int
    nelec: Tuple[int, int]
    n_opt_iter: int = 30

    @partial(jit, static_argnums=0)
    def calc_overlap(self, walker_up, walker_dn, wave_data):
        return jnp.linalg.det(
            wave_data[0][:, : self.nelec[0]].T @ walker_up
        ) * jnp.linalg.det(wave_data[1][:, : self.nelec[1]].T @ walker_dn)

    def calc_overlap_vmap(self, walkers, wave_data):
        return vmap(self.calc_overlap, in_axes=(0, 0, None))(
            walkers[0], walkers[1], wave_data
        )

    @partial(jit, static_argnums=0)
    def calc_green(self, walker_up, walker_dn, wave_data):
        green_up = (
            walker_up.dot(
                jnp.linalg.inv(wave_data[0][:, : self.nelec[0]].T.dot(walker_up))
            )
        ).T
        green_dn = (
            walker_dn.dot(
                jnp.linalg.inv(wave_data[1][:, : self.nelec[1]].T.dot(walker_dn))
            )
        ).T
        return [green_up, green_dn]

    def calc_green_vmap(self, walkers, wave_data):
        return vmap(self.calc_green, in_axes=(0, 0, None))(
            walkers[0], walkers[1], wave_data
        )

    @partial(jit, static_argnums=0)
    def calc_force_bias(self, walker_up, walker_dn, rot_chol, wave_data):
        green_walker = self.calc_green(walker_up, walker_dn, wave_data)
        fb_up = jnp.einsum(
            "gij,ij->g", rot_chol[0], green_walker[0], optimize="optimal"
        )
        fb_dn = jnp.einsum(
            "gij,ij->g", rot_chol[1], green_walker[1], optimize="optimal"
        )
        return fb_up + fb_dn

    def calc_force_bias_vmap(self, walkers, ham, wave_data):
        return vmap(self.calc_force_bias, in_axes=(0, 0, None, None))(
            walkers[0], walkers[1], ham["rot_chol"], wave_data
        )

    @partial(jit, static_argnums=0)
    def calc_energy(self, h0, rot_h1, rot_chol, walker_up, walker_dn, wave_data):
        ene0 = h0
        green_walker = self.calc_green(walker_up, walker_dn, wave_data)
        ene1 = jnp.sum(green_walker[0] * rot_h1[0]) + jnp.sum(
            green_walker[1] * rot_h1[1]
        )
        f_up = jnp.einsum(
            "gij,jk->gik", rot_chol[0], green_walker[0].T, optimize="optimal"
        )
        f_dn = jnp.einsum(
            "gij,jk->gik", rot_chol[1], green_walker[1].T, optimize="optimal"
        )
        c_up = vmap(jnp.trace)(f_up)
        c_dn = vmap(jnp.trace)(f_dn)
        exc_up = jnp.sum(vmap(lambda x: x * x.T)(f_up))
        exc_dn = jnp.sum(vmap(lambda x: x * x.T)(f_dn))
        ene2 = (
            jnp.sum(c_up * c_up)
            + jnp.sum(c_dn * c_dn)
            + 2.0 * jnp.sum(c_up * c_dn)
            - exc_up
            - exc_dn
        ) / 2.0

        return ene2 + ene1 + ene0

    def calc_energy_vmap(self, ham, walkers, wave_data):
        return vmap(self.calc_energy, in_axes=(None, None, None, 0, 0, None))(
            ham["h0"], ham["rot_h1"], ham["rot_chol"], walkers[0], walkers[1], wave_data
        )

    def get_rdm1(self, wave_data):
        dm_up = (wave_data[0][:, : self.nelec[0]]).dot(
            wave_data[0][:, : self.nelec[0]].T
        )
        dm_dn = (wave_data[1][:, : self.nelec[1]]).dot(
            wave_data[1][:, : self.nelec[1]].T
        )
        return jnp.array([dm_up, dm_dn])

    @partial(jit, static_argnums=0)
    def optimize_orbs(self, ham_data, wave_data):
        h1 = ham_data["h1"]
        h1 = h1.at[0].set((h1[0] + h1[0].T) / 2.0)
        h1 = h1.at[1].set((h1[1] + h1[1].T) / 2.0)
        h2 = ham_data["chol"]
        h2 = h2.reshape((h2.shape[0], h1.shape[1], h1.shape[1]))
        nelec = self.nelec

        def scanned_fun(carry, x):
            dm = carry
            f_up = jnp.einsum("gij,ik->gjk", h2, dm[0])
            c_up = vmap(jnp.trace)(f_up)
            vj_up = jnp.einsum("g,gij->ij", c_up, h2)
            vk_up = jnp.einsum("glj,gjk->lk", f_up, h2)
            f_dn = jnp.einsum("gij,ik->gjk", h2, dm[1])
            c_dn = vmap(jnp.trace)(f_dn)
            vj_dn = jnp.einsum("g,gij->ij", c_dn, h2)
            vk_dn = jnp.einsum("glj,gjk->lk", f_dn, h2)
            fock_up = h1[0] + vj_up + vj_dn - vk_up
            fock_dn = h1[1] + vj_up + vj_dn - vk_dn
            mo_energy_up, mo_coeff_up = linalg_utils._eigh(fock_up)
            mo_energy_dn, mo_coeff_dn = linalg_utils._eigh(fock_dn)

            nmo = mo_energy_up.size

            idx_up = jnp.argmax(abs(mo_coeff_up.real), axis=0)
            mo_coeff_up = jnp.where(
                mo_coeff_up[idx_up, jnp.arange(len(mo_energy_up))].real < 0,
                -mo_coeff_up,
                mo_coeff_up,
            )
            e_idx_up = jnp.argsort(mo_energy_up)
            mo_occ_up = jnp.zeros(nmo)
            nocc_up = nelec[0]
            mo_occ_up = mo_occ_up.at[e_idx_up[:nocc_up]].set(1)
            mocc_up = mo_coeff_up[:, jnp.nonzero(mo_occ_up, size=nocc_up)[0]]
            dm_up = (mocc_up * mo_occ_up[jnp.nonzero(mo_occ_up, size=nocc_up)[0]]).dot(
                mocc_up.T
            )

            idx_dn = jnp.argmax(abs(mo_coeff_dn.real), axis=0)
            mo_coeff_dn = jnp.where(
                mo_coeff_dn[idx_dn, jnp.arange(len(mo_energy_dn))].real < 0,
                -mo_coeff_dn,
                mo_coeff_dn,
            )
            e_idx_dn = jnp.argsort(mo_energy_dn)
            mo_occ_dn = jnp.zeros(nmo)
            nocc_dn = nelec[1]
            mo_occ_dn = mo_occ_dn.at[e_idx_dn[:nocc_dn]].set(1)
            mocc_dn = mo_coeff_dn[:, jnp.nonzero(mo_occ_dn, size=nocc_dn)[0]]
            dm_dn = (mocc_dn * mo_occ_dn[jnp.nonzero(mo_occ_dn, size=nocc_dn)[0]]).dot(
                mocc_dn.T
            )

            return jnp.array([dm_up, dm_dn]), jnp.array([mo_coeff_up, mo_coeff_dn])

        dm_up = (wave_data[0][:, : nelec[0]]).dot(wave_data[0][:, : nelec[0]].T)
        dm_dn = (wave_data[1][:, : nelec[1]]).dot(wave_data[1][:, : nelec[1]].T)
        dm0 = jnp.array([dm_up, dm_dn])
        _, mo_coeff = lax.scan(scanned_fun, dm0, None, length=self.n_opt_iter)

        return mo_coeff[-1]

    def __hash__(self):
        return hash(
            (
                self.norb,
                self.nelec,
                self.n_opt_iter,
            )
        )


@dataclass
class noci:
    norb: int
    nelec: Tuple[int, int]
    ndets: int

    @partial(jit, static_argnums=0)
    def calc_overlap_single_det(self, walker_up, walker_dn, trial_up, trial_dn):
        return jnp.linalg.det(
            trial_up[:, : self.nelec[0]].T @ walker_up
        ) * jnp.linalg.det(trial_dn[:, : self.nelec[1]].T @ walker_dn)

    @partial(jit, static_argnums=0)
    def calc_overlap(self, walker_up, walker_dn, wave_data):
        ci_coeffs = wave_data[0]
        dets = wave_data[1]
        overlaps = vmap(self.calc_overlap_single_det, in_axes=(None, None, 0, 0))(
            walker_up, walker_dn, dets[0], dets[1]
        )
        return jnp.sum(ci_coeffs * overlaps)

    @partial(jit, static_argnums=0)
    def calc_overlap_vmap(self, walkers, wave_data):
        return vmap(self.calc_overlap, in_axes=(0, 0, None))(
            walkers[0], walkers[1], wave_data
        )

    @partial(jit, static_argnums=0)
    def calc_green_single_det(self, walker_up, walker_dn, trial_up, trial_dn):
        green_up = (
            walker_up.dot(jnp.linalg.inv(trial_up[:, : self.nelec[0]].T.dot(walker_up)))
        ).T
        green_dn = (
            walker_dn.dot(jnp.linalg.inv(trial_dn[:, : self.nelec[1]].T.dot(walker_dn)))
        ).T
        return [green_up, green_dn]

    @partial(jit, static_argnums=0)
    def calc_green(self, walker_up, walker_dn, wave_data):
        ci_coeffs = wave_data[0]
        dets = wave_data[1]
        overlaps = vmap(self.calc_overlap_single_det, in_axes=(None, None, 0, 0))(
            walker_up, walker_dn, dets[0], dets[1]
        )
        overlap = jnp.sum(ci_coeffs * overlaps)
        up_greens, dn_greens = vmap(
            self.calc_green_single_det, in_axes=(None, None, 0, 0)
        )(walker_up, walker_dn, dets[0], dets[1])
        return up_greens, dn_greens, overlaps

    @partial(jit, static_argnums=0)
    def calc_force_bias(self, walker_up, walker_dn, rot_chol, wave_data):
        ci_coeffs = wave_data[0]
        dets = wave_data[1]
        up_greens, dn_greens, overlaps = self.calc_green(
            walker_up, walker_dn, wave_data
        )
        overlap = jnp.sum(ci_coeffs * overlaps)
        fb_up = (
            jnp.einsum(
                "ngij,nij,n->g",
                rot_chol[0],
                up_greens,
                ci_coeffs * overlaps,
                optimize="optimal",
            )
            / overlap
        )
        fb_dn = (
            jnp.einsum(
                "ngij,nij,n->g",
                rot_chol[1],
                dn_greens,
                ci_coeffs * overlaps,
                optimize="optimal",
            )
            / overlap
        )
        return fb_up + fb_dn

    @partial(jit, static_argnums=0)
    def calc_force_bias_vmap(self, walkers, ham, wave_data):
        return vmap(self.calc_force_bias, in_axes=(0, 0, None, None))(
            walkers[0], walkers[1], ham["rot_chol"], wave_data
        )

    @partial(jit, static_argnums=0)
    def calc_energy_single_det(
        self,
        h0,
        rot_h1_up,
        rot_h1_dn,
        rot_chol_up,
        rot_chol_dn,
        walker_up,
        walker_dn,
        trial_up,
        trial_dn,
    ):
        ene0 = h0
        green_walker = self.calc_green_single_det(
            walker_up, walker_dn, trial_up, trial_dn
        )
        ene1 = jnp.sum(green_walker[0] * rot_h1_up) + jnp.sum(
            green_walker[1] * rot_h1_dn
        )
        f_up = jnp.einsum(
            "gij,jk->gik", rot_chol_up, green_walker[0].T, optimize="optimal"
        )
        f_dn = jnp.einsum(
            "gij,jk->gik", rot_chol_dn, green_walker[1].T, optimize="optimal"
        )
        c_up = vmap(jnp.trace)(f_up)
        c_dn = vmap(jnp.trace)(f_dn)
        exc_up = jnp.sum(vmap(lambda x: x * x.T)(f_up))
        exc_dn = jnp.sum(vmap(lambda x: x * x.T)(f_dn))
        ene2 = (
            jnp.sum(c_up * c_up)
            + jnp.sum(c_dn * c_dn)
            + 2.0 * jnp.sum(c_up * c_dn)
            - exc_up
            - exc_dn
        ) / 2.0

        return ene2 + ene1 + ene0

    @partial(jit, static_argnums=0)
    def calc_energy(self, h0, rot_h1, rot_chol, walker_up, walker_dn, wave_data):
        ci_coeffs = wave_data[0]
        dets = wave_data[1]
        overlaps = vmap(self.calc_overlap_single_det, in_axes=(None, None, 0, 0))(
            walker_up, walker_dn, dets[0], dets[1]
        )
        overlap = jnp.sum(ci_coeffs * overlaps)
        energies = vmap(
            self.calc_energy_single_det, in_axes=(None, 0, 0, 0, 0, None, None, 0, 0)
        )(
            h0,
            rot_h1[0],
            rot_h1[1],
            rot_chol[0],
            rot_chol[1],
            walker_up,
            walker_dn,
            dets[0],
            dets[1],
        )
        ene = jnp.sum(ci_coeffs * overlaps * energies) / overlap
        return ene

    @partial(jit, static_argnums=0)
    def calc_energy_vmap(self, ham, walkers, wave_data):
        return vmap(self.calc_energy, in_axes=(None, None, None, 0, 0, None))(
            ham["h0"], ham["rot_h1"], ham["rot_chol"], walkers[0], walkers[1], wave_data
        )

    @partial(jit, static_argnums=0)
    def get_trans_rdm1_single_det(self, sd_0_up, sd_0_dn, sd_1_up, sd_1_dn):
        dm_up = (
            (sd_0_up[:, : self.nelec[0]])
            .dot(
                jnp.linalg.inv(
                    sd_1_up[:, : self.nelec[0]].T.dot(sd_0_up[:, : self.nelec[0]])
                )
            )
            .dot(sd_1_up[:, : self.nelec[0]].T)
        )
        dm_dn = (
            (sd_0_dn[:, : self.nelec[1]])
            .dot(
                jnp.linalg.inv(
                    sd_1_dn[:, : self.nelec[1]].T.dot(sd_0_dn[:, : self.nelec[1]])
                )
            )
            .dot(sd_1_dn[:, : self.nelec[1]].T)
        )
        return [dm_up, dm_dn]

    @partial(jit, static_argnums=0)
    def get_rdm1(self, wave_data):
        ci_coeffs = wave_data[0]
        dets = wave_data[1]
        overlaps = vmap(
            vmap(self.calc_overlap_single_det, in_axes=(None, None, 0, 0)),
            in_axes=(0, 0, None, None),
        )(dets[0], dets[1], dets[0], dets[1])
        overlap = jnp.sum(jnp.outer(ci_coeffs, ci_coeffs) * overlaps)
        up_rdm1s, dn_rdm1s = vmap(
            vmap(self.get_trans_rdm1_single_det, in_axes=(0, 0, None, None)),
            in_axes=(None, None, 0, 0),
        )(dets[0], dets[1], dets[0], dets[1])
        up_rdm1 = (
            jnp.einsum(
                "hg,hgij->ij", jnp.outer(ci_coeffs, ci_coeffs) * overlaps, up_rdm1s
            )
            / overlap
        )
        dn_rdm1 = (
            jnp.einsum(
                "hg,hgij->ij", jnp.outer(ci_coeffs, ci_coeffs) * overlaps, dn_rdm1s
            )
            / overlap
        )
        return up_rdm1 + dn_rdm1

    # not implemented
    @partial(jit, static_argnums=0)
    def optimize_orbs(self, ham_data, wave_data):
        return wave_data

    def __hash__(self):
        return hash(
            (
                self.norb,
                self.nelec,
                self.ndets,
            )
        )

@dataclass
class hci:
    norb: int
    nelec: Tuple[int, int]
#    ndets: int

#    @partial(jit, static_argnums=0)
#    def det(self,A):
#        return jnp.linalg.det(A)

    @partial(jit, static_argnums=0) 
    def det2x2(self,A):
        return A[0,0] * A[1,1] - A[0,1] * A[1,0]
    
    @partial(jit, static_argnums=0)
    def det3x3(self,a):
        return (a[ 0, 0] * a[ 1, 1] * a[ 2, 2] +
              a[ 0, 1] * a[ 1, 2] * a[ 2, 0] +
              a[ 0, 2] * a[ 1, 0] * a[ 2, 1] -
              a[ 0, 2] * a[ 1, 1] * a[ 2, 0] -
              a[ 0, 0] * a[ 1, 2] * a[ 2, 1] -
              a[ 0, 1] * a[ 1, 0] * a[ 2, 2])
        
    @partial(jit, static_argnums=0)
    def det(self,A):
         #if (A.shape[0] == 2):
         #    return self.det2x2(A)
         #elif (A.shape[0] == 3):
         #    return self.det3x3(A)
         #else:
            return jnp.linalg.det(A)



    @partial(jit, static_argnums=0)
    def ovlp_nA_0B_or_0A_nB(self,GF, Acre, Ades):
        return self.det(GF[jnp.ix_(Acre, Ades)]) 
    
    @partial(jit, static_argnums=0)
    def ovlp_nA_mB(self,GF, Acre, Ades, Bcre, Bdes):
        return self.det(GF[jnp.ix_(Acre, Ades)]) * self.det(GF[jnp.ix_(Bcre, Bdes)])


    @partial(jit, static_argnums=0)
    def calc_green(self,walker):
      return (walker.dot(jnp.linalg.inv(walker[:walker.shape[1], :]))).T
    
    calc_green_vmap = vmap(calc_green)

    @partial(jit, static_argnums=0)
    def calc_overlap(self, walker, wave_data):
      Acre, Ades, Bcre, Bdes, coeff = wave_data[0], wave_data[1], wave_data[2], wave_data[3], wave_data[4]
      #import pdb;pdb.set_trace()
      GF =  (walker.dot(jnp.linalg.inv(walker[:walker.shape[1], :]))).T #self.calc_green(walker)
      
      ovlp0 = jnp.linalg.det(walker[:walker.shape[1], :])**2
      
      
      ovlp = coeff[(0,0)]
      
      
      for i in range(1,MAX_EXCITATION+1):
        #vals = vmap(ovlp_nA_0B_or_0A_nB, in_axes=(None, 0, 0))(GF, Acre[(i,0)], Ades[(i,0)])
        ovlp += vmap(self.ovlp_nA_0B_or_0A_nB, in_axes=(None, 0, 0))(GF, Acre[(i,0)], Ades[(i,0)]).dot(coeff[(i,0)]) 
        ovlp += vmap(self.ovlp_nA_0B_or_0A_nB, in_axes=(None, 0, 0))(GF, Bcre[(0,i)], Bdes[(0,i)]).dot(coeff[(0,i)]) 
        
        for j in range(1, MAX_EXCITATION+1):
          if (i+j <= MAX_EXCITATION):
            ovlp += vmap(self.ovlp_nA_mB, in_axes=(None, 0, 0, 0, 0))(GF, Acre[(i,j)], Ades[(i,j)], Bcre[(i,j)], Bdes[(i,j)]).dot(coeff[(i,j)]) 
      
      
      
      return (ovlp * ovlp0)[0] #jnp.linalg.det(walker[:walker.shape[1], :])**2

    @partial(jit, static_argnums=0)
    def calc_overlap_vmap(self, walkers, wave_data):
        return vmap(self.calc_overlap, in_axes=(0, None))(
            walkers, wave_data
        )

    @partial(jit, static_argnums=0) 
    def overlap_with_rot_SD(self, x_gamma, walker, chol, wave_data):
        Onebody = jnp.einsum('gij,g->ij', chol, x_gamma)
        #walker2 = jsp.linalg.expm(Onebody).dot(walker)
        walker2 = walker + Onebody.dot(walker)
        #return jnp.linalg.det(walker2)
        return self.calc_overlap(walker2, wave_data)

    @partial(jit, static_argnums=0)
    def calc_force_bias(self, walker, chol, wave_data):
         val, grad = vjp(self.overlap_with_rot_SD, jnp.zeros((chol.shape[0],))+0.j, walker, chol, wave_data)
         return grad(1.+0.j)[0]/val 

    @partial(jit, static_argnums=0)
    def calc_force_bias_vmap(self, walkers, ham, wave_data):
        return vmap(self.calc_force_bias, in_axes=(0, None, None))(
            walkers, ham["chol"].reshape(-1, self.norb, self.norb), wave_data
        )

    @partial(jit, static_argnums=0)
    def overlap_with_singleRot(self,x, h1, walker, excitations):
        walker2 = walker + (x)*h1.dot(walker)
        #walker2 = jsp.linalg.expm(x*h1).dot(walker)
        return self.calc_overlap(walker2, excitations)

    @partial(jit, static_argnums=0)
    def overlap_with_doubleRot(self,x, chol_i, walker, excitations):
        walker2 = walker + x * chol_i.dot(walker) + x**2/2. * chol_i.dot(chol_i.dot(walker))
        return self.calc_overlap(walker2, excitations)

    @partial(jit, static_argnums=0) 
    def calc_energy_FD(self, h0, h1, chol, walker, excitations):
        x = 0.
    
        v0 = -0.5 * jnp.einsum("gik,gjk->ij",chol,chol,optimize="optimal")
    
        ##ONE BODY TERM
        f1 = lambda a : self.overlap_with_singleRot(a, h1+v0, walker, excitations)
        val1, dx1 = jvp(f1, [x], [1.])
    
        vmap_fun = vmap(self.overlap_with_doubleRot, in_axes = (None, 0, None, None))
     
        eps, zero = 1.e-4, 0.
        dx = (vmap_fun(eps, chol, walker, excitations) -2.*vmap_fun(zero, chol, walker, excitations) + vmap_fun(-1.*eps, chol, walker, excitations))/eps/eps
    
        return (dx1+jnp.sum(dx)/2.)/val1 + h0 ##why factor of 1./2. in 2-electron


    @partial(jit, static_argnums=0)
    def calc_energy_vmap(self, ham, walkers, wave_data):
        return vmap(self.calc_energy_FD, in_axes=(None, None, None, 0, None))(
            ham["h0"], ham["h1"], ham["chol"].reshape(-1,self.norb,self.norb), walkers, wave_data
        )

    @partial(jit, static_argnums=0)
    def get_trans_rdm1_single_det(self, sd_0_up, sd_0_dn, sd_1_up, sd_1_dn):
        dm_up = (
            (sd_0_up[:, : self.nelec[0]])
            .dot(
                jnp.linalg.inv(
                    sd_1_up[:, : self.nelec[0]].T.dot(sd_0_up[:, : self.nelec[0]])
                )
            )
            .dot(sd_1_up[:, : self.nelec[0]].T)
        )
        dm_dn = (
            (sd_0_dn[:, : self.nelec[1]])
            .dot(
                jnp.linalg.inv(
                    sd_1_dn[:, : self.nelec[1]].T.dot(sd_0_dn[:, : self.nelec[1]])
                )
            )
            .dot(sd_1_dn[:, : self.nelec[1]].T)
        )
        return [dm_up, dm_dn]

    @partial(jit, static_argnums=0)
    def get_rdm1(self, wave_data):
        rdm1 = 2 * np.eye(self.norb, self.nelec).dot(np.eye(self.norb, self.nelec).T)
        return rdm1

#        ci_coeffs = wave_data[0]
#        dets = wave_data[1]
#        overlaps = vmap(
#            vmap(self.calc_overlap_single_det, in_axes=(None, None, 0, 0)),
#            in_axes=(0, 0, None, None),
#        )(dets[0], dets[1], dets[0], dets[1])
#        overlap = jnp.sum(jnp.outer(ci_coeffs, ci_coeffs) * overlaps)
#        up_rdm1s, dn_rdm1s = vmap(
#            vmap(self.get_trans_rdm1_single_det, in_axes=(0, 0, None, None)),
#            in_axes=(None, None, 0, 0),
#        )(dets[0], dets[1], dets[0], dets[1])
#        up_rdm1 = (
#            jnp.einsum(
#                "hg,hgij->ij", jnp.outer(ci_coeffs, ci_coeffs) * overlaps, up_rdm1s
#            )
#            / overlap
#        )
#        dn_rdm1 = (
#            jnp.einsum(
#                "hg,hgij->ij", jnp.outer(ci_coeffs, ci_coeffs) * overlaps, dn_rdm1s
#            )
#            / overlap
#        )
#        return up_rdm1 + dn_rdm1

    # not implemented
    @partial(jit, static_argnums=0)
    def optimize_orbs(self, ham_data, wave_data):
        return wave_data

    def __hash__(self):
        return hash(
            (
                self.norb,
                self.nelec,
        #        self.ndets,
            )
        )
