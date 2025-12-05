import os
import numpy as np
import pickle
from ad_afqmc import utils

class PrepAfqmc:
    __slots__ = (
        "mol", "ao_basis", "mo_basis", "options",
        "ci", "cc", "trial", "path", "tmp", "io"
    )
    def __init__(self):
        self.mol = Mol()
        self.ao_basis = AoBasis()
        self.mo_basis = MoBasis()
        self.options = None
        self.ci = None
        self.cc = None
        self.trial = None
        self.path = Path()
        self.tmp = Dummy()
        self.io = IO()

    def get_setup_data(self):
        return (
            self.tmp.ham_data,
            self.tmp.ham,
            self.tmp.prop,
            self.tmp.trial,
            self.tmp.wave_data,
            self.tmp.trial_ket,
            self.tmp.wave_data_ket,
            self.tmp.sampler,
            self.tmp.observable,
            self.options,
        )

    def setup_afqmc(self):
        self.from_pyscf_prep()

        self.set_options()
        self.set_integrals()
        self.set_trial_coeff()
        self.set_amplitudes()

        self.tmp.pyscf_prep = self.to_pyscf_prep()
        self.set_ham()
        self.apply_symmetry_mask()
        self.read_observable()
        self.set_trial()
        self.set_prop()
        self.set_sampler()
        self.setup_print()

    def setup_print(self):
        print(f"# norb: {self.mo_basis.norb}")
        print(f"# nelec: {(self.mo_basis.nelec_sp)}")
        print("#")
        for op in self.options:
            if self.options[op] is not None:
                print(f"# {op}: {self.options[op]}")
        print("#")

    def read_observable(self):
        self.tmp.observable = utils.read_observable(
            self.mo_basis.norb,
            self.options,
            self.path.tmpdir,
        )

    def set_ham(self):
        mo = self.mo_basis
        mol = self.mol
        ham, ham_data = utils.set_ham(mo.norb, mo.h0, mo.h1, mo.chol, mol.ene0)
        self.tmp.ham = ham
        self.tmp.ham_data = ham_data

    def apply_symmetry_mask(self):
        self.tmp.ham_data = utils.apply_symmetry_mask(self.tmp.ham_data, self.options)

    def set_trial(self):
        self.tmp.trial, self.tmp.wave_data = utils.set_trial(
            self.options,
            self.options["trial"], # TODO trial
            self.mo_basis.trial_coeff,
            self.mo_basis.norb,
            self.mo_basis.nelec_sp,
            self.path.tmpdir,
            self.tmp.pyscf_prep,
        )
        if "trial_ket" in self.options:
            self.tmp.trial_ket, self.tmp.wave_data_ket = utils.set_trial(
                self.options,
                self.options["trial_ket"], # TODO trial
                self.mo_basis.trial_coeff,
                self.mo_basis.norb,
                self.mo_basis.nelec_sp,
                self.path.tmpdir,
                self.tmp.pyscf_prep,
            )

    def set_prop(self):
        self.tmp.prop = utils.set_prop(self.options)

    def set_sampler(self):
        self.tmp.sampler = utils.set_sampler(self.options)

    ##################
    ### pyscf_prep ###
    ##################
    def from_pyscf_prep(self):
        if not hasattr(self.tmp, "pyscf_prep"): return
        if self.tmp.pyscf_prep is None: return

        [nelec, norb, ms, nchol] = self.tmp.pyscf_prep["header"]
        h0 = np.array(self.tmp.pyscf_prep.get("energy_core"))
        h1 = np.array(self.tmp.pyscf_prep.get("hcore")).reshape(norb, norb)
        h1_mod = np.array(self.tmp.pyscf_prep.get("hcore_mod")).reshape(norb, norb)
        chol = np.array(self.tmp.pyscf_prep.get("chol")).reshape(-1, norb, norb)
        assert type(ms) is np.int64
        assert type(nelec) is np.int64
        assert type(norb) is np.int64
        ms, nelec, norb = int(ms), int(nelec), int(norb)
        nelec_sp = ((nelec + abs(ms)) // 2, (nelec - abs(ms)) // 2)
        mo_coeff = np.array(self.tmp.pyscf_prep["trial_coeffs"])

        if "amplitudes" in self.tmp.pyscf_prep:
            self.tmp.amplitudes = self.tmp.pyscf_prep["amplitudes"]

        self.mo_basis.norb = norb
        self.mo_basis.h0 = h0
        self.mo_basis.h1 = h1
        self.mo_basis.h1_mod = h1_mod
        self.mo_basis.chol = chol
        self.mo_basis.nelec_sp = nelec_sp
        self.mo_basis.trial_coeff = mo_coeff
        self.tmp.observable = None

        self.set_options()

    def to_pyscf_prep(self):
        pyscf_prep = {}

        pyscf_prep["header"] = np.array([
            sum(self.mo_basis.nelec_sp),
            self.mo_basis.norb,
            self.mol.spin,
            self.mo_basis.chol.shape[0]
        ])
        pyscf_prep["hcore"] = self.mo_basis.h1.flatten()
        pyscf_prep["hcore_mod"] = self.mo_basis.h1_mod.flatten()
        pyscf_prep["chol"] = self.mo_basis.chol.flatten()
        pyscf_prep["energy_core"] = self.mo_basis.h0
        #if self.mo_basis.trial_coeff is not None:
        pyscf_prep["trial_coeffs"] = self.mo_basis.trial_coeff
        if hasattr(self.tmp, "amplitudes"): # Super dirty
            pyscf_prep["amplitudes"] = self.tmp.amplitudes

        return pyscf_prep

    ###############
    ### Options ###
    ###############
    def set_options(self):
        io = self.io.options
        # Read
        if io == IO.Read:
            self.read_options()
        # Compute
        elif io == IO.Write or io == IO.NoIO:
            self.options = utils.get_options(self.options)
            self.mol.ene0 = self.options["ene0"]
        else:
            raise TypeError(f"self.io.options is '{io}' instead of IO.Read/Write/NoIO.")

        # Write
        if io == IO.Write:
            self.write_options()
            self.io.set_read_options()

    def read_options(self):
        self.options = utils.read_options(self.options, self.path.options)
        self.mol.ene0 = self.options["ene0"]

    def write_options(self):
        assert self.path.options is not None
        with open(self.path.options+"/options.bin", "wb") as f:
            pickle.dump(self.options, f)

    #################
    ### Integrals ###
    #################
    # TODO add asserts
    def set_integrals(self):
        io = self.io.fcidump
        # Read
        if io == IO.Read:
            self.read_fcidump()
        # Compute
        elif io == IO.Write or io == IO.NoIO:
            if self.mo_basis.chol is None:
                self.compute_integrals()
        else:
            raise TypeError(f"self.io.fcidump is '{io}' instead of IO.Read/Write/NoIO.")

        # Write
        if self.io.fcidump == IO.Write:
            self.write_fcidump()
            self.io.set_read_fcidump()

    def compute_integrals(self):
        # TODO the function compute_cholesky_integrals should be split
        # mol and mf should be removed
        h1e, chol, nelec, enuc, _, _ = utils.compute_cholesky_integrals(
            self.tmp.mol,
            self.tmp.mf,
            self.mo_basis.basis_coeff,
            self.ao_basis.custom_integrals,
            self.mo_basis.norb_frozen,
            self.mo_basis.chol_cut,
        )
        print("# Finished calculating Cholesky integrals\n#")
        
        nbasis = h1e.shape[-1]
        print("# Size of the correlation space:")
        print(f"# Number of electrons: {nelec}")
        print(f"# Number of basis functions: {nbasis}")
        print(f"# Number of Cholesky vectors: {chol.shape[0]}\n#")
        chol = chol.reshape((-1, nbasis, nbasis))
        v0 = 0.5 * np.einsum("nik,njk->ij", chol, chol, optimize="optimal")
        h1e_mod = h1e - v0
        chol = chol.reshape((chol.shape[0], -1))
        self.mo_basis.norb = nbasis
        self.mo_basis.nelec_sp = nelec
        self.mo_basis.h0 = enuc
        self.mo_basis.h1 = h1e
        self.mo_basis.h1_mod = h1e_mod
        self.mo_basis.chol = chol

    def read_fcidump(self):
        h0, h1, h1_mod, chol, norb, nelec_sp = utils.read_fcidump(self.path.fcidump)
        self.mo_basis.norb = norb
        self.mo_basis.h0 = h0
        self.mo_basis.h1 = h1
        self.mo_basis.h1_mod = h1_mod
        self.mo_basis.chol = chol
        self.mo_basis.nelec_sp = nelec_sp

    def write_fcidump(self):
        utils.write_dqmc(
            self.mo_basis.h1,
            self.mo_basis.h1_mod,
            self.mo_basis.chol,
            sum(self.mo_basis.nelec_sp),
            self.mo_basis.norb,
            self.mo_basis.h0,
            ms=self.mol.spin,
            filename=self.path.fcidump + "/FCIDUMP_chol",
            mo_coeffs=self.mo_basis.trial_coeff,
        )

    ###################
    ### Trial coeff ###
    ###################
    # TODO add asserts
    def set_trial_coeff(self):
        io = self.io.trial_coeff
        # Read
        if io == IO.Read:
            self.read_trial_coeff()
        # Compute
        elif io == IO.Write or io == IO.NoIO:
            if self.mo_basis.trial_coeff is None:
                self.mo_basis.trial_coeff = utils.get_trial_coeffs(
                    self.tmp.mol, # TODO Remove, only needed for the overlap
                    self.tmp.mf, # TODO Replace with MoType
                    self.mo_basis.basis_coeff,
                    self.mo_basis.norb,
                    self.mo_basis.norb_frozen,
                )
        else:
            raise TypeError(f"self.io.trial_coeff is '{io}' instead of IO.Read/Write/NoIO.")

        # Write
        if self.io.trial_coeff == IO.Write:
            self.write_trial_coeff()
            self.io.set_read_trial_coeff()

    def read_trial_coeff(self):
        self.mo_basis.trial_coeff = utils.load_mo_coefficients(self.path.tmpdir)

    # TODO write should only write them
    def write_trial_coeff(self):
        utils.write_trial_coeffs(
            self.tmp.mol,
            self.tmp.mf,
            self.mo_basis.basis_coeff,
            self.mo_basis.norb,
            self.mo_basis.norb_frozen,
            self.path.tmpdir,
        )

    ##################
    ### Amplitudes ###
    ##################
    def set_amplitudes(self):
        if not hasattr(self.tmp, "cc"): return

        io = self.io.amplitudes
        # Read
        if self.io.amplitudes == IO.Read:
            # path should also contain the filename
            self.tmp.amplitudes = np.load(self.path.amplitudes + "/amplitudes.npz")
        # Compute
        elif self.io.amplitudes == IO.Write or self.io.amplitudes == IO.NoIO:
            if not hasattr(self.tmp, "amplitudes"): # Super dirty
                self.set_ci_from_cc()
        else:
            raise TypeError(f"self.io.amplitudes is '{io}' instead of IO.Read/Write/NoIO.")
        # Write
        if self.io.amplitudes == IO.Write:
            self.write_amplitudes()
            self.io.set_read_amplitudes()

    def set_ci_from_cc(self):
        self.tmp.amplitudes = utils.get_ci_amplitudes_from_cc(self.tmp.cc)

    # TODO should only write them
    def write_amplitudes(self):
        if hasattr(self.tmp, "cc"): # Super dirty
            utils.write_pyscf_ccsd(self.tmp.cc, self.path.tmpdir)

    def prep(self):
        self.set_amplitudes()
        self.set_integrals() 
        self.set_trial_coeff()
        self.set_options()
 
class Path:
    __slots__ = ("options", "fcidump", "tmpdir", "amplitudes")

    def __init__(self):
        self.options = None
        self.fcidump = None
        self.tmpdir = None
        self.amplitudes = None

    def set(self, path):
        for attr in self.__slots__:
            setattr(self, attr, path)

class Dummy: pass

class Mol:
    __slots__ = ("n_a", "n_b", "spin", "ene0")

    def __init__(self):
        self.n_a = None
        self.n_b = None
        self.spin = None
        self.ene0 = None

class AoBasis:
    __slots__ = ("n_ao", "overlap", "custom_integrals")

    def __init__(self):
        self.n_ao = None
        self.overlap = None
        self.custom_integrals = None

class MoBasis:
    __slots__ = (
        "mo_type", "norb", "nelec_sp", "norb_frozen",
        "mo_coeff", "basis_coeff", "trial_coeff", "n_chol", "chol",
        "chol_cut", "h0", "h1", "h1_mod",
    )
    def __init__(self):
        self.mo_type = None
        self.norb = None # Number of active MOs
        self.nelec_sp = None # Number of active electron
        self.norb_frozen = None
        self.mo_coeff = None
        self.basis_coeff = None
        self.trial_coeff = None
        self.n_chol = None
        self.chol = None
        self.chol_cut = None
        self.h0 = None
        self.h1 = None
        self.h1_mod = None

    class MoType:
        class Restricted: pass
        class Unrestricted: pass
        class Generalized: pass

class Trial:
    def __init__(self):
        self.bra = None
        self.ket = None
        self.walker = None

    class Bra:
        class RHF: pass
        class UHF: pass
        class GHF: pass
        class GHFComplex: pass
        class NOCI: pass
        class CISD: pass
        class UCISD: pass
        class GCISD: pass
        class PKL: pass
    
    class Ket:
        class RHF: pass
        class UHF: pass
        class GHF: pass
        class CCSD: pass
        class UCCSD: pass
    
    class Walker:
        class Restricted: pass
        class Unrestricted: pass
        class Generalized: pass

class CI:
    def __init__(self):
        self.type = None
        c1 = None
        c2 = None

    class Restricted: pass
    class Unrestricted: pass
    class Generalized: pass

class CC:
    def __init__(self):
        self.type = None
        t1 = None
        t2 = None

    class Restricted: pass
    class Unrestricted: pass
    class Generalized: pass

class IO:
    __slots__ = ("options", "fcidump", "trial_coeff", "amplitudes")

    def __init__(self):
        self.options = None
        self.fcidump = None
        self.trial_coeff = None
        self.amplitudes = None

    def set(self, io_mode):
        assert io_mode in (self.Read, self.Write, self.NoIO)
        for attr in self.__slots__:
            setattr(self, attr, io_mode)

    def set_no_io(self):
        self.set(IO.NoIO)

    def set_read(self):
        self.set(IO.Read)

    def set_write(self):
        self.set(IO.Write)

    class Read:
        def __eq__(self, other):
            return isinstance(other, self.Read)
    class Write:
        def __eq__(self, other):
            return isinstance(other, self.Write)
    class NoIO:
        def __eq__(self, other):
            return isinstance(other, self.NoIO)

    def set_read_options(self):
        self.options = self.Read
    def set_read_fcidump(self):
        self.fcidump = self.Read
    def set_read_trial_coeff(self):
        self.trial_coeff = self.Read
    def set_read_amplitudes(self):
        self.amplitudes = self.Read

    def set_write_options(self):
        self.options = self.Write
    def set_write_fcidump(self):
        self.fcidump = self.Write
    def set_write_trial_coeff(self):
        self.trial_coeff = self.Write
    def set_write_amplitudes(self):
        self.amplitudes = self.Write

    def set_no_io_options(self):
        self.options = self.NoIO
    def set_no_io_fcidump(self):
        self.fcidump = self.NoIO
    def set_no_io_trial_coeff(self):
        self.trial_coeff = self.NoIO
    def set_no_io_amplitudes(self):
        self.amplitudes = self.NoIO
