import os
from enum import Enum, auto
import numpy as np
import pickle
from ad_afqmc import utils

from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD
from pyscf.cc.gccsd import GCCSD
from pyscf.scf.hf import RHF
from pyscf.scf.rohf import ROHF
from pyscf.scf.uhf import UHF
from pyscf.scf.ghf import GHF

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
        self.tmp = Tmp()
        self.io = IO()

    # Returns data needed for the afqmc calculation
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

    # Prepares the data for the afqmc calculation
    def setup_afqmc(self):
        self.set_options()
        self.set_integrals()
        self.set_trial_coeff()
        self.set_amplitudes()

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
            self
        )
        if "trial_ket" in self.options:
            self.tmp.trial_ket, self.tmp.wave_data_ket = utils.set_trial(
                self
            )

    def set_prop(self):
        self.tmp.prop = utils.set_prop(self.options)

    def set_sampler(self):
        self.tmp.sampler = utils.set_sampler(self.options)

    def set_mol(self, mol):
        self.tmp.mol = mol
        self.mol.spin = mol.spin
        self.mol.n_a, self.mol.n_b = mol.nelec

    def set_pyscf_mf_cc(self, mf_or_cc, mf_or_cc_ket):
        if isinstance(mf_or_cc, (CCSD, UCCSD, GCCSD)):
            self.tmp.mf = mf_or_cc._scf
            self.tmp.cc = mf_or_cc
        elif isinstance(mf_or_cc_ket, (CCSD, UCCSD, GCCSD)):
            self.tmp.mf = mf_or_cc_ket._scf
            self.tmp.cc = mf_or_cc_ket
        elif isinstance(mf_or_cc, (RHF, ROHF, UHF, GHF)):
            self.tmp.mf = mf_or_cc
        else:
            raise TypeError(f"Unexpected object '{mf_or_cc}'.")

    def set_basis_coeff(self, basis_coeff):
        if basis_coeff is None:
            if isinstance(self.tmp.mf, UHF):
                basis_coeff = self.tmp.mf.mo_coeff[0]
            elif isinstance(self.tmp.mf, (RHF, ROHF, GHF)):
                basis_coeff = self.tmp.mf.mo_coeff
            else:
                raise TypeError(f"Unexpected object '{self.tmp.mf}'.")

        self.mo_basis.basis_coeff = basis_coeff

    ##################
    ### pyscf_prep ###
    ##################
    def from_pyscf_prep(self):
        assert hasattr(self.tmp, "pyscf_prep")
        assert self.tmp.pyscf_prep is not None

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
        self.ao_basis.n_chol = chol.shape[0]
        self.mo_basis.nelec_sp = nelec_sp
        self.mo_basis.trial_coeff = mo_coeff
        self.tmp.observable = None

        #self.set_options()

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

        # Check
        IOMode.check(io)

        if io.is_read():
            self.path.check(self.path.options + "/options.bin")

        # Read
        if io.is_read():
            self.read_options()
        # Compute
        elif io.is_write() or io.is_no_io():
            self.options = utils.get_options(self.options)
            self.mol.ene0 = self.options["ene0"]
        else:
            raise ValueError(f"io should be IOMode.Read/Write/NoIO, not {io}.")

        # Write
        if io.is_write():
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

        # Check
        IOMode.check(io)

        if io.is_read():
            self.path.check(self.path.fcidump + "/FCIDUMP_chol")

        # Read
        if io.is_read():
            self.read_fcidump()
        # Compute
        elif io.is_write() or io.is_no_io():
            if self.mo_basis.chol is None:
                self.compute_integrals()
        else:
            raise ValueError(f"io should be IOMode.Read/Write/NoIO, not {io}.")

        # Write
        if io.is_write():
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
            self.ao_basis.chol_cut,
        )
        print("# Finished calculating Cholesky integrals\n#")
        
        norb = h1e.shape[-1]
        n_chol = chol.shape[0]
        print("# Size of the correlation space:")
        print(f"# Number of electrons: {nelec}")
        print(f"# Number of orbitals: {norb}")
        print(f"# Number of Cholesky vectors: {n_chol}\n#")
        chol = chol.reshape((-1, norb, norb))
        v0 = 0.5 * np.einsum("nik,njk->ij", chol, chol, optimize="optimal")
        h1e_mod = h1e - v0
        chol = chol.reshape((n_chol, norb * norb))
        self.mo_basis.norb = norb
        self.mo_basis.nelec_sp = nelec
        self.mo_basis.h0 = enuc
        self.mo_basis.h1 = h1e
        self.mo_basis.h1_mod = h1e_mod
        self.mo_basis.chol = chol
        self.ao_basis.n_chol = n_chol

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

        # Check
        IOMode.check(io)

        if io.is_read():
            self.path.check(self.path.trial_coeff + "/mo_coeff.npz")

        # Read
        if io.is_read():
            self.read_trial_coeff()

        # Compute
        elif io.is_write() or io.is_no_io():
            if self.mo_basis.trial_coeff is None:
                self.mo_basis.trial_coeff = utils.get_trial_coeffs(
                    self.tmp.mol, # TODO Remove, only needed for the overlap
                    self.tmp.mf, # TODO Replace with MoType
                    self.mo_basis.basis_coeff,
                    self.mo_basis.norb,
                    self.mo_basis.norb_frozen,
                )
        else:
            raise ValueError(f"io should be IOMode.Read/Write/NoIO, not {io}.")

        # Write
        if io.is_write():
            self.write_trial_coeff()
            self.io.set_read_trial_coeff()

    def read_trial_coeff(self):
        self.mo_basis.trial_coeff = utils.load_mo_coefficients(self.path.trial_coeff)

    # TODO write should only write them
    def write_trial_coeff(self):
        utils.write_trial_coeffs(
            self.tmp.mol,
            self.tmp.mf,
            self.mo_basis.basis_coeff,
            self.mo_basis.norb,
            self.mo_basis.norb_frozen,
            self.path.trial_coeff,
        )

    ##################
    ### Amplitudes ###
    ##################
    def set_amplitudes(self):
        bra = self.options["trial"]
        ket = self.options["trial_ket"]

        trial = ["cisd", "ucisd", "gcisd", "gcisd_complex", "ccsd", "uccsd"]
        if not bra in trial and not ket in trial:
            return

        io = self.io.amplitudes

        # Check
        IOMode.check(io)

        if io.is_read():
            self.path.check(self.path.amplitudes + "/amplitudes.npz")

        # Read
        if io.is_read():
            # path should also contain the filename
            self.tmp.amplitudes = np.load(self.path.amplitudes + "/amplitudes.npz")
        # Compute
        elif io.is_write() or io.is_no_io():
            if not hasattr(self.tmp, "amplitudes"): # Super dirty
                if not hasattr(self.tmp, "cc"): # Super dirty
                    raise AttributeError(f"self.tmp.cc must exist and point to the cc pyscf object in order to compute the amplitudes.")
                self.set_ci_from_cc()
        else:
            raise ValueError(f"io should be IOMode.Read/Write/NoIO, not {io}.")

        # Write
        if io.is_write():
            self.write_amplitudes()
            self.io.set_read_amplitudes()

    def set_ci_from_cc(self):
        self.tmp.amplitudes = utils.get_ci_amplitudes_from_cc(self.tmp.cc)

    # TODO should only write them
    def write_amplitudes(self):
        utils.write_pyscf_ccsd(self.tmp.cc, self.path.amplitudes)

    # Read/Compute/Write what is needed for the calculation
    def prep(self):
        self.set_options()
        self.set_integrals() 
        self.set_trial_coeff()
        self.set_amplitudes()

class Path:
    __slots__ = ("tmpdir", "options", "fcidump", "trial_coeff", "amplitudes")

    def __init__(self):
        self.tmpdir = None
        self.options = None
        self.fcidump = None
        self.trial_coeff = None
        self.amplitudes = None

    # Brute-force
    def set(self, path):
        for attr in self.__slots__:
            setattr(self, attr, path)

    # Check if the path exists
    def check(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File '{path}' does not exist.")

class Tmp: pass

class Mol:
    __slots__ = ("n_a", "n_b", "spin", "ene0")

    def __init__(self):
        self.n_a = None
        self.n_b = None
        self.spin = None
        self.ene0 = None

class AoBasis:
    __slots__ = ("n_ao", "n_chol", "chol_cut", "overlap", "custom_integrals")

    def __init__(self):
        self.n_ao = None
        self.n_chol = None
        self.chol_cut = None
        self.overlap = None
        self.custom_integrals = None

class MoBasis:
    __slots__ = (
        "mo_type", "norb", "nelec_sp", "norb_frozen",
        "mo_coeff", "basis_coeff", "trial_coeff","chol",
        "h0", "h1", "h1_mod",
    )
    def __init__(self):
        self.mo_type = None
        self.norb = None # Number of active MOs
        self.nelec_sp = None # Number of active electron
        self.norb_frozen = None
        self.mo_coeff = None
        self.basis_coeff = None
        self.trial_coeff = None
        self.chol = None
        self.h0 = None
        self.h1 = None
        self.h1_mod = None

    class MoType:
        class Restricted: pass
        class Unrestricted: pass
        class Generalized: pass

# Options should be divided, it does not make any sense to have AD, LNO, nuclear
# gradient, symmetry, ... keywords here as their number is becoming large.
# I don't think mode should be here.
class Options:
    # To catch typos...
    __slots__ = ("dt", "n_prop_steps", "n_ene_blocks",
    "n_walkers", "n_sr_blocks", "n_blocks", "n_ene_blocks_eql",
    "n_sr_blocks_eql", "n_eql", "seed", "ad_mode", "orbital_rotation",
    "do_sr", "walker_type", "symmetry_projector", "ngrid", "optimize_trial",
    "target_spin", "symmetry", "save_walkers", "dR", "free_projection",
    "ene0", "n_chunks", "vhs_mixed_precision", "trial_mixed_precision",
    "memory_mode", "write_to_disk", "prjlo",
    )

    def __init__(self, mode):
        self.dt = 0.005
        self.n_prop_steps = 50
        self.n_ene_blocks = 1
        if mode == "small":
            self.n_walkers = 50
            self.n_sr_blocks = 1
            self.n_blocks = 200
            self.n_ene_blocks_eql = 1
            self.n_sr_blocks_eql = 5
            self.n_eql = 10
        elif mode == "production":
            self.n_walkers = 200
            self.n_sr_blocks = 20
            self.n_blocks = 500
            self.n_ene_blocks_eql = 5
            self.n_sr_blocks_eql = 10
            self.n_eql = 3
        self.seed = np.random.randint(1, int(1e6))
        self.ad_mode = None
        self.orbital_rotation = True
        self.do_sr = True
        self.walker_type = "restricted"

        # this can be tr, s2 or sz for time-reversal, S^2, or S_z symmetry projection, respectively
        self.symmetry_projector = None
        self.ngrid = 4 # Number of grid point for the quadrature
        self.optimize_trial = False
        self.target_spin = 0  # 2S and is only used when symmetry_projector is s2
        self.symmetry = False
        self.save_walkers = False
        self.dR = 1e-5  # displacement used in finite difference to calculate integral gradients for ad_mode = nuc_grad
        self.free_projection = False

        self.ene0 = 0.0
        self.n_chunks = 1
        self.vhs_mixed_precision = False
        self.trial_mixed_precision = False
        self.memory_mode = "low"
        self.write_to_disk = False # Write FCIDUMP and ci/cc coeff to disk
        self.prjlo = None  # used in LNO, need to fix

    def to_dict(self):
        return {slot: getattr(self, slot) for slot in self.__slots__}

    def from_dict(self, options: dict):
        for key, val in options.items():
            setattr(self, key, val)

class IOMode(Enum):
    Read = auto()
    Write = auto()
    NoIO = auto()

    def is_read(self):
        return self is self.Read

    def is_write(self):
        return self is self.Write

    def is_no_io(self):
        return self is self.NoIO

    def check(io_mode):
        if not isinstance(io_mode, IOMode):
            raise TypeError(f"Type '{io}' instead of IOMode.Read/Write/NoIO.")

class IO:
    __slots__ = ("options", "fcidump", "trial_coeff", "amplitudes")

    def __init__(self):
        self.options = None
        self.fcidump = None
        self.trial_coeff = None
        self.amplitudes = None

    def set(self, io_mode):
        assert io_mode in IOMode
        for attr in self.__slots__:
            setattr(self, attr, io_mode)

    def set_no_io(self):
        self.set(IOMode.NoIO)

    def set_read(self):
        self.set(IOMode.Read)

    def set_write(self):
        self.set(IOMode.Write)

# Creates the method set_read/write/noio_ for all the IO fields, i.e.,
# set_read_options, set_read_fcidump, ...
def _make_setter(field, io_mode):
    def setter(self):
        setattr(self, field, io_mode)
    #setter.__name__ = f"set_{io_mode.name.lower()}_{field}"
    return setter

for field in IO.__slots__:
    for io_mode in IOMode:
        fname = f"set_{io_mode.name.lower()}_{field}"
        setattr(IO, fname, _make_setter(field, io_mode))
