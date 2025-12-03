import numpy as np
from ad_afqmc import utils

class PrepAfqmc:
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

    def setup_afqmc(self, from_disk):
        if from_disk:
            assert self.path.options != None
            assert self.path.fcidump != None
            assert self.path.tmpdir  != None
            self.read_options()
            self.read_fcidump()
            self.read_observable()
            self.load_mo_coefficients()
            self.tmp.pyscf_prep = None
        else:
            self.from_pyscf_prep()

        self.set_ham()
        self.apply_symmetry_mask()
        self.set_trial()
        self.set_prop()
        self.set_sampler()
        self.setup_print()

    def setup_print(self):
        print(f"# norb: {self.mo_basis.norb}")
        print(f"# nelec: {(self.mol.n_a, self.mol.n_b)}")
        print("#")
        for op in self.options:
            if self.options[op] is not None:
                print(f"# {op}: {self.options[op]}")
        print("#")

    def from_pyscf_prep(self):
        assert self.tmp.pyscf_prep != None
        assert self.options != None
        [nelec, norb, ms, nchol] = self.tmp.pyscf_prep["header"]
        h0 = np.array(self.tmp.pyscf_prep.get("energy_core"))
        h1 = np.array(self.tmp.pyscf_prep.get("hcore")).reshape(norb, norb)
        chol = np.array(self.tmp.pyscf_prep.get("chol")).reshape(-1, norb, norb)
        assert type(ms) is np.int64
        assert type(nelec) is np.int64
        assert type(norb) is np.int64
        ms, nelec, norb = int(ms), int(nelec), int(norb)
        nelec_sp = ((nelec + abs(ms)) // 2, (nelec - abs(ms)) // 2)
        mo_coeff = np.array(self.tmp.pyscf_prep["trial_coeffs"])

        self.mo_basis.norb = norb
        self.mo_basis.h0 = h0
        self.mo_basis.h1 = h1
        self.mo_basis.chol = chol
        self.mol.n_a, self.mol.n_b = nelec_sp
        self.mo_basis.trial_coeff = mo_coeff
        self.tmp.observable = None

        self.get_options()

    def get_options(self):
        self.options = utils.get_options(self.options)
        self.mol.ene0 = self.options["ene0"]

    def read_options(self):
        self.options = utils.read_options(self.options, self.path.options)
        self.mol.ene0 = self.options["ene0"]

    def read_fcidump(self):
        h0, h1, chol, norb, nelec_sp = utils.read_fcidump(self.path.fcidump) # TODO change to FCIDUMP_PATH
        self.mo_basis.norb = norb
        self.mo_basis.h0 = h0
        self.mo_basis.h1 = h1
        self.mo_basis.chol = chol
        self.mol.n_a, self.mol.n_b = nelec_sp

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

    def load_mo_coefficients(self):
        self.mo_basis.trial_coeff = utils.load_mo_coefficients(self.path.tmpdir)

    def set_trial(self):
        self.tmp.trial, self.tmp.wave_data = utils.set_trial(
            self.options,
            self.options["trial"],
            self.mo_basis.trial_coeff,
            self.mo_basis.norb,
            (self.mol.n_a, self.mol.n_b),
            self.path.tmpdir,
            self.tmp.pyscf_prep,
        )
        if "trial_ket" in self.options:
            self.tmp.trial_ket, self.tmp.wave_data_ket = utils.set_trial(
                self.options,
                self.options["trial_ket"],
                self.mo_basis.trial_coeff,
                self.mo_basis.norb,
                (self.mol.n_a, self.mol.n_b),
                self.path.tmpdir,
                self.tmp.pyscf_prep,
            )

    def set_prop(self):
        self.tmp.prop = utils.set_prop(self.options)

    def set_sampler(self):
        self.tmp.sampler = utils.set_sampler(self.options)

class Path:
    def __init__(self):
        self.options = None
        self.fcidump = None
        self.tmpdir = None

class Dummy: pass

class Mol:
    def __init__(self):
        self.n_a = None
        self.n_b = None
        self.spin = None
        self.h0 = None
        self.ene0 = None

class AoBasis:
    def __init__(self):
        self.n_ao = None
        self.basis_coeff = None
        self.overlap = None

class MoBasis:
    def __init__(self):
        self.mo_type = None
        self.norb = None
        self.trial_coeff = None
        self.n_chol = None
        self.chol = None
        self.h1e = None
        self.h1e_mod = None
    def __init__(self):
        self.bra = None

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
        class Rhf: pass
        class Uhf: pass
        class Cisd: pass
        class Ucisd: pass
    
    class Ket:
        class Rhf: pass
        class Uhf: pass
        class Cisd: pass
        class Ucisd: pass
    
    class Walker:
        class Restricted: pass
        class Unrestricted: pass
        class Generalized: pass

class Ci:
    def __init__(self):
        self.type = None
        c1 = None
        c2 = None

    class Restricted: pass
    class Unrestricted: pass
    class Generalized: pass

class Cc:
    def __init__(self):
        self.type = None
        t1 = None
        t2 = None

    class Restricted: pass
    class Unrestricted: pass
    class Generalized: pass

