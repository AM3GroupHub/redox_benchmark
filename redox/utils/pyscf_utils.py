import os
from types import MethodType
from typing import Dict

import numpy as np
import ase.io
from ase.calculators.calculator import Calculator, all_changes
from pyscf import gto, lib, symm, scf, dft
from pyscf.hessian import thermo
from ase import Atoms, units
from sella import Sella


def read_pcm_eps() -> Dict[str, float]:
    # from https://gaussian.com/scrf/
    pcm_eps_txt = os.path.join(os.path.dirname(__file__), "pcm_eps.txt")
    with open(pcm_eps_txt, "r") as f:
        lines = f.readlines()
    eps_dict = {}
    for line in lines:
        solvent, eps = line.split(": ")
        eps_dict[solvent.strip().lower()] = float(eps.strip())
    return eps_dict


def build_dft(mol: gto.Mole, **kwargs) -> scf.hf.SCF:
    """
    Build a PySCF mean field object with the given molecule and parameters.
    Args:
        mol (gto.Mole): The molecule object.
        kwargs: Additional parameters for the mean field object.
    Returns:
        mf (pyscf.dft.KS): The mean field object.
    """
    # convert memory from GB to MB
    max_memory = kwargs.get("max_memory", None)
    if max_memory is not None:
        max_memory *= 1024  # convert GB to MB

    # build mole
    basis = kwargs.get("basis", "def2-SVP")
    charge = kwargs.get("charge", 0)
    spin = kwargs.get("spin", 0)
    mol.build(basis=basis, charge=charge, spin=spin, max_memory=max_memory)
    
    # set exchange-correlation functional
    xc = kwargs.get("xc", "B3LYP")
    mf = dft.KS(mol, xc=xc)

    # density fitting
    density_fit = kwargs.get("density_fit", False)
    if density_fit:
        aux_basis = kwargs.get("aux_basis", None)
        mf = mf.density_fit(auxbasis=aux_basis)
    
    # set solvation model
    solvation = kwargs.get("solvation", None)
    solvent = kwargs.get("solvent", None)
    solvent_params = kwargs.get("solvent_params", None)
    assert solvent is None or solvent_params is None, \
        "You can only specify one of --solvent or --solvent-param"
    pcm_models = {"C-PCM", "IEF-PCM", "SS(V)PE", "COSMO"}
    if solvation in pcm_models:
        eps_dict = read_pcm_eps()
        mf = mf.PCM()
        mf.with_solvent.method = solvation
        if solvent is not None:
            assert solvent.lower() in eps_dict, \
                f"Solvent {solvent} not found in predefined solvents"
            eps = eps_dict[solvent.lower()]
        elif solvent_params is not None:
            assert len(solvent_params) == 1, \
                "You must provide exactly one parameter of dielectric constant for PCM model"
            eps = solvent_params[0]
        mf.with_solvent.eps = eps
    elif solvation == "SMD":
        mf = mf.SMD()
        if solvent is not None:
            mf.with_solvent.solvent = solvent
        elif solvent_params is not None:
            assert len(solvent_params) == 8, \
                """
                You must provide exactly 8 parameters for SMD solvation model:
                [n, n25, alpha, beta, gamma, epsilon, phi, psi]
                """
            mf.with_solvent.solvent_descriptors = solvent_params
    
    # set other parameters
    disp = kwargs.get("disp", None)
    if disp is not None:
        mf.disp = disp.lower()
    mf.conv_tol = kwargs.get("scf_conv", 1e-8)
    mf.grids.level = kwargs.get("grid", 9)
    mf.max_cycle = kwargs.get("scf_max_cycle", 50)
    
    return mf


def dump_normal_mode(mol: gto.Mole, results: Dict[str, np.ndarray]) -> None:
    """
    The function in PySCF does not dump imagnary frequencies.
    We made a custom function to dump all frequencies and normal modes.
    Args:
        mol (gto.Mole): The molecule object.
        results (Dict[str, np.ndarray]): A dictionary containing frequencies and normal modes.
    """

    dump = mol.stdout.write
    freq_wn = results['freq_wavenumber']
    if np.iscomplexobj(freq_wn):
        freq_wn = freq_wn.real - abs(freq_wn.imag)
    nfreq = freq_wn.size

    r_mass = results['reduced_mass']
    force = results['force_const_dyne']
    vib_t = results['vib_temperature']
    mode = results['norm_mode']
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]

    def inline(q, col0, col1):
        return ''.join('%20.4f' % q[i] for i in range(col0, col1))
    def mode_inline(row, col0, col1):
        return '  '.join('%6.2f%6.2f%6.2f' % (mode[i,row,0], mode[i,row,1], mode[i,row,2])
                         for i in range(col0, col1))

    for col0, col1 in lib.prange(0, nfreq, 3):
        dump('Mode              %s\n' % ''.join('%20d'%i for i in range(col0,col1)))
        dump('Irrep\n')
        dump('Freq [cm^-1]          %s\n' % inline(freq_wn, col0, col1))
        dump('Reduced mass [au]     %s\n' % inline(r_mass, col0, col1))
        dump('Force const [Dyne/A]  %s\n' % inline(force, col0, col1))
        dump('Char temp [K]         %s\n' % inline(vib_t, col0, col1))
        #dump('IR\n')
        #dump('Raman\n')
        dump('Normal mode            %s\n' % ('       x     y     z'*(col1-col0)))
        for j, at in enumerate(symbols):
            dump('    %4d%4s               %s\n' % (j, at, mode_inline(j, col0, col1)))


def build_method(config: dict):
    """
    Create a PySCF mean field object from a configuration dictionary.

    Args:
        config (dict): Configuration dictionary containing parameters for the mean field object.

    Returns:
        mf (pyscf.dft.KS): The mean field object.
    """
    xc = config.get("xc", "B3LYP")
    basis = config.get("basis", "def2-SVP")
    ecp = config.get("ecp", None)
    nlc = config.get("nlc", "")
    disp = config.get("disp", None)
    grids = config.get("grids", {"atom_grid": (99, 590)})
    nlcgrids = config.get("nlcgrids", {"atom_grid": (50, 194)})
    verbose = config.get("verbose", 2)
    scf_conv_tol = config.get("scf_conv_tol", 1e-8)
    direct_scf_tol = config.get("direct_scf_tol", 1e-8)
    scf_max_cycle = config.get("scf_max_cycle", 50)
    with_df = config.get("with_df", True)
    auxbasis = config.get("auxbasis", "def2-universal-jkfit")
    with_gpu = config.get("with_gpu", True)
    
    with_solvent = config.get("with_solvent", False)
    solvent = config.get("solvent", {"method": "ief-pcm", "eps": 78.3553, "solvent": "water"})
    
    max_memory = config.get("max_memory", None)
    if max_memory is not None:
        max_memory *= 1024  # convert GB to MB
    threads = config.get("threads", os.environ.get("OMP_NUM_THREADS", os.cpu_count()))
    lib.num_threads(threads)

    atom = config.get("inputfile", "mol.xyz")
    charge = config.get("charge", 0)
    spin = config.get("spin", None)

    # build molecule
    mol = gto.M(
        atom=atom,
        basis=basis,
        ecp=ecp,
        max_memory=max_memory,
        verbose=verbose,
        charge=charge,
        spin=spin,
    )
    mol.build()

    # build Kohn-Sham object
    mf = dft.KS(mol, xc=xc)
    mf.nlc = nlc
    mf.disp = disp
    # set grids
    if "atom_grid" in grids:
        mf.grids.atom_grid = grids["atom_grid"]
    if "level" in grids:
        mf.grids.level = grids["level"]
    if mf._numint.libxc.is_nlc(mf.xc) or nlc is not None:
        if "atom_grid" in nlcgrids:
            mf.nlcgrids.atom_grid = nlcgrids["atom_grid"]
        if "level" in nlcgrids:
            mf.nlcgrids.level = nlcgrids["level"]
    # set density fitting
    if with_df:
        mf = mf.density_fit(auxbasis=auxbasis)
    
    # move to GPU if available
    if with_gpu:
        try:
            import cupy
            cupy.get_default_memory_pool().free_all_blocks()
            mf = mf.to_gpu()
        except ImportError:
            print("GPU support is not available. Proceeding with CPU.")
    
    # solvation model
    if with_solvent:
        solvent = solvent
        if solvent["method"].upper() in {"C-PCM", "IEF-PCM", "SS(V)PE", "COSMO"}:
            mf = mf.PCM()
            mf.with_solvent.lebedev_order = 29
            mf.with_solvent.method = solvent["method"]
            if "eps" in solvent:
                mf.with_solvent.eps = solvent["eps"]
            elif "solvent" in solvent:
                eps_dict = read_pcm_eps()
                assert solvent["solvent"].lower() in eps_dict, \
                    f"Solvent {solvent['solvent']} not found in predefined solvents"
                mf.with_solvent.eps = eps_dict[solvent["solvent"].lower()]
            else:
                raise ValueError("You must provide either 'eps' or 'solvent' for PCM model.")
        elif solvent["method"].upper() == "SMD":
            mf = mf.SMD()
            mf.with_solvent.lebedev_order = 29
            mf.with_solvent.method = "SMD"
            if "solvent_descriptors" in solvent:
                mf.with_solvent.solvent_descriptors = solvent["solvent_descriptors"]
            elif "solvent" in solvent:
                mf.with_solvent.solvent = solvent["solvent"]
            else:
                raise ValueError("You must provide either 'solvent_descriptors' or 'solvent' for SMD model.")
        else:
            raise ValueError(f"Solvation method {solvent['method']} not recognized.")
        
    mf.direct_scf_tol = float(direct_scf_tol)
    mf.chkfile = None
    mf.conv_tol = float(scf_conv_tol)
    mf.max_cycle = scf_max_cycle

    return mf


def build_3c_method(config: dict):
    """
    Special cases for 3c methods, e.g., B97-3c
    """
    xc = config.get("xc", "B97-3c")
    if not xc.endswith("3c"):
        raise ValueError("The xc functional must be a 3c method, e.g., B97-3c.")
    from gpu4pyscf.drivers.dft_3c_driver import parse_3c, gen_disp_fun
    
    # modify config dictionary
    pyscf_xc, nlc, basis, ecp, (xc_disp, disp), xc_gcp = parse_3c(xc.lower())
    config["xc"] = pyscf_xc
    config["nlc"] = nlc
    config["basis"] = basis
    config["ecp"] = ecp

    # build method
    mf = build_method(config)

    # attach 3c specific functions
    mf.get_dispersion = MethodType(gen_disp_fun(xc_disp, xc_gcp), mf)
    mf.do_disp = lambda: True

    return mf


def get_gradient_method(mf, xc_3c=None):
    """
    Get the gradient method from a mean field object.
    Args:
        mf (pyscf.dft.KS): The mean field object.
    Returns:
        grad (pyscf.grad.KS): The gradient method.
    """
    # 3c methods
    if xc_3c is not None:
        if not xc_3c.endswith("3c"):
            raise ValueError("The xc functional must be a 3c method, e.g., B97-3c.")
        from gpu4pyscf.drivers.dft_3c_driver import parse_3c, gen_disp_grad_fun
        _, _, _, _, (xc_disp, disp), xc_gcp = parse_3c(xc_3c.lower())
        g = mf.nuc_grad_method()
        g.get_dispersion = MethodType(gen_disp_grad_fun(xc_disp, xc_gcp), g)
        return g
    
    return mf.nuc_grad_method()


def get_Hessian_method(mf, xc_3c=None):
    """
    Get the Hessian method from a mean field object.
    Args:
        mf (pyscf.dft.KS): The mean field object.
    Returns:
        hess (pyscf.hessian.KS): The Hessian method.
    """
    # 3c methods
    if xc_3c is not None:
        if not xc_3c.endswith("3c"):
            raise ValueError("The xc functional must be a 3c method, e.g., B97-3c.")
        from gpu4pyscf.drivers.dft_3c_driver import parse_3c, gen_disp_hess_fun
        _, _, _, _, (xc_disp, disp), xc_gcp = parse_3c(xc_3c.lower())
        h = mf.Hessian()
        h.get_dispersion = MethodType(gen_disp_hess_fun(xc_disp, xc_gcp), h)
        h.auxbasis_response = 2
        return h
    
    h = mf.Hessian()
    h.auxbasis_response = 2
    return h


class PySCFCalculator(Calculator):
    """
    PySCF calculator for ASE.
    This calculator uses PySCF to compute the energy and forces of a system.
    It can be used with various mean field methods provided by PySCF.
    """
    implemented_properties = ["energy", "forces"]
    default_parameters = {}
    def __init__(self, method, xc_3c=None, **kwargs):
        self.method = method
        self.g_scanner: lib.GradScanner = get_gradient_method(self.method, xc_3c).as_scanner()
        Calculator.__init__(self, **kwargs)

    def set(self, **kwargs):
        changed_parameters = Calculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()

    def calculate(
        self,
        atoms: Atoms = None,
        properties=None, 
        system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties
        
        Calculator.calculate(self, atoms, properties, system_changes)
        
        mol: gto.Mole = self.method.mol
        positions = atoms.get_positions()
        atomic_numbers = atoms.get_atomic_numbers()
        Z = np.array([gto.charge(x) for x in mol.elements])
        if all(Z == atomic_numbers):
            _atoms = positions
        else:
            _atoms = list(zip(atomic_numbers, positions))
        
        mol.set_geom_(_atoms, unit="Angstrom")
        
        energy, gradients = self.g_scanner(mol)

        # store the energy and forces
        self.results["energy"] = energy * units.Hartree
        self.results["forces"] = -gradients * (units.Hartree / units.Bohr)


def hessian_function(atoms: Atoms, method: scf.hf.SCF, xc_3c=None)-> np.ndarray:
    """Calculate the Hessian matrix for the given atoms using the provided method."""
    method.mol.set_geom_(atoms.get_positions(), unit="Angstrom")
    method.run()
    hessian = get_Hessian_method(method, xc_3c=xc_3c).kernel()
    natom = method.mol.natm
    hessian = hessian.transpose(0, 2, 1, 3).reshape(3 * natom, 3 * natom)
    hessian *= (units.Hartree / units.Bohr**2)  # Convert from Hartree/Bohr^2
    return hessian


def optimize_geometry(
    atoms: Atoms,
    charge: int = 0,
    spin: int = 0,
    config: dict = None,
    outputfile: str = "mol_opt.xyz",
) -> dict:
    if config is None:
        config = {}
    # set symmetry tolerance (hardcoded in Angstrom)
    symm_geom_tol = config.get("symm_geom_tol", 0.05)  # Angstrom
    symm.geom.TOLERANCE = symm_geom_tol / units.Bohr

    # build method
    config["charge"] = charge
    config["spin"] = spin
    input_atoms_list = [(ele, coord) for ele, coord in zip(atoms.get_chemical_symbols(), atoms.get_positions())]
    config["inputfile"] = input_atoms_list
    if "xc" in config and config["xc"].endswith("3c"):
        xc_3c = config["xc"]
        mf = build_3c_method(config)
    else:
        xc_3c = None
        mf = build_method(config)
        
    # set calculator
    calc = PySCFCalculator(mf, xc_3c=xc_3c)
    atoms.calc = calc

    # parameters for Sella
    sella_opt = Sella(
        atoms=atoms,
        order=0,
        internal=config.get("internal", True),
        delta0=config.get("delta0", None),
        eta=float(config.get("eta", 1e-4)),
        gamma=float(config.get("gamma", 0.1)),
        eig=config.get("calc_hess", False),
        threepoint=True,
        diag_every_n=config.get("diag_every_n", None),
        hessian_function=lambda x: hessian_function(x, mf, xc_3c=xc_3c),
    )
    energy_criteria = float(config.get("energy", 1e-6 * units.Hartree))
    fmax_criteria = float(config.get("fmax", 4.5e-4 * units.Hartree / units.Bohr))
    frms_criteria = float(config.get("frms", 3.0e-4 * units.Hartree / units.Bohr))
    dmax_criteria = float(config.get("dmax", 1.8e-3))
    drms_criteria = float(config.get("drms", 1.2e-3))
    max_steps: int = config.get("max_steps", 1000)
    last_pos = atoms.get_positions().copy()
    last_energy = np.inf
    for i in sella_opt.irun(fmax=0, steps=max_steps):
        delta_pos = np.linalg.norm(atoms.get_positions() - last_pos, axis=1)
        delta_energy = abs(atoms.get_potential_energy() - last_energy)
        fmax = np.max(np.abs(atoms.get_forces()))
        frms = np.sqrt(np.mean(atoms.get_forces()**2))
        dmax = np.max(delta_pos)
        drms = np.sqrt(np.mean(delta_pos**2))
        if (delta_energy < energy_criteria and
            fmax < fmax_criteria and
            frms < frms_criteria and
            dmax < dmax_criteria and
            drms < drms_criteria):
            print("Optimization converged based on given criteria.")
            break
        last_pos = atoms.get_positions().copy()
        last_energy = atoms.get_potential_energy()
    else:
        Warning("Optimization did not converge within the maximum number of steps.")
        print(f"Final Energy Change   : {delta_energy:.6e} Eh")
        print(f"Final MAX force       : {fmax * units.Bohr / units.Hartree:.6e} Eh/Bohr")
        print(f"Final RMS force       : {frms * units.Bohr / units.Hartree:.6e} Eh/Bohr")
        print(f"Final MAX displacement: {dmax:.6e} Angstrom")
        print(f"Final RMS displacement: {drms:.6e} Angstrom")
    # save final structure
    ase.io.write(outputfile, atoms, format="xyz")
    print(f"Optimized geometry saved to {outputfile}")

    # single point
    mf.kernel()
    if not mf.converged:
        Warning("SCF calculation did not converge after optimization.")

    # frequency calculation
    numerical = config.get("numerical", False)
    if numerical:
        try:
            import finite_diff_gpu as finite_diff
        except ImportError:
            from pyscf.tools import finite_diff
        print("Running numerical Hessian calculation...")
        displacement = float(config.get("displacement", 1e-3))
        h = finite_diff.Hessian(get_gradient_method(mf, xc_3c=xc_3c))
        h.displacement = displacement
        hessian = h.kernel()
    else:
        print("Running analytical Hessian calculation...")
        h = get_Hessian_method(mf, xc_3c=xc_3c)
        h.auxbasis_response = 2
        hessian = h.kernel()
    

    # vibrational analysis
    freq_info = thermo.harmonic_analysis(mf.mol, hessian, imaginary_freq=False)
    # imaginary frequencies
    freq_au = freq_info["freq_au"]
    num_imag = np.sum(freq_au < 0)
    if num_imag > 0:
        print(f"Note: {num_imag} imaginary frequencies detected!")
    temp = config.get("temp", 298.15)
    press = config.get("press", 101325)
    thermo_info = thermo.thermo(mf, freq_au, temp, press)
    # log thermo info
    dump_normal_mode(mf.mol, freq_info)
    thermo.dump_thermo(mf.mol, thermo_info)

    return thermo_info


def run_single_point(
    atoms: Atoms,
    charge: int = 0,
    spin: int = 0,
    config: dict = None,
) -> float:
    # build method
    config["charge"] = charge
    config["spin"] = spin
    input_atoms_list = [(ele, coord) for ele, coord in zip(atoms.get_chemical_symbols(), atoms.get_positions())]
    config["inputfile"] = input_atoms_list
    if "xc" in config and config["xc"].endswith("3c"):
        mf = build_3c_method(config)
    else:
        mf = build_method(config)
    
    mf.mol.set_geom_(atoms.get_positions(), unit="Angstrom")

    e_tot = mf.kernel()
    if not mf.converged:
        Warning("SCF calculation did not converge.")
    e1 = mf.scf_summary.get("e1", 0.0)
    e_coul = mf.scf_summary.get("coul", 0.0)
    e_xc = mf.scf_summary.get("exc", 0.0)
    e_disp = mf.scf_summary.get("dispersion", 0.0)
    e_solvent = mf.scf_summary.get("e_solvent", 0.0)

    # log results
    print(f"Total Energy        [Eh]: {e_tot:16.10f}")
    print(f"One-electron Energy [Eh]: {e1:16.10f}")
    print(f"Coulomb Energy      [Eh]: {e_coul:16.10f}")
    print(f"XC Energy           [Eh]: {e_xc:16.10f}")
    if abs(e_disp) > 1e-10:
        print(f"Dispersion Energy   [Eh]: {e_disp:16.10f}")
    if abs(e_solvent) > 1e-10:
        print(f"Solvent Energy      [Eh]: {e_solvent:16.10f}")

    return e_tot
