import os
from types import MethodType
from typing import Dict

import numpy as np
from pyscf import gto, lib, scf, dft


def write_xyz(mol: gto.Mole, filename: str) -> None:
    elements = mol.elements
    coords = mol.atom_coords(unit="Angstrom")
    with open(filename, "w") as f:
        f.write(f"{len(elements)}\n")
        # Convert spin (2S) to multiplicity (2S+1) format, as required by quantum chemistry conventions.
        f.write(f"{mol.charge} {mol.spin + 1}\n")
        for ele, coord in zip(elements, coords):
            f.write(f"{ele: <2}    {coord[0]:10.6f}    {coord[1]:10.6f}    {coord[2]:10.6f}\n")


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
    verbose = config.get("verbose", 4)
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