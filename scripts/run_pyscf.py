import argparse
import time

import numpy as np
import ase.io
import yaml
import h5py
from pyscf import symm, scf
from pyscf.hessian import thermo
from ase import Atoms, units
from sella import Sella, IRC, Constraints

from utils import build_method, build_3c_method, dump_normal_mode, get_gradient_method, get_Hessian_method
from ase_interface import PySCFCalculator


CACHED_POSITION = None
CACHED_HESSIAN = None

def hessian_function(atoms: Atoms, method: scf.hf.SCF, xc_3c=None)-> np.ndarray:
    """Calculate the Hessian matrix for the given atoms using the provided method."""
    if CACHED_POSITION is not None and np.allclose(atoms.get_positions(), CACHED_POSITION):
        return CACHED_HESSIAN
    method.mol.set_geom_(atoms.get_positions(), unit="Angstrom")
    method.run()
    hessian = get_Hessian_method(method, xc_3c=xc_3c).kernel()
    natom = method.mol.natm
    hessian = hessian.transpose(0, 2, 1, 3).reshape(3 * natom, 3 * natom)
    hessian *= (units.Hartree / units.Bohr**2)  # Convert from Hartree/Bohr^2
    return hessian


def main():
    global CACHED_POSITION, CACHED_HESSIAN
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="pyscf_config.yaml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config: dict = yaml.safe_load(f)

    # setup files
    inputfile: str = config.get("inputfile", "mol.xyz")
    filename = inputfile.rsplit('.', 1)[0]
    datafile: str = config.get("datafile", f"{filename}_data.h5")
    # empty datafile if save anything
    for key in config:
        if isinstance(key, str) and key.startswith("save_") and config[key]:
            h5py.File(datafile, 'w').close()
            break

    # set symmetry tolerance (hardcoded in Angstrom)
    if "symm_geom_tol" in config:
        symm.geom.TOLERANCE = config["symm_geom_tol"] / units.Bohr

    # build method
    if "xc" in config and config["xc"].endswith("3c"):
        xc_3c = config["xc"]
        mf = build_3c_method(config)
    else:
        xc_3c = None
        mf = build_method(config)
    
    use_newton = config.get("newton", False)
    if use_newton:
        mf = mf.newton()
    
    # set calculator
    calc = PySCFCalculator(mf, xc_3c=xc_3c)
    atoms = ase.io.read(config["inputfile"])
    atoms.calc = calc

    # task 1: optimization
    run_opt = config.get("opt", False)
    if run_opt:
        # record start time
        start_time = time.time()
        # parameters for Sella
        opt_config: dict = config.get("opt_config", {})
        optts: bool = opt_config.get("ts", False)
        if optts:
            eig = opt_config.get("calc_hessian", True)
            order = 1
        else:
            eig = opt_config.get("calc_hessian", False)
            order = 0
        # constraints
        if "constraints" in opt_config:
            cons = Constraints(atoms)
            cons_dict: dict = opt_config["constraints"]
            # translation
            if "fix_translation" in cons_dict:
                for atom_idx in cons_dict["fix_translation"]:
                    cons.fix_translation(atom_idx)
                print(f"Applied translation constraints: {cons_dict['fix_translation']}")
            # bond
            if "fix_bond" in cons_dict:
                for bond in cons_dict["fix_bond"]:
                    cons.fix_bond((bond[0], bond[1]))
                print(f"Applied bond constraints: {cons_dict['fix_bond']}")
            # angle
            if "fix_angle" in cons_dict:
                for angle in cons_dict["fix_angle"]:
                    cons.fix_angle((angle[0], angle[1], angle[2]))
                print(f"Applied angle constraints: {cons_dict['fix_angle']}")
            # dihedral
            if "fix_dihedral" in cons_dict:
                for dihedral in cons_dict["fix_dihedral"]:
                    cons.fix_dihedral((dihedral[0], dihedral[1], dihedral[2], dihedral[3]))
                print(f"Applied dihedral constraints: {cons_dict['fix_dihedral']}")
        else:
            cons = None
        sella_opt = Sella(
            atoms=atoms,
            trajectory=opt_config.get("trajectory", f"{filename}_opt.traj"),
            order=order,  # 0 for minimum, 1 for saddle point
            internal=opt_config.get("internal", True),
            constraints=cons,
            constraints_tol=opt_config.get("constraints_tol", 1e-5),
            delta0=opt_config.get("delta0", None),
            eta=float(opt_config.get("eta", 1e-4)),
            gamma=float(opt_config.get("gamma", 0.1)),
            eig=eig,
            threepoint=True,
            diag_every_n=opt_config.get("diag_every_n", None),
            hessian_function=lambda x: hessian_function(x, mf, xc_3c=xc_3c),
        )
        fmax: float = opt_config.get("fmax", 4.5e-4)
        max_steps: int = opt_config.get("max_steps", 1000)
        opt_converged = sella_opt.run(fmax=fmax, steps=max_steps)
        if not opt_converged:
            Warning("Optimization did not converge within the maximum number of steps.")
        opt_outputfile = opt_config.get("outputfile", f"{filename}_opt.xyz")
        ase.io.write(opt_outputfile, atoms, format="xyz")
        # record end time
        end_time = time.time()
        print(f"Optimization completed in {end_time - start_time:.2f} seconds.")

    # task 2: single point energy
    start_time = time.time()
    e_tot = mf.kernel()
    if not mf.converged:
        Warning("SCF calculation did not converge.")
    e1 = mf.scf_summary.get("e1", 0.0)
    e_coul = mf.scf_summary.get("coul", 0.0)
    e_xc = mf.scf_summary.get("exc", 0.0)
    e_disp = mf.scf_summary.get("dispersion", 0.0)
    e_solvent = mf.scf_summary.get("e_solvent", 0.0)
    end_time = time.time()

    # log results
    print(f"Single point energy calculation completed in {end_time - start_time:.2f} seconds.")
    print(f"Total Energy        [Eh]: {e_tot:16.10f}")
    print(f"One-electron Energy [Eh]: {e1:16.10f}")
    print(f"Coulomb Energy      [Eh]: {e_coul:16.10f}")
    print(f"XC Energy           [Eh]: {e_xc:16.10f}")
    if abs(e_disp) > 1e-10:
        print(f"Dispersion Energy   [Eh]: {e_disp:16.10f}")
    if abs(e_solvent) > 1e-10:
        print(f"Solvent Energy      [Eh]: {e_solvent:16.10f}")

    # properties about the wavefunction
    dm = mf.make_rdm1()
    if not isinstance(dm, np.ndarray):  # which means dm is cupy.array
        dm = dm.get()  # convert cupy array to numpy array
    mo_energy = mf.mo_energy
    if not isinstance(mo_energy, np.ndarray):
        mo_energy = mo_energy.get()
    if dm.ndim == 3:  # open-shell case
        mo_energy[0].sort()
        mo_energy[1].sort()
        na, nb = mf.nelec
        print(f"LUMO Alpha [Eh]: {mo_energy[0][na]:12.6f}")
        print(f"LUMO Beta  [Eh]: {mo_energy[1][nb]:12.6f}")
        print(f"HOMO Alpha [Eh]: {mo_energy[0][na-1]:12.6f}")
        print(f"HOMO Beta  [Eh]: {mo_energy[1][nb-1]:12.6f}")
    else:  # closed-shell case
        mo_energy.sort()
        nocc = mf.mol.nelectron // 2
        print(f"LUMO [Eh]: {mo_energy[nocc]:12.6f}")
        print(f"HOMO [Eh]: {mo_energy[nocc-1]:12.6f}")
    

    # (optional) save density
    save_dm: bool = config.get("save_dm", False)
    if save_dm:
        with h5py.File(datafile, 'w') as h5f:
            h5f.create_dataset("dm", data=dm)
    save_density: bool = config.get("save_density", False)
    if save_density:
        weights = mf.grids.weights
        coords = mf.grids.coords
        dm0 = dm[0] + dm[1] if dm.ndim == 3 else dm
        rho = mf._numint.get_rho(mf.mol, dm0, mf.grids)
        if not isinstance(weights, np.ndarray):
            weights = weights.get()
        if not isinstance(coords, np.ndarray):
            coords = coords.get()
        if not isinstance(rho, np.ndarray):
            rho = rho.get()
        with h5py.File(datafile, 'a') as h5f:
            h5f.create_dataset("grids_weights", data=weights)
            h5f.create_dataset("grids_coords", data=coords)
            h5f.create_dataset("grids_rho", data=rho)
    
    # if not converged, skip the following tasks
    if not mf.converged:
        return

    # task 3: forces (gradients)
    run_forces: bool = config.get("forces", False)
    if run_forces:
        start_time = time.time()
        g = get_gradient_method(mf, xc_3c=xc_3c)
        if "with_df" in config and config["with_df"]:
            g.auxbasis_response = True
        forces = -g.kernel()
        end_time = time.time()
        print(f"Force calculation completed in {end_time - start_time:.2f} seconds.")
        print(f"Forces  [Eh/Bohr]:")
        elements = mf.mol.elements
        for i, (ele, force) in enumerate(zip(elements, forces)):
            print(f"{i+1:3d} {ele:2s} {force[0]:12.6f} {force[1]:12.6f} {force[2]:12.6f}")
        save_forces = config.get("save_forces", False)
        if save_forces:
            with h5py.File(datafile, 'a') as h5f:
                h5f.create_dataset("forces", data=forces)
                h5f.create_dataset("forces_unit", data="Eh/Bohr")
    
    # task 4: vibrational frequency analysis
    run_freq = config.get("freq", False)
    freq_config = config.get("freq_config", {})
    if run_freq:
        # calculation Hessian matrix
        start_time = time.time()
        numerical = freq_config.get("numerical", False)
        if numerical:
            try:
                import finite_diff_gpu as finite_diff
            except ImportError:
                from pyscf.tools import finite_diff
            displacement = float(freq_config.get("displacement", 1e-3))
            h = finite_diff.Hessian(get_gradient_method(mf, xc_3c=xc_3c))
            h.displacement = displacement
            hessian = h.kernel()
        else:
            h = get_Hessian_method(mf, xc_3c=xc_3c)
            h.auxbasis_response = 2
            hessian = h.kernel()
        end_time = time.time()
        print(f"Hessian calculation completed in {end_time - start_time:.2f} seconds.")
        
        CACHED_POSITION = mf.mol.atom_coords(unit="Angstrom").copy()
        _hessian = hessian.transpose(0, 2, 1, 3).reshape(3 * mf.mol.natm, 3 * mf.mol.natm)
        CACHED_HESSIAN = _hessian * (units.Hartree / units.Bohr**2)  # Convert from Hartree/Bohr^2
        # (optional) save Hessian matrix (a.u.)
        save_hess: bool = config.get("save_hess", False)
        if save_hess:
            with h5py.File(datafile, 'a') as h5f:
                h5f.create_dataset("hessian", data=hessian)
                h5f.create_dataset("hessian_unit", data="Eh/Bohr^2")

        # vibrational analysis
        start_time = time.time()
        freq_info = thermo.harmonic_analysis(mf.mol, hessian, imaginary_freq=False)
        # imaginary frequencies
        freq_au = freq_info["freq_au"]
        num_imag = np.sum(freq_au < 0)
        if num_imag > 0:
            print(f"Note: {num_imag} imaginary frequencies detected!")
        temp = freq_config.get("temp", 298.15)
        press = freq_config.get("press", 1.0)
        thermo_info = thermo.thermo(mf, freq_au, temp, press)
        end_time = time.time()
        print(f"Vibrational frequency analysis completed in {end_time - start_time:.2f} seconds.")
        # log thermo info
        dump_normal_mode(mf.mol, freq_info)
        thermo.dump_thermo(mf.mol, thermo_info)
        # save frequencies and normal modes
        save_freq: bool = config.get("save_freq", False)
        if save_freq:
            with h5py.File(datafile, 'a') as h5f:
                h5f.create_dataset("freq_wavenumber", data=freq_info["freq_wavenumber"])
                h5f.create_dataset("freq_wavenumber_unit", data="cm^-1")
                h5f.create_dataset("norm_mode", data=freq_info["norm_mode"])
    
    # task 5: IRC
    run_irc = config.get("irc", False)
    if run_irc:
        start_time = time.time()
        irc_config: dict = config.get("irc_config", {})
        sella_irc = IRC(
            atoms=atoms,
            trajectory=irc_config.get("irc_trajectory", f"{filename}_irc.traj"),
            ninner_iter=irc_config.get("ninner_iter", 10),
            eta=float(irc_config.get("eta", 1e-4)),
            peskwargs={"threepoint": True},
            hessian_function=lambda x: hessian_function(x, mf, xc_3c=xc_3c),
            keep_going=irc_config.get("keep_going", False),
        )
        fmax: float = float(irc_config.get("fmax", 4.5e-4))
        irc_steps: int = irc_config.get("irc_steps", 10)
        direction: str = irc_config.get("direction", "both")
        assert direction in ["forward", "reverse", "both"], "Invalid IRC direction. Choose from 'forward', 'reverse', or 'both'."
        # forward direction
        if direction in ["forward", "both"]:
            irc_converged = sella_irc.run(fmax=fmax, steps=irc_steps, direction="forward")
            if not irc_converged:
                Warning("Forward IRC did not converge within the maximum number of steps.")
            ase.io.write(f"{filename}_irc_forward.xyz", sella_irc.atoms, format="xyz")
        
        # reverse direction
        if direction in ["reverse", "both"]:
            irc_converged = sella_irc.run(fmax=fmax, steps=irc_steps, direction="reverse")
            if not irc_converged:
                Warning("Reverse IRC did not converge within the maximum number of steps.")
            ase.io.write(f"{filename}_irc_reverse.xyz", sella_irc.atoms, format="xyz")
        
        end_time = time.time()
        print(f"IRC calculation completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
