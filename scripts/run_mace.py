import argparse
import time
from types import SimpleNamespace

import numpy as np
import ase.io
import yaml
import h5py
from mace.calculators import mace_omol
from ase import units
from sella import Sella, IRC, Constraints
from pyscf import gto, symm
from pyscf.hessian import thermo

from utils import dump_normal_mode


def main():
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
    
    # load MACE model
    atoms = ase.io.read(inputfile)
    charge = config.get("charge", 0)
    spin = config.get("spin", 0)
    atoms.info["charge"] = charge
    atoms.info["spin"] = spin + 1  # Convert PySCF's 2S (args.spin) to ASE's 2S+1 by adding 1
    atoms.calc = mace_omol(model=config.get("model", "default"))
    n_atoms = len(atoms)

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
            order=order,
            internal=opt_config.get("internal", True),
            constraints=cons,
            constraints_tol=opt_config.get("constraints_tol", 1e-5),
            eig=eig,
            threepoint=True,
            diag_every_n=opt_config.get("diag_every_n", None),
            hessian_function=lambda x: x.calc.get_hessian().reshape(n_atoms * 3, n_atoms * 3)
        )
        fmax: float = opt_config.get("fmax", 4.5e-4)
        max_steps = opt_config.get("max_steps", 1000)
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
    energy = atoms.get_potential_energy()
    end_time = time.time()
    print(f"Energy prediction completed in {end_time - start_time:.2f} seconds.")
    print(f"MACE energy  [eV]: {energy:16.10f}")
    print(f"MACE energy  [Eh]: {energy / units.Hartree:16.10f}")

    # task 3: forces (gradients)
    run_forces: bool = config.get("forces", False)
    if run_forces:
        start_time = time.time()
        forces = atoms.get_forces()
        end_time = time.time()
        print(f"Forces prediction completed in {end_time - start_time:.2f} seconds.")
        print("MACE forces [eV/Ang]:")
        elements = atoms.get_chemical_symbols()
        for i, (ele, force) in enumerate(zip(elements, forces)):
            print(f"{i+1:3d} {ele:2s} {force[0]:12.6f} {force[1]:12.6f} {force[2]:12.6f}")
        save_forces: bool = config.get("save_forces", False)
        if save_forces:
            with h5py.File(datafile, 'a') as h5f:
                h5f.create_dataset("forces", data=forces)
                h5f.create_dataset("forces_unit", data="eV/Ang")

    # task 4: vibrational frequency analysis
    run_freq: bool = config.get("freq", False)
    if run_freq:
        start_time = time.time()
        hessian = atoms.calc.get_hessian().reshape(n_atoms * 3, n_atoms * 3)
        end_time = time.time()
        print(f"Hessian prediction completed in {end_time - start_time:.2f} seconds.")
        save_hess: bool = config.get("save_hess", False)
        if save_hess:
            with h5py.File(datafile, 'a') as h5f:
                h5f.create_dataset("hessian", data=hessian)
                h5f.create_dataset("hessian_unit", data="eV/Ang^2")
        
        # convert hessian to Hartree/Bohr^2
        _hessian = hessian.reshape(n_atoms, 3, n_atoms, 3).transpose(0, 2, 1, 3)
        _hessian *= (units.Bohr**2 / units.Hartree)  # Convert from eV/Ang^2 to Hartree/Bohr^2

        # create a temporary Mole()
        start_time = time.time()
        mol = gto.M(
            atom=[(ele, coord) for ele, coord in zip(atoms.get_chemical_symbols(), atoms.get_positions())],
            charge=charge,
            spin=spin,
        )
        freq_info = thermo.harmonic_analysis(mol, _hessian, imaginary_freq=False)
        # imaginary frequencies
        freq_au = freq_info["freq_au"]
        num_imag = np.sum(freq_au < 0)
        if num_imag > 0:
            print(f"Note: {num_imag} imaginary frequencies detected!")
        dummy_mf = SimpleNamespace(mol=mol, e_tot=energy / units.Hartree)
        temp = config.get("temp", 298.15)
        press = config.get("press", 1.0)
        thermo_info = thermo.thermo(dummy_mf, freq_au, temp, press)
        end_time = time.time()
        print(f"Vibrational frequency analysis completed in {end_time - start_time:.2f} seconds.")
        # log thermo info
        dump_normal_mode(mol, freq_info)
        thermo.dump_thermo(mol, thermo_info)
        # save frequencies and normal modes
        save_freq: bool = config.get("save_freq", False)
        if save_freq:
            with h5py.File(datafile, 'a') as h5f:
                h5f.create_dataset("freq_wavenumber", data=freq_info["freq_wavenumber"])
                h5f.create_dataset("freq_wavenumber_unit", data="cm^-1")
                h5f.create_dataset("normal_modes", data=freq_info["normal_modes"])

    # task 5: IRC
    run_irc: bool = config.get("irc", False)
    if run_irc:
        start_time = time.time()
        irc_config: dict = config.get("irc_config", {})
        sella_irc = IRC(
            atoms=atoms,
            trajectory=irc_config.get("irc_trajectory", f"{filename}_irc.traj"),
            ninner_iter=irc_config.get("ninner_iter", 10),
            peskwargs={"threepoint": True},
            hessian_function=lambda x: x.calc.get_hessian().reshape(n_atoms * 3, n_atoms * 3),
            keep_going=irc_config.get("keep_going", False),
        )
        # forward direction
        fmax: float = irc_config.get("fmax", 4.5e-4)
        max_steps = irc_config.get("max_steps", 100)
        direction: str = irc_config.get("direction", "both")
        assert direction in ["forward", "reverse", "both"], "Invalid IRC direction. Choose from 'forward', 'reverse', or 'both'."
        # forward direction
        if direction in ["forward", "both"]:
            irc_converged = sella_irc.run(fmax=fmax, steps=max_steps, direction="forward")
            if not irc_converged:
                Warning("Forward IRC did not converge within the maximum number of steps.")
            ase.io.write(f"{filename}_irc_forward.xyz", sella_irc.atoms, format="xyz")
        
        # reverse direction
        if direction in ["reverse", "both"]:
            irc_converged = sella_irc.run(fmax=fmax, steps=max_steps, direction="reverse")
            if not irc_converged:
                Warning("Reverse IRC did not converge within the maximum number of steps.")
            ase.io.write(f"{filename}_irc_reverse.xyz", sella_irc.atoms, format="xyz")
        
        end_time = time.time()
        print(f"IRC calculation completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()