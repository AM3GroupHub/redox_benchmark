import os
from types import SimpleNamespace

import numpy as np
import ase.io
from ase import Atoms, units
from sella import Sella
from pyscf import gto, symm
from pyscf.hessian import thermo
try:
    import torch
except ImportError:
    print("PyTorch not found. MLIP calculations will not work.")

from redox.utils.pyscf_utils import dump_normal_mode


def optimize_geometry(
    atoms: Atoms,
    charge: int,
    multiplicity: int,
    config: dict,
    outputfile: str = "opt.xyz",
) -> dict:
    """Optimize geometry and calculate frequencies."""
    if config is None:
        config = {}
    # set symmetry tolerance (hardcoded in Angstrom)
    symm_geom_tol = config.get("symm_geom_tol", 0.05)  # Angstrom
    symm.geom.TOLERANCE = symm_geom_tol / units.Bohr

    atoms.info["charge"] = charge
    atoms.info["spin"] = multiplicity

    mlip: str = config.get("mlip", "mace")
    device: str = config.get("device", None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"No device specified. Using device: {device}")
    precision: str = config.get("precision", "float64")
    task_name: str = config.get("task_name", "omol")
    
    if mlip.lower() == "mace":
        from mace.calculators import mace_omol
        mace_calc = mace_omol(
            model=config["model"],
            device=device,
            default_dtype=precision,
        )
        atoms.calc = mace_calc
        hessian_function = lambda x: mace_calc.get_hessian(x).reshape(len(x) * 3, len(x) * 3)
    elif mlip.lower() == "uma":
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        from fairchem.core.datasets import data_list_collater
        from omegaconf import OmegaConf
        model: str = config["model"]
        atom_refs = OmegaConf.load(os.path.join(model.rsplit('/', 1)[0], "iso_atom_elem_refs.yaml"))
        predictor = pretrained_mlip.load_predict_unit(model, device=device, atom_refs=atom_refs)
        # predictor = pretrained_mlip.get_predict_unit(model, device=device, cache_dir="/home/users/nus/zhongpc/scratch/models/uma")
        uma_calc = FAIRChemCalculator(predictor, task_name=task_name)
        atoms.calc = uma_calc
        batch_size = config.get("batch_size", 128)
        finite_diff_eps = config.get("finite_diff_eps", 5e-3)
        def uma_hess_function(atoms: ase.Atoms):
            eps = finite_diff_eps
            data_list = []
            for i in range(len(atoms)):
                for j in range(3):
                    displaced_plus = atoms.copy()
                    displaced_minus = atoms.copy()
                    displaced_plus.positions[i, j] += eps
                    displaced_minus.positions[i, j] -= eps
                    data_plus = uma_calc.a2g(displaced_plus)
                    data_minus = uma_calc.a2g(displaced_minus)
                    data_list.extend([data_plus, data_minus])
            # batch and predict
            forces_list = []
            for i in range(0, len(data_list), batch_size):
                if i + batch_size > len(data_list):
                    data_list_batch = data_list[i:]
                else:
                    data_list_batch = data_list[i:i+batch_size]
                batch = data_list_collater(data_list_batch, otf_graph=True)
                pred = predictor.predict(batch)
                batch_forces = pred["forces"].detach()
                forces_list.append(batch_forces)
            forces = torch.cat(forces_list, dim=0).reshape(-1, len(atoms), 3)
            # calculated hessian using finite differences
            hessian = np.zeros((len(atoms) * 3, len(atoms) * 3))
            for i in range(len(atoms)):
                for j in range(3):
                    idx = i * 3 + j
                    forces_plus = forces[2 * idx].flatten().cpu().numpy()
                    forces_minus = forces[2 * idx + 1].flatten().cpu().numpy()
                    hessian[:, idx] = (forces_minus - forces_plus) / (2 * eps) # forces is the negative graidents
            return hessian

        hessian_function = uma_hess_function
    else:
        raise ValueError(f"Unsupported MLIP model: {mlip}")

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
        hessian_function=hessian_function,
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

    energy = atoms.get_potential_energy()
    n_atoms = len(atoms)
    hessian = hessian_function(atoms)
    # convert hessian to Hartree/Bohr^2
    _hessian = hessian.reshape(n_atoms, 3, n_atoms, 3).transpose(0, 2, 1, 3)
    _hessian *= (units.Bohr**2 / units.Hartree)  # Convert from eV/Ang^2 to Hartree/Bohr^2
    mol = gto.M(
        atom=[(ele, coord) for ele, coord in zip(atoms.get_chemical_symbols(), atoms.get_positions())],
        charge=charge,
        spin=multiplicity - 1,
    )
    freq_info = thermo.harmonic_analysis(mol, _hessian, imaginary_freq=False)
    # imaginary frequencies
    freq_au = freq_info["freq_au"]
    num_imag = np.sum(freq_au < 0)
    if num_imag > 0:
        print(f"Note: {num_imag} imaginary frequencies detected!")
    dummy_mf = SimpleNamespace(mol=mol, e_tot=energy / units.Hartree)
    temp = config.get("temp", 298.15)
    press = config.get("press", 101325)
    thermo_info = thermo.thermo(dummy_mf, freq_au, temp, press)
    # log thermo info
    dump_normal_mode(mol, freq_info)
    thermo.dump_thermo(mol, thermo_info)

    return thermo_info


def run_single_point(
    atoms: Atoms,
    charge: int,
    multiplicity: int,
    config: dict,
) -> float:
    """Run single point energy calculation."""
    if config is None:
        config = {}
    atoms.info["charge"] = charge
    atoms.info["spin"] = multiplicity

    mlip: str = config.get("mlip", "mace")
    device: str = config.get("device", None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"No device specified. Using device: {device}")
    precision: str = config.get("precision", "float64")

    if mlip.lower() == "mace":
        from mace.calculators import mace_omol
        mace_calc = mace_omol(
            model=config["model"],
            device=device,
            default_dtype=precision,
        )
        atoms.calc = mace_calc
    elif mlip.lower() == "uma":
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        from omegaconf import OmegaConf
        model: str = config["model"]
        atom_refs = OmegaConf.load(os.path.join(model.rsplit('/', 1)[0], "iso_atom_elem_refs.yaml"))
        predictor = pretrained_mlip.load_predict_unit(model, device=device, atom_refs=atom_refs)
        # predictor = pretrained_mlip.get_predict_unit(model, device=device, cache_dir="/home/users/nus/zhongpc/scratch/models/uma")
        uma_calc = FAIRChemCalculator(predictor, task_name="omol")
        atoms.calc = uma_calc
    else:
        raise ValueError(f"Unsupported MLIP model: {mlip}")

    energy = atoms.get_potential_energy()
    return energy