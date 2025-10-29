import os
import shutil
import argparse
import subprocess
import tempfile

import yaml
import ase.io
from ase.units import Hartree
from rdkit import Chem
from rdkit.Chem import AllChem

from run_pyscf import optimize_geometry, run_single_point


def read_input(input_str: str) -> str:
    if input_str.endswith(".xyz"):
        return input_str
    print(f"Converting SMILES to XYZ for {input_str}...")
    mol = Chem.MolFromSmiles(input_str)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    mol = Chem.AddHs(mol)
    # to xyz
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    xyzfile = f"{input_str}.xyz"
    Chem.MolToXYZFile(mol, xyzfile)
    print(f"Wrote XYZ file to {xyzfile}")
    return xyzfile


def run_crest(
    xyzfile: str,
    charge: int = 0,
    uhf: int = 0,
    methods: str = "gfn2",
    solvation: str = None,
    solvent: str = None,
    threads: int = 4,
) -> ase.Atoms:
    # create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # copy the xyz file to the temp directory
        temp_xyz = os.path.join(tmpdir, "mol.xyz")
        shutil.copy(xyzfile, temp_xyz)
        # build the crest command
        cmd = [
            "crest",
            temp_xyz,
            f"--{methods.lower()}",
            "-c", str(charge),
            "-u", str(uhf),
            "-T", str(threads),
        ]
        if solvation and solvent:
            cmd.extend([
                f"--{solvation.lower()}", str(solvent.lower()),
            ])
        # run the crest command
        print(f"Running CREST for {xyzfile}...")
        # mute the output
        subprocess.run(cmd, cwd=tmpdir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # load the results
        atoms = ase.io.read(os.path.join(tmpdir, "crest_best.xyz"))
        # copy the best structure back to current directory
        name = xyzfile.split("/")[-1].split(".xyz")[0]
        shutil.copy(os.path.join(tmpdir, "crest_best.xyz"), f"{name}_crest.xyz")
        print(f"Wrote CREST optimized structure to {name}_crest.xyz")
    return atoms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="path to config file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config: dict = yaml.safe_load(f)

    # step1: read input molecuoles
    ox_input_config: dict = config.get("oxidized", {})
    red_input_config: dict = config.get("reduced", {})
    # convert smiles to xyz if needed
    ox_input = read_input(ox_input_config["input"])
    red_input = read_input(red_input_config["input"])
    # read the charge and multiplicity (here, spin = multiplicity - 1)
    ox_charge = ox_input_config.get("charge", 0)
    ox_spin = ox_input_config.get("multiplicity", 1) - 1
    red_charge = red_input_config.get("charge", 0)
    red_spin = red_input_config.get("multiplicity", 1) - 1

    # step2: use CREST to generate lowest energy conformer
    crest_config: dict = config.get("crest", {})
    methods = crest_config.get("methods", "gfn2")
    solvation = crest_config.get("solvation", None)
    solvent = crest_config.get("solvent", None)
    threads = crest_config.get("threads", 4)
    ox_atoms = run_crest(
        ox_input,
        charge=ox_charge,
        uhf=ox_spin,
        methods=methods,
        solvation=solvation,
        solvent=solvent,
        threads=threads,
    )
    red_atoms = run_crest(
        red_input,
        charge=red_charge,
        uhf=red_spin,
        methods=methods,
        solvation=solvation,
        solvent=solvent,
        threads=threads,
    )

    # step3: use MACE or DFT to do optimization and frequency calculation
    opt_config = config.get("optimization", {})
    opt_config["inputfile"] = ox_input.split('.')[0] + '_crest.xyz'
    ox_thermo_info = optimize_geometry(
        ox_atoms,
        charge=ox_charge,
        spin=ox_spin,
        config=opt_config,
        outputfile=f"{ox_input.split('.')[0]}_opt.xyz",
    )
    opt_config["inputfile"] = red_input.split('.')[0] + '_crest.xyz'
    red_thermo_info = optimize_geometry(
        red_atoms,
        charge=red_charge,
        spin=red_spin,
        config=opt_config,
        outputfile=f"{red_input.split('.')[0]}_opt.xyz",
    )

    # step4: single point energy calculation
    sp_config = config.get("single_point", None)
    if sp_config is not None:
        sp_config["inputfile"] = ox_input.split('.')[0] + '_opt.xyz'
        ox_sp = run_single_point(
            ox_atoms,
            charge=ox_charge,
            spin=ox_spin,
            config=sp_config,
        )
        sp_config["inputfile"] = red_input.split('.')[0] + '_opt.xyz'
        red_sp = run_single_point(
            red_atoms,
            charge=red_charge,
            spin=red_spin,
            config=sp_config,
        )
    else:  # use the energies from optimization step
        ox_sp = ox_thermo_info["E_elec"][0]
        red_sp = red_thermo_info["E_elec"][0]
    
    # step5: calculate SMD solvation energy
    solvent = config.get("solvent", "water")
    gas_config = {
        "xc": "M062X",
        "basis": "6-31Gs",
        "newton": True,
    }
    gas_config["inputfile"] = ox_input.split('.')[0] + '_opt.xyz'
    ox_gas_energy = run_single_point(
        ox_atoms,
        charge=ox_charge,
        spin=ox_spin,
        config=gas_config,
    )
    gas_config["inputfile"] = red_input.split('.')[0] + '_opt.xyz'
    red_gas_energy = run_single_point(
        red_atoms,
        charge=red_charge,
        spin=red_spin,
        config=gas_config,
    )
    smd_config = gas_config.copy()
    smd_config.update({
        "with_solvent": True,
        "solvent": {"method": "SMD", "solvent": solvent},
    })
    smd_config["inputfile"] = ox_input.split('.')[0] + '_opt.xyz'
    ox_smd_energy = run_single_point(
        ox_atoms,
        charge=ox_charge,
        spin=ox_spin,
        config=smd_config,
    )
    smd_config["inputfile"] = red_input.split('.')[0] + '_opt.xyz'
    red_smd_energy = run_single_point(
        red_atoms,
        charge=red_charge,
        spin=red_spin,
        config=smd_config,
    )

    ox_G_solv = ox_smd_energy - ox_gas_energy
    red_G_solv = red_smd_energy - red_gas_energy

    E_ref = config.get("E_ref", 4.44)  # default to SHE
    n_electrons = config["n_electrons"]
    n_protons = config["n_protons"]
    ox_delta_G = (ox_thermo_info["G_tot"][0] - ox_thermo_info["E_elec"][0])
    red_delta_G = (red_thermo_info["G_tot"][0] - red_thermo_info["E_elec"][0])
    print("\n=== Final Results ===")
    print(f"Single Point Energy (Ox)           [eV]: {ox_sp * Hartree:.6f}")
    print(f"Single Point Energy (Red)          [eV]: {red_sp * Hartree:.6f}")
    print(f"Gibbs Free Energy correction (Ox)  [eV]: {ox_delta_G * Hartree:.6f}")
    print(f"Gibbs Free Energy correction (Red) [eV]: {red_delta_G * Hartree:.6f}")
    print(f"Solvation Free Energy (Ox)         [eV]: {ox_G_solv * Hartree:.6f}")
    print(f"Solvation Free Energy (Red)        [eV]: {red_G_solv * Hartree:.6f}")
    delta_G_redox = (
        (red_sp + red_delta_G + red_G_solv)
        - (ox_sp + ox_delta_G + ox_G_solv)
    ) * Hartree + n_protons * 11.4473
    E_redox = -delta_G_redox / n_electrons - E_ref
    print(f"Calculated Free Energy Change      [eV]: {delta_G_redox:.6f}")
    print(f"Calculated Redox Potential         [V]: {E_redox:.6f}")


if __name__ == "__main__":
    main()
