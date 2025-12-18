import os
import shutil
import subprocess
import tempfile

import ase.io


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
