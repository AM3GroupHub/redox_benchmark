# redox_benchmark
A repo for benchmark test of the redox potential of molecules

## Prerequisites
- **CUDA 12.x**: Required for gpu4pyscf GPU acceleration
- **CREST**: Required for conformational sampling (See [CREST](https://crest-lab.github.io/crest-docs/))
- **Python**: ≥ 3.9
- **Git**: For cloning the repository

## Install
```shell
git clone https://github.com/AM3GroupHub/redox_benchmark.git
cd redox_benchmark
```
Basic Installation
```shell
pip install -e .
```
Install MACE (for MACE-OMol-0)
```shell
pip install -e .[mace]
```
Install FairChem (for UMA)
```shell
pip install -e .[fairchem]
```
If your CUDA version is not 12, you can manually install [GPU4PySCF](https://github.com/pyscf/gpu4pyscf) CUDA 11 or 13 version.


## Usage
```bash
calc_redox --config config.yaml
```

Config file example for using DFT
```yaml
oxidized:                  # input setting for oxidized species
  input: O=C1C=CC(=O)C=C1  # smiles string or xyz file name
  charge: 0                # net charge
  multiplicity: 1          # spin multiplicity, 2S+1
reduced:                   # input setting for reduced species
  input: Oc1ccc(O)cc1
  charge: 0
  multiplicity: 1
crest:                     # setting for CREST for conformation sampling
  method: gfn2             # xTB method
  solvation: gbsa          # solvation model
  solvent: water           # solvent name
  threads: 16              # number of threads used for CREST
optimization:              # setting for geometry optimization and frequency analysis
  xc: wB97M-D3BJ           # name of DFT functional
  basis: def2-SVP          # name of basis set
  grids:                   # grid setting
    atom_grid: [99, 590]
  nlcgrids:                # grid setting for VV10
    atom_grid: [50, 194]
  verbose: 2               # print level of PySCF
  scf_conv_tol: 1e-8       # convergence criteria (Eh)
  scf_max_cycle: 100       # max cycle for SCF iterations
  with_df: true            # using density fitting or not
  auxbasis: def2-universal-jkfit  # auxiliary basis
  with_gpu: true           # whether using GPU
single_point:              # setting for single point energy
  xc: wB97M-V
  basis: def2-TZVP
  grids:
    atom_grid: [99, 590]
  nlcgrids:
    atom_grid: [50, 194]
  verbose: 2
  scf_conv_tol: 1e-8
  scf_max_cycle: 100
  with_df: true
  auxbasis: def2-universal-jkfit
  with_gpu: true
solvent: water             # solvent name for SMD
E_ref: 4.44                # potential for reference electrode
n_electrons: 2             # number of transfered electrons
n_protons: 2               # number of transfered protons
```
Config file example for using MACE
```yaml
oxidized:                  # input setting for oxidized species
  input: O=C1C=CC(=O)C=C1  # smiles string or xyz file name
  charge: 0                # net charge
  multiplicity: 1          # spin multiplicity, 2S+1
reduced:                   # input setting for reduced species
  input: Oc1ccc(O)cc1
  charge: 0
  multiplicity: 1
crest:                     # setting for CREST for conformation sampling
  method: gfn2             # xTB method
  solvation: gbsa          # solvation model
  solvent: water           # solvent name
  threads: 16              # number of threads used for CREST
optimization:              # setting for geometry optimization and frequency analysis
  mlip: mace               # name of MLIP
  model: ./MACE-omol-0-extra-large-1024.model  # path to the model
single_point:              # setting for single point energy
  xc: wB97M-V
  basis: def2-TZVP
  grids:
    atom_grid: [99, 590]
  nlcgrids:
    atom_grid: [50, 194]
  verbose: 2
  scf_conv_tol: 1e-8
  scf_max_cycle: 100
  with_df: true
  auxbasis: def2-universal-jkfit
  with_gpu: true
solvent: water             # solvent name for SMD
E_ref: 4.44                # potential for reference electrode
n_electrons: 2             # number of transfered electrons
n_protons: 2               # number of transfered protons
```
