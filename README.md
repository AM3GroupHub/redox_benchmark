# redox_benchmark
A repo for benchmark test of the redox potential of molecules

## Install requirements
Requirements
- Conda/Mamba
- CUDA version greater than 12
```shell
conda env create -f environment.yml
```
or
```shell
mamba env create -f environment.yml
```

If your CUDA version is under 12, you can manually install the following pacakge with CUDA 11 version.
- PyTorch
- GPU4PySCF

## TODO: scripts