### Fine-tuning and transfer learning Datasets

We have prepared the relevant scripts in [here](https://doi.org/10.5281/zenodo.15473642).

For Formation Energy, Band Gap fine-tuning datasets, please run:

```
python bin/cif2dataset_finetune_megnet.py
```

For Bulk Moduli and Shear Moduli fine-tuning datasets, please download the package from [here](https://figshare.com/projects/Bulk_and_shear_datasets/165430).

```
python bin/cif2dataset_bulk_moduli.py
python bin/cif2dataset_shear_moduli.py
```

For Formation Energy, Bandgap (OPT), Total Energy, Bandgap (MBJ) and Ehull fine-tuning datasets, please run:

```
python bin/cif2dataset_finetune_dft_3d.py
```

For Alloy-GMAE, FG-GMAE and OCD-GMAE fine-tuning datasets, please download the package from [here](https://zenodo.org/records/12104162).

```
python bin/cif2dataset_GMAE.py
```
