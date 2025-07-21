### Pretraining Dataset 

We have prepared the relevant scripts in [here](https://doi.org/10.5281/zenodo.15473642).

For OQMD pretraining dataset, please download the processed CIF files from [train](https://zenodo.org/records/10642388/files/cifs_v1_train.pkl.gz),  [val](https://zenodo.org/records/10642388/files/cifs_v1_val.pkl.gz),  [test](https://zenodo.org/records/10642388/files/cifs_v1_tset.pkl.gz). 

```
python bin/cif2dataset_OQMD_pretrained.py
```

For GMAE pretraining dataset, please download the package from [here](https://zenodo.org/records/12104162).

```
python bin/cif2dataset_GMAE_pretrained.py
```
