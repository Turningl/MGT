# Universal crystal material property prediction via multi-view geometric fusion in graph transformers

![Fig.1.png](assert%2FFig.1.png)

## Installation

Set up conda environment

```
conda env create -f environment.yml
conda activate MGT
```

## Pre-trained models

The pre-trained MGT for pretraining can be found in `ckpt/pretraining` folder. 

All downstream tasks of MGT for `tutorial.ipynb` can be found in `ckpt/finetuned` folder.

## Pretraining

To train the MGT framework, where the configurations and detailed explaination for each variable can be found in `config/pretraining.yml` folder.

```
python pretraining.py  config/pretraining.yml
```

## Fine-tuning 

To fine-tune the pre-trained model on downstream prediction tasks, where the configurations and detailed explaination for each variable can be found in `config/finetune.yml`

```
python finetune.py config/finetune.yml
```

## Interferce

A tutorial notebook for interferce process is available in `tutorial.ipynb`.

```
jupyter notebook tutorial.ipynb
```

## Dataset 

We have prepared the relevant processed datasets, which can be used directly for your convenience. Please download the pre-training, fine-tuning, and transfer datasets used in [here](https://doi.org/10.5281/zenodo.15473642).

Once you have successfully downloaded the datasets, please follow these steps for organization:

#### Pretraining Datasets: 

Extract the pre-training dataset and unzip it under the `./dataset/pretrained` folder. Additionally, we have provided a pre-training debug dataset to assist you in debugging your code.

#### Fine-tuning and Transfer Learning Datasets:

Extract the fine-tuning and transfer learning datasets and unzip them under the `./dataset/fine-tuning` folder.

#### Process dataset:

If you prefer to handle each pre-training and fine-tuning dataset independently, we have provided relevant command lines and detailed instructions. You can find more information in the `./dataset/README.md` file.


## License

MGT is released under the [MIT](LICENSE) license.

## Contact

If you have any questions, please reach out to liang36365@mail.ustc.edu.cn
