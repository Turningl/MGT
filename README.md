# DGM is a dual-graph multimodal transformer for crystal material property prediction

Accurate and efficient representation of crystal structures is essential for enabling machine learning in large-scale crystal materials simulations, however, effectively capturing and leveraging the intricate geometric and topological features of crystal structures remains a significant challenge for most current methods in crystal property prediction. Herein, we propose DGM, a novel dual-graph multimodal transformer framework that explicitly incorporates both SE3 invariant and SO3 equivariant graph features. To dynamically integrate these complementary geometric representations, a mixture of experts module is employed in DGM to adaptively adjust the weight assigned to SE3 and SO3 embeddings based on the specific prediction task. Through multi-task self-supervised learning pretraining, DGM significantly improves the performance when fine-tuned on various downstream tasks. Ablation experiments and interpretable investigations effectively underscore the crucial role of each technique implemented in our DGM framework. In addition, the transfer learning scenario demonstrates the generalizability and scalability of DGM in crystal catalyst screening application. As evidenced by a comprehensive set of studies, DGM can serve as a powerful and generalizable model that fully leverages geometric information for crystal representation learning, providing a valuable tool for accelerating the discovery of novel materials.


## Installation

Set up conda environment

```
conda env create -f environment.yml
conda activate DGM
```

## Dataset 

We have prepared the relevant processed datasets, which can be used directly for your convenience. Please download the pre-training, fine-tuning, and transfer datasets used in the paper [here](https://doi.org/10.5281/zenodo.15473642).

Once you have successfully downloaded the datasets, please follow these steps for organization:

#### Pretraining Datasets: 

Extract the pre-training dataset and unzip it under the `./dataset/pretrained` folder. Additionally, we have provided a pre-training debug dataset to assist you in debugging your code.

#### Fine-tuning and Transfer Learning Datasets:

Extract the fine-tuning and transfer learning datasets and unzip them under the `./dataset/fine-tuning` folder.

#### Process dataset:

If you prefer to handle each pre-training and fine-tuning dataset independently, we have provided relevant command lines and detailed instructions. You can find more information in the `./bin/README.md` file.

## Pretrained models

The pre-trained DGM for pretraining can be found in `ckpt/pretraining` folder. 

All downstream tasks of DGM for `tutorial.ipynb` can be found in `ckpt/finetuned` folder.

## Pretraining

To train the DGM framework, where the configurations and detailed explaination for each variable can be found in `config/pretraining.yml` folder.

```
python pretraining.py
```

## Fine-tuning 

To fine-tune the pre-trained model on downstream prediction tasks, where the configurations and detailed explaination for each variable can be found in `config/finetune.yml`

```
python finetune.py
```

## Interferce

A tutorial notebook for interferce process is available in `tutorial.ipynb`.

```
jupyter notebook tutorial.ipynb
```


## License

DGM is released under the [MIT](LICENSE) license.

## Contact

If you have any questions, please reach out to zl16035056@gmail.com
