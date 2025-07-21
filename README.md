# Universal crystal material property prediction via multi-view geometric fusion in graph transformers

![Fig.1.png](assert%2FFig.1.png)

Accurately and comprehensively representing crystal structures is critical for advancing machine learning in large-scale crystal materials simulations, however, effectively capturing and leveraging the intricate geometric and topological characteristics of crystal structures remains a core, long-standing challenge for most existing methods in crystal property prediction. Here, we propose MGT, a multi-view graph transformer framework that synergistically fuses SE3 invariant and SO3 equivariant graph representations, which respectively captures rotation-translation invariance and rotation equivariance in crystal geometries. To strategically incorporate these complementary geometric representations, we employ a lightweight mixture of experts router in MGT to adaptively adjust the weight assigned to SE3 and SO3 embeddings based on the specific target task. Compared with previous state-of-the-art models, MGT reduces the mean absolute error by up to 21% on crystal property prediction tasks through multi-task self-supervised pretraining. Ablation experiments and interpretable investigations confirm the effectiveness of each technique implemented in our framework. Additionally, in transfer learning scenarios including crystal catalyst adsorption energy and hybrid perovskite bandgap prediction, MGT achieves performance improvements of up to 58% over existing baselines, demonstrating domain-agnostic scalability across diverse application domains. As evidenced by the above series of studies, we believe that MGT can serve as useful model for crystal material property prediction, providing a valuable tool for the discovery of novel materials.

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
