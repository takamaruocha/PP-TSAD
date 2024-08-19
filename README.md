# **PP-TSAD**

## 1. Overview
We develop a framework that balances the trade-off between privacy preservation and anomaly detection performance and maintains sufficient accuracy for practical purposes, even with anonymized time series.

## 2. Code Description
- pp_tsad_runner.py: Main file. The trained models are saved in folder `saved_models/`
- parameter.py: Parameter setting file.
- models: Definition folder of the anonymization function, privacy model, and anomaly detection model.
- TimeSeriesProject: Definition folder of the classification model.
- data_loader.py: Pre-processing file for datasets.
- augmentations.py: Processing file for data augmentation.
- datasets: Dataset folder.

## 3. Dataset
We use ECG, TODS, SWaT, PSM datasets.  
You can use `train.csv` and `test.csv` of the TODS dataset from folder `datasets/TODS/`
The TODS dataset is a synthetic time series generated using the method proposed in [here](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/ec5decca5ed3d6b8079e2e7e7bacc9f2-Paper-round1.pdf).
We created five types of five-dimensional time series with different function types (e.g., sine and cosine functions), frequencies, and amplitudes, and assigned five corresponding IDs to each.

If you want to use the ECG dataset, you need to download it from [here](https://physionet.org/content/mitdb/1.0.0/).
If you want to use the SWaT dataset, you need to request it [here](https://itrust.sutd.edu.sg/itrust-labs_datasets/).
If you want to use the PSM dataset, you need to download it from [here](https://drive.google.com/drive/folders/1RaIJQ8esoWuhyphhmMaH-VCDh-WIluRR).

## 4. Reproducibility
1. Download data and put them in folder `datasets/`.
2. To set parameters in parameter.py
- Parameters in parameter.py
```
data: Name of the dataset to be used
root_path: Root path for the dataset
saved_models_dir: Directory for saving models
in_len: Length of the sequence
step_size: Step size for creating sequences
n_classes: Number of ID classes
d_model: Embedding dimension
d_ff: Feed-forward layer dimension
n_heads: Number of attention heads
e_layers: Number of encoder layers
dropout: Dropout rate
seg_len_fa: Segment length for the anonymization function
data_dim: Dimension of the data
jitter_ratio: Ratio for jittering in data augmentation settings
temperature: Temperature parameter for contrastive learning
batch_size: Batch size
learning_rate: Learning rate
num_epochs: Number of epochs
```
3. To train and evaluate the proposed framework, run:  
```python
python3 pp_tsad_runner.py
```
The trained models are saved in folder `saved_models/`
