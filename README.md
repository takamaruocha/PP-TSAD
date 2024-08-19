# **PP-TSAD**

## 1. Overview
We develop a framework that balances the trade-off between privacy preservation and anomaly detection performance and maintains sufficient accuracy for practical purposes, even with anonymized time series.

## 2. Code Description
- pp_tsad_runner.py: The main file. The trained modela are saved in folder `saved_models/`
- parameter.py: You can set all parameters.
- models: The definition folder of the proposed framework.
- data_loader.py: The pre-processing file for datasets.
- augmentations.py: The processing file for data augmentation.
- datasets: The dataset folder.

## 3. Dataset
We use ECG, TODS, SWaT, PSM datasets.  
You can use `train.csv` and `test.csv` of the TODS dataset from folder `datasets/TODS/`
The TODS dataset is a synthetic time series generated using the method proposed in [here](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/ec5decca5ed3d6b8079e2e7e7bacc9f2-Paper-round1.pdf).
We created five types of five-dimensional time series with different function types (e.g., sine and cosine functions), frequencies, and amplitudes, and assigned five corresponding IDs to each.

If you want to use the ECG dataset, you need to download it from [here](https://physionet.org/content/mitdb/1.0.0/).
If you want to use the SWaT dataset, you need to request to　[here](https://itrust.sutd.edu.sg/itrust-labs_datasets/).
If you want to use the ECG dataset, you need to download it from [here](https://drive.google.com/drive/folders/1RaIJQ8esoWuhyphhmMaH-VCDh-WIluRR).

## 4. Reproducibility
1. Download data and put them in folder `datasets/`.
2. To set parameters in parameter.py
3. To train and evaluate the proposed framework, run:  
```python
python3 pp_tsad_runner.py
```
The trained models are saved in folder `saved_models/`
- Parameter options
```
--data: dataset
--root_path: The root path of the data file
--checkpoints: The location to store the trained model
--in_len: The input length
--out_len: The prediction length
--step_size: The step size
--seg_len: The segment length
--data_dim: The dimensions of data
--d_model: The dimension of hidden states
--d_ff: The dimension of feedforward network
--n_heads: The number of heads
--e_layers: The number of encoder layers
--dropout: The dropout
--attn_ratio: The attention ratio in the attention block
--itr: The experiments times
```

```
## 4. Citation
