# RNN with Particle Flow for Probabilistic Spatio-temporal Forecasting

This is an implementation of the methodology in "RNN with Particle Flow for Probabilistic Spatio-temporal Forecasting".
You can 
- reproduce the results for AGCGRU+flow, DCGRU+flow, and GRU+flow algorithms in our paper and supplementary material with the checkpoints of our trained model.
- re-train the models from scratch with the same hyperparameter setting and evaluate.

&nbsp;
&nbsp;
&nbsp;

---
## To reproduce the results in the paper with pre-trained checkpoints

1. download the preprocessed datasets and checkpoints from ```https://drive.google.com/file/d/1tnaTTBVxUOdSpc5hxYUpCqUriYVVr5BP/view?usp=sharing``` 
2. decompress the downloaded zip file inside the folder ```rnn_flow```
3. run script ```python inference_for_dir.py --search_dir ckpt/pretrained --output_dir ./result_pretrained --gpu_id 0```

&nbsp;

Then the results will be saved in the ```./result_pretrained```.


&nbsp;
&nbsp;
&nbsp;

## To train the model
1. download the preprocessed datasets and checkpoints and decompress the downloaded zip file inside the folder ```rnn_flow```
2. train model with script (e.g.) : ```python training.py --dataset PEMS07 --rnn_type agcgru --cost mae --gpu_id 0  --trial_id  --ckpt_dir ./ckpt --data_dir ./data ```
3. evaluate with script: ```python inference.py --dataset PEMS07 --rnn_type agcgru --cost mae --gpu_id 0 --trial_id 0 --ckpt_dir ./ckpt --output_dir ./result```
&nbsp;

Note that:
- the setting of ```--dataset```, ```--rnn_type```, ```--cost```, ```--trial_id``` and ```--ckpt_dir``` should be consistent in training and evaluation.
- ```--trial_id```  controls the random seed; by changing it, we can get multiple-trials of results with different random seeds.

&nbsp;


### Options of Setting
```--rnn_type```:
- ```agcgru```
- ```dcgru```
- ```gru```

```--cost```:
- ```mae``` (for MAE, MAPE and RMSE computation)
- ```nll``` (for CRPS, P10QL and P90QL computation)

&nbsp;
&nbsp;

### Other Setting
The adjacency matrices of the graphs of the datasets are stored in ```./data/sensor_graph``` (```--graph_dir```). 

The model configs are stored in ```./data/model_config``` (```--config_dir```)

If the folders are changed, please make the arguments of ```train.py``` and ```inference.py``` consistent.

&nbsp;
&nbsp;
&nbsp;

----


## Dependencies
The code has been tested running under ```python 3.7.0``` with the following packages installed (along with their dependencies):
- ```tensorflow-gpu==1.15.0```
- ```numpy==1.19.2```
- ```networkx==2.5```
- ```pandas==1.1.3```
- ```scikit-learn==0.23.2```
- ```PyYAML==5.3.1```
- ```scipy==1.5.4```

In addition, for running with gpu, the code has been tested with ```CUDA 10.0``` and ```cuDNN 7.4```.

## Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{pal2021, 
author={S. Pal and L. Ma and Y. Zhang and M. Coates}, 
booktitle={Proc. Int. Conf. Machine Learning}, 
title={{RNN} with Particle Flow for Probabilistic Spatio-temporal Forecasting},
month={Jul.},
year={2021},
address={Virtual Conference}}
```
