# Robust Unsupervised Image Categorization Based on Variational Autoencoder with Disentangled Latent Representations
## Requirements
  
To install requirements:

```setup
conda env create -f environment.yml
conda activate Test
```

## File

    datasets/  # container of data  
    fsvae/     # core code  
    train.py   # training code   
    runs/      # runing result  

## Training

To train the method(s) in the paper, run this command:  

    __params:__  
    -r   # name of runing folder, default is fsvae  
    -n   # training epoch, default is 500  
    -o   # ratios of outlier, default is 0
    -s   # data set name, default is mnist  
    -v   # version of training, default is v1  

```train
python train.py -s mnist -n 500 -v v1 -r fsvae -o 0
```



