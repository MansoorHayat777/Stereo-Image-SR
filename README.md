**In the final result, we integrate ten models, four of which are from NAFSSR, three from SwinIR_LTE, and the remaining three from the RDN_LTE, SSRDEFNET, LIIF, and the fusion strategy is explained in detail in factsheet.**

# Requirements
```shell
pip install -r requirements.txt 
```
# Train
## 1. Prepare training data 


### model1: NAFSSR
```shell
​    datasets
​    ├── StereoSR
​    │   ├── patches_x4
​    │   │   ├── 0001
​    │   │   │   ├── hr0.png
​    │   │   │   ├── hr1.png
​    │   │   │   ├── lr0.png
​    │   │   │   └── lr1.png
​    │   │   ├── ...
​    │   │   ├── 
​    │   │   └── 0800
​    │   │       ├── hr0.png
​    │   │       ├── hr1.png
​    │   │       ├── lr0.png
​    │   │       └── lr1.png
​    │   ├── test
​    │   │   ├── Flickr1024
​    │   │       ├── hr
​    │   │       │   ├── 0001
​    │   │       │   │   ├── lr0.png
​    │   │       │   │   └── lr1.png
​    │   │       │   ├── ...
​    │   │       │   ├──	
​    │   │       ├── lr_x4
​    │   │       │   ├── 0001
​    │   │       │   │   ├── lr0.png
​    │   │       │   │   └── lr1.png
​    │   │       │   ├── ...
​    │   │       │   ├──
```
### model2: SSRDEFNet
```shell
​    data
​    ├── train
​    │	    └── Flickr1024_patches
​    │		└── patches_x4
​    │		    ├── 0001
​    │		    │   ├── hr0.png
​    │		    │   ├── hr1.png
​    │		    │   ├── lr0.png
​    │		    │   └── lr1.png
​    │		    ├── 0002
​    │		    │   ├── hr0.png
​    │		    │   ├── hr1.png
​    │		    │   ├── lr0.png
​    │		    │   └── lr1.png
​    │		    ├── ...
​    │
​    ├── test
​    │   ├── Flickr1024
​    │       ├── hr
​    │       │   ├── 0001
​    │       │   │   ├── lr0.png
​    │       │   │   └── lr1.png
​    │       │   ├── ...
​    │       │   ├──	
​    │       ├── lr_x4
​    │       │   ├── 0001
​    │       │   │   ├── lr0.png
​    │       │   │   └── lr1.png
​    │       │   ├── ...
​    │       │   ├──
```
### model3: SwinIR-LTE
```shell
​    data
    ├── Flickr1024_train
    │   ├── HR
    │   │   ├── 0001_L.png
    │   │   ├── 0001_R.png
    │   │   ├── 0002_L.png
    │   │   ├── 0002_R.png
    │   │   │    ...
    │   ├── LR
    │   │   ├── 0001_L.png
    │   │   ├── 0001_R.png
    │   │   ├── 0002_L.png
    │   │   ├── 0002_R.png
    │   │   │    ...
​    │
    ├── Flickr1024_val
    │   ├── HR
    │   │   ├── 0001_L.png
    │   │   ├── 0001_R.png
    │   │   ├── 0002_L.png
    │   │   ├── 0002_R.png
    │   │   │    ...
    │   ├── LR
    │   │   ├── 0001_L.png
    │   │   ├── 0001_R.png
    │   │   ├── 0002_L.png
    │   │   ├── 0002_R.png
    │   │   │    ...
```
### model4: RDN_LTE
```shell
​    data
    ├── Flickr1024_train
    │   ├── HR
    │   │   ├── 0001_L.png
    │   │   ├── 0001_R.png
    │   │   ├── 0002_L.png
    │   │   ├── 0002_R.png
    │   │   │    ...
    │   ├── LR
    │   │   ├── 0001_L.png
    │   │   ├── 0001_R.png
    │   │   ├── 0002_L.png
    │   │   ├── 0002_R.png
    │   │   │    ...
​    │
    ├── Flickr1024_val
    │   ├── HR
    │   │   ├── 0001_L.png
    │   │   ├── 0001_R.png
    │   │   ├── 0002_L.png
    │   │   ├── 0002_R.png
    │   │   │    ...
    │   ├── LR
    │   │   ├── 0001_L.png
    │   │   ├── 0001_R.png
    │   │   ├── 0002_L.png
    │   │   ├── 0002_R.png
    │   │   │    ...
```
### model5: LIIF
```shell
​    data
    ├── Flickr1024_train
    │   ├── HR
    │   │   ├── 0001_L.png
    │   │   ├── 0001_R.png
    │   │   ├── 0002_L.png
    │   │   ├── 0002_R.png
    │   │   │    ...
    │   ├── LR
    │   │   ├── 0001_L.png
    │   │   ├── 0001_R.png
    │   │   ├── 0002_L.png
    │   │   ├── 0002_R.png
    │   │   │    ...
​    │
    ├── Flickr1024_val
    │   ├── HR
    │   │   ├── 0001_L.png
    │   │   ├── 0001_R.png
    │   │   ├── 0002_L.png
    │   │   ├── 0002_R.png
    │   │   │    ...
    │   ├── LR
    │   │   ├── 0001_L.png
    │   │   ├── 0001_R.png
    │   │   ├── 0002_L.png
    │   │   ├── 0002_R.png
    │   │   │    ...
```



## 2. Begin to train

### model1: NAFSSR
```shell
cd NAF/NAFNet/
python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/NAFSSR/NAFSSR-L_x4.yml --launcher pytorch
```

### model2: SSRDEFNet

```shell
cd SSRDEFNet/SSRDEFNet-PyTorch
python train.py --scale_factor 4
```

### model3: SwinIR-LTE

```shell
cd LTE/lte
python train.py --config configs/train/train_swinir-lte.yaml --gpu 0,1,2,3,4,5,6,7
```

### model4: RDN_LTE

```shell
cd LTE/lte
python train.py --config configs/train/train_rdn-lte.yaml --gpu 0,1,2,3,4,5,6,7
```

### model5: LIIF

```shell
cd mmedit
bash tools/dist_train.sh EXP/LIIF/liif.py 8
```

# Test
## 1. Prepare test data 
The test set format is consistent with the validation set format.

## 2. Begin to test
### model1: NAFSSR

```shell
cd NAF/NAFNet/
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt options/test/NAFSSR/NAFSSR-L_4x_1.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt options/test/NAFSSR/NAFSSR-L_4x_2.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt options/test/NAFSSR/NAFSSR-L_4x_3.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt options/test/NAFSSR/NAFSSR-L_4x_4.yml --launcher pytorch
```

### model2: SSRDEFNet

```shell
cd SSRDEFNet/SSRDEFNet-PyTorch
python test_sr.py
```

### model3: SwinIR-LTE

```shell
cd LTE/lte
python test-save.py --config configs/test/test.yaml --model save/_train_swinir-lte/epoch_1.pth --window 8 --gpu 0
python test-save.py --config configs/test/test.yaml --model save/_train_swinir-lte/epoch_2.pth --window 8 --gpu 0
python test-save.py --config configs/test/test.yaml --model save/_train_swinir-lte/epoch_3.pth --window 8 --gpu 0
```

### model4: RDN_LTE

```shell
cd LTE/lte
python test-save.py --config configs/test/test.yaml --model save/_train_rdn-lte/epoch_1.pth --window 8 --gpu 0
```

### model5: LIIF

```shell
cd mmedit
python tools/test.py EXP/LIIF/liif.py EXP/LIIF/iter_588000.pth --save-path EXP/save
```
