# CityUCS5182ComputerGraphics Project

Compatible with pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.1 and Ubuntu 20.04

## Download Datasets:
### 2. Data Preparation

#### a. Prepare Patches

First, follow the official repo, download patches in HDF5 format from [GoogleDrive](https://drive.google.com/file/d/1wMtNGvliK_pUTogfzMyrz57iDb_jSQR8/view?usp=sharing) and put it into `./datas/`. Patches are splitted for training (3200) and testing (800). See `./datas/train_list.txt` and `./datas/test_list.txt`.

#### b. Prepare Datas for Visualization

Objects with 5k points for testing can be downloaded from the official repo, [link](https://github.com/yulequan/PU-Net/tree/master/data/test_data/our_collected_data/MC_5k). Put them into `./datas/test_data/our_collected_data/MC_5k`.

#### c. Prepare Datas for NUC Calculation

The training and testing mesh files can be downloaded from [GoogleDrive](https://drive.google.com/file/d/1R21MD1O6q8E7ANui8FR0MaABkKc30PG4/view?usp=sharing). Put test mesh files into `./datas/test_data/test_mesh`.

The `./datas` folder should be organized as follows:

```shell
PU-Net_pytorch
├── datas
│   ├── Patches_noHole_and_collected.h5
│   ├── test_list.txt
│   ├── train_list.txt
│   ├── test_data
│   │  │   ├── test_mesh
│   │  │   │   ├── *.off
│   │  │   ├── our_collected_data/MC_5k
│   │  │   │   ├── *.xyz
```


## Run Training
```shell
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.1 -c pytorch

# Point2
cd pointnet2
python setup.py install

# in root(PUNET)
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
bash train_punet.sh
```

### 3. Train

Run the following commands for training.

```shell
mkdir logs
bash train_punet.sh
```

### 4. Evaluation (EMD and CD)

Run the following commands for evaluation.

```shell
python eval.py --gpu 0 --resume logs/punet_baseline/punet_epoch_99.pth
```

### 5. Visualization and Test (NUC)

Run the following commands to generate upsampled datas from full mesh objects with 5k points. Upsampled point clouds are saved in `./outputs/punet_baseline/*.ply`. And the dumpped `*.xyz` files are used for NUC calculation.

```shell
mkdir outputs
bash test_punet.sh
```
