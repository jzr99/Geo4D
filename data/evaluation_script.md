# Dataset Preparation for Evaluation

We provide scripts to download and prepare the datasets for evaluation. The datasets include: **Sintel**, **Bonn**, **KITTI**, **NYU-v2**, **TUM-dynamics**, **ScanNetv2**, and **DAVIS**.

> [!NOTE]
> The scripts provided here are for reference only. Please ensure you have obtained the necessary licenses from the original dataset providers before proceeding.


## Download Datasets

### Sintel
To download and prepare the **Sintel** dataset, execute:
```bash
cd data
bash download_sintel.sh
cd ..

# (optional) generate the GT dynamic mask
cd ..
python datasets_preprocess/sintel_get_dynamics.py --threshold 0.1 --save_dir dynamic_label_perfect 
```

### Bonn
To download and prepare the **Bonn** dataset, execute:
```bash
cd data
bash download_bonn.sh
cd ..

# create the subset for video depth evaluation, following depthcrafter
cd datasets_preprocess
python prepare_bonn.py
cd ..
```

### KITTI
To download and prepare the **KITTI** dataset, execute:
```bash
cd data
bash download_kitti.sh
cd ..

# create the subset for video depth evaluation, following depthcrafter
cd datasets_preprocess
python prepare_kitti.py
cd ..
```

### NYU-v2
To download and prepare the **NYU-v2** dataset, execute:
```bash
cd data
bash download_nyuv2.sh
cd ..

# prepare the dataset for depth evaluation
cd datasets_preprocess
python prepare_nyuv2.py
cd ..
```

### TUM-dynamics
To download and prepare the **TUM-dynamics** dataset, execute:
```bash
cd data
bash download_tum.sh
cd ..

# prepare the dataset for pose evaluation
cd datasets_preprocess
python prepare_tum.py
cd ..
```

### ScanNet
To download and prepare the **ScanNet** dataset, execute:
```bash
cd data
bash download_scannetv2.sh
cd ..

# prepare the dataset for pose evaluation
cd datasets_preprocess
python prepare_scannet.py
cd ..
```

### DAVIS
To download and prepare the **DAVIS** dataset, execute:
```bash
cd data
python download_davis.py
cd ..
```