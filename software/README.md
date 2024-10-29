# Evaluating the ConvNeXt-ViViT Hybrid Architecture in Predicting Vehicular Crashes from Video Data

## Introduction

Implementation and evaluation of a hybrid accident anticipation model for dashcam footage using deep learning architectures of Video Vision Transformer and ConvNeXt.

## Requirements

For this project CUDA Toolkit and Nvidia drivers compatible with pytorch must be installed. For this research university cluster was used with Driver Version: 470.74 & CUDA Version: 11.4.

Conda environment package list and usage are provided in `package-list.txt`

Alternatively, manually install latest (please adjust pytorch-cuda version):

```sh
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

conda install opencv scikit-learn numpy -c conda-forge -y

pip install transformers
```

## Dataset

Dashcam Accident Dataset (DAD): http://aliensunmin.github.io/project/dashcam/

## Best Model Data

Best model details and attached files listed below:

```
combined_crash_classifier_1024_512_256.pth (not provided due to file size)
slurm-1017304.out - train
slurm-1017479.out - test

Average Precision: 0.7279758534598242
Precision: 0.5975833990018388
Recall: 0.6893939393939394
Average guess: 0.5742291808128357
Test runtime:  8635.208868980408
```

## Usage

### Important!

The main file is named `combined_crash_classifier.py`, additional files for `ConvNeXt` and `ViViT` are named respectively inside of the `/individual_models` directory. TTA calcuation script file is located inside of the `/extra_scripts` directory.

</br>

Please replace to correct path directories for all files in training, testing, validation and ouput:

`/cs/home/psyjn4/db/DAD/...`

`./datasets/DAD/Dashcam_dataset/videos/...`

### Environmental Variables:

`USE_DIST`: 0/1, set to 1 if multiple GPUs should be used

`TEST`: 0/1, set to 0 if training, set to 1 if testing

`TEST_TTA`: 0/1, set to 0 if training maximising Average Precision (AP) on the last 10 frame subsets of the video, set to 1 if testing on the entire video for Time-To-Accident (TTA).

### Cluster

Slurm workload manager CPU, RAM, GPU and priviledge settings.

[Slurm Documentation](https://slurm.schedmd.com/documentation.html)

```
#SBATCH -c8 --mem=32g
#SBATCH --gpus 2
#SBATCH -p cs -q csug
```

`nproc_per_node`: in batch scripts should be adjusted based on the number of GPUs used

### Training

OPTIONAL: Adjust environment variables as needed.

#### Local:

`python combined_crash_classifier.py`

#### Cluster:

`sbatch train.sh`

### Testing

#### Local:

Linux: `TEST=0 python combined_crash_classifier.py`

Windows CMD: `set TEST=0 && python combined_crash_classifier.py`

#### Cluster:

`sbatch test.sh`
