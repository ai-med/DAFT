# Combining 3D Image and Tabular Data via the Dynamic Affine Feature Map Transform

[![Journal Paper](https://img.shields.io/static/v1?label=DOI&message=10.1016%2fj.neuroimage.2022.119505&color=3a7ebb)](https://dx.doi.org/10.1016/j.neuroimage.2022.119505)
[![Conference Paper](https://img.shields.io/static/v1?label=DOI&message=10.1007%2f978-3-030-87240-3_66&color=3a7ebb)](https://dx.doi.org/10.1007/978-3-030-87240-3_66)
[![Preprint](https://img.shields.io/badge/arXiv-2107.05990-b31b1b)](https://arxiv.org/abs/2107.05990)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

This repository contains the code to the paper "DAFT: A Universal Module to Interweave Tabular Data and 3D Images in CNNs"
```
@article{Wolf2022-daft,
  title = {{DAFT: A Universal Module to Interweave Tabular Data and 3D Images in CNNs}},
  author = {Wolf, Tom Nuno and P{\"{o}}lsterl, Sebastian and Wachinger, Christian},
  journal = {NeuroImage},
  pages = {119505},
  year = {2022},
  issn = {1053-8119},
  doi = {10.1016/j.neuroimage.2022.119505},
  url = {https://www.sciencedirect.com/science/article/pii/S1053811922006218},
}
```
and the paper "Combining 3D Image and Tabular Data via the Dynamic Affine Feature Map Transform"
```
@inproceedings(Poelsterl2021-daft,
  title     = {{Combining 3D Image and Tabular Data via the Dynamic Affine Feature Map Transform}},
  author    = {P{\"{o}}lsterl, Sebastian and Wolf, Tom Nuno and Wachinger, Christian},
  booktitle = {International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  pages     = {688--698},
  year      = {2021},
  url       = {https://arxiv.org/abs/2107.05990},
  doi       = {10.1007/978-3-030-87240-3_66},
}
```
If you are using this code, please cite the papers above.

## Installation

1. Use [conda](https://conda.io/miniconda.html) to create an environment called `daft` with all dependencies:

```bash
conda env create -n daft --file requirements.yaml
```

2. (Optional) If you want to run the ablation study, you need to setup a
[Ray cluster](https://docs.ray.io/en/releases-1.1.0/cluster/index.html) to execute the
experiments in parallel. Please refer to the Ray documentation for details.

## Data

We used data from the [Alzheimer's Disease Neuroimaging Initiative (ADNI)](http://adni.loni.usc.edu/).
Since we are not allowed to share our data, you would need to process the data yourself.
Data for training, validation, and testing should be stored in separate
[HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) files,
using the following hierarchical format:

1. First level: A unique identifier, e.g. image ID.
2. The second level always has the following entries:
    1. A group named `Left-Hippocampus`, which itself has the
       [dataset](https://docs.h5py.org/en/stable/high/dataset.html) named `vol_with_bg` as child:
       The cropped ROI around the left hippocampus of size 64x64x64.
    2. A dataset named `tabular` of size 14:
       The tabular non-image data.
    3. A scalar [attribute](https://docs.h5py.org/en/stable/high/attr.html) `RID` with the *patient* ID.
    4. A string attribute `VISCODE` with ADNI's visit code.
    5. Additional attributes, depending on the task.
        1. For *classification*, a string attribute `DX` containing the diagnosis:
           `CN`, `MCI`, or `Dementia`.
        2. For *time-to-dementia* analysis.
           A string attribute `event` indicating whether conversion to dementia
           was observed (`yes` or `no`), and a scalar attribute `time` with
           the time to dementia onset or the time of censoring.

One entry in the resulting HDF5 file should have the following structure:
```
/1010012                 Group
    Attribute: RID scalar
        Type:      native long
        Data:  1234
    Attribute: VISCODE scalar
        Type:      variable-length null-terminated UTF-8 string
        Data:  "bl"
    Attribute: DX scalar
        Type:      variable-length null-terminated UTF-8 string
        Data:  "CN"
    Attribute: event scalar
        Type:      variable-length null-terminated UTF-8 string
        Data:  "no"
    Attribute: time scalar
        Type:      native double
        Data:  123
/1010012/Left-Hippocampus Group
/1010012/Left-Hippocampus/vol_with_bg Dataset {64, 64, 64}
/1010012/tabular         Dataset {14}
```

Finally, the HDF5 file should also contain the following meta-information
in a separate group named `stats`:

```
/stats/tabular           Group
/stats/tabular/columns   Dataset {14}
/stats/tabular/mean      Dataset {14}
/stats/tabular/stddev    Dataset {14}
```

They are the names of the features in the tabular data,
their mean, and standard deviation.


## Usage

## Training

To train DAFT, or any of the baseline models, execute the `train.py` script to begin training.
The essential command line arguments are:

  - `--task`: The type of loss to optimize. Can be `clf` for classification, and `surv` for time-to-dementia analysis.
  - `--train_data`: Path to HDF5 file containing *training* data.
  - `--val_data`: Path to HDF5 file containing *validation* data.
  - `--test_data`: Path to HDF5 file containing *test* data.

Model checkpoints will be written to the `experiments_clf` or `experiments_surv` folder,
depending on the value of `--task`.

For a full list of command line arguments, execute:
```
python train.py --help
```

## Ablation Study

Performing the ablation study
is computationally quite expensive, and requires a Ray cluster to be setup
to execute experiments in parallel.
Please refer to the [Ray documentation](https://docs.ray.io/en/releases-1.1.0/cluster/index.html)
on how to do that.
We expect that `ray-cluster.yaml` refers to a valid
[Ray cluster config](https://docs.ray.io/en/releases-1.1.0/cluster/cloud.html).

There are 2 scripts that you can execute:
  - `ablation_adni_classification.py`: Ablation study for classification experiments.
    Will create `results_adni_classification_ablation_split-*.csv` files on the Ray head node.
  - `ablation_adni_survival.py`: Ablation study for time-to-dementia experiments.
    Will create `results_adni_survival_ablation_split-*.csv` files on the Ray head node.

These are the steps you need to execute:

1. Start the Ray cluster:
```
ray up ray-cluster.yaml
```

2. Execute the `ablation_adni_classification.py` (or `ablation_adni_survival.py`):
```
ray exec ray-cluster.yaml --tmux \
  '/usr/bin/python /path/to/workspace/ablation_adni_classification.py --data_dir=/path/to/data/ --ray_address=<hostname>:<port>'
```

3. Monitor the progress:
```
ray attach ray-cluster.yaml --tmux
```

4. Inspect the final results for the first fold:
```
cat results_adni_classification_ablation_split-0.csv
```
