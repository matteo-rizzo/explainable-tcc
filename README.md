# interpretable-tcc

Interpretable neural network architecture for Temporal Color Constancy (TCC)

## Installation

This implementation uses [Pytorch](http://pytorch.org/). It was developed and tested using torch >= 1.7.1 and python 3.6
on Ubuntu 18.04.

Requirements can be installed using the following command:

```shell
pip install -r requirements.txt
```

## Dataset

The TCC dataset was introduced in the paper [A Benchmark for Temporal Color Constancy](https://arxiv.org/abs/2003.03763) and can be downloaded and preprocessed following the instructions reported in the related [code repository](https://github.com/yanlinqian/Temporal-Color-Constancy). A polished version of the code for
processing the sequences is available at `dataset/tcc/img2npy.py` in [this repository](https://github.com/matteo-rizzo/cctcc). The script generates the preprocessed data at `dataset/tcc/preprocessed` and can be used to generate the splits for a 3-folds CV based on a CSV file containing the metadata (i.e., which items belong to which split). The metadata for the CV can be generated running the `dataset/tcc/cv_split.py` script and the file needed to replicate the splits used in the experiments is provided as `dataset/tcc/3_folds_experiment.csv`. The preprocessed sequences must be placed in `dataset/tcc/preprocessed` with respect to the root folder.

The script for preprocessing can be run with `python3 img2npy.py` from within the `dataset/tcc/` folder.

The  `dataset/tcc/img2npy.py` file contains the following global variables that can be edited to configured the
preprocessing of the data:

```python
# Whether or not to use the CV metadata file for generating the preprocessed files
USE_CV_METADATA = False

# Whether or not the CV metadada contain a validation set
USE_VAL_SET = False

# The id of the CV split to be generated (i.e, the corresponding index in the CSV with the metadata) 
FOLD_NUM = 2

# The name of the CSV file containing the metadata for the CV
CV_METADATA_FILE = "3_folds_experiment.csv"

# Whether or not to save data using float64 (results in a relevant increase in space disk required)
USE_HIGH_PRECISION = False

# Whether or not to truncate the sequences keeping only the last SUBSEQUENCE_LEN frames
TRUNCATE = False

# The number of frames to be kept in the truncation process
SUBSEQUENCE_LEN = 2

# Base path to the folder containing the preprocessed data
BASE_PATH_TO_DATA = os.path.join("preprocessed", "fold_" + str(FOLD_NUM) if USE_CV_METADATA else "tcc_split")

# Path to the folder containing the sequences in numpy format
PATH_TO_NUMPY_SEQ = os.path.join(BASE_PATH_TO_DATA, "ndata_seq")

# Path to the folder containing the labels (i.e., ground truth illuminants) in numpy format
PPATH_TO_NUMPY_LABEL = os.path.join(BASE_PATH_TO_DATA, "nlabel")

# The base path to the raw sequences to be preprocessed
BASE_PATH_TO_DATASET = "raw"

# The path to the raw training set to be preprocessed
PATH_TO_TRAIN = os.path.join(BASE_PATH_TO_DATASET, "train")

# The path to the raw test set to be preprocessed
PATH_TO_TEST = os.path.join(BASE_PATH_TO_DATASET, "test")

# The name of the file containing the groundtruth of a sequence (located at, e.g., "raw/train/1/")
GROUND_TRUTH_FILE = "groundtruth.txt"
```

## Pretrained models

Pretrained models in PTH format can be downloaded from [here](https://ubcca-my.sharepoint.com/:u:/r/personal/marizzo_student_ubc_ca/Documents/Models/itccnet.zip?csf=1&web=1&e=aZIFFt). To reproduce the results reported in the paper, the path to the pretrained models must be specified in the corresponding testing script.

## Structure of the project

The code in this project is mainly structured following object-oriented programming. The core code for the project is stored under `classes` . The implemented neural network architectures (i.e., AttTCCNet, ConfTCCNet and ConfAttTCCNet) are located at `classes/modules/multiframe`.

Each module at `classes/modules` features a **network** (i.e., a subclass of `nn.Module`) and a **model** (i.e., a subclass of the custom `classes/modules/common/Model.py` handling the prediction step). Note that each model *[has a](https://en.wikipedia.org/wiki/Has-a)* network and acts as interface towards it for training and inference, handling the weights update and the loss computation. Each network implements three dimensions of attentions, namely spatial, temporal and spatiotemporal.

The `auxiliary/settings.py` file features two functions:

* `get_device`: instantiates the Torch DEVICE (i.e., either CPU or GPU) for training and testing. The DEVICE type can be edited to the corresponding global variable at the top of the file.
* `make_deterministic`: sets the random seed. Note that models have been trained using a mix of Tesla P100 and NVidia
  GeForce GTX 1080 Ti GPUs from local lab equipment and cloud services. Please refer to the [official PyTorch docs](https://pytorch.org/docs/stable/notes/randomness.html) for an explanation on how randomness is handled.

**Important:** the path to the dataset folder must specified editing the `PATH_TO_DATASET` global variable at `auxiliary/settings.py`.

## Running the code

### Training

**TCCNet** can be trained with the `python3 train/tcc/train_tccnet.py` command. The file includes the following global variables that can be edited to configure the training:

```python
# The folder at "dataset/tcc/preprocessed" containing the data the model must be trained on 
DATA_FOLDER = "tcc_split"

# Whether or not to use the shot-frame branch in TCCNet
USE_SHOT_BRANCH = False

# The number of iterations over data
EPOCHS = 2000

# The learning rate to which the optimizer must be initialized
LEARNING_RATE = 0.00003

# The random seed for reproducibility
RANDOM_SEED = 0

# The path where the log data will be stored at
PATH_TO_LOGS = os.path.join("train", "tcc", "logs")

# Whether or not to resume the training based on an existing checkpoint model in PTH format
RELOAD_CHECKPOINT = False

# The path to the PTH file to be used as checkpoint
PATH_TO_PTH_CHECKPOINT = os.path.join("trained_models", "{}_{}".format("tccnet", DATA_FOLDER), "model.pth")
```

The **Interpretable TCCNet** (`itccnet`) can be trained with the `python3 train/tcc/train_itccnet.py` command. The file includes the following global variables that can be edited to configure the training:

```python
# The random seed for reproducibility
RANDOM_SEED = 0

# Which type of model to train - Values: "att_tccnet", "conf_tccnet", "conf_att_tccnet"
MODEL_TYPE = "att_tccnet"

# Which dataset slit to use - Values (TCC): "tcc_split", "fold_0", "fold_1", "fold_2"
DATA_FOLDER = "tcc_split"

# Which attention/confidence module should be deactivated - Values: "spat", "temp", empty string
DEACTIVATE = ""

# The size in the hidden layer of the ConvLSTM module
HIDDEN_SIZE = 128

# The kernel size of the convolutions in the ConvLSTM module
KERNEL_SIZE = 5

# The number of iterations over data
EPOCHS = 1000

# The learning rate to which the optimizer must be initialized
LEARNING_RATE = 0.00003

# Whether or not to resume the training based on an existing checkpoint model in PTH format
RELOAD_CHECKPOINT = True

# The path to the PTH file to be used as checkpoint
PATH_TO_PTH_CHECKPOINT = os.path.join("trained_models", "{}_{}".format(MODEL_TYPE, DATA_FOLDER), "model.pth")
```

**Note:** all models can also be trained executing the corresponding scripts from terminal and specifying additional parameters corresponding to the global variables described above.

### Testing

The **Interpretable TCCNet** (`itccnet`) can be tested with the `python3 test/tcc/test_itccnet.py` command. The file includes the following global variables that can be edited to configure the testing:

```python
# The random seed for reproducibility
RANDOM_SEED = 0

# Which type of model to train - Values: "att_tccnet", "conf_tccnet", "conf_att_tccnet"
MODEL_TYPE = "conf_tccnet"

# Which dataset slit to use - Values (TCC): "tcc_split", "fold_0", "fold_1", "fold_2"
DATA_FOLDER = "tcc_split"

# The path to the pretrained model to be used for testing
PATH_TO_PTH = os.path.join("trained_models")

# Which attention/confidence module should be deactivated - Values: "spat", "temp", empty string
DEACTIVATE = ""

# The size in the hidden layer of the ConvLSTM module
HIDDEN_SIZE = 128

# The kernel size of the convolutions in the ConvLSTM module
KERNEL_SIZE = 5

# Whether to save the predictions made during testing
SAVE_PRED = False

# Whether to save the attention scored generated during testing
SAVE_ATT = True
```
