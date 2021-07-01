import os
from time import time

import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader

from LossTracker import LossTracker
from auxiliary.settings import DEVICE, make_deterministic
from auxiliary.utils import print_val_metrics
from classes.core.Evaluator import Evaluator
from classes.data.datasets.TCC import TCC
from classes.modules.multiframe.att_tccnet.ModelAttTCCNet import ModelAttTCCNet
from classes.modules.multiframe.conf_att_tccnet.ModelConfAttTCCNet import ModelConfAttTCCNet
from classes.modules.multiframe.conf_tccnet.ModelConfTCCNet import ModelConfTCCNet

# ----------------------------------------------------------------------------------------------------------------

RANDOM_SEED = 0

MODEL_TYPE = "att_tccnet"
DATA_FOLDER = "tcc_split"
# PATH_TO_PTH = os.path.join("trained_models", "tcc", MODEL_TYPE, DATA_FOLDER)
PATH_TO_PTH = "trained_models/full_seq/spatiotemporal/att_tccnet/tcc_split"

HIDDEN_SIZE = 128
KERNEL_SIZE = 5
DEACTIVATE = ""

SAVE_PRED = True
SAVE_ATT = True
USE_TRAINING_SET = True

# ----------------------------------------------------------------------------------------------------------------

MODELS = {"att_tccnet": ModelAttTCCNet, "conf_tccnet": ModelConfTCCNet, "conf_att_tccnet": ModelConfAttTCCNet}


# ----------------------------------------------------------------------------------------------------------------

def main():
    path_to_pred, path_to_spat_att, path_to_temp_att = None, None, None

    if SAVE_PRED:
        path_to_pred = os.path.join("test", "pred", "{}_{}".format("train" if USE_TRAINING_SET else "test", time()))
        print("\n Saving predictions at {}".format(path_to_pred))
        os.makedirs(path_to_pred)

    if SAVE_ATT:
        path_to_att = os.path.join("test", "att", "{}_{}".format("train" if USE_TRAINING_SET else "test", time()))
        print("\n Saving attention weights at {}".format(path_to_att))

        path_to_spat_att = os.path.join(path_to_att, "spatial")
        os.makedirs(path_to_spat_att)

        path_to_temp_att = os.path.join(path_to_att, "temporal")
        os.makedirs(path_to_temp_att)

    print("\n Loading data from '{}':".format(DATA_FOLDER))
    dataset = TCC(mode="train" if USE_TRAINING_SET else "test", data_folder=DATA_FOLDER)
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=8)
    dataset_size = len(dataset)
    print("\n -> Data loaded! Dataset size is {}".format(dataset_size))

    model = MODELS[MODEL_TYPE](HIDDEN_SIZE, KERNEL_SIZE, DEACTIVATE)
    path_to_pth = os.path.join(PATH_TO_PTH, "model.pth")
    print('\n Reloading pretrained model stored at: {} \n'.format(path_to_pth))
    model.load(path_to_pth)
    model.evaluation_mode()

    print("\n--------------------------------------------------------------")
    print("\t\t Testing Model '{}'".format(MODEL_TYPE))
    print("--------------------------------------------------------------\n")

    evaluator = Evaluator()
    test_loss = LossTracker()
    start = time()

    with torch.no_grad():

        for i, (x, m, y, path_to_seq) in enumerate(dataloader):
            file_name = path_to_seq[0].split(os.sep)[-1]
            x, m, y = x.to(DEVICE), m.to(DEVICE), y.to(DEVICE)

            pred, spat_att, temp_att = model.predict(x, m, return_steps=True)
            loss = model.get_loss(pred, y).item()
            test_loss.update(loss)
            evaluator.add_error(loss)

            print("- {}/{} [ item: {} | AE: {:.4f} ]".format(i, dataset_size, file_name, loss))

            if SAVE_PRED:
                np.save(os.path.join(path_to_pred, file_name), pred)
            if SAVE_ATT:
                np.save(os.path.join(path_to_spat_att, file_name), spat_att)
                np.save(os.path.join(path_to_temp_att, file_name), temp_att)

    test_time = time() - start

    metrics = evaluator.compute_metrics()
    print("\n********************************************************************")
    print("....................................................................")
    print(" Test Time ... : {:.4f}".format(test_time))
    print(" Test Loss ... : {:.4f}".format(test_loss.avg))
    print("....................................................................")
    print_val_metrics(metrics, metrics)
    print("********************************************************************\n")


if __name__ == '__main__':
    make_deterministic(RANDOM_SEED)
    main()
