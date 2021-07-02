import argparse
import os
from time import time

import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader

from auxiliary.settings import DEVICE, make_deterministic
from auxiliary.utils import print_val_metrics
from classes.core.Evaluator import Evaluator
from classes.core.LossTracker import LossTracker
from classes.data.datasets.TCC import TCC
from classes.modules.multiframe.att_tccnet.ModelAttTCCNet import ModelAttTCCNet
from classes.modules.multiframe.conf_att_tccnet.ModelConfAttTCCNet import ModelConfAttTCCNet
from classes.modules.multiframe.conf_tccnet.ModelConfTCCNet import ModelConfTCCNet

# ----------------------------------------------------------------------------------------------------------------

RANDOM_SEED = 0

MODEL_TYPE = "att_tccnet"
DATA_FOLDER = "tcc_split"
PATH_TO_PTH = os.path.join("trained_models", "full_seq", "spatiotemporal")

HIDDEN_SIZE = 128
KERNEL_SIZE = 5
DEACTIVATE = ""

SAVE_PRED = True
SAVE_ATT = True
USE_TRAINING_SET = False

# ----------------------------------------------------------------------------------------------------------------

MODELS = {"att_tccnet": ModelAttTCCNet, "conf_tccnet": ModelConfTCCNet, "conf_att_tccnet": ModelConfAttTCCNet}


# ----------------------------------------------------------------------------------------------------------------

def main(opt):
    model_type, data_folder, path_to_pth = opt.model_type, opt.data_folder, opt.path_to_pth
    hidden_size, kernel_size, deactivate = opt.hidden_size, opt.kernel_size, opt.deactivate
    save_pred, save_att, use_training_set = opt.save_pred, opt.save_att, opt.use_training_set

    path_to_pred, path_to_spat_att, path_to_temp_att = None, None, None

    if save_pred:
        path_to_pred = os.path.join("test", "logs", "{}_{}_{}_pred".format(model_type, data_folder, time()))
        print("\n Saving predictions at {}".format(path_to_pred))
        os.makedirs(path_to_pred)

    if save_att:
        path_to_att = os.path.join("test", "logs", "{}_{}_{}_att".format(model_type, data_folder, time()))
        print("\n Saving attention weights at {}".format(path_to_att))

        path_to_spat_att = os.path.join(path_to_att, "spatial")
        os.makedirs(path_to_spat_att)

        path_to_temp_att = os.path.join(path_to_att, "temporal")
        os.makedirs(path_to_temp_att)

    print("\n Loading data from '{}':".format(data_folder))
    dataset = TCC(mode="train" if use_training_set else "test", data_folder=data_folder)
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=8)
    dataset_size = len(dataset)
    print("\n -> Data loaded! Dataset size is {}".format(dataset_size))

    model = MODELS[model_type](hidden_size, kernel_size, deactivate)
    path_to_pth = os.path.join(path_to_pth, "model.pth")
    print('\n Reloading pretrained model stored at: {} \n'.format(path_to_pth))
    model.load(path_to_pth)
    model.evaluation_mode()

    print("\n------------------------------------------------------------------------------------------")
    print("\t\t Testing Model '{}'".format(model_type))
    print("------------------------------------------------------------------------------------------\n")

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

            if save_pred:
                np.save(os.path.join(path_to_pred, file_name), pred.numpy())
            if save_att:
                np.save(os.path.join(path_to_spat_att, file_name), spat_att.numpy())
                np.save(os.path.join(path_to_temp_att, file_name), temp_att.numpy())

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default=MODEL_TYPE)
    parser.add_argument('--data_folder', type=str, default=DATA_FOLDER)
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--hidden_size', type=int, default=HIDDEN_SIZE)
    parser.add_argument('--kernel_size', type=int, default=KERNEL_SIZE)
    parser.add_argument('--deactivate', type=str, default=DEACTIVATE)
    parser.add_argument('--save_pred', type=bool, default=SAVE_PRED)
    parser.add_argument('--save_att', type=bool, default=SAVE_ATT)
    parser.add_argument('--use_training_set', type=bool, default=USE_TRAINING_SET)
    parser.add_argument('--path_to_pth', type=str, default=PATH_TO_PTH)
    opt = parser.parse_args()

    opt.path_to_pth = os.path.join(opt.path_to_pth, opt.model_type, opt.data_folder)

    print("\n *** Test configuration ***")
    print("\t Model type ......... : {}".format(opt.model_type))
    print("\t Data folder ........ : {}".format(opt.data_folder))
    print("\t Random seed ........ : {}".format(opt.random_seed))
    print("\t Hidden size ........ : {}".format(opt.hidden_size))
    print("\t Kernel size ........ : {}".format(opt.kernel_size))
    print("\t Deactivate ......... : {}".format(opt.deactivate))
    print("\t Save predictions ... : {}".format(opt.save_pred))
    print("\t Save attention ..... : {}".format(opt.save_att))
    print("\t Use training set ... : {}".format(opt.use_training_set))
    print("\t Path to PTH ........ : {}".format(opt.path_to_pth))

    make_deterministic(opt.random_seed)
    main(opt)
