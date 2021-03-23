import os
import time

import torch.utils.data
from torch.utils.data import DataLoader

from auxiliary.settings import DEVICE
from auxiliary.utils import log_experiment, log_metrics, print_metrics, log_time
from classes.data.datasets.TemporalColorConstancy import TemporalColorConstancy
from classes.modules.multiframe.conf_tccnet.ModelConfTCCNet import ModelConfTCCNet
from classes.training.Evaluator import Evaluator
from classes.training.LossTracker import LossTracker

MODEL_TYPE = "conf_tccnet"
DATA_FOLDER = "tcc_split"
EPOCHS = 2000
BATCH_SIZE = 1
LEARNING_RATE = 0.00003

# The ids of the sequences to be monitored (e.g., ["0", "1", "2"])
TEST_VIS_IMG = []

RELOAD_CHECKPOINT = False
PATH_TO_PTH_CHECKPOINT = os.path.join("trained_models", "{}_{}".format(MODEL_TYPE, DATA_FOLDER), "model.pth")


def main():
    evaluator = Evaluator()

    path_to_log = os.path.join("train", "tcc", "logs", "", "{}_{}_{}".format(MODEL_TYPE, DATA_FOLDER, + time.time()))
    os.makedirs(path_to_log)

    path_to_metrics_log = os.path.join(path_to_log, "metrics.csv")
    path_to_experiment_log = os.path.join(path_to_log, "experiment.json")
    log_experiment(MODEL_TYPE, DATA_FOLDER, LEARNING_RATE, path_to_experiment_log)

    path_to_vis = os.path.join(path_to_log, "test_vis")
    if TEST_VIS_IMG:
        print("Test vis for monitored sequences {} will be saved at {}\n".format(TEST_VIS_IMG, path_to_vis))
        os.makedirs(path_to_vis)

    print("\nLoading data from '{}':".format(DATA_FOLDER))

    training_set = TemporalColorConstancy(mode="train", data_folder=DATA_FOLDER)
    train_loader = DataLoader(dataset=training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    test_set = TemporalColorConstancy(mode="test", data_folder=DATA_FOLDER)
    test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, num_workers=8)

    training_set_size, test_set_size = len(training_set), len(test_set)
    print("Training set size: ... {}".format(training_set_size))
    print("Test set size: ....... {}\n".format(test_set_size))

    model = ModelConfTCCNet()

    if RELOAD_CHECKPOINT:
        print('\n Reloading checkpoint - pretrained model stored at: {} \n'.format(PATH_TO_PTH_CHECKPOINT))
        model.load(PATH_TO_PTH_CHECKPOINT)

    model.print_network()
    model.log_network(path_to_log)
    model.set_optimizer(learning_rate=LEARNING_RATE)

    best_val_loss, best_metrics = 100.0, evaluator.get_best_metrics()
    train_loss, val_loss = LossTracker(), LossTracker()

    for epoch in range(EPOCHS):

        print("\n--------------------------------------------------------------")
        print("\t\t Training epoch {}/{}".format(epoch + 1, EPOCHS))
        print("--------------------------------------------------------------\n")

        model.train_mode()
        train_loss.reset()
        start = time.time()

        for i, (x, m, y, file_name) in enumerate(train_loader):

            model.reset_gradient()
            x, m, y = x.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
            loss = model.compute_loss(x, y, m)
            model.optimize()

            train_loss.update(loss)

            if i % 5 == 0:
                print("[ Epoch: {}/{} - Batch: {}/{} ] | [ Train loss: {:.4f} ]"
                      .format(epoch + 1, EPOCHS, i + 1, training_set_size, loss))

        train_time = time.time() - start
        log_time(time=train_time, time_type="train", path_to_log=path_to_experiment_log)

        val_loss.reset()
        start = time.time()

        if epoch % 5 == 0:

            print("\n--------------------------------------------------------------")
            print("\t\t Validation")
            print("--------------------------------------------------------------\n")

            with torch.no_grad():

                model.evaluation_mode()
                evaluator.reset_errors()

                for i, (x, m, y, file_name) in enumerate(test_loader):

                    sequence_id = file_name[0].split(".")[0]
                    x, m, y = x.to(DEVICE), m.to(DEVICE), y.to(DEVICE)

                    pred, rgb, confidence = model.predict(x, m, return_steps=True)
                    loss = model.get_loss(pred, y)
                    val_loss.update(loss)
                    evaluator.add_error(loss)

                    if sequence_id in TEST_VIS_IMG:
                        model.vis_confidence({"x": x, "y": y, "pred": pred, "rgb": rgb, "c": confidence},
                                             os.path.join(path_to_vis, sequence_id, "epoch_{}.png".format(epoch)))

                    if i % 5 == 0:
                        print("[ Epoch: {}/{} - Batch: {}/{}] | Val loss: {:.4f} ]"
                              .format(epoch, EPOCHS, i, test_set_size, loss))

            print("\n--------------------------------------------------------------\n")

        val_time = time.time() - start
        log_time(time=val_time, time_type="val", path_to_log=path_to_experiment_log)

        metrics = evaluator.compute_metrics()
        print("\n********************************************************************")
        print(" Train Time ... : {:.4f}".format(train_time))
        print(" Train Loss ... : {:.4f}".format(train_loss.avg))
        if val_time > 0.1:
            print("....................................................................")
            print(" Val Time ..... : {:.4f}".format(val_time))
            print(" Val Loss ..... : {:.4f}".format(val_loss.avg))
            print("....................................................................")
            print_metrics(metrics, best_metrics)
        print("********************************************************************\n")

        if 0 < val_loss.avg < best_val_loss:
            best_val_loss = val_loss.avg
            best_metrics = evaluator.update_best_metrics()
            print("Saving new best model... \n")
            model.save(os.path.join(path_to_log, "model.pth"))

        log_metrics(train_loss.avg, val_loss.avg, metrics, best_metrics, path_to_metrics_log)


if __name__ == '__main__':
    main()
