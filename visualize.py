import os
import time

import matplotlib.pyplot as plt
import torch.utils.data
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from auxiliary.settings import DEVICE
from auxiliary.utils import correct, rescale, scale
from classes.data.datasets.TemporalColorConstancy import TemporalColorConstancy
from multiframe.conf_tccnet.ModelConfTCCNet import ModelSTAttTCCNet
from classes.training.Evaluator import Evaluator

MODEL_TYPE = "sta_tccnet"
DATA_FOLDER = "tcc_split"

# Where to save the generated visualizations
PATH_TO_SAVED = os.path.join("vis", "{}_{}_test_set_{}".format(MODEL_TYPE, DATA_FOLDER, time.time()))

# Where to find the pretrained model to be tested
PATH_TO_PTH = os.path.join("trained_models", "full_seq", MODEL_TYPE, DATA_FOLDER, "model.pth")

# Set to -1 to process all the samples in the test set of the current fold
NUM_SAMPLES = -1

MODELS = {"sta_tccnet": ModelSTAttTCCNet}


def main():
    test_set = TemporalColorConstancy(mode="test", data_folder=DATA_FOLDER)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=20)
    print('Test set size: {}'.format(len(test_set)))

    model = MODELS[MODEL_TYPE]()

    if os.path.exists(PATH_TO_PTH):
        print('\n Loading pretrained {} model stored at: {} \n'.format(MODEL_TYPE, PATH_TO_PTH))
        model.load(PATH_TO_PTH)
    else:
        raise ValueError("No pretrained {} model found at {}".format(MODEL_TYPE, PATH_TO_PTH))

    print("\n *** Testing model {} on {} *** \n".format(MODEL_TYPE, DATA_FOLDER))

    evaluator = Evaluator()
    model.evaluation_mode()

    with torch.no_grad():
        for i, (x, m, y, file_name) in enumerate(test_loader):
            x, m, y = x.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
            pred, rgb, confidence = model.predict(x, m)
            loss = model.get_angular_loss(pred, y).item()
            evaluator.add_error(loss)

            file_name = file_name[0].split(os.sep)[-1].split(".")[0]
            print("Item {}: {}, AE: {:.4f}".format(i, file_name, loss))

            for j, frame in enumerate(x.squeeze(0)):
                original = transforms.ToPILImage()(frame.squeeze()).convert("RGB")
                gt_corrected, est_corrected = correct(original, y), correct(original, pred)

                size = original.size[::-1]

                scaled_rgb = rescale(rgb, size).squeeze(0).permute(1, 2, 0)
                scaled_confidence = rescale(confidence, size).squeeze(0).permute(1, 2, 0)

                weighted_est = scale(rgb * confidence)
                scaled_weighted_est = rescale(weighted_est, size).squeeze().permute(1, 2, 0)

                masked_original = scale(F.to_tensor(original).permute(1, 2, 0) * scaled_confidence)

                fig, axs = plt.subplots(2, 3)

                axs[0, 0].imshow(original)
                axs[0, 0].set_title("Original")
                axs[0, 0].axis("off")

                axs[0, 1].imshow(masked_original, cmap="gray")
                axs[0, 1].set_title("Confidence Mask")
                axs[0, 1].axis("off")

                axs[0, 2].imshow(est_corrected)
                axs[0, 2].set_title("Correction")
                axs[0, 2].axis("off")

                axs[1, 0].imshow(scaled_rgb)
                axs[1, 0].set_title("Per-patch Estimate")
                axs[1, 0].axis("off")

                axs[1, 1].imshow(scaled_confidence, cmap="gray")
                axs[1, 1].set_title("Confidence")
                axs[1, 1].axis("off")

                axs[1, 2].imshow(scaled_weighted_est)
                axs[1, 2].set_title("Weighted Estimate")
                axs[1, 2].axis("off")

                fig.suptitle("Seq ID: {} | Error: {:.4f}".format(file_name, loss))
                fig.tight_layout(pad=0.25)

                path_to_save = os.path.join(PATH_TO_SAVED, file_name)
                os.makedirs(path_to_save)

                fig.savefig(os.path.join(path_to_save, "stages.png"), bbox_inches='tight', dpi=200)
                original.save(os.path.join(path_to_save, "original.png"))
                est_corrected.save(os.path.join(path_to_save, "est_corrected.png"))
                gt_corrected.save(os.path.join(path_to_save, "gt_corrected.png"))

                plt.clf()

    metrics = evaluator.compute_metrics()
    print("\n Mean ............ : {}".format(metrics["mean"]))
    print(" Median .......... : {}".format(metrics["median"]))
    print(" Trimean ......... : {}".format(metrics["trimean"]))
    print(" Best 25% ........ : {}".format(metrics["bst25"]))
    print(" Worst 25% ....... : {}".format(metrics["wst25"]))
    print(" Percentile 95 ... : {} \n".format(metrics["wst5"]))


if __name__ == '__main__':
    main()
