import argparse
import configparser
from seamseg.data.dataset import ISSTestDataset

import torch
from seamseg.data import (
    ISSTransform,
    ISSTestTransform,
    ISSDataset,
    ResultDataset,
    MapillaryToTarget,
)
from seamseg.utils.panoptic import panoptic_stats, PanopticPreprocessing

parser = argparse.ArgumentParser(
    description="Simple Script to measure panotic quality between a ground truth set and a testset"
)
parser.add_argument("--gt", help="Path to the ground truth root data dir", type=str)
parser.add_argument(
    "--result", help="Path the results of the source folder of the results", type=str
)
parser.add_argument(
    "--conf", help="Path to the config file to read in the configs for panoptic", type=str
)


def main(args):
    # config = configparser.ConfigParser()
    # config.read(args.conf)

    transform = ISSTransform(shortest_size=512, longest_max_size=1024)

    trans_map_vista = MapillaryToTarget(void_value=255)

    gt_dataset = ISSDataset(args.gt, "val", transform)
    num_stuff = gt_dataset.num_stuff
    num_classes = gt_dataset.num_categories

    res_dataset = ResultDataset(args.result, transform=trans_map_vista, file_suffix="_leftImg8bit")

    # Iterate over the whole dataset
    for i in range(len(gt_dataset)):

        gt_out = gt_dataset[i]
        msk_gt = gt_out["msk"].cpu()
        cat_gt = gt_out["cat"].cpu()
        iscrowd = gt_out["iscrowd"].cpu()

        msk_gt = msk_gt.squeeze(0)
        sem_gt = cat_gt[msk_gt]
        cmap = msk_gt.new_zeros(cat_gt.numel())
        cmap[~iscrowd] = torch.arange(
            0, (~iscrowd).long().sum().item(), dtype=cmap.dtype, device=cmap.device
        )
        msk_gt = cmap[msk_gt]
        cat_gt = cat_gt[~iscrowd]

        res_out = res_dataset[gt_out["idx"]]
        panoptic_preprocessing = PanopticPreprocessing()
        panoptic_out = panoptic_preprocessing(
            res_out["sem_pred"].cpu(),
            res_out["bbx_pred"].cpu(),
            res_out["cls_pred"].cpu(),
            res_out["obj_pred"].cpu(),
            res_out["msk_pred"].cpu(),
            num_stuff,
        )

        stats = panoptic_stats(msk_gt, cat_gt, panoptic_out, num_classes, num_stuff)

    print(len(panoptic_out))


if __name__ == "__main__":
    main(parser.parse_args())
