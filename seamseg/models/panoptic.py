from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from seamseg.utils.sequence import pad_packed_images

NETWORK_INPUTS = ["img", "msk", "cat", "iscrowd", "bbx"]


class PanopticNet(nn.Module):
    def __init__(
        self,
        body,
        rpn_head,
        roi_head,
        sem_head,
        rpn_algo,
        instance_seg_algo,
        semantic_seg_algo,
        classes,
        extra_resolutions=[],
    ):
        super(PanopticNet, self).__init__()
        self.num_stuff = classes["stuff"]
        self.extra_resolutions = extra_resolutions

        # Modules
        self.body = body
        self.rpn_head = rpn_head
        self.roi_head = roi_head
        self.sem_head = sem_head

        # Algorithms
        self.rpn_algo = rpn_algo
        self.instance_seg_algo = instance_seg_algo
        self.semantic_seg_algo = semantic_seg_algo

    def _prepare_inputs(self, msk, cat, iscrowd, bbx):
        cat_out, iscrowd_out, bbx_out, ids_out, sem_out = [], [], [], [], []
        for msk_i, cat_i, iscrowd_i, bbx_i in zip(msk, cat, iscrowd, bbx):
            msk_i = msk_i.squeeze(0)
            thing = (cat_i >= self.num_stuff) & (cat_i != 255)
            valid = thing & ~iscrowd_i

            if valid.any().item():
                cat_out.append(cat_i[valid])
                bbx_out.append(bbx_i[valid])
                ids_out.append(torch.nonzero(valid))
            else:
                cat_out.append(None)
                bbx_out.append(None)
                ids_out.append(None)

            if iscrowd_i.any().item():
                iscrowd_i = iscrowd_i & thing
                iscrowd_out.append(iscrowd_i[msk_i])
            else:
                iscrowd_out.append(None)

            sem_out.append(cat_i[msk_i])

        return cat_out, iscrowd_out, bbx_out, ids_out, sem_out

    def forward(
        self, img, msk=None, cat=None, iscrowd=None, bbx=None, do_loss=False, do_prediction=True
    ):
        # Pad the input images
        img, valid_size = pad_packed_images(img)
        base_size = img.shape[-2:]
        img_sizes = []
        for extra_res in self.extra_resolutions:
            img_sizes.append((base_size[0], extra_res))

        images = [F.interpolate(img, size=extra_size) for extra_size in img_sizes]
        images.insert(0, img)
        img_sizes.insert(0, base_size)
        # Convert ground truth to the internal format
        if do_loss:
            cat, iscrowd, bbx, ids, sem = self._prepare_inputs(msk, cat, iscrowd, bbx)

        # Resize the image to three different resolutions

        # Run network body
        x = [self.body(i) for i in images]

        # RPN part
        if do_loss:
            obj_loss, bbx_loss, proposals = self.rpn_algo.training(
                self.rpn_head,
                x[0],
                bbx,
                iscrowd,
                valid_size,
                training=self.training,
                do_inference=True,
            )
        elif do_prediction:
            proposals = self.rpn_algo.inference(self.rpn_head, x[0], valid_size, self.training)
            obj_loss, bbx_loss = None, None
        else:
            obj_loss, bbx_loss, proposals = None, None, None

        # ROI part
        if do_loss:
            roi_cls_loss, roi_bbx_loss, roi_msk_loss = self.instance_seg_algo.training(
                self.roi_head, x[0], proposals, bbx, cat, iscrowd, ids, msk, img_sizes[0]
            )
        else:
            roi_cls_loss, roi_bbx_loss, roi_msk_loss = None, None, None
        if do_prediction:
            bbx_pred, cls_pred, obj_pred, msk_pred = self.instance_seg_algo.inference(
                self.roi_head, x[0], proposals, valid_size, img_sizes[0]
            )
        else:
            bbx_pred, cls_pred, obj_pred, msk_pred = None, None, None, None

        # Segmentation part
        if do_loss:
            sem_loss, conf_mat, sem_pred = self.semantic_seg_algo.training(
                self.sem_head, x[0], sem, valid_size, img_sizes[0]
            )
        elif do_prediction:
            sem_pred = self.semantic_seg_algo.inference(self.sem_head, x, valid_size, img_sizes)
            sem_loss, conf_mat = None, None
        else:
            sem_loss, conf_mat, sem_pred = None, None, None

        # Prepare outputs
        loss = OrderedDict(
            [
                ("obj_loss", obj_loss),
                ("bbx_loss", bbx_loss),
                ("roi_cls_loss", roi_cls_loss),
                ("roi_bbx_loss", roi_bbx_loss),
                ("roi_msk_loss", roi_msk_loss),
                ("sem_loss", sem_loss),
            ]
        )
        pred = OrderedDict(
            [
                ("bbx_pred", bbx_pred),
                ("cls_pred", cls_pred),
                ("obj_pred", obj_pred),
                ("msk_pred", msk_pred),
                ("sem_pred", sem_pred),
            ]
        )
        conf = OrderedDict([("sem_conf", conf_mat)])
        return loss, pred, conf
