# Copyright (c) Megvii Inc. All rights reserved.

import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou, cxcywh2xyxy, meshgrid, visualize_assign

from .losses import IouLoss
from .network_blocks import BaseConv, DWConv


class YoloxPoseHead(nn.Module):
    def __init__(
        self,
        num_kpts,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()
        self.num_classes = 1
        self.num_kpts = num_kpts
        self.decode_in_inference = True  # for deploy, set to False
        #self.kpt_vis_conf_thr = 0.0

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.kpt_convs = nn.ModuleList()

        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.kpt_cls_preds = nn.ModuleList()
        self.kpt_regr_preds = nn.ModuleList()

        self.stems = nn.ModuleList()

        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.kpt_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )

            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.kpt_cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.num_kpts,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.kpt_regr_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=2*self.num_kpts,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IouLoss(reduction="none")

        # Keypoint specific losses
        self.kpt_regr_loss = nn.L1Loss(reduction="none") # L1 for keypoint regression
        self.kpt_vis_loss = nn.BCEWithLogitsLoss(reduction="none") # BCE for visibility classification

        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        # Define output channel counts for clarity
        reg_ch = 4
        obj_ch = 1
        cls_ch = self.num_classes
        kpt_vis_ch = self.num_kpts
        kpt_regr_ch = 2 * self.num_kpts
        # Total channels per anchor prediction
        # Order: [reg(4), obj(1), cls(1), kpt_vis(Nk), kpt_regr(2*Nk)]
        self.n_ch_pred = reg_ch + obj_ch + cls_ch + kpt_vis_ch + kpt_regr_ch

        for k, (cls_conv, reg_conv, kpt_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.kpt_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x
            kpt_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            kpt_feat = kpt_conv(kpt_x)
            kpt_cls_output = self.kpt_cls_preds[k](kpt_feat)
            kpt_regr_output = self.kpt_regr_preds[k](kpt_feat)

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output, kpt_cls_output, kpt_regr_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, 1, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid(), kpt_cls_output.sigmoid(), kpt_regr_output], 1
                )

            outputs.append(output)

        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        # output shape: [B, N_CH_PRED, H, W]
        grid = self.grids[k]
        batch_size = output.shape[0]
        # n_ch = 5 + self.num_classes + self.num_kpts * 3 # Old calculation
        n_ch = self.n_ch_pred # Use the stored total channel count
        hsize, wsize = output.shape[-2:]

        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid # Cache grid

        # Reshape output: [B, N_CH_PRED, H, W] -> [B, H*W, N_CH_PRED]
        output = output.flatten(start_dim=2).permute(0, 2, 1)
        # Reshape grid: [1, 1, H, W, 2] -> [1, H*W, 2]
        grid = grid.view(1, -1, 2)

        # Decode bounding box predictions (relative to grid cell and stride)
        # output[:, :, :2] are cx, cy offsets
        # output[:, :, 2:4] are log(w), log(h) offsets
        output[..., :2] = (output[..., :2] + grid) * stride # Decode cx, cy
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride # Decode w, h

        return output, grid

    def decode_outputs(self, outputs, dtype):
        # outputs shape: [B, N_ANCHORS_ALL, N_CH_PRED]
        # This function assumes get_output_and_grid was *not* called during inference forward pass
        # It decodes the raw network outputs here.
        # If get_output_and_grid *was* called, this function needs adjustment or removal.
        # Based on the original YOLOX structure, decoding happens here if decode_in_inference=True.

        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2) # [1, H*W, 2]
            grids.append(grid)
            shape = grid.shape[:2] # [1, H*W]
            strides.append(torch.full((*shape, 1), stride)) # [1, H*W, 1]

        grids = torch.cat(grids, dim=1).type(dtype) # [1, N_ANCHORS_ALL, 2]
        strides = torch.cat(strides, dim=1).type(dtype) # [1, N_ANCHORS_ALL, 1]

        # Decode Boxes: (pred_offset_xy + grid_xy) * stride, exp(pred_log_wh) * stride
        box_cxcy = (outputs[..., 0:2] + grids) * strides
        box_wh = torch.exp(outputs[..., 2:4]) * strides
        decoded_boxes = torch.cat((box_cxcy, box_wh), dim=-1) # [B, N_ANCHORS_ALL, 4]

        # Decode Keypoints: (pred_offset_xy + grid_xy) * stride
        # Reshape kpt predictions: [B, N_ANCHORS_ALL, 2*Nk] -> [B, N_ANCHORS_ALL, Nk, 2]
        kpt_preds = outputs[..., -2*self.num_kpts:].view(outputs.shape[0], -1, self.num_kpts, 2)
        # Expand grid and strides for keypoints
        # grids: [1, N_ANCHORS_ALL, 2] -> [1, N_ANCHORS_ALL, Nk, 2]
        # strides: [1, N_ANCHORS_ALL, 1] -> [1, N_ANCHORS_ALL, Nk, 1]
        grids_expanded = grids.unsqueeze(2).repeat(1, 1, self.num_kpts, 1)
        strides_expanded = strides.unsqueeze(2).repeat(1, 1, self.num_kpts, 1)
        decoded_kpts = (kpt_preds + grids_expanded) * strides_expanded
        # Reshape back: [B, N_ANCHORS_ALL, Nk, 2] -> [B, N_ANCHORS_ALL, 2*Nk]
        decoded_kpts = decoded_kpts.view(outputs.shape[0], -1, 2*self.num_kpts)

        # Concatenate decoded boxes, obj, cls, kpt_vis, decoded kpts
        # Note: obj, cls, kpt_vis were already sigmoided in the forward pass for inference
        outputs = torch.cat([
            decoded_boxes, # 4
            outputs[..., 4:5], # obj
            outputs[..., 5:6], # cls
            outputs[..., 6:6+self.num_kpts], # kpt_vis
            decoded_kpts # 2*Nk
        ], dim=-1)

        return outputs

    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels, # Shape: [B, max_obj, 5 + 3*Nk]
        outputs, # Shape: [B, N_ANCHORS_ALL, N_CH_PRED] (Decoded Boxes, Raw Kpt Offsets)
        origin_preds, # List of raw box preds per level, if self.use_l1
        dtype,
    ):
        # --- Extract Predictions ---
        # outputs contains: Decoded Boxes [cx,cy,w,h], Raw Logits [obj, cls, kpt_vis], Raw Offsets [kpt_regr]
        bbox_preds_decoded = outputs[:, :, :4]          # [B, N_anchors, 4] (decoded cx, cy, w, h)
        obj_preds = outputs[:, :, 4:5]                  # [B, N_anchors, 1] (logits)
        cls_preds = outputs[:, :, 5:6]                  # [B, N_anchors, N_cls] (logits)
        kpt_vis_preds = outputs[:, :, 6:6+self.num_kpts] # [B, N_anchors, Nk] (logits)
        # Directly use the raw keypoint regression offsets from the output tensor
        kpt_regr_preds_raw = outputs[:, :, -2*self.num_kpts:] # [B, N_anchors, 2*Nk] (raw offsets)

        # --- Calculate Targets ---
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of valid objects per image [B]

        total_num_anchors = outputs.shape[1]
        # Flatten shifts and strides for assignment
        x_shifts_flat = torch.cat(x_shifts, 1)  # [1, N_anchors]
        y_shifts_flat = torch.cat(y_shifts, 1)  # [1, N_anchors]
        expanded_strides_flat = torch.cat(expanded_strides, 1) # [1, N_anchors]
        if self.use_l1:
            # origin_preds contains the raw box offsets from the network conv layers
            origin_preds = torch.cat(origin_preds, 1) # [B, N_anchors, 4] raw box offsets

        cls_targets = []
        reg_targets = [] # GT boxes [cx, cy, w, h] for IoU loss
        l1_targets = [] # Target box offsets for L1 loss
        obj_targets = []
        fg_masks = []
        # New targets for keypoints
        kpt_regr_targets = [] # Target keypoint offsets
        kpt_vis_targets = []
        kpt_loss_masks = [] # Mask for calculating regression loss (only for visible GT kpts)

        num_fg = 0.0 # Total foreground anchors across batch
        num_gts = 0.0 # Total ground truth objects across batch

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
                kpt_regr_target = outputs.new_zeros((0, 2 * self.num_kpts))
                kpt_vis_target = outputs.new_zeros((0, self.num_kpts))
                kpt_loss_mask = outputs.new_zeros((0, self.num_kpts))
            else:
                # Extract GT for this image
                gt_labels_per_image = labels[batch_idx, :num_gt, :] # [N_gt, 5 + 3*Nk]
                gt_classes = gt_labels_per_image[:, 0]             # [N_gt]
                gt_bboxes_per_image = gt_labels_per_image[:, 1:5]   # [N_gt, 4] (cx, cy, w, h)
                gt_kpts_per_image = gt_labels_per_image[:, 5:].view(num_gt, self.num_kpts, 3) # [N_gt, Nk, 3] (x, y, vis)

                # Extract predictions for this image (needed for assignment)
                # Use the decoded box predictions for assignment
                bboxes_preds_per_image = bbox_preds_decoded[batch_idx] # [N_anchors, 4] (decoded)
                obj_preds_per_image = obj_preds[batch_idx]     # [N_anchors, 1] (logits)
                cls_preds_per_image = cls_preds[batch_idx]     # [N_anchors, N_cls] (logits)

                try:
                    (
                        gt_matched_classes, fg_mask, pred_ious_this_matching,
                        matched_gt_inds, num_fg_img,
                    ) = self.get_assignments(
                        batch_idx, num_gt, gt_bboxes_per_image, gt_classes,
                        bboxes_preds_per_image, # Use decoded boxes for assignment
                        expanded_strides_flat, x_shifts_flat, y_shifts_flat,
                        cls_preds, obj_preds,
                    )
                except RuntimeError as e:
                    if "CUDA out of memory. " not in str(e): raise
                    logger.error("OOM during label assignment. Using CPU fallback.")
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes, fg_mask, pred_ious_this_matching,
                        matched_gt_inds, num_fg_img,
                    ) = self.get_assignments(
                        batch_idx, num_gt, gt_bboxes_per_image, gt_classes,
                        bboxes_preds_per_image, expanded_strides_flat, x_shifts_flat, y_shifts_flat,
                        cls_preds_per_image, obj_preds_per_image, "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                # --- Standard Targets (Box, Obj, Cls) ---
                obj_target = fg_mask.unsqueeze(-1)
                # reg_target is for IoU loss, uses GT boxes directly
                reg_target = gt_bboxes_per_image[matched_gt_inds] # [N_fg, 4] (cx, cy, w, h)
                cls_target = F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes) * \
                             pred_ious_this_matching.unsqueeze(-1)

                if self.use_l1:
                    # l1_target is for box L1 loss, uses target offsets
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        reg_target, # Use matched GT boxes to calculate target offsets
                        expanded_strides_flat[0][fg_mask],
                        x_shifts=x_shifts_flat[0][fg_mask],
                        y_shifts=y_shifts_flat[0][fg_mask],
                    ) # [N_fg, 4] (target offsets for box L1 loss)

                # --- Keypoint Targets ---
                if num_fg_img > 0:
                    matched_gt_kpts = gt_kpts_per_image[matched_gt_inds] # [N_fg, Nk, 3]
                    gt_kpt_coords = matched_gt_kpts[..., :2] # [N_fg, Nk, 2]
                    gt_kpt_vis = matched_gt_kpts[..., 2]   # [N_fg, Nk]

                    # 1. Keypoint Visibility Target
                    kpt_vis_target = (gt_kpt_vis > 0).float() # [N_fg, Nk]

                    # 2. Keypoint Regression Target (Offsets relative to grid cell center)
                    # Calculation remains the same as before
                    x_shifts_fg = x_shifts_flat[0][fg_mask]
                    y_shifts_fg = y_shifts_flat[0][fg_mask]
                    strides_fg = expanded_strides_flat[0][fg_mask]
                    x_shifts_fg_exp = x_shifts_fg.unsqueeze(1).repeat(1, self.num_kpts)
                    y_shifts_fg_exp = y_shifts_fg.unsqueeze(1).repeat(1, self.num_kpts)
                    strides_fg_exp = strides_fg.unsqueeze(1).repeat(1, self.num_kpts)
                    target_kpt_x_offset = (gt_kpt_coords[..., 0] / strides_fg_exp) - x_shifts_fg_exp
                    target_kpt_y_offset = (gt_kpt_coords[..., 1] / strides_fg_exp) - y_shifts_fg_exp
                    kpt_regr_target = torch.stack((target_kpt_x_offset, target_kpt_y_offset), dim=-1)\
                                         .view(num_fg_img, -1) # [N_fg, 2*Nk]

                    # 3. Keypoint Loss Mask
                    kpt_loss_mask = (gt_kpt_vis == 2).float() # [N_fg, Nk]
                else:
                    kpt_regr_target = outputs.new_zeros((0, 2 * self.num_kpts))
                    kpt_vis_target = outputs.new_zeros((0, self.num_kpts))
                    kpt_loss_mask = outputs.new_zeros((0, self.num_kpts))

            # Append targets
            cls_targets.append(cls_target)
            reg_targets.append(reg_target) # GT boxes for IoU loss
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target) # Target offsets for box L1 loss
            kpt_regr_targets.append(kpt_regr_target) # Target offsets for kpt loss
            kpt_vis_targets.append(kpt_vis_target)
            kpt_loss_masks.append(kpt_loss_mask)

        # --- Concatenate Targets Across Batch ---
        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0) # GT boxes [N_fg_total, 4]
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0) # Target box offsets [N_fg_total, 4]
        kpt_regr_targets = torch.cat(kpt_regr_targets, 0) # Target kpt offsets [N_fg_total, 2*Nk]
        kpt_vis_targets = torch.cat(kpt_vis_targets, 0)
        kpt_loss_masks = torch.cat(kpt_loss_masks, 0)

        # --- Calculate Losses ---
        num_fg = max(num_fg, 1)
        num_anchors_total = outputs.shape[0] * outputs.shape[1]

        # Select foreground predictions
        # Use decoded box predictions for IoU loss calculation
        bbox_preds_decoded_fg = bbox_preds_decoded.view(-1, 4)[fg_masks] # [N_fg_total, 4]
        obj_preds_all = obj_preds.view(-1, 1)
        cls_preds_fg = cls_preds.view(-1, self.num_classes)[fg_masks]
        kpt_vis_preds_fg = kpt_vis_preds.view(-1, self.num_kpts)[fg_masks]
        # Use raw keypoint regression predictions (offsets) for kpt regression loss
        kpt_regr_preds_raw_fg = kpt_regr_preds_raw.view(-1, 2 * self.num_kpts)[fg_masks] # [N_fg_total, 2*Nk]

        # --- Standard Losses ---
        # IoU loss uses decoded predictions vs GT boxes
        loss_iou = (self.iou_loss(bbox_preds_decoded_fg, reg_targets)).sum() / num_fg
        loss_obj = (self.bcewithlog_loss(obj_preds_all, obj_targets)).sum() / num_fg
        loss_cls = (self.bcewithlog_loss(cls_preds_fg, cls_targets)).sum() / num_fg

        if self.use_l1:
            # L1 box loss uses raw box offsets vs target box offsets
            origin_preds_fg = origin_preds.view(-1, 4)[fg_masks] # Raw box offsets for fg anchors
            loss_l1 = (self.l1_loss(origin_preds_fg, l1_targets)).sum() / num_fg
        else:
            loss_l1 = 0.0

        # --- Keypoint Losses ---
        loss_kpt_vis = 0.0
        loss_kpt_regr = 0.0
        if cls_targets.shape[0] > 0:
            # 1. Keypoint Visibility Loss (BCE) - uses logits
            loss_kpt_vis = (self.kpt_vis_loss(kpt_vis_preds_fg, kpt_vis_targets)).sum() / num_fg

            # 2. Keypoint Regression Loss (L1) - uses raw offsets
            kpt_loss_masks_expanded = kpt_loss_masks.repeat_interleave(2, dim=1)
            # Compare raw predicted offsets with target offsets
            loss_kpt_regr_unmasked = self.kpt_regr_loss(kpt_regr_preds_raw_fg, kpt_regr_targets)
            loss_kpt_regr = (loss_kpt_regr_unmasked * kpt_loss_masks_expanded).sum()
            num_visible_kpts = kpt_loss_masks.sum()
            loss_kpt_regr = loss_kpt_regr / max(num_visible_kpts, 1)

        # --- Total Loss ---
        reg_weight = 5.0
        kpt_loss_weight = 1.0
        kpt_vis_loss_weight = 1.0
        
        loss = reg_weight * loss_iou + \
               loss_obj + \
               loss_cls + \
               loss_l1 + \
               kpt_vis_loss_weight * loss_kpt_vis + \
               kpt_loss_weight * loss_kpt_regr

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            kpt_vis_loss_weight * loss_kpt_vis,
            kpt_loss_weight * loss_kpt_regr,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        obj_preds,
        mode="gpu",
    ):

        if mode == "cpu":
            print("-----------Using CPU for the Current Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, geometry_relation = self.get_geometry_constraint(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
        )
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()
            ).sqrt()
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + float(1e6) * (~geometry_relation)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_geometry_constraint(
        self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts,
    ):
        """
        Calculate whether the center of an object is located in a fixed range of
        an anchor. This is used to avert inappropriate matching. It can also reduce
        the number of candidate anchors so that the GPU memory is saved.
        """
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)

        # in fixed center
        center_radius = 1.5
        center_dist = expanded_strides_per_image.unsqueeze(0) * center_radius
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0:1]) - center_dist
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0:1]) + center_dist
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1:2]) - center_dist
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1:2]) + center_dist

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        anchor_filter = is_in_centers.sum(dim=0) > 0
        geometry_relation = is_in_centers[:, anchor_filter]

        return anchor_filter, geometry_relation

    def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        # deal with the case that one anchor matches multiple ground-truths
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[:, multiple_match_mask] *= 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1
        fg_mask_inboxes = anchor_matching_gt > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

    def visualize_assign_result(self, xin, labels=None, imgs=None, save_prefix="assign_vis_"):
        # original forward logic
        outputs, x_shifts, y_shifts, expanded_strides = [], [], [], []
        # TODO: use forward logic here.

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)
            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat([reg_output, obj_output, cls_output], 1)
            output, grid = self.get_output_and_grid(output, k, stride_this_level, xin[0].type())
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(
                torch.full((1, grid.shape[1]), stride_this_level).type_as(xin[0])
            )
            outputs.append(output)

        outputs = torch.cat(outputs, 1)
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)

        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
        for batch_idx, (img, num_gt, label) in enumerate(zip(imgs, nlabel, labels)):
            img = imgs[batch_idx].permute(1, 2, 0).to(torch.uint8)
            num_gt = int(num_gt)
            if num_gt == 0:
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = label[:num_gt, 1:5]
                gt_classes = label[:num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                _, fg_mask, _, matched_gt_inds, _ = self.get_assignments(  # noqa
                    batch_idx, num_gt, gt_bboxes_per_image, gt_classes,
                    bboxes_preds_per_image, expanded_strides, x_shifts,
                    y_shifts, cls_preds, obj_preds,
                )

            img = img.cpu().numpy().copy()  # copy is crucial here
            coords = torch.stack([
                ((x_shifts + 0.5) * expanded_strides).flatten()[fg_mask],
                ((y_shifts + 0.5) * expanded_strides).flatten()[fg_mask],
            ], 1)

            xyxy_boxes = cxcywh2xyxy(gt_bboxes_per_image)
            save_name = save_prefix + str(batch_idx) + ".png"
            img = visualize_assign(img, xyxy_boxes, coords, matched_gt_inds, save_name)
            logger.info(f"save img to {save_name}")
