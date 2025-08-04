import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader

import copy
import numpy as np
import shutil

from .seg import SegTrainer
from utils.metric import compute_iou, compute_acc, compute_dice
from utils.losses import (
    multi_level_consistency_loss,
    multi_level_propgation_consistency_loss,
)

class FeatureCosineLoss(nn.Module):
    def __init__(self, beta=1.0):
        super(FeatureCosineLoss, self).__init__()
        self.beta = beta

    def forward(self, teacher_feats, student_feats):
        total_loss = 0
        for t_feat, s_feat in zip(teacher_feats, student_feats):
            cos_sim = F.cosine_similarity(s_feat, t_feat, dim=1, eps=1e-6)
            total_loss += torch.mean(1 - cos_sim)
        total_loss = total_loss/len(teacher_feats)
        return self.beta * total_loss

def mask_uncertain_regions_with_numpy(image, pseudo_label_1, pseudo_label_2, mask_ratio=0.3, block_size_range=(3, 10)):
    """
    使用 NumPy 对图像中伪标签不确定的区域进行随机掩码处理。

    Args:
        image (torch.Tensor): 输入图像，形状为 (B, C, H, W)
        pseudo_label_1 (torch.Tensor): 第一组伪标签，形状为 (B, 1, H, W)
        pseudo_label_2 (torch.Tensor): 第二组伪标签，形状为 (B, 1, H, W)
        mask_ratio (float): 掩码区域占所有不确定区域的比例，范围 [0, 1]
        block_size_range (tuple): 随机掩码块的尺寸范围，例如 (3, 10)

    Returns:
        masked_image (torch.Tensor): 掩码处理后的图像
        masked_mask (torch.Tensor): 对应的掩码区域（掩盖后的伪标签）
    """
    B, C, H, W = image.shape
    masked_image = image.clone()
    masked_mask = pseudo_label_2.clone()

    # 将伪标签转换成 NumPy 数组以便计算
    pseudo_label_1 = pseudo_label_1.squeeze(1).cpu().numpy()  # (B, H, W)
    pseudo_label_2 = pseudo_label_2.squeeze(1).cpu().numpy()  # (B, H, W)

    # 找出两个伪标签之间不确定（不同）的区域
    uncertain_mask = pseudo_label_2 != pseudo_label_1  # (B, H, W)，布尔数组

    # 获取所有不确定区域的坐标
    uncertain_coords = np.argwhere(uncertain_mask)  # 不确定区域的坐标，(N, 3) 表示 (batch_idx, y, x)
    total_uncertain_pixels = uncertain_coords.shape[0]

    # 如果没有不确定区域，直接返回原图
    if total_uncertain_pixels == 0:
        return masked_image, masked_mask

    # 计算要掩盖的像素总数
    total_pixels_to_mask = int(total_uncertain_pixels * mask_ratio)

    masked_pixels = 0  # 已掩盖的像素计数
    np.random.shuffle(uncertain_coords)  # 随机打乱不确定坐标顺序

    for coord in uncertain_coords:
        if masked_pixels >= total_pixels_to_mask:
            break  # 掩盖像素数达到目标值则退出

        batch_idx, center_y, center_x = coord

        # 随机选择掩码块大小
        block_size = np.random.randint(block_size_range[0], block_size_range[1] + 1)

        # 计算掩码块的范围，确保在图像边界内
        start_y = max(0, center_y - block_size // 2)
        end_y = min(H, center_y + block_size // 2 + 1)
        start_x = max(0, center_x - block_size // 2)
        end_x = min(W, center_x + block_size // 2 + 1)

        # 对图像和掩码标签进行掩盖操作（置零）
        masked_image[batch_idx, :, start_y:end_y, start_x:end_x] = 0
        masked_mask[batch_idx, 0, start_y:end_y, start_x:end_x] = 0

        # 更新已掩盖像素计数
        masked_pixels += (end_y - start_y) * (end_x - start_x)

    return masked_image, masked_mask


class ConsistencyWeight(object):
    """
    ramp_types = ['sigmoid_rampup', 'linear_rampup', 'cosine_rampup', 'log_rampup', 'exp_rampup']
    """

    def __init__(self):
        self.final_w = 1  # Fixed unsupervised weight
        self.iter_per_epoch = 100  # Fixed iterations per epoch
        self.start_iter = 0 * self.iter_per_epoch  # Fixed rampup start
        self.rampup_length = 40 * self.iter_per_epoch  # Fixed rampup length
        self.rampup_func = self.sigmoid  # Default rampup type
        self.current_rampup = 0

    def __call__(self, current_idx):
        if current_idx <= self.start_iter:
            return 0.0

        self.current_rampup = self.rampup_func(current_idx - self.start_iter,
                                               self.rampup_length)

        return self.final_w * self.current_rampup

    @staticmethod
    def gaussian(start, current, rampup_length):
        assert rampup_length >= 0
        if current == 0:
            return 0.0
        if current < start:
            return 0.0
        if current >= rampup_length:
            return 1.0
        return np.exp(-5 * (1 - current / rampup_length) ** 2)

    @staticmethod
    def sigmoid(current, rampup_length):
        if rampup_length == 0:
            return 1.0
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length

        return float(np.exp(-5.0 * phase * phase))

    @staticmethod
    def linear(current, rampup_length):
        assert current >= 0 and rampup_length >= 0
        if current >= rampup_length:
            return 1.0
        return current / rampup_length

def crc_loss(inputs, targets,
                  threshold=0.8,
                  neg_threshold=0.2,
                  conf_mask=True):
    if not conf_mask:
        raise NotImplementedError
    targets_prob = F.softmax(targets, dim=1)

    # for positive
    pos_weight = targets_prob.max(1)[0]
    pos_mask = (pos_weight >= threshold)
    # for negative
    neg_weight = targets_prob.min(1)[0]
    neg_mask = (neg_weight < neg_threshold)
    y_tilde = torch.argmax(targets, dim=1)
    if not torch.any(pos_mask):
        positive_loss_mat = torch.tensor([.0], device=targets.device)
    else:
        positive_loss_mat = F.nll_loss(torch.log_softmax(inputs, dim=1),
                                       y_tilde, reduction="none")
        positive_loss_mat = positive_loss_mat * pos_weight
        positive_loss_mat = positive_loss_mat[pos_mask]
    
    if not torch.any(neg_mask):
        negative_loss_mat = torch.tensor([.0], device=targets.device)
    else:
        inverse_prob = torch.clamp(1-F.softmax(inputs, dim=1), min=1e-6, max=1.0)
        negative_loss_mat = F.nll_loss(inverse_prob.log(), (1-y_tilde), reduction="none")
        negative_loss_mat = negative_loss_mat * neg_weight
        negative_loss_mat = negative_loss_mat[neg_mask]
    
    return positive_loss_mat.mean() + negative_loss_mat.mean()


class GazeSupTrainer(SegTrainer):
    def __init__(self, *argv, **kargs):
        super().__init__(*argv, **kargs)
        self.main_metric = "mdice_2"
        self.weight_scheduler = ConsistencyWeight()
        self.iter_num = 0
        feat_loss_weight = 0.5
        self.feature_distill_loss_fn = FeatureCosineLoss(beta=feat_loss_weight)

    @staticmethod
    def rand_bbox(size, lam=None):
        # past implementation
        W = size[2]
        H = size[3]
        B = size[0]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = np.random.randint(size=[B, ], low=int(W / 8), high=W)
        cy = np.random.randint(size=[B, ], low=int(H / 8), high=H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

    def cut_mix(self, volume=None, mask=None, feature = None):
        mix_volume = volume.clone()
        mix_target = mask.clone()
      
        u_rand_index = torch.randperm(volume.size()[0])[:volume.size()[0]].cuda()
        u_bbx1, u_bby1, u_bbx2, u_bby2 = self.rand_bbox(volume.size(), lam=np.random.beta(4, 4))

        for i in range(0, mix_volume.shape[0]):
            mix_volume[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                volume[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
          
            if len(mix_target.shape) > 3:
                mix_target[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                    mask[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
            
            else:
                mix_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                    mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        return mix_volume, mix_target
    
    def _update(self, minibatch):
        self.iter_num += 1
        image = minibatch["image"].cuda()
        embedding = minibatch["embedding"].cuda()
        model1, model2 = self.model
        # model1 = model1.cuda
        # model2 = model2.cuda
        optimizer1, optimizer2 = self.optimizer
        loss_dict = {}
        loss_dict["lr"] = optimizer1.param_groups[0]["lr"]
        model1.train()
        model2.train()

        with autocast(enabled=self.args.fp16):

            pseudo_label_1 = minibatch.get("pseudo_label_1", None)
            pseudo_label_2 = minibatch.get("pseudo_label_2", None)
            use_revise = True  # Flag to determine whether to apply the revision logic
            if use_revise == True:
                # Get the number of samples in the batch
                B = pseudo_label_1.shape[0]

                # Flatten each mask and compute the sum of elements (area) for each sample mask.
                area_1 = pseudo_label_1.view(B, -1).sum(dim=1)  # (B,) tensor representing the "area" for each pseudo_label_1 mask
                area_2 = pseudo_label_2.view(B, -1).sum(dim=1)  # (B,) tensor representing the "area" for each pseudo_label_2 mask

                # Count the number of samples where pseudo_label_1 has a smaller area than pseudo_label_2
                num_smaller_masks = (area_1 < area_2).sum().item()  # Number of samples where pseudo_label_1's area is less than pseudo_label_2's

                # Count the number of samples where pseudo_label_1 has a larger area than pseudo_label_2
                num_larger_masks = (area_1 > area_2).sum().item()  # Number of samples where pseudo_label_1's area is greater than pseudo_label_2's

                # Create a boolean mask that marks samples where pseudo_label_1 has a larger area than pseudo_label_2.
                swap_mask = area_1 > area_2  # True for positions where pseudo_label_1's mask should be swapped

                # Swap pseudo_label_1 and pseudo_label_2 for those samples marked by swap_mask.
                pseudo_label_1[swap_mask], pseudo_label_2[swap_mask] = pseudo_label_2[swap_mask], pseudo_label_1[swap_mask]

                # The following code is for debugging purposes (commented out)
                num_swaps = swap_mask.sum().item()  # Total number of samples where swapping occurred
                # print("\nRevised mask information:")
                # print(f"Number of samples with smaller pseudo_label_1 masks: {num_smaller_masks}/{B}")
                # print(f"Number of samples with larger pseudo_label_1 masks: {num_larger_masks}/{B}")
                # print(f"Number of swaps performed: {num_swaps}/{B}")

            if pseudo_label_1 is not None:
                pseudo_label_1 = pseudo_label_1.cuda()

            if pseudo_label_2 is not None:
                pseudo_label_2 = pseudo_label_2.cuda()

            masked_image, masked_mask= mask_uncertain_regions_with_numpy(image, pseudo_label_1, pseudo_label_2)
            if masked_image is not None and masked_image is not None:
                masked_image = masked_image.cuda()
                masked_mask = masked_mask.cuda()

            pred_dict1 = model1(image, embedding)
            pred_dict2 = model2(masked_image)
            
            loss = 0
            # for i in range(self.args.num_levels):
            #     pseudo_label = minibatch[f"pseudo_label_{i+1}"].cuda()
            #     loss_cls = self.criterion(pred_dict[f"logits_{i+1}"], pseudo_label.float())

            #     loss_dict[f"loss_cls_{i+1}"] = loss_cls.item()

            #     loss += loss_cls
            logits_1 = pred_dict1["logits"][:, 1:2, :, :]  # �~O~V�~_~P个�~@~Z�~A~S
            logits_2 = pred_dict2["logits"][:, 1:2, :, :]
            loss_cls_1 = self.criterion(logits_1, pseudo_label_1.float()) if pseudo_label_1 is not None else 0
            loss_cls_2 = self.criterion(logits_2, masked_mask.float()) if masked_mask is not None else 0
           
            # if self.args.cons_weight > 0 and self.args.num_levels > 1:
            #     if self.args.cons_mode == "pure":
            #         loss_consistency = multi_level_consistency_loss(
            #             [pred_dict[f"logits_{i+1}"] for i in range(self.args.num_levels)]
            #         )
            #     elif self.args.cons_mode == "prop":
            #         loss_consistency = multi_level_propgation_consistency_loss(
            #             [pred_dict[f"logits_{i+1}"] for i in range(self.args.num_levels)],
            #             [pred_dict[f"logits_prop_{i+1}"] for i in range(self.args.num_levels)],
            #         )

                # loss_dict["loss_cons"] = loss_consistency.mean().item()

                # loss += self.args.cons_weight * loss_consistency.sum()
            teacher_feats = [
                pred_dict1["encoder_features"]["x1"].clone().detach(),
                pred_dict1["encoder_features"]["x2"].clone().detach(),
                pred_dict1["encoder_features"]["x3"].clone().detach(),
                pred_dict1["encoder_features"]["x4"].clone().detach(),
            ]
            student_feats = [
                pred_dict2["encoder_features"]["x1"],
                pred_dict2["encoder_features"]["x2"],
                pred_dict2["encoder_features"]["x3"],
                pred_dict2["encoder_features"]["x4"],
            ]
            loss_feat = self.feature_distill_loss_fn(teacher_feats, student_feats)
            consistency_weight = self.weight_scheduler(self.iter_num)

            pseudo_label_for_model1 = pred_dict2["logits"].clone().detach() 

            noise = torch.zeros_like(image).uniform_(-.2, .2)
            cut_mix_input1, cut_mix_pseudo_label_1 = self.cut_mix(image + noise, pseudo_label_for_model1)

            cons_outputs_pred1 = model1(cut_mix_input1, embedding)
  

            loss1_crc =  crc_loss(inputs=cons_outputs_pred1["logits"],
                                         targets=cut_mix_pseudo_label_1,
                                         threshold=0.8,
                                         neg_threshold=0.2,
                                         conf_mask=True)
            
            pseudo_label_for_model2 = pred_dict1["logits"].clone().detach() 


            noise = torch.zeros_like(image).uniform_(-.2, .2)
            cut_mix_input2, cut_mix_pseudo_label_2 = self.cut_mix(image+ noise, pseudo_label_for_model2)

            cons_outputs_pred2 = model2(cut_mix_input2)
            loss2_crc = crc_loss(inputs=cons_outputs_pred2["logits"],
                                         targets=cut_mix_pseudo_label_2,
                                         threshold=0.8,
                                         neg_threshold=0.2,
                                         conf_mask=True)
            loss = loss_cls_1 + loss_cls_2+ consistency_weight * (loss1_crc + loss2_crc ) + 0.1  * consistency_weight * loss_feat
            loss_dict["loss_cls_1"] = loss_cls_1.item()
            loss_dict["loss_cls_2"] = loss_cls_2.item()
            loss_dict["loss1_crc"] = loss1_crc.item()
            loss_dict["loss2_crc"] = loss2_crc.item()
            loss_dict["loss_feat"] = loss_feat.item()
            loss_dict["loss"] = loss.item()
            # if self.iter_num % 100 == 0:
            #     print({
            #         "iter_num": self.iter_num,
            #         "loss_cls_1": float(loss_cls_1),
            #         "loss_cls_2": float(loss_cls_2),

            #         "loss": float(loss),
            #     })

        if self.args.fp16:
            # for param in self.model.parameters():
            #     param.grad = None
            # self.scaler.scale(loss).backward()
            # self.scaler.step(self.optimizer)
            for param in model1.parameters():
                param.grad = None
            for param in model2.parameters():
                param.grad = None

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer1)
            self.scaler.step(optimizer2)
            self.scaler.update()

        else:
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()
            optimizer1.step()
            optimizer2.step()

        return loss_dict

    def validate(self, dataloader, model=None, save_pred=False, save_root=None):
        model1, model2 = self.model
        model1.eval()
        model2.eval()


        iou_sub_l = [[] for _ in range(2)]
        dice_sub_l = [[] for _ in range(2)]

        iou_l, dice_l = [], []

        with torch.no_grad():
            for minibatch in dataloader:
                image, label, embedding = minibatch["image"].cuda(), minibatch["label"].cuda(), minibatch["embedding"].cuda()

                mask = ~minibatch["trimap"].cuda() if "trimap" in minibatch.keys() else None

                with autocast(enabled=self.args.fp16):

                    pred_dict1 = model1(image, embedding)
                    pred_dict2 = model2(image)
                    logits_1 = pred_dict1["logits"]
                    logits_2 = pred_dict2["logits"]
                    pred_sub_l = [
                        F.interpolate(logits_1, size=label.shape[2:], mode="bilinear"),
                        F.interpolate(logits_2, size=label.shape[2:], mode="bilinear"),
                    ]
                    # �~@�~M~U平�~]~G
                    # pred = torch.stack(pred_sub_l, dim=0).mean(dim=0)

                    # pred_dict = model(image)
                    # pred_sub_l = [
                    #     F.interpolate(
                    #         pred_dict[f"logits_{i+1}"],
                    #         size=label.shape[2:],
                    #         mode="bilinear",
                    #     )
                    #     for i in range(self.args.num_levels)
                    # ]

                    pred = torch.stack(pred_sub_l, dim=0).mean(dim=0)

                if save_pred and save_root is not None:
                    self.save_pred_batch(
                        pred.clone(),
                        save_root=save_root,
                        save_filenames=minibatch["subject_id"],
                    )

                    for i in range(2):
                        save_root_i = os.path.join(save_root, f"pred_level_{i+1}")

                        self.save_pred_batch(
                            pred_sub_l[i].clone(),
                            save_root=save_root_i,
                            save_filenames=minibatch["subject_id"],
                        )

                for i in range(2):
                    iou_sub_l[i].append(compute_iou(pred_sub_l[i][:, 1:2, :, :], label, mask=mask, do_threshold=True).cpu().numpy())
                    dice_sub_l[i].append(
                        compute_dice(pred_sub_l[i][:, 1:2, :, :], label, mask=mask, do_threshold=True).cpu().numpy()
                    )

                iou_l.append(compute_iou(pred[:, 1:2, :, :], label, mask=mask, do_threshold=True).cpu().numpy())
                dice_l.append(compute_dice(pred[:, 1:2, :, :], label, mask=mask, do_threshold=True).cpu().numpy())

        iou_l = np.concatenate(iou_l, axis=0)
        dice_l = np.concatenate(dice_l, axis=0)

        performance_dict = {}
        performance_dict["miou"] = np.mean(iou_l)
        performance_dict["miou_std"] = np.std(iou_l)

        performance_dict["mdice"] = np.mean(dice_l)
        performance_dict["mdice_std"] = np.std(dice_l)

        for i in range(self.args.num_levels):
            iou_sub = np.concatenate(iou_sub_l[i], axis=0)
            dice_sub = np.concatenate(dice_sub_l[i], axis=0)

            performance_dict[f"miou_{i+1}"] = np.mean(iou_sub)
            performance_dict[f"miou_std_{i+1}"] = np.std(iou_sub)

            performance_dict[f"mdice_{i+1}"] = np.mean(dice_sub)
            performance_dict[f"mdice_std_{i+1}"] = np.std(dice_sub)

        return performance_dict
