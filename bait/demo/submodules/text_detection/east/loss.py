import torch
import torch.nn as nn
import torch.nn.functional as F


def get_dice_loss(gt_score, pred_score, training_map):
    eps = 1e-5
    intersection = torch.sum(gt_score * pred_score * training_map)
    union = (
        torch.sum(gt_score * training_map) + torch.sum(pred_score * training_map) + eps
    )
    loss = 1.0 - (2 * intersection / union)

    return loss


def get_ce_loss(gt_score, pred_score):
    return F.cross_entropy(pred_score, gt_score)


def get_geo_loss(gt_geo, pred_geo):
    d1_gt, d2_gt, d3_gt, d4_gt, angle_gt = torch.split(gt_geo, 1, 1)
    d1_pred, d2_pred, d3_pred, d4_pred, angle_pred = torch.split(pred_geo, 1, 1)
    area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
    area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)
    w_union = torch.min(d3_gt, d3_pred) + torch.min(d4_gt, d4_pred)
    h_union = torch.min(d1_gt, d1_pred) + torch.min(d2_gt, d2_pred)
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    iou_loss_map = -torch.log((area_intersect + 1.0) / (area_union + 1.0))
    angle_loss_map = 1 - torch.cos(angle_pred - angle_gt)
    return iou_loss_map, angle_loss_map


class Loss(nn.Module):
    def __init__(self, weight_angle=20):
        super(Loss, self).__init__()
        self.weight_angle = weight_angle

    def forward(self, gt_score, pred_score, gt_geo, pred_geo, ignored_map):
        # if torch.sum(gt_score) < 1:
        #     return torch.sum(pred_score + pred_geo) * 0

        training_map = 1 - ignored_map
        dice_loss = get_dice_loss(gt_score, pred_score, training_map)
        # scale classification loss to match the iou loss part
        classify_loss = dice_loss * 0.01
        # ce_loss = get_ce_loss(gt_score, pred_score * training_map)
        # classify_loss = dice_loss + ce_loss

        iou_loss_map, angle_loss_map = get_geo_loss(gt_geo, pred_geo)

        # angle_loss = torch.mean(angle_loss_map * gt_score * training_map)
        # iou_loss = torch.mean(iou_loss_map * gt_score * training_map)
        geo_loss_map = iou_loss_map + self.weight_angle * angle_loss_map
        geo_loss = torch.mean(geo_loss_map * gt_score * training_map)
        # print('classify loss is {:.8f}, angle loss is {:.8f}, iou loss is {:.8f}'.format(classify_loss, angle_loss, iou_loss))
        return geo_loss + classify_loss
