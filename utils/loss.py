# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import Concat
from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):
    """Returns label smoothing BCE targets for reducing overfitting; pos: `1.0 - 0.5*eps`, neg: `0.5*eps`. For details
    see https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441.
    """
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    """Modified BCEWithLogitsLoss to reduce missing label effects in YOLOv5 training with optional alpha smoothing."""

    def __init__(self, alpha=0.05):
        """Initializes a modified BCEWithLogitsLoss with reduced missing label effects, taking optional alpha smoothing
        parameter.
        """
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        """Computes modified BCE loss for YOLOv5 with reduced missing label effects, taking pred and true tensors,
        returns mean loss.
        """
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    """Applies focal loss to address class imbalance by modifying BCEWithLogitsLoss with gamma and alpha parameters."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes FocalLoss with specified loss function, gamma, and alpha values; modifies loss reduction to
        'none'.
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Calculates the focal loss between predicted and true labels using a modified BCEWithLogitsLoss."""
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    """Implements Quality Focal Loss to address class imbalance by modulating loss based on prediction confidence."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes Quality Focal Loss with given loss function, gamma, alpha; modifies reduction to 'none'."""
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Computes the focal loss between `pred` and `true` using BCEWithLogitsLoss, adjusting for imbalance with
        `gamma` and `alpha`.
        """
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    """Computes the total loss for YOLOv5 model predictions, including classification, box, and objectness losses."""

    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False, teacher_model=None):
        """Initializes ComputeLoss with model and autobalance option, autobalances losses if True."""
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance

        self.model = model

        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device
        self.teacher_model = teacher_model
        self.distill_ok = self.teacher_model is not None
        if self.distill_ok:
            # 1. freezes teacher model
            self.teacher_model.float()  # 强制 teacher 用 FP32
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.eval()
            de_parallel(self.teacher_model).model[-1].train()
            # 2. 蒸馏超参数（从 hyp 读取，方便调优）
            # 1. 稍微提高总开关权重，让蒸馏起效
            self.distill_w_max = h.get("distill_w", 0.1)  # 总体蒸馏权重（最大值，由 warmup 调度器控制）
            self.distill_warmup_ratio = h.get(
                "distill_warmup_ratio", 0.3
            )  # 蒸馏 warmup 比例（训练前多少比例 epoch 为 0）
            self.distill_w = 0.0  # 当前蒸馏权重（通过 set_epoch 更新），默认 0（warmup 初始）
            # 2. 温度稍微调高一点点，使软标签更平滑
            self.distill_temp = h.get("distill_temp", 3.0)

            self.distill_box_w = h.get("distill_box_w", 0.05)  # 降低框权重
            self.distill_cls_w = h.get("distill_cls_w", 0.1)  # 提高分类权重
            self.distill_obj_w = h.get("distill_obj_w", 0.0)  # 提高置信度权重，默认关闭以保护 Recall

            # 各类蒸馏子项是否启用（可通过 hyp 控制实验组 A/B/C）
            self.distill_box_enabled = h.get("distill_box_enabled", True)
            self.distill_cls_enabled = h.get("distill_cls_enabled", False)
            self.distill_obj_enabled = h.get("distill_obj_enabled", False)

            self.distill_cls_criterion = nn.KLDivLoss(reduction="batchmean")  # 推荐用 batchmean
            self.distill_box_loss_type = h.get(
                "distill_box_loss_type", "l1"
            )  # 支持多种框蒸馏类型（L1/MSE），默认 L1 更稳定
            if self.distill_box_loss_type == "mse":
                self.distill_box_criterion = nn.MSELoss(reduction="mean")
            else:
                self.distill_box_criterion = nn.L1Loss(reduction="mean")
            # [重要修改] 置信度改用 BCE，梯度更健康
            self.distill_obj_criterion = nn.BCEWithLogitsLoss(reduction="mean")

            # 4. 中间层特征蒸馏（可选，v5s→v5l 推荐开启，提升小模型特征提取能力）
            self.feat_distill_enabled = h.get("feat_distill_enabled", True)

            self.distill_block_size = h.get("distill_block_size", 4)
            self.distill_topk_ratio = h.get("distill_topk_ratio", 0.3)
            # [重要修改] 特征蒸馏权重降级，防止淹没主损失！
            self.feat_distill_w = 0.005
            # YOLOv5 中间特征层：取 Detect 头前的 3 个多尺度特征层（P3、P4、P5，对应 model.model[17]、[20]、[23]，需根据 yaml 确认）
            self.student_feat_layers = [6, 8, 10]  # 学生模型的特征层索引（yolov5s.yaml 对应 C3 输出）
            self.teacher_feat_layers = [6, 8, 10]  # 教师模型的特征层索引（yolov5l.yaml 同架构，索引一致）
            # Class-aware distillation settings
            self.wbc_class_id = h.get("wbc_class_id", 1)  # 默认 WBC=1
            self.wbc_distill_factor = h.get("wbc_distill_factor", 2.0)  # WBC 蒸馏增强系数

            dummy_img = torch.zeros(1, 3, 640, 640).to(self.device)

            with torch.no_grad():
                s_feats = self._get_intermediate_feats(self.model, dummy_img, self.student_feat_layers)
                student_channels = [f.shape[1] for f in s_feats]

                t_feats = self._get_intermediate_feats(self.teacher_model, dummy_img, self.teacher_feat_layers)
                teacher_channels = [f.shape[1] for f in t_feats]

            print(f"Distillation Channels detected: Student={student_channels}, Teacher={teacher_channels}")

            # 3. 传入计算好的通道数
            self.feat_projectors = self._build_feat_projectors(student_channels, teacher_channels)

        # 记录 epoch/total 用于 warmup 调度
        self.current_epoch = 0
        self.epochs = None

    def set_epoch(self, epoch, epochs):
        """Set current epoch and total epochs for warmup scheduling of distill weight."""
        self.current_epoch = int(epoch)
        self.epochs = int(epochs) if epochs is not None else None
        # compute linear warmup scaling
        if self.epochs is None or self.epochs <= 0:
            self.distill_w = self.distill_w_max
            return
        warmup_epochs = max(int(self.epochs * float(self.distill_warmup_ratio)), 1)
        # if within warmup period -> 0
        if self.current_epoch < warmup_epochs:
            self.distill_w = 0.0
        else:
            # linear ramp from 0 at end of warmup to distill_w_max at final epoch
            progress = (self.current_epoch - warmup_epochs) / max(1, (self.epochs - warmup_epochs))
            self.distill_w = float(self.distill_w_max) * float(progress)
            # clamp
            if self.distill_w > self.distill_w_max:
                self.distill_w = float(self.distill_w_max)

    def _compute_topk_patch_loss(self, s_feat, t_feat):
        """计算 Top-K Patch 蒸馏损失 1. 计算像素级 MSE (B, C, H, W) 2. 将 H, W 划分为若干个 block_size * block_size 的块 3. 计算每个块的总误差能量 4.
        选取 Top-K 个误差最大的块，仅计算这些块的 Loss.
        """
        # 1. 计算逐像素的平方差 (B, C, H, W) -> (B, H, W) [在通道维度求和]
        diff = (s_feat - t_feat) ** 2
        diff_spatial = diff.sum(dim=1)  # (B, H, W)

        B, H, W = diff_spatial.shape
        bs = self.distill_block_size

        # 如果特征图太小，无法切分，则退化为普通 MSE
        if H < bs or W < bs:
            return F.mse_loss(s_feat, t_feat)

        # 2. 利用 AvgPool 计算每个 Block 的平均误差能量
        # 输出形状: (B, H//bs, W//bs)
        block_energy = F.avg_pool2d(diff_spatial, kernel_size=bs, stride=bs, count_include_pad=False)

        b_h, b_w = block_energy.shape[1], block_energy.shape[2]
        num_blocks = b_h * b_w
        k = int(num_blocks * self.distill_topk_ratio)

        if k == 0:  # 保护机制
            return torch.tensor(0.0, device=self.device)

        # 3. 展平并选取 Top-K
        # (B, num_blocks)
        flat_energy = block_energy.view(B, -1)

        # 获取 Top-k 的值和索引 (我们只需要索引来生成掩码，或者直接利用值)
        # vals: (B, k), indices: (B, k)
        _topk_vals, topk_indices = torch.topk(flat_energy, k, dim=1)

        # 4. 生成 Mask (掩码)
        mask_flat = torch.zeros_like(flat_energy)
        # 将 top-k 的位置置为 1
        mask_flat.scatter_(1, topk_indices, 1.0)

        # 还原 Mask 形状 -> (B, 1, b_h, b_w) -> 上采样回 (B, 1, H, W)
        mask = mask_flat.view(B, 1, b_h, b_w)

        # 使用最近邻插值将 Mask 放大回原特征图大小，这样被选中的 Block 区域全是 1，其余是 0
        mask_expanded = F.interpolate(mask, size=(H, W), mode="nearest")

        # 5. 计算最终 Loss
        # 只计算 mask 为 1 的区域的 loss。
        # 注意：diff 已经是平方差了。
        # 为了数值稳定性，除以有效元素的个数 (mask.sum() * C) 或者直接取 mean

        # 这里的 diff 是 (B, C, H, W)，mask_expanded 是 (B, 1, H, W)，广播机制生效
        masked_diff = diff * mask_expanded

        # 计算非零元素的平均值 (防止 k 很小时 loss 变得极小)
        # 总有效像素数 = B * C * (k * bs * bs)
        total_active_elements = mask_expanded.sum() * s_feat.shape[1]

        loss = masked_diff.sum() / (total_active_elements + 1e-8)

        return loss

    def _build_feat_projectors(self, s_channels, t_channels):
        """构建特征通道投影器（v5l 特征通道数 → v5s 特征通道数，因为 v5l 通道数是 v5s 的 2 倍）."""
        projectors = []

        for t_ch, s_ch in zip(t_channels, s_channels):
            # 1x1 卷积降维 + BatchNorm
            projector = nn.Sequential(
                nn.Conv2d(t_ch, s_ch, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(s_ch)
            ).to(self.device)
            nn.init.xavier_uniform_(projector[0].weight)
            projectors.append(projector)
        return nn.ModuleList(projectors)  # 建议使用 ModuleList 以便正确注册参数

    def _get_intermediate_feats(self, model, x, layer_indices):
        """获取模型中间层特征（适配 YOLOv5 结构，跳过 Concat 等多输入模块）."""
        feats = []

        # 假设 Concat 模块已经被导入 (见步骤 1)

        for idx, m in enumerate(model.model):
            # 检查模块类型，如果是 Concat 模块，则跳过前向传播
            # 否则，Concat 模块的输入 x 此时是一个 Tensor 而非 Tensor 列表，会导致 TypeError: cat()
            if isinstance(m, Concat):
                continue  # 跳过 Concat 模块，继续下一个模块

            # 前向传播到当前层（只针对单输入模块：Conv, C3, SPPF 等）
            x = m(x)

            # 保存特征
            if idx in layer_indices:
                feats.append(x)

            if idx == layer_indices[-1]:  # 到最后一个特征层后停止，提升效率
                break
        return feats

    def __call__(self, p, targets, imgs=None):  # predictions, targets
        """Performs forward pass, calculating class, box, and object loss for given predictions and targets."""
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        ldistill = torch.zeros(1, device=self.device)
        lfeat_distill = torch.zeros(1, device=self.device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            if self.distill_ok is True:
                # -------- Class-aware weight (WBC stronger distillation) --------
                # tcls[i]: GT class for each positive sample
                class_weight = torch.ones_like(tcls[i], dtype=torch.float, device=self.device)

                # 强化 WBC 的蒸馏权重
                class_weight[tcls[i] == self.wbc_class_id] = self.wbc_distill_factor

            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            if n := b.shape[0]:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
                # distill loss（只在训练阶段做；验证阶段直接跳过）
        if self.distill_ok and imgs is not None and self.model.training:
            # 确保输入给 teacher 的是 FP32
            imgs_fp32 = imgs.float()

            with torch.no_grad():
                # teacher forward 固定 FP32，并显式关掉 autocast
                with torch.amp.autocast("cuda", enabled=False):
                    teacher_p = self.teacher_model(imgs_fp32)

                    # 教师中间特征（同样在 FP32 下）
                    if self.feat_distill_enabled:
                        teacher_feats = self._get_intermediate_feats(
                            self.teacher_model, imgs_fp32, self.teacher_feat_layers
                        )
            for i, (student_pi, teacher_pi) in enumerate(zip(p, teacher_p)):
                b, a, gj, gi = indices[i]
                n = b.shape[0]
                if n == 0:
                    continue  # 无目标层跳过，节省计算

                # 3.1 提取学生/教师的目标位置预测（仅聚焦有真实目标的位置，提升效率）
                # 学生预测
                s_pxy, s_pwh, _, s_pcls = student_pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)
                s_obj = student_pi[b, a, gj, gi, 4:5]  # 置信度预测（logits）

                # 教师预测（解码格式与学生完全一致）
                t_pxy, t_pwh, _, t_pcls = teacher_pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)
                t_obj = teacher_pi[b, a, gj, gi, 4:5]  # 教师置信度（logits）

                # 3.2 框回归蒸馏（对齐解码后的预测框位置），仅在启用且使用正样本位置计算
                distill_box_loss = torch.tensor(0.0, device=self.device)
                if getattr(self, "distill_box_enabled", True):
                    # student/teacher box are decoded; 对齐后计算 L1/MSE
                    s_box = torch.cat([s_pxy.sigmoid() * 2 - 0.5, (s_pwh.sigmoid() * 2) ** 2], 1)
                    t_box = torch.cat([t_pxy.sigmoid() * 2 - 0.5, (t_pwh.sigmoid() * 2) ** 2], 1)
                    if getattr(self, "distill_box_loss_type", "l1") == "mse":
                        distill_box_loss = F.mse_loss(s_box, t_box, reduction="mean") * self.distill_box_w
                    else:
                        # per-sample L1 loss: (n, 4) -> (n,)
                        per_sample_box_loss = F.l1_loss(s_box, t_box, reduction="none").mean(dim=1)

                        # class-aware weighted distillation
                        distill_box_loss = (per_sample_box_loss).mean() * self.distill_box_w

                # 3.3 分类蒸馏（KL散度 + 温度平滑），可配置是否启用
                distill_cls_loss = torch.tensor(0.0, device=self.device)
                if getattr(self, "distill_cls_enabled", False):
                    s_cls_logsoftmax = F.log_softmax(s_pcls / self.distill_temp, dim=-1)
                    t_cls_softmax = F.softmax(t_pcls / self.distill_temp, dim=-1)
                    # 乘温度平方：抵消 KL 散度在高温度下的损失缩放（参考蒸馏原论文）
                    # per-sample KL divergence
                    per_sample_kl = F.kl_div(s_cls_logsoftmax, t_cls_softmax, reduction="none").sum(dim=1)  # (n,)

                    distill_cls_loss = (per_sample_kl).mean() * (self.distill_temp**2) * self.distill_cls_w

                # 3.4 置信度蒸馏（对齐 sigmoid 后的概率），默认关闭
                distill_obj_loss = torch.tensor(0.0, device=self.device)
                if getattr(self, "distill_obj_enabled", False):
                    distill_obj_loss = self.distill_obj_criterion(s_obj, torch.sigmoid(t_obj)) * self.distill_obj_w

                # 3.5 累加该层输出蒸馏损失（使用总体权重 self.distill_w 来控制）
                layer_distill = distill_box_loss + distill_cls_loss + distill_obj_loss
                ldistill += layer_distill * getattr(self, "distill_w", 0.0)

            # 4. 中间层特征蒸馏（对齐学生与教师的特征分布）
            # if self.feat_distill_enabled and len(teacher_feats) == len(self.student_feat_layers):
            # 获取学生中间层特征
            #     student_feats = self._get_intermediate_feats(de_parallel(self.model), imgs, self.student_feat_layers)
            # 只在训练阶段做特征蒸馏（val/test 禁用）
            if (
                self.model.training
                and self.feat_distill_enabled
                and len(teacher_feats) == len(self.student_feat_layers)
            ):
                student_feats = self._get_intermediate_feats(
                    de_parallel(self.model), imgs_fp32, self.student_feat_layers
                )

                # 逐特征层计算蒸馏损失（MSE 对齐特征图）
                for idx, (s_feat, t_feat, projector) in enumerate(
                    zip(student_feats, teacher_feats, self.feat_projectors)
                ):
                    t_feat_proj = projector(t_feat)
                    if s_feat.shape[2:] != t_feat_proj.shape[2:]:
                        t_feat_proj = F.interpolate(
                            t_feat_proj, size=s_feat.shape[2:], mode="bilinear", align_corners=False
                        )
                    # 累加特征蒸馏损失
                    # lfeat_distill += F.mse_loss(s_feat, t_feat_proj)
                    # 将特征蒸馏权重和总体蒸馏权重同时考虑进来（feat_distill_w * distill_w）
                    lfeat_distill += (
                        self._compute_topk_patch_loss(s_feat, t_feat_proj)
                        * getattr(self, "feat_distill_w", 0.0)
                        * getattr(self, "distill_w", 0.0)
                    )

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # batch size

        total_loss = (lbox + lobj + lcls + ldistill + lfeat_distill) * bs

        if imgs is not None:
            return total_loss, torch.cat((lbox, lobj, lcls, ldistill)).detach()
        else:
            return total_loss, torch.cat((lbox, lobj, lcls)).detach()

        # return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        """Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
        indices, and anchors.
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
