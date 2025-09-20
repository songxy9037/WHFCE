import numpy as np
import matplotlib.pyplot as plt
import os

import utils
from models.networks import *

import torch
import torch.optim as optim
import numpy as np
from misc.metric_tool import ConfuseMatrixMeter
from models.losses import cross_entropy, NewLoss
import models.losses as losses
from models.losses import get_alpha, softmax_helper, FocalLoss, mIoULoss, mmIoULoss
from thop import profile
from misc.logger_tool import Logger, Timer

from utils import de_norm
from tqdm import tqdm
import torch
from torch.nn import DataParallel
import torch
import torch.nn.functional as F

def compute_similarity_loss_with_adaptive_weights(feat_A, feat_B, labels, epsilon=1e-8):
    """
    计算相似性损失，加入正负样本自适应调节。

    参数：
    - feat_A, feat_B: (B, C, H, W)，输入的特征图
    - labels: (B, H, W)，变化掩码，0 表示未变化，1 表示变化
    - epsilon: 防止除零错误的小值
    """
    # 调整特征和标签尺寸
    B, C, H, W = feat_A.size()
    labels = labels.view(B, 1, H, W)  # (B, 1, H, W)

    # 展开特征和标签
    feat_A_flat = feat_A.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
    feat_B_flat = feat_B.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
    labels_flat = labels.view(B, -1)  # (B, N)

    # 特征归一化
    feat_A_flat = F.normalize(feat_A_flat, dim=2)
    feat_B_flat = F.normalize(feat_B_flat, dim=2)

    # 计算特征距离（欧氏距离）
    distances = torch.norm(feat_A_flat - feat_B_flat, p=2, dim=2)  # (B, N)

    # 构建正负样本掩码
    mask_positive = (labels_flat == 0).float()  # 未变化区域
    mask_negative = (labels_flat == 1).float()  # 变化区域

    # 计算正负样本数量
    num_positive = mask_positive.sum() + epsilon
    num_negative = mask_negative.sum() + epsilon

    # 计算正负样本的动态权重
    pos_weight = num_negative / (num_positive + num_negative)  # (B,)
    neg_weight = num_positive / (num_positive + num_negative)  # (B,)


    # 计算正负样本损失
    loss_positive = pos_weight * mask_positive * distances ** 2
    loss_negative = neg_weight * mask_negative * F.relu(1.0 - distances) ** 2  # 假设 margin 为 1.0

    # 计算每个批次的平均损失
    loss_positive = loss_positive.sum(dim=1) / num_positive  # (B,)
    loss_negative = loss_negative.sum(dim=1) / num_negative  # (B,)

    # 总损失为正负损失之和的平均
    loss = (loss_positive + loss_negative).mean() / 2

    return loss

class CDTrainer(nn.Module):

    def __init__(self, args, dataloaders,):
        super(CDTrainer, self).__init__()
        self.args = args
        self.dataloaders = dataloaders

        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)

        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids) > 0
                                   else "cpu")
        print(self.device)

        # Learning rate and Beta1 for Adam optimizers
        self.lr = args.lr

        # define optimizers
        if args.optimizer == "sgd":
            self.optimizer_G = torch.compile(self.net_G.parameters(), lr=self.lr,
                                         momentum=0.9,
                                         weight_decay=5e-4)
        elif args.optimizer == "adam":
            self.optimizer_G = optim.Adam(self.net_G.parameters(), lr=self.lr,
                                          weight_decay=0)
        elif args.optimizer == "adamw":
            self.optimizer_G = optim.AdamW(self.net_G.parameters(), lr=self.lr,
                                           betas=(0.9, 0.999), weight_decay=0.01)


        # define lr schedulers
        self.exp_lr_scheduler_G = get_scheduler(self.optimizer_G, args)

        self.running_metric = ConfuseMatrixMeter(n_class=2)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)
        # define timer
        self.timer = Timer()
        self.batch_size = args.batch_size

        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_epochs

        self.global_step = 0
        self.steps_per_epoch = len(dataloaders['train'])
        self.total_steps = (self.max_num_epochs - self.epoch_to_start) * self.steps_per_epoch

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.G_loss = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        self.shuffle_AB = args.shuffle_AB

        # define the loss functions
        self.multi_scale_train = args.multi_scale_train
        self.multi_scale_infer = args.multi_scale_infer
        self.weights = tuple(args.multi_pred_weights)
        if args.loss == 'ce':
            self._pxl_loss = cross_entropy
        elif args.loss == 'bce':
            self._pxl_loss = losses.binary_ce
        elif args.loss == 'NewLoss':
            self._pxl_loss = losses.NewLoss()
        else:
            raise NotImplemented(args.loss)

        self.VAL_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_acc.npy')):
            self.VAL_ACC = np.load(os.path.join(self.checkpoint_dir, 'val_acc.npy'))
        self.TRAIN_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'train_acc.npy')):
            self.TRAIN_ACC = np.load(os.path.join(self.checkpoint_dir, 'train_acc.npy'), allow_pickle=True)

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)

    def _load_checkpoint(self, ckpt_name='last_ckpt.pt'):
        print("\n")
        if os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, ckpt_name),
                                    map_location=self.device)

            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(checkpoint['exp_lr_scheduler_G_state_dict'])

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.total_steps = (self.max_num_epochs - self.epoch_to_start) * self.steps_per_epoch

            self.logger.write('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)\n' %
                              (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')
        elif self.args.pretrain is not None:
            print("Initializing backbone weights from: " + self.args.pretrain)
            self.net_G.load_state_dict(torch.load(self.args.pretrain), strict=False)
            self.net_G.to(self.device)
            self.net_G.eval()
        else:
            print('training from scratch...')
        print("\n")

    def _timer_update(self):
        self.global_step = (self.epoch_id - self.epoch_to_start) * self.steps_per_epoch + self.batch_id

        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
        return imps, est

    def _visualize_pred(self):
        pred = torch.argmax(self.G_final_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def _save_checkpoint(self, ckpt_name):
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    def _update_lr_schedulers_G(self):
        self.exp_lr_scheduler_G.step()

    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_final_pred.detach()

        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score


    def _collect_running_batch_states(self):
        running_acc = self._update_metric()  # 获取两个输出的指标

        m = len(self.dataloaders['train'])
        if self.is_training is False:
            m = len(self.dataloaders['val'])

        imps, est = self._timer_update()
        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d][%d,%d], imps: %.2f, est: %.2fh, G_loss: %.5f, running_mf1: %.5f\n' % \
                      (self.is_training, self.epoch_id, self.max_num_epochs - 1, self.batch_id, m,
                       imps * self.batch_size, est,
                       self.G_loss.item(), running_acc)
            self.logger.write(message)

        if np.mod(self.batch_id, 500) == 1:
            vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
            vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))

            vis_pred = utils.make_numpy_grid(self._visualize_pred())

            vis_gt = utils.make_numpy_grid(self.batch['L'])
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(
                self.vis_dir, 'istrain_' + str(self.is_training) + '_' +
                              str(self.epoch_id) + '_' + str(self.batch_id) + '.jpg')
            plt.imsave(file_name, vis)

    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']
        self.logger.write('Is_training: %s. Epoch %d / %d, epoch_mF1= %.5f\n' %
                          (self.is_training, self.epoch_id, self.max_num_epochs - 1, self.epoch_acc))
        message = ''
        for k, v in scores.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write(message + '\n')
        self.logger.write('\n')

    def _update_checkpoints(self):

        # save current model
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write('Lastest model updated. Epoch_acc=%.4f, Historical_best_acc=%.4f (at epoch %d)\n'
                          % (self.epoch_acc, self.best_val_acc, self.best_epoch_id))
        self.logger.write('\n')

        # update the best model (based on eval acc)
        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc = self.epoch_acc
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*' * 10 + 'Best model updated!\n')
            self.logger.write('\n')

    def _update_training_acc_curve(self):
        # update train acc curve
        self.TRAIN_ACC = np.append(self.TRAIN_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'train_acc.npy'), self.TRAIN_ACC)

    def _update_val_acc_curve(self):
        # update val acc curve
        self.VAL_ACC = np.append(self.VAL_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'val_acc.npy'), self.VAL_ACC)

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)

        # 获取 net_G 的输出
        outputs = self.net_G(img_in1, img_in2)

        # 根据返回值的个数动态处理
        if isinstance(outputs, tuple):
            self.G_final_pred, self.encoder_features = outputs
        else:
            self.G_final_pred = outputs
            self.encoder_features = None  # 或者设置默认值

    def _backward_G(self):
        gt = self.batch['L'].to(self.device).float()  # (B, H, W)
        # 主损失：分类损失
        self.G_loss_ce = self._pxl_loss(self.G_final_pred, gt)

        self.G_loss_contrast = 0.0


        if self.encoder_features is not None:

            layer_weights = {
                'layer2': 0.5,  # 第二层权重
                'layer3': 0.5,  # 第三层权重
                'layer4': 0.5  # 第四层权重
            }

            # 遍历多层特征，计算每层的对比损失
            for layer_name, (feat_A, feat_B) in self.encoder_features.items():
                # 下采样标签以匹配特征尺寸
                H, W = feat_A.size(2), feat_A.size(3)
                gt_downsampled = F.interpolate(gt, size=(H, W), mode='nearest')

                # 确保特征和标签在同一设备上
                feat_A = feat_A.to(self.device)
                feat_B = feat_B.to(self.device)
                gt_downsampled = gt_downsampled.to(self.device)

                # 计算每层的对比损失
                loss_contrast = compute_similarity_loss_with_adaptive_weights(feat_A, feat_B, gt_downsampled)

                # 将损失按层权重加入总对比损失
                if layer_name in layer_weights:
                    self.G_loss_contrast += layer_weights[layer_name] * loss_contrast
            # 计算总损失
            contrastive_weight = 0.15 # 全局对比损失的权重，可以调整
            self.G_loss = self.G_loss_ce + contrastive_weight * self.G_loss_contrast
        else:
            self.G_loss = self.G_loss_ce

        # 反向传播
        self.G_loss.backward()

    def train_models(self):
        self._load_checkpoint()
        ########################
        ########################
        # loop over the dataset multiple times
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):

            ################## train #################
            ##########################################
            self._clear_cache()
            self.is_training = True

            self.net_G.train()  # Set model to training mode

            # Iterate over data.
            total = len(self.dataloaders['train'])
            self.logger.write('lr: %0.7f\n \n' % self.optimizer_G.param_groups[0]['lr'])
            for self.batch_id, batch in tqdm(enumerate(self.dataloaders['train'], 0), total=total):
                self._forward_pass(batch)

                ### 更新 net_G ###
                self.optimizer_G.zero_grad()  # 清除 net_G 的梯度
                self._backward_G()  # 反向传播 net_G 的损失
                self.optimizer_G.step()  # 更新 net_G 的权重

                # 记录和更新训练状态
                self._collect_running_batch_states()
                self._timer_update()

            self._collect_epoch_states()
            self._update_training_acc_curve()
            self._update_lr_schedulers_G()
            ################## Eval ##################
            ##########################################
            self.logger.write('Begin evaluation...\n')
            self._clear_cache()
            self.is_training = False
            self.net_G.eval()

            # Iterate over data.
            for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    self._forward_pass(batch)
                self._collect_running_batch_states()
            self._collect_epoch_states()

            ########### Update_Checkpoints ###########
            ##########################################
            self._update_val_acc_curve()
            self._update_checkpoints()
