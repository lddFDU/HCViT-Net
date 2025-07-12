from torch.utils.data import DataLoader
from seg.losses import *
from seg.datasets.isic_2018_dataset import *
from tools.utils import process_model_params


# training hparam
max_epoch = 300
ignore_index = len(CLASSES)
train_batch_size = 2
val_batch_size = 1
lr = 0.001
weight_decay = 1e-2
accumulate_n = 1
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "hcvitnet-refine-1-dice-celoss"
weights_path = "/mnt/nvme1n1p1/zrh/2d_seg/log/log_isic2018/{}".format(weights_name)
test_weights_name = weights_name
log_name = "/mnt/nvme1n1p1/zrh/2d_seg/log/log_isic2018/{}".format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
gpus = [0]
pretrained_ckpt_path = None
resume_ckpt_path = None

#  define the network
from seg.models.HCViTNet import HCViTNet
net = HCViTNet(256)

# define the loss
loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)

use_aux_loss = False

# define the dataloader
train_dataset = ISIC2018Dataset(mode='train',transform=train_aug)
val_dataset = ISIC2018Dataset(mode='val',transform=val_aug)
test_dataset = ISIC2018Dataset(mode='test',transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)


test_loader = DataLoader(dataset=test_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
net_params = process_model_params(net)
optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=300,eta_min=1e-5,last_epoch=-1)