from Main_model_2 import TileDataset, TileDatasetTest, get_model_instance_segmentation, albumentations_transforms
import os

import torch
import torchvision
import utils
import torch.profiler
from tensorboardX import SummaryWriter
import numpy as np
import transforms as T
import albumentations as A
from torchvision.models.detection import MaskRCNN
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
from engine import train_one_epoch, evaluate
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import nvsmi
import json
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers
from args import get_train_args
from json import dumps


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2
    args.save_dir = utils.get_save_dir(args.save_dir, args.name, training=True)

    log = utils.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')

    log.info('Building dataset...')
    dataset = TileDataset("C:/Users/ghait/Desktop/NYU_Project/Datasets/Training/coco_training/", transform=albumentations_transforms)
    dataset_test = TileDatasetTest("C:/Users/ghait/Desktop/NYU_Project/Datasets/Testing/")

    # define training and validation data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=3, shuffle=True, num_workers=3,
        collate_fn=utils.collate_fn)

    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=3, shuffle=True, num_workers=3,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    log.info('Building model...')
    model = get_model_instance_segmentation(num_classes).to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.02,
                                momentum=0.9, weight_decay=0.0001)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=(3,5,7,9),
                                                   gamma=0.85)

    step = 0
    epoch = 0
    num_epoch = 12
    model = model.to(device)


    log.info('Training...')

    steps_till_eval = 22140

    while epoch != num_epoch:
        epoch += 1
        model.train()
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), tqdm(total=len(train_loader.dataset)) as progress_bar:
            for images, targets in train_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                optimizer.zero_grad()
                loss_dict = model(images, targets)
                batch_size = len(images)
                #print('Len of images: ', batch_size)

                losses = sum(loss for loss in loss_dict.values())

                losses.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 8)
                optimizer.step()

                step += batch_size

                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         NLL=losses)

                tbx.add_scalar('train/loss_classifier', loss_dict['loss_classifier'], step)
                tbx.add_scalar('train/loss_box_reg', loss_dict['loss_box_reg'], step)
                tbx.add_scalar('train/loss_mask', loss_dict['loss_mask'], step)
                tbx.add_scalar('train/loss_objectness', loss_dict['loss_objectness'], step)
                tbx.add_scalar('train/loss_rpn_box_reg', loss_dict['loss_rpn_box_reg'], step)

                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)

                tbx.add_scalar('nv-smi GPU used memory', int(json.loads(next(nvsmi.get_gpus()).to_json())['mem_used']),
                               step)
            lr_scheduler.step()

            steps_till_eval -= batch_size
            if steps_till_eval <= 0:
                steps_till_eval = 22140
                IoUs = evaluate(model, test_loader, device)
                for i in IoUs:
                    tbx.add_scalar(f'dev/', i, step)
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_sched': lr_scheduler}
            torch.save(checkpoint, os.path.join(args.save_dir, f'checkpoint{args.name}_{epoch}.pth'))

            model.train()

def evaluate(model, eval_loder, device):
    model.eval()
    IoUs = []
    with torch.no_grad(), \
            tqdm(total=len(eval_loder.dataset)) as progress_bar:
        for images, targets in eval_loder:
            pred = model(images)

            IoUs.append(utils.compute_IoU(pred,targets))

        return IoUs

if __name__ == "__main__":
    main(get_train_args())
