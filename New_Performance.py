import os
import torch
import torchvision
import utils
import csv
import numpy as np
import pandas as pd
from itertools import product
from engine import evaluate
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from Main_model_2 import TileDataset, TileDatasetTest, get_model_instance_segmentation

def main():

    mask_scores = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    pixel_scores = [0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    comm = product(mask_scores,pixel_scores)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_list = {'Epoch_8': 'C:/Users/ghait/PycharmProjects/NYU_CBRD/save/train/M1-01/checkpointM1_8.pth',
                  'Epoch_7': 'C:/Users/ghait/PycharmProjects/NYU_CBRD/save/train/M1-01/checkpointM1_7.pth',
                  'Epoch_6': 'C:/Users/ghait/PycharmProjects/NYU_CBRD/save/train/M1-01/checkpointM1_6.pth',
                  'Epoch_5': 'C:/Users/ghait/PycharmProjects/NYU_CBRD/save/train/M1-01/checkpointM1_5.pth',

                  }

    results = {}



    num_classes = 2
    dataset_test = TileDatasetTest("C:/Users/ghait/Desktop/NYU_Project/Datasets/Testing/")
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=3, shuffle=True, num_workers=3,
        collate_fn=utils.collate_fn)
    step = len(test_loader)/3
    model = get_model_instance_segmentation(num_classes).to(device)


    for model_name, path in tqdm(model_list.items()):

        for com in comm:
            results[model_name + str(com[0]) + "_" + str(com[1])] = 0.0

        ckpt_dict = torch.load(path, map_location=device)
        model.load_state_dict(ckpt_dict['model'])
        model.eval()



        for images, targets in tqdm(test_loader):
            mask_scores = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
            pixel_scores = [0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
            comm = product(mask_scores, pixel_scores)

            #print(" test 1")
            images = list(image.to(device) for image in images)
            #print(" test 2")
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            #print(" test 3")
            pred = model(images)
            for c in comm:
                #print(" test 4")
                u = utils.compute_IoU(pred, targets, device, c[0], c[1])
                #print(u)
                results[model_name + str(c[0]) + "_" + str(c[1])] += u

        for k, v in results.items():
            results[k] = results[k]/step


        with open('C:/Users/ghait/PycharmProjects/NYU_CBRD/save/train/M1-01/IoU' + model_name + '.csv', 'w') as f:
            for key in results.keys():
                f.write("%s,%s\n" % (key, results[key]))









if __name__ == "__main__":
    main()