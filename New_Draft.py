import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import utils
import torch.profiler
from Main_model_2 import TileDataset, TileDatasetTest, get_model_instance_segmentation, albumentations_transforms
import PIL
import numpy as np
import transforms as T
import albumentations as A
from torchvision.models.detection import MaskRCNN
import matplotlib.pyplot as plt
from PIL import Image
from engine import train_one_epoch, evaluate
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from dash import Dash, html, dcc, Output, Input
import plotly.express as px
from matplotlib import image
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from dash import Dash, html, dcc

def show(t ,m=None):
    B = len(t)
    if B > 1 and m is None:
        f, axarr = plt.subplots(1, B)
        for i in range(B):
            axarr[i].imshow((t[i].permute(1, 2, 0)))
        plt.show()

    elif B > 1 and m is not None:
        f, axarr = plt.subplots(2, B)
        for i in range(B):
            axarr[0, i].imshow((t[i].permute(1, 2, 0)))
            axarr[1, i].imshow((m[i]['masks'].permute(1, 2, 0)))
        plt.show()

    elif B == 1 and m is None:
        plt.imshow(t[0].permute(1, 2, 0))
        plt.show()
    else:
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow((t[0].permute(1, 2, 0)))
        axarr[1].imshow((m[0]['masks'].permute(1, 2, 0)))
        plt.show()
    return

app = Dash(__name__)

final_score = np.ones((500,500))
fig = px.imshow(final_score)


app.layout = html.Div([
    html.Div(children=[
        html.H1('NYU Langone’s Center for Biospecimen Research and Development', style={"color": "DarkViolet"}),
        html.Label('confidence threshold selector'),
        dcc.Slider(
            min=0.5,
            max=1,
            marks={0.5:'0.5', 0.6: '0.6', 0.7:'0.7', 0.8: '0.8', 0.9:'0.9', 1.0: '1.0'},
            value=10, included=False, tooltip={'always_visible': True}
        ),
        html.Br(),
        dcc.Graph(
            id='example-graph',
            figure=fig
        )
    ], style={'padding': 10, 'flex': 1}),


], style={'display': 'flex', 'flex-direction': 'row'})

app.run_server(debug=True)

def main():
    torch.set_printoptions(edgeitems=20, precision=6)
    backbone = resnet_fpn_backbone('resnet101', True, trainable_layers=4)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model_instance_segmentation(2).to(device)
    model.eval()

    ckpt_dict = torch.load(r'C:\Users\ghait\PycharmProjects\NYU_CBRD\save\train\M1-01\checkpointM1_8.pth', map_location=device)
    model.load_state_dict(ckpt_dict['model'])
    model.eval()

    dataset_test = TileDatasetTest("C:/Users/ghait/Desktop/NYU_Project/Datasets/Testing/")
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=3, collate_fn=utils.collate_fn)


    while(True):
        images, targets = next(iter(test_loader))

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if targets[0]['masks'].sum() < 50000:
            print("Mask is black...")
            continue
        else:
            pred = model(images)
            print(pred[0]['scores'][0])
            print(pred[0]['masks'][0])
            #print(targets[0]['masks'])
            final_score = pred[0]['masks'][0] * pred[0]['scores'][0]
            print(final_score)

            #print(pred[0]['masks'][0].squeeze().detach().cpu().numpy().shape)
            #print(targets[0]['masks'].squeeze().detach().cpu().numpy().shape)
            a2 = sns.heatmap(targets[0]['masks'].squeeze().detach().cpu().to(torch.float).numpy())
            plt.show()
            ax1 = sns.heatmap(final_score.squeeze().detach().cpu().to(torch.float).numpy() > 0.75)
            plt.show()

        input('Enter...')



if __name__ == "__main__":
    main()