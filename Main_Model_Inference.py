import os
import math
import torch
import torchvision
import argparse
import torch.utils.data
import torch.profiler
import slideai.slideai

import cv2 as cv
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
from PIL import ImageFilter
from PIL import ImageDraw
from tqdm import tqdm
from datetime import datetime
from xml.etree import ElementTree
from os.path import exists
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers
from torchvision.models.detection import MaskRCNN


def parse_args():
    parser = argparse.ArgumentParser(description='Instance Segmentation Inference')
    parser.add_argument('--model', type=int, default=2, help='1 for resnet50_fpn, 2 for resnet101_fpn')
    parser.add_argument('--path', type=str, default="", help='Path to trained models')
    parser.add_argument('--trained', type=int, default=7, help='Key of trained model dic')
    parser.add_argument('--classifier', type=str, default="", help='path to trained classifier')
    parser.add_argument('--tile_size', type=int, default=500, help='tile size')
    parser.add_argument('--mask_score', type=float, default=0.95, help='Mask score threshold')
    parser.add_argument('--pixel_threshold', type=float, default=0.95, help='Mask score threshold')
    parser.add_argument('--scale', type=int, default=1, help='scale factor')
    parser.add_argument('--wsi', type=str, default=None, help='WSI file path or directory containing WSI files')

    args = parser.parse_args()
    return args


def get_trained_classifier(path, device, num_classes=2):
    """Return trained tile classifier """
    classifier = models.vgg11_bn(pretrained=False, progress=True, num_classes=2)
    classifier.load_state_dict(torch.load(path))
    classifier.eval()
    classifier.to(device)
    return classifier


def get_segmentation_model_resnet50_fpn():

    num_classes = 2
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def get_model_instance_segmentation_resnet101(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    trainable_backbone_layers = torchvision.models.detection.backbone_utils._validate_trainable_layers(False,2 , 5, 3)
    backbone = resnet_fpn_backbone('resnet101', True, trainable_layers=trainable_backbone_layers)
    model = MaskRCNN(backbone, num_classes)
    print(model)
    #model = torchvision.models.detection.maskrcnn_resnet101_fpn()
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_model_instance_segmentation(model_backbone, num_classes=2):
    try:
        if model_backbone == "resnet50_fpn":
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        elif model_backbone == "resnet101_fpn":
            trainable_backbone_layers = torchvision.models.detection.backbone_utils._validate_trainable_layers(False,
                                                                                                               2, 5, 3)
            backbone = resnet_fpn_backbone('resnet101', True, trainable_layers=trainable_backbone_layers)
            model = MaskRCNN(backbone, num_classes)

        else:
            raise ValueError()

    except:
        print('Valid argument must be selected.')

    else:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           num_classes)
        return model

def load_trained_model(model_backbone, path_trained_model, device, num_classes=2):
    trained_model = get_model_instance_segmentation(model_backbone, num_classes)
    trained_model.load_state_dict(torch.load(path_trained_model))
    trained_model.eval()
    trained_model.to(device)
    return  trained_model

def segment_wsi(wsi, classifier_model, main_model, tile_size=500, mask_score=0.975, pixel_score=0.955, scale=10, output_path=None):


    outputs_path = os.path.dirname(wsi) if output_path is None else output_path
    sld = slideai.slideai.SlideAi(wsi)
    file_name = os.path.basename(wsi)[:-4]

    Pred_seg_mask = Image.new('RGB',
                              (int(sld.dimensions[0]),
                               int(sld.dimensions[1])))

    draw_on_pred_mask = ImageDraw.Draw(Pred_seg_mask)

    h_idx, v_idx = math.floor(sld.dimensions[0] / tile_size),\
                   math.floor(sld.dimensions[1] / tile_size)

    device = ("cuda" if torch.cuda.is_available() else "cpu")

    if tile_size == 500:
        TT = transforms.Compose(
            [transforms.ToTensor(),                           # Removed transforms.Resize((tile_size, tile_size)),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

    if tile_size != 500:
        TT = transforms.Compose(
            [transforms.Resize((tile_size, tile_size)),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    for H in tqdm(range(h_idx), desc=" progress"):
        for V in range(v_idx):

            tile_RGB = (sld.read_region((0 + tile_size * H,
                                         0 + tile_size * V),
                                        0,
                                        (tile_size, tile_size))).convert('RGB')
            img_tensor = TT(tile_RGB)
            img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(device)
            class_output = classifier_model(img_tensor)
            pred_class = torch.argmax(class_output)
            if pred_class == 1:
                with torch.no_grad():
                    test_img_np = np.array(tile_RGB)
                    test_img_T = F.to_tensor(test_img_np)
                    test_img_T2 = torch.unsqueeze(test_img_T, 0)
                    test_img_T_GPU = test_img_T2.to(device)
                    main_output = main_model(test_img_T_GPU)

                    if main_output[0]["scores"][0] > mask_score:
                        pred_mask = main_output[0]["masks"][0].cpu().numpy()
                        pred_mask = pred_mask[0, :, :]
                        max_value = np.amax(pred_mask)
                        pred_mask = pred_mask/max_value
                        pred_mask = pred_mask > pixel_score
                        pred_bbx = main_output[0]["boxes"][0].cpu().numpy().astype(int)
                        pred_mask_PIL = Image.fromarray(pred_mask)
                        Pred_seg_mask.paste(pred_mask_PIL, (0 + tile_size * H, 0 + tile_size * V))

    original_size = Pred_seg_mask.size
    Pred_seg_mask_scaled = Pred_seg_mask.resize((math.ceil(original_size[0]/scale),math.ceil(original_size[1]/scale)))
    Pred_seg_mask_scaled.save(outputs_path + str(file_name) +
                              "_pred_scaled_P"+str(pixel_score) +
                              "_S" +
                              str(mask_score)+"_B4.png")
    return Pred_seg_mask


def segment_wsi_updated(wsi, classifier_model, main_model, device, tile_size=500, mask_score=0.975, pixel_score=0.955, scale=1):

    outputs_path, file_name = (os.path.dirname(wsi), os.path.basename(wsi)[:-4])

    sld = slideai.slideai.SlideAi(wsi)

    Pred_seg_mask = Image.new('RGB',
                              (int(sld.dimensions[0]),
                               int(sld.dimensions[1])))

    #draw_on_pred_mask = ImageDraw.Draw(Pred_seg_mask)

    h_idx, v_idx = (math.floor(sld.dimensions[0] / tile_size),
                    math.floor(sld.dimensions[1] / tile_size))


    if tile_size == 500:
        TT = transforms.Compose(
            [transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

    elif tile_size != 500:
        TT = transforms.Compose(
            [transforms.Resize((tile_size, tile_size)),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    My_to_tensor = transforms.Compose(
            [transforms.ToTensor()])

    for H in tqdm(range(h_idx), desc=" progress"):
        for V in range(v_idx):

            tile_RGB = (sld.read_region((0 + tile_size * H,
                                         0 + tile_size * V),
                                        0,
                                        (tile_size, tile_size))).convert('RGB')

            img_tensor = My_to_tensor(tile_RGB)
            img_tensor = img_tensor.to(device)
            img_tensor = TT(img_tensor)
            img_tensor = img_tensor.unsqueeze(0)

            class_output = classifier_model(img_tensor)
            pred_class = torch.argmax(class_output)

            if pred_class == 1:
                with torch.no_grad():
                    #test_img_np = np.array(tile_RGB)
                    #test_img_T = F.to_tensor(tile_RGB)
                    test_img_T = My_to_tensor(tile_RGB)
                    test_img_T = test_img_T.to(device)
                    test_img_T2 = test_img_T.unsqueeze(0)


                    main_output = main_model(test_img_T2)

                    if main_output[0]["scores"][0] > mask_score:
                        pred_mask = main_output[0]["masks"][0].cpu().numpy()
                        pred_mask = pred_mask[0, :, :]
                        max_value = np.amax(pred_mask)
                        pred_mask = pred_mask/max_value
                        pred_mask = pred_mask > pixel_score
                        pred_bbx = main_output[0]["boxes"][0].cpu().numpy().astype(int)
                        pred_mask_PIL = Image.fromarray(pred_mask)
                        Pred_seg_mask.paste(pred_mask_PIL, (0 + tile_size * H, 0 + tile_size * V))

    original_size = Pred_seg_mask.size
    Pred_seg_mask_scaled = Pred_seg_mask.resize((math.ceil(original_size[0]/scale),math.ceil(original_size[1]/scale)))

    Pred_seg_mask_scaled.save(outputs_path + "/" + file_name +
                              "_pred_scaled_P"+str(pixel_score) +
                              "_S" + str(mask_score)+".png")
    return Pred_seg_mask


def fill_gaps(path, filter_size=25, scale=1):
    img = Image.open(path)
    original_size = img.size
    img_small = img.resize((original_size[0]/scale,original_size[1]/scale))
    img_small_new = img_small.filter(ImageFilter.MaxFilter(size=7))
    img_small_new2 = img_small_new.filter(ImageFilter.MaxFilter(size=filter_size))
    #img_original_size = img_small_new2.resize(original_size)

    return img_small_new2


def get_main_contours(img) -> list:
    im = cv.imread(img)

    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    main_contours_indices = list([i for i, x in enumerate(hierarchy[0]) if x[3] == -1])
    main_contours = []

    for i in main_contours_indices:

        one_contour = np.squeeze(contours[i])
        one_contour_list = one_contour.tolist()
        first_point = one_contour_list[0]
        one_contour_list.append(first_point)
        main_contours.append(one_contour_list)
        del one_contour, first_point, one_contour_list
    return main_contours


def get_xml_of_AI_anno(lists, output_path, file_name, scale=10):
    ver_list = lists

    anno_root = ElementTree.Element('Annotations', MicronsPerPixel="0.252500")
    tree_AI = ElementTree.ElementTree(anno_root)

    Annotation_node = ElementTree.Element('Annotation')
    Annotation_node.set('Id', '1')
    Annotation_node.set('Name', '')
    Annotation_node.set('ReadOnly', '0')
    Annotation_node.set('NameReadOnly', '0')
    Annotation_node.set('LineColorReadOnly', '0')
    Annotation_node.set('Incremental', '0')
    Annotation_node.set('Incremental', '0')
    Annotation_node.set('Type', '4')
    Annotation_node.set('LineColor', '65280')
    Annotation_node.set('Visible', '1')
    Annotation_node.set('Selected', '1')
    Annotation_node.set('MarkupImagePath', '')
    Annotation_node.set('MacroName', '')

    Attributes_node = ElementTree.Element('Attributes')
    regions_node = ElementTree.Element('Regions')
    Plots_node = ElementTree.Element('Plots')

    anno_root.append(Annotation_node)
    Annotation_node.append(Attributes_node)
    Annotation_node.append(regions_node)
    Annotation_node.append(Plots_node)

    for index, one_list in enumerate(ver_list):
        region_node = ElementTree.Element('Region')
        Vertices_node = ElementTree.Element('Vertices')
        regionAttributeHeaders_node = ElementTree.Element('RegionAttributeHeaders')
        AttributeHeader1_node = ElementTree.Element('AttributeHeader', Name="Region", Id="1", ColumnWidth="-1")
        AttributeHeader2_node = ElementTree.Element('AttributeHeader', Name="Length", Id="1", ColumnWidth="-1")
        AttributeHeader3_node = ElementTree.Element('AttributeHeader', Name="Area", Id="1", ColumnWidth="-1")
        AttributeHeader4_node = ElementTree.Element('AttributeHeader', Name="Text", Id="1", ColumnWidth="-1")

        regions_node.append(regionAttributeHeaders_node)
        regions_node.append(region_node)
        region_node.append(Vertices_node)
        regionAttributeHeaders_node.append(AttributeHeader1_node)
        regionAttributeHeaders_node.append(AttributeHeader2_node)
        regionAttributeHeaders_node.append(AttributeHeader3_node)
        regionAttributeHeaders_node.append(AttributeHeader4_node)

        region_node.set('Id', str(index+1 * 4000))
        region_node.set('Type', '0')
        region_node.set('Zoom', '1')
        region_node.set('Selected', '1')
        region_node.set('ImageLocation', '')
        region_node.set('ImageFocus', '-1')
        region_node.set('Length', '0.0')
        region_node.set('Area', '0.0')
        region_node.set('LengthMicrons', '0.0')
        region_node.set('AreaMicrons', '0.0')
        region_node.set('Text', "")
        region_node.set('NegativeROA', '0')
        region_node.set('InputRegionId', '0')
        region_node.set('Analyze', '1')
        region_node.set('DisplayId', str(index+1))

        for v in one_list:
            if isinstance(v, list):
                One_Vertice_node = ElementTree.Element('Vertices', Z="0", Y=str(v[1]*scale), X=str(v[0]*scale))
                Vertices_node.append(One_Vertice_node)
                del One_Vertice_node


    tree_AI.write(output_path+file_name)





def main():

    model_backbones = [
            "resnet50_fpn",
            "resnet101_fpn"
            ]

    trained_models = {
            1: "M_RCNN_ResN50_Epoch12.pth",
            2: "M_RCNN_ResN101_Epoch5.pth",
            3: "M_RCNN_ResN101_Epoch6.pth",
            4: "M_RCNN_ResN101_Epoch7.pth",
            5: "M_RCNN_ResN101_Epoch8.pth",
            6: "M_RCNN_ResN101_Epoch9.pth",
            7: "M_RCNN_ResN101_Epoch10.pth",
            8: "M_RCNN_ResN101_Epoch11.pth"
    }

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    args = parse_args()

    tile_classifier = get_trained_classifier(args.classifier, device)

    segmentation_model = load_trained_model(model_backbones[args.model], args.path, device)




if __name__ == "__main__":
    main()