import os
import openslide
import numpy
import math
import numpy as np

import torch

from xml.etree import ElementTree
from PIL import Image, ImageDraw
from tqdm import tqdm

import torchvision.models as models
import torchvision.transforms as transforms

import gc


import torch
import torchvision.models as models
import torchvision.transforms as transforms

device = ("cuda" if torch.cuda.is_available() else "cpu")

def build_classifier_transforms(size):
    return transforms.Compose(
        [transforms.Resize((size, size)),
         transforms.ToTensor(),
         transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])


def build_classifier(path_to_pth):
    ''' Load a Pytorch pretrained classification model to classify tile as:
            0: Background/ Empty tile
            1: Tissue Tile (Tile contain at least 5% tissue

     '''
    classifier_device = ("cuda" if torch.cuda.is_available() else "cpu")
    B_T_Classifier_inf = models.vgg11_bn(pretrained=False, progress=True, num_classes=2)
    B_T_Classifier_inf.load_state_dict(torch.load(path_to_pth))
    B_T_Classifier_inf.eval()
    B_T_Classifier_inf.to(classifier_device)
    return B_T_Classifier_inf


class SlideAi (openslide.OpenSlide):
    """Class based on OpenSlide for slicing .svs WSI, creating tiles, converting .xml annotation to COCO .json format"""

    def __init__(self, filename):
        super().__init__(filename)
        self._filename = filename
        self._image_name = os.path.basename(self._filename) + "/"
        self.anno_dic = None                    # Dic to store annotation regions, their class, and vertices
        self.annotation_layers = None           # Store the number of annotation layer imported from .xml file
        self.annotation_regions = None          # Store the number of annotation regions across all layers
        self.annotation_binary_mask = None      # A numpy 2d binary array to represent annotation mask
        self.annotation_PIL_mask = None         # A PIL image to represent annotation mask
        self.classifier = None
        self.tiles_dir = None
        self.tiles_dir = None



    @property
    def slide_format(self) -> str:
        """The format of WSI file."""
        return self.detect_format(str(self._filename))


    @property
    def annotation_filename(self) -> str:
        """The name of annotation file if exists."""
        return os.path.splitext(self._filename)[0]+'.xml'

    @property
    def annotation_exists(self) -> bool:
        """Flag indicates annotation file exists."""
        return os.path.exists(self.annotation_filename)

    def read_annotation_file(self):
        if self.annotation_exists:
            try:
                self._annotation_tree = ElementTree.parse(self.annotation_filename)
            except:
                print("Can't open annotation file")
            else:
                self.anno_dic = {}
                self._annotation_root = self._annotation_tree.getroot()

                for idx_l, layer in enumerate(self._annotation_root):
                    for idx_r, region in enumerate(layer.iter('Region')):
                        self.anno_dic[str(idx_l + 1) + str(idx_r + 1)] = {}
                        self.anno_dic[str(idx_l + 1) + str(idx_r + 1)]['Class'] = region.attrib['Text']
                        self.anno_dic[str(idx_l + 1) + str(idx_r + 1)]['Layer'] = idx_l + 1
                        self.anno_dic[str(idx_l + 1) + str(idx_r + 1)]['Region'] = idx_r + 1
                        self.anno_dic[str(idx_l + 1) + str(idx_r + 1)]['Vertices'] = []
                        for idx_vl, region_vertex_list in enumerate(region.iter('Vertices')):
                            for idx_v, vertex_point in enumerate(region_vertex_list.iter('Vertex')):
                                self.anno_dic[str(idx_l + 1) + str(idx_r + 1)]['Vertices'].append(
                                    (int(vertex_point.attrib["X"]), int(vertex_point.attrib["Y"])))
        self.annotation_layers = idx_l
        self.annotation_regions = idx_l * idx_r


    def make_annotation_binary_mask(self):
        """Save binary mask as .png."""
        if self.anno_dic is None:
            self.read_annotation_file()
        if self.anno_dic is not None:
            self.list_of_regions_vertices = []
            for key in self.anno_dic:
                self.list_of_regions_vertices.append(self.anno_dic[key]['Vertices'])
        self.temp_image = Image.new('1', self.dimensions, 0)
        for polygon in self.list_of_regions_vertices:
            ImageDraw.Draw(self.temp_image).polygon(polygon, outline=1, fill=1)

        self.annotation_PIL_mask = self.temp_image

        self.temp_image.save(os.path.splitext(self._filename)[0]+'.png')


        #self.annotation_binary_mask = numpy.array(self.temp_image, dtype=numpy.bool)


    def build_tile_classifier(self, path_to_pth):
        if self.classifier is None:
            self.classifier = build_classifier(path_to_pth)

    def make_tiles(self, tile_sizes=(500), output_tile_size=(500,500), output_path=None, classify_tile=False, masked_only=False):

        self.outputs_path = os.path.dirname(self._filename) if output_path is None else output_path
        if classify_tile and self.classifier is None:
            print("Classifier is not found. build a classifier model with self.build_tile_classifier")
            return
        if classify_tile and self.classifier is not None:
            self.classifier_transforms = build_classifier_transforms(output_tile_size)

        if self.annotation_PIL_mask is not None:
            try:
                self.tiles_dir = self.outputs_path + "/tiles"
                self.masks_dir = self.outputs_path + "/masks"
                os.mkdir(self.tiles_dir)
                os.mkdir(self.masks_dir)
            except FileExistsError:
                pass
            except:

                print("Error - output directories not created")
                return

            #self.new_whole_image = Image.new('RGB', (int(self.dimensions[0]),int(self.dimensions[1])))

            #self.draw_on_new_image = ImageDraw.Draw(self.new_whole_image)



            for tile_size in tile_sizes:
                self.h_idx, self.v_idx = math.floor(self.dimensions[0] / tile_size), math.floor(
                    self.dimensions[1] / tile_size)
                for H in tqdm(range(self.h_idx), desc=" progress"):
                    for V in range(self.v_idx):
                        self.tile_RGB = (self.read_region((0 + tile_size * H, 0 + tile_size * V), 0, (tile_size, tile_size))).convert('RGB')
                        self.tile_RGB = self.tile_RGB.resize(output_tile_size)

                        if not classify_tile:
                            if not masked_only:
                                self.tile_RGB.save(self.tiles_dir + "/" +self._image_name[:-4]+"_"+ str(tile_size) +"_"+str(0+tile_size*H)+"_"+str(0+tile_size*V)+".jpg")
                                self.tile_mask = self.annotation_PIL_mask.crop((0 + tile_size * H,
                                                                                0 + tile_size * V,
                                                                                (0 + tile_size * H) + tile_size,
                                                                                (0 + tile_size * V) + tile_size))
                                self.tile_mask.save(
                                    self.masks_dir + "/"+ self._image_name[:-5] + "_" + str(tile_size) +"_"+ str(0 + tile_size * H) + "_" + str(0 + tile_size * V) + "_mask.png")
                                #self.new_whole_image.paste(self.tile_RGB, (0 + tile_size * H, 0 + tile_size * V))

                            if masked_only:
                                self.tile_mask = self.annotation_PIL_mask.crop((0 + tile_size * H,
                                                                                0 + tile_size * V,
                                                                                (0 + tile_size * H) + tile_size,
                                                                                (0 + tile_size * V) + tile_size))
                                self.tile_mask.resize(output_tile_size)

                                tile_mask_np = np.array(self.tile_mask)
                                if np.max(tile_mask_np) == 0: pass
                                if np.max(tile_mask_np) >  0:
                                    self.tile_mask.save(
                                        self.masks_dir + "/" + self._image_name[:-5] + "_" + str(tile_size) +"_"+ str(
                                            0 + tile_size * H) + "_" + str(0 + tile_size * V) + "_mask.png")
                                self.tile_RGB.save(
                                    self.tiles_dir + "/" + self._image_name[:-4] + "_" + str(tile_size) +"_"+ str(0 + tile_size * H) + "_" + str(
                                        0 + tile_size * V) + ".jpg")





                        if classify_tile:
                            if not masked_only:
                                self.tile_mask = self.annotation_PIL_mask.crop((0 + tile_size * H,
                                                                                0 + tile_size * V,
                                                                                (0 + tile_size * H) + tile_size,
                                                                                (0 + tile_size * V) + tile_size))
                                self.tile_mask.resize(output_tile_size)
                                self.img_tensor = self.classifier_transforms(self.tile_RGB)
                                self.img_tensor = self.img_tensor.unsqueeze(0)
                                self.img_tensor = self.img_tensor.to(tile_classifier.device)
                                self.class_output = self.classifier(self.img_tensor)
                                self.pred_class = torch.argmax(self.class_output)
                                if self.pred_class == 1:

                                    self.tile_RGB.save(
                                        self.tiles_dir + "/" + self._image_name[:-5] + "_" + str(tile_size) +"_"+ "T" + "_" + str(0 + tile_size * H) + "_" + str(
                                            0 + tile_size * V) + ".jpg")

                                    self.tile_mask.save(
                                        self.masks_dir + "/"+ self._image_name[:-5] + "_" + str(tile_size) +"_"+"T" + "_" + str(0 + tile_size * H) + "_" + str(
                                            0 + tile_size * V) + "_mask.png")

                            if masked_only:
                                self.tile_mask = self.annotation_PIL_mask.crop((0 + tile_size * H,
                                                                                    0 + tile_size * V,
                                                                                    (0 + tile_size * H) + tile_size,
                                                                                    (0 + tile_size * V) + tile_size))
                                self.tile_mask.resize(output_tile_size)

                                tile_mask_np = np.array(self.tile_mask)
                                if np.max(tile_mask_np) == 0:
                                    pass
                                else:
                                    self.img_tensor = self.classifier_transforms(self.tile_RGB)
                                    self.img_tensor = self.img_tensor.unsqueeze(0)
                                    self.img_tensor = self.img_tensor.to(tile_classifier.device)
                                    self.class_output = self.classifier(self.img_tensor)
                                    self.pred_class = torch.argmax(self.class_output)
                                    if self.pred_class == 1:
                                        self.tile_RGB.save(
                                            self.tiles_dir + "/" + self._image_name[:-5] + "_" + str(tile_size) +"_"+ "T" + "_" + str(
                                                0 + tile_size * H) + "_" + str(
                                                0 + tile_size * V) + ".jpg")

                                        self.tile_mask.save(
                                            self.masks_dir + "/" + self._image_name[:-5] + "_" + str(tile_size) +"_"+ "T" + "_" + str(
                                                0 + tile_size * H) + "_" + str(
                                                0 + tile_size * V) + "_mask.png")



                            #self.new_whole_image.paste(self.tile_RGB, (0 + tile_size * H, 0 + tile_size * V))
                #del self.tile_RGB, self.tile_mask

        #self.new_whole_image.save(self.outputs_path + "/"+ self._image_name[:-5] + "new_WSI.png")







