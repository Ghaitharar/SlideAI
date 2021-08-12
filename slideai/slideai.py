import os
import openslide
import numpy
import math
from xml.etree import ElementTree
from PIL import Image, ImageDraw
import tile_classifier
import torch


class SlideAi (openslide.OpenSlide):
    """Class based on OpenSlide for slicing WSI, creating tiles, converting .xml annotation to COCO .json format"""

    def __init__(self, filename):
        super().__init__(filename)
        self._filename = filename
        self._image_name = os.path.basename(self._filename)
        self.anno_dic = None                    # Dic to store annotation regions, their class, and vertices
        self.annotation_layers = None           # Store the number of annotation layer imported from .xml file
        self.annotation_regions = None          # Store the number of annotation regions across all layers
        self.annotation_binary_mask = None      # A numpy 2d binary array to represent annotation mask
        self.annotation_PIL_mask = None         # A PIL image to represent annotation mask
        self.classifier = None

    @property
    def slide_format(self) -> str:
        return self.detect_format(str(self._filename))

    @property
    def annotation_filename(self) -> str:
        return os.path.splitext(self._filename)[0]+'.xml'

    @property
    def annotation_exists(self) -> bool:
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
        self.annotation_binary_mask = numpy.array(self.temp_image)

    def build_tile_classifier(self, path_to_pth):
        if self.classifier is None:
            self.classifier = tile_classifier.build_classifier(path_to_pth)

    def make_tiles(self, tile_size=500, output_path=None, classify_tile=False):
        if classify_tile and self.classifier is None:
            print("Classifier is not found. build a classifier model with self.build_tile_classifier")
            return
        if classify_tile and self.classifier is not None:
            self.classifier_transforms = tile_classifier.build_classifier_transforms(tile_size)

        if self.annotation_PIL_mask is not None:
            self.new_whole_image = Image.new('RGB', (int(self.dimensions[0]),
                                                     int(self.dimensions[1])))
            self.draw_on_new_image = ImageDraw.Draw(self.new_whole_image)
            self.outputs_path = os.path.dirname(self._filename) if output_path is None else output_path
            self.h_idx, self.v_idx = math.floor(self.dimensions[0]/tile_size), math.floor(self.dimensions[1]/tile_size)

            for H in range(self.h_idx):
                for V in range(self.v_idx):
                    self.tile_RGB = (slide.read_region((0 + tile_size * H, 0 + tile_size * V), 0, (tile_size, tile_size))).convert('RGB')

                    if not classify_tile:
                        self.tile_RGB.save(self.outputs_path+self._image_name[:-4]+"_"+str(0+tile_size*H)+"_"+str(0+tile_size*V)+".jpg")
                        self.tile_mask = self.annotation_PIL_mask.crop((0 + tile_size * H,
                                                                        0 + tile_size * V,
                                                                        (0 + tile_size * H) + tile_size,
                                                                        (0 + tile_size * V) + tile_size))
                        self.tile_mask.save(
                            self.outputs_path + self._image_name[:-4] + "_" + str(0 + tile_size * H) + "_" + str(0 + tile_size * V) + "_mask.png")
                        self.new_whole_image.paste(self.tile_RGB, (0 + tile_size * H, 0 + tile_size * V))
                    if classify_tile:
                        self.tile_mask = self.annotation_PIL_mask.crop((0 + tile_size * H,
                                                                        0 + tile_size * V,
                                                                        (0 + tile_size * H) + tile_size,
                                                                        (0 + tile_size * V) + tile_size))
                        self.img_tensor = self.classifier_transforms(self.tile_RGB)
                        self.img_tensor = self.img_tensor.unsqueeze(0)
                        self.img_tensor = self.img_tensor.to(tile_classifier.device)
                        self.class_output = self.classifier(self.img_tensor)
                        self.pred_class = torch.argmax(self.class_output)
                        if self.pred_class == 1:
                            self.tile_RGB.save(
                                self.outputs_path + self._image_name[:-4] + "_" + "T" + "_" + str(0 + tile_size * H) + "_" + str(
                                    0 + tile_size * V) + ".jpg")
                            self.tile_mask.save(
                                self.outputs_path + self._image_name[:-4] + "_" + "T" + "_" + str(0 + tile_size * H) + "_" + str(
                                    0 + tile_size * V) + "_mask.png")
                            self.new_whole_image.paste(self.tile_RGB, (0 + tile_size * H, 0 + tile_size * V))
        self.new_whole_image.save(self.outputs_path + self._image_name[:-4] + "new_WSI.png")







