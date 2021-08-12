# SlideAi

SlideAi is a tool to read Aperio scanner histology whole slide images .svs files and their associated .xml annotation files, slice WSI into tiles, and make tile masks for segmentation.

## Built With:
OpenSlide-python

Pillow

PyTorch

## Usage
Converting Aperio scanner histology WSI in .svs format and their annotation .xml files to .jpg tiles and .png binary masks. This could be a part of a larger data-preprocessing pipeline for instance segmentation training/inference. The slicer can classify each tile as background/tissue tile and keep only tissue tiles; however, user needs to pass a pre-trained model to the classifier function which uses TorchVision vgg11_bn with two output classes.

## Example
This example show the result of slicing a WSI .svs file obtain from a public and open source. The WSI has a resulotion of 32001 x 38474 with a mock annotation .xml file that contains 5 random annotations. 

`
slide = slideai.SlideAi("test_1.svs")
slide.read_annotation_file()
slide.make_annotation_binary_mask()
slide.build_tile_classifier("trained_classification_model.pth")  # two input pre-trianed tile classifier
slide.make_tiles(tile_size=500, output_path="output_path/", classify_tile=True)
`

![image](https://user-images.githubusercontent.com/54161236/129137381-3ad0e516-bb36-426f-a6d0-d8ca550170ba.png)

![image](https://user-images.githubusercontent.com/54161236/129137501-376deaa2-e208-4801-9632-9b5c63a87aa7.png)

Each tile is 500 x 500; the classifier disrecarded background tiles.

![image](https://user-images.githubusercontent.com/54161236/129137888-2bcd9fb6-12d5-4508-af01-7fa8532aa1f0.png)

for each tile, a corresponding 500x500 .png binary mask is generated.

![image](https://user-images.githubusercontent.com/54161236/129137789-28dd4e97-6f6f-46c6-992e-3d781a3c10a6.png)

