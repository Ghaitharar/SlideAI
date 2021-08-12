# SlideAi

SlideAi is a tool to read Aperio scanner histology whole slide images .svs files and their associated .xml annotation files, slice WSI into tiles, and making tile masks for segmentation.

## Built With:
OpenSlide-python
Pillow
PyTorch

## Usage
Converting Aperio scanner histology WSI in .svs format and their annotation .xml files to .jpg tiles and .png binary masks. This could be a part of a larger data-preprocessing pipeline for instance segmentation training/inference. The slicer can classify each tile as background/tissue tile and keep only tissue tiles; however, user needs to pass a pre-trained model to the classifier function which uses TorchVision vgg11_bn with two output classes.
