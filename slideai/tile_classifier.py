
import numpy as np
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

