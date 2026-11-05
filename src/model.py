import torch
from torch import nn
from torchvision import models


def cria_modelo(num_class = 15):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_class)
    return model
    

if __name__ == "__main__":
    modelo = cria_modelo()
    x = torch.randn(4, 3, 224, 224)
    saida = modelo(x)
    print(saida.shape)