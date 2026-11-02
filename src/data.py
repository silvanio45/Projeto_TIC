import json
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
image_dataset_path = BASE_DIR / 'data' / 'raw' / 'PlantVillage'
SPLITS_DIR = BASE_DIR / 'config' / 'splits'

transform_treino = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_teste = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
def carrega_split(nome_split):
    ''' carregando os splits a partir dos arquivos JSON gerados pelo script split.py '''
    with open(SPLITS_DIR / f"{nome_split}.json", "r") as f:
        dados = json.load(f)
    return [(image_dataset_path, item['rotulo']) for item in dados]

if __name__ == "__main__":

    df_treino =carrega_split("treino")

    print(f"Splits carregados de {SPLITS_DIR}")
    print(df_treino[0])
    print(f"Total de imagens de treino: {len(df_treino)}")