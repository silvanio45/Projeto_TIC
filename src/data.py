import json
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
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
    return [(image_dataset_path / item['caminho'], item['rotulo']) for item in dados]

class FolhasDataset(Dataset):
    def __init__(self, amostras, transform=None):
        self.amostras = amostras
        self.transform = transform

    def __len__(self):  
        
        return len(self.amostras)
    
    def __getitem__(self, idx):
        caminho_imagem, rotulo = self.amostras[idx]
        imagem = Image.open(caminho_imagem).convert("RGB")
        if self.transform:
            imagem = self.transform(imagem)
        return imagem, rotulo

def get_dataloaders(batch_size=32, num_workers=4):
    df_treino = carrega_split("treino")
    df_teste = carrega_split("teste")
    df_validacao = carrega_split("validacao")

    ds_treino = FolhasDataset(df_treino, transform_treino)
    ds_teste = FolhasDataset(df_teste, transform_teste)
    ds_validacao = FolhasDataset(df_validacao, transform_teste)

    dataloader_treino = DataLoader(ds_treino, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader_teste = DataLoader(ds_teste, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dataloader_validacao = DataLoader(ds_validacao, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataloader_treino, dataloader_validacao, dataloader_teste

if __name__ == "__main__":
    loader_treino, loader_val, loader_teste = get_dataloaders()
    imagens, rotulos = next(iter(loader_treino))
    print(imagens.shape, rotulos[:10])