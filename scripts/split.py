import json
from pathlib import Path
from torchvision import datasets
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
image_dataset_path = BASE_DIR / 'data' / 'raw' / 'PlantVillage'
splits_dir = BASE_DIR / 'config' / 'splits'

dataset = datasets.ImageFolder(image_dataset_path)

X_temp, X_test, y_temp, y_test = train_test_split(
    dataset.samples, dataset.targets,
    test_size=0.2, random_state=42, stratify=dataset.targets)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.125, random_state=42, stratify=y_temp)

splits_dir.mkdir(parents=True, exist_ok=True)

def salvar_split(nome, X, y):
    # caminhos relativos a data/raw/PlantVillage, para funcionar em qualquer máquina
    dados = [{"caminho": Path(caminho).relative_to(image_dataset_path).as_posix(),
              "rotulo": int(rotulo)}
             for (caminho, _), rotulo in zip(X, y)]
    with open(splits_dir / f"{nome}.json", "w") as f:
        json.dump(dados, f, indent=2)

salvar_split("treino", X_train, y_train)
salvar_split("validacao", X_val, y_val)
salvar_split("teste", X_test, y_test)

print(f"Splits salvos em {splits_dir}")
print(f"Treino: {len(X_train)} | Validação: {len(X_val)} | Teste: {len(X_test)}")