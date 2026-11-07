from pathlib import Path

import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score

from data import get_dataloaders
from model import cria_modelo

EPOCAS = 10
LR = 1e-4
BATCH_SIZE = 32

BASE_DIR = Path(__file__).resolve().parent.parent
CAMINHO_MODELO = BASE_DIR / "models" / "baseline_resnet50.pth"


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
    model = cria_modelo().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    CAMINHO_MODELO.parent.mkdir(exist_ok=True)
    melhor_f1 = 0.0

    for epoca in range(EPOCAS):
        loss_treino = treinar_epoca(model, train_loader, loss_fn, optimizer, device)
        loss_val, acc_val, f1_val = avaliar(model, val_loader, loss_fn, device)

        msg = (f"Época {epoca+1}/{EPOCAS} | loss treino {loss_treino:.4f} | "
               f"loss val {loss_val:.4f} | acc val {acc_val:.4f} | F1 val {f1_val:.4f}")

        if f1_val > melhor_f1:
            melhor_f1 = f1_val
            torch.save(model.state_dict(), CAMINHO_MODELO)
            msg += "  ✔ salvo"
        print(msg)

    print(f"Melhor F1 macro na validação: {melhor_f1:.4f} -> {CAMINHO_MODELO}")


def treinar_epoca(model, loader, loss_fn, optimizer, device):
    model.train()
    loss_total = 0.0
    for imagens, rotulo in loader:
        imagens, rotulo = imagens.to(device), rotulo.to(device)
        optimizer.zero_grad()
        saida = model(imagens)
        loss_batch = loss_fn(saida, rotulo)
        loss_batch.backward()
        optimizer.step()
        loss_total += loss_batch.item()
    return loss_total / len(loader)


def avaliar(model, loader, loss_fn, device):
    model.eval()
    loss_total = 0.0
    todas_preds, todos_rotulos = [], []
    with torch.no_grad():
        for imagens, rotulo in loader:
            imagens, rotulo = imagens.to(device), rotulo.to(device)
            saida = model(imagens)
            loss_total += loss_fn(saida, rotulo).item()
            todas_preds.extend(saida.argmax(dim=1).cpu().tolist())
            todos_rotulos.extend(rotulo.cpu().tolist())

    acc = accuracy_score(todos_rotulos, todas_preds)
    f1 = f1_score(todos_rotulos, todas_preds, average="macro")
    return loss_total / len(loader), acc, f1


if __name__ == "__main__":
    main()
