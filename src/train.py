
import torch
from data import get_dataloaders


if __name__ == "__main__":
    torch.device("cuda" if torch.cuda.is_available() else "cpu")