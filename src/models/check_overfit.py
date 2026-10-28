import torch
from torch import nn
from src.data.waveform_data import load_split
from src.models.cnn import ECGCNN

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
tr = load_split("train")
x, y = tr.x[:64].to(dev), tr.y[:64].to(dev)
model = ECGCNN().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()
for step in range(1, 201):
    loss = loss_fn(model(x), y)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0:
        print(step, round(loss.item(), 4))