# ============================================================
# HW3 - Vision Transformer on CIFAR-10
# test.py: 方便測試不同架構的版本
# 可修改區域已標注 [MODIFIABLE]
# ============================================================

# ─────────────────────────────────────────────────────────────
# 學號設定
# ─────────────────────────────────────────────────────────────
student_id = 'r14725055'
assert student_id != 'xxxxx', "Please fill in your student ID."

# ─────────────────────────────────────────────────────────────
# Part 1: Imports  [DO NOT MODIFY]
# ─────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from scipy.io import loadmat

from torch.optim.optimizer import Optimizer

import torchvision.transforms as trns
from PIL import Image

from math import sqrt

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# ─────────────────────────────────────────────────────────────
# Part 2: Global variables  [DO NOT MODIFY]
# ─────────────────────────────────────────────────────────────
batch_size = 16
num_classes = 10
input_size = (32, 32, 3)
patch_size = 4
num_epoch = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")

# ─────────────────────────────────────────────────────────────
# Part 3: Dataloader  [DO NOT MODIFY]
# ─────────────────────────────────────────────────────────────
train_transform = trns.Compose([
    trns.ToTensor(),
])

test_transform = trns.Compose([
    trns.ToTensor(),
])

data_train = datasets.CIFAR10(root='./dataset/', train=True,  transform=train_transform, download=True)
data_test  = datasets.CIFAR10(root='./dataset/', train=False, transform=test_transform,  download=True)

train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(data_test,  batch_size=batch_size, shuffle=False)

# ─────────────────────────────────────────────────────────────
# Part 4: Tokenization  [DO NOT MODIFY]
# ─────────────────────────────────────────────────────────────
class myTokenization(nn.Module):
    def __init__(self, output_dim, patch_size, channels):
        super().__init__()
        patch_dim = patch_size * patch_size * channels
        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, output_dim)
        )

    def forward(self, x):
        return self.to_patch_tokens(x)


# ╔══════════════════════════════════════════════════════════╗
# ║  Part 5: myFFN  [MODIFIABLE]                            ║
# ║  FFN(x) = max(0, x W1 + b1) W2 + b2                    ║
# ╚══════════════════════════════════════════════════════════╝
class myFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        out = self.ffn(x)
        return out


# ╔══════════════════════════════════════════════════════════╗
# ║  Part 6: myAttention  [MODIFIABLE]                      ║
# ║  Multi-Head Scaled Dot-Product Attention                ║
# ║  DO NOT use torch.nn.MultiheadAttention()               ║
# ╚══════════════════════════════════════════════════════════╝
class myAttention(nn.Module):
    def __init__(self, input_dim, heads, head_dim, attn_dropout=0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner_dim = heads * head_dim
        self.scale = head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_dropout)

        self.to_qkv = nn.Linear(input_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, input_dim)

    def forward(self, x):
        b, n, _ = x.shape
        h = self.heads

        qkv = self.to_qkv(x)
        qkv = qkv.chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv
        )

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = self.attn_drop(dots.softmax(dim=-1))

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


# ╔══════════════════════════════════════════════════════════╗
# ║  Part 7: myTransformer  [MODIFIABLE]                    ║
# ║  自由設計層數、Dropout、LayerNorm 位置等                  ║
# ╚══════════════════════════════════════════════════════════╝
class myTransformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim):
        super().__init__()

        num_layers = 8  # ← 可調整層數

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                myAttention(dim, heads, dim_head),
                nn.Dropout(0.1),
                nn.LayerNorm(dim),
                myFFN(dim, mlp_dim),
                nn.Dropout(0.1),
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for norm1, attn, drop1, norm2, ffn, drop2 in self.layers:
            x = drop1(attn(norm1(x))) + x
            x = drop2(ffn(norm2(x))) + x
        return self.norm(x)


# ─────────────────────────────────────────────────────────────
# Part 8: myViT  [DO NOT MODIFY]
# ─────────────────────────────────────────────────────────────
class myViT(nn.Module):
    def __init__(self, input_size, patch_size, hidden_dim, heads, head_dim, mlp_dim, num_classes):
        super().__init__()

        image_height, image_width, channels = input_size
        patch_height, patch_width = patch_size, patch_size

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = patch_height * patch_width * channels

        self.to_input_token = myTokenization(hidden_dim, patch_size, channels)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim))

        self.transformer = myTransformer(hidden_dim, heads, head_dim, mlp_dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.to_input_token(x)

        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        x = self.transformer(x)
        x = x[:, 0]

        return self.mlp_head(x)


# ╔══════════════════════════════════════════════════════════╗
# ║  Part 9: 超參數  [MODIFIABLE]                            ║
# ╚══════════════════════════════════════════════════════════╝
model = myViT(
    input_size  = input_size,
    patch_size  = patch_size,
    hidden_dim  = 64,   # ← Token embedding 維度
    heads       = 16,   # ← Attention head 數量
    head_dim    = 4,    # ← 每個 head 的維度
    mlp_dim     = 96,   # ← FFN 中間層維度
    num_classes = num_classes,
).to(device)

# 模型大小確認（目標 < 1MB = 250K params）
total_params = sum(p.numel() for p in model.parameters())
size_mb = total_params * 4 / (1024 ** 2)
print(f"參數量: {total_params:,}")
print(f"模型大小: {size_mb:.3f} MB")

# ─────────────────────────────────────────────────────────────
# Part 10: Loss  [DO NOT MODIFY]
# ─────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()

# ─────────────────────────────────────────────────────────────
# Part 11: Optimizer  [DO NOT MODIFY]
# ─────────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# ─────────────────────────────────────────────────────────────
# Part 12: Training  [DO NOT MODIFY]
# ─────────────────────────────────────────────────────────────
model.train()

for epoch in range(num_epoch):
    losses = []

    for batch_num, input_data in enumerate(train_loader):
        optimizer.zero_grad()

        x, y = input_data
        x = x.to(device).float()
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()

        if batch_num % 500 == 0:
            print('\tEpoch %d | Batch %d | Loss %6.4f' % (epoch, batch_num, loss.item()))

    print('Epoch %d | Loss %6.4f' % (epoch, sum(losses) / len(losses)))

torch.save(model, student_id + '_submission.pt')

# ─────────────────────────────────────────────────────────────
# Part 13: Evaluation  [DO NOT MODIFY]
# ─────────────────────────────────────────────────────────────
import csv
model.eval()

with open(student_id + '_submission.csv', 'w') as f:
    fieldnames = ['ImageId', 'Prediction']
    writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator='\n')
    writer.writeheader()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, t in test_loader:
            x = x.to(device).float()
            output = model(x).argmax(dim=1)

            for y, l in zip(output, t):
                writer.writerow({fieldnames[0]: (total + 1),
                                 fieldnames[1]: y.item()})
                total += 1
                if y.item() == l.item():
                    correct += 1

    print('Accuracy: %6.4f' % (correct / total))
