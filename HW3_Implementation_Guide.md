# HW3 Transformer 實作指南

> CIFAR-10 分類任務，使用 Vision Transformer (ViT)  
> 需要實作：Part 5 (FFN)、Part 6 (Attention)、Part 7 (Transformer)、Part 9 (超參數)

---

## 目錄

1. [Part 5 — myFFN](#part-5--myffn)
2. [Part 6 — myAttention](#part-6--myattention)
3. [Part 7 — myTransformer](#part-7--mytransformer)
4. [Part 9 — 超參數設計](#part-9--超參數設計)
5. [模型大小 vs 準確度的取捨](#模型大小-vs-準確度的取捨)

---

## Part 5 — myFFN

### 概念說明

FFN（Position-wise Feed-Forward Network）是 Transformer 的基本組成之一。  
它對 sequence 中**每個 token 獨立**地做兩層線性變換，中間加 ReLU 激活：

$$FFN(x) = \max(0,\ xW_1 + b_1)W_2 + b_2$$

- $W_1$：將維度從 `input_dim` 升維到 `hidden_dim`
- $W_2$：將維度從 `hidden_dim` 降回 `input_dim`
- 通常 `hidden_dim` 會是 `input_dim` 的 2～4 倍（升維再降維）

### 實作方式

```python
class myFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),   # W1, b1
            nn.ReLU(),                           # max(0, ...)
            nn.Linear(hidden_dim, input_dim)     # W2, b2
        )

    def forward(self, x):
        out = self.ffn(x)
        return out
```

### 設計建議

| 選項 | 說明 | 建議 |
|------|------|------|
| 加 Dropout | 在 ReLU 後加 `nn.Dropout(p)` 可防止 overfitting | ✅ 建議加，p=0.1 |
| hidden_dim 大小 | 通常為 input_dim 的 2～4 倍 | `mlp_dim` 在 Part 9 控制 |
| 其他激活函數 | GELU 比 ReLU 在 Transformer 中更常見 | 可試，但作業公式規定用 ReLU |

### ⚠️ 注意事項

- 公式要求用 ReLU（$\max(0, \cdot)$），不要換成其他激活函數（避免被扣分）
- FFN 的 **output 維度必須等於 input 維度**，因為 Transformer block 有殘差連接
- `hidden_dim`（即 `mlp_dim`）是升維用的中間維度，不是最終輸出維度

---

## Part 6 — myAttention

### 概念說明

#### Single-Head Attention（必做，20%）

Scaled Dot-Product Attention：

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- $Q$（Query）、$K$（Key）、$V$（Value）都是從輸入 $x$ 線性投影出來的
- $\sqrt{d_k}$ 是縮放因子，防止點積值過大導致 softmax 梯度消失
- 直觀理解：Q 問「我想要什麼？」，K 回答「我有什麼？」，注意力分數決定對每個 V 取多少

#### Multi-Head Attention（Bonus，+20%）

$$MultiHead(Q,K,V) = Concat(head_1, \dots, head_h)W^O$$
$$head_i = Attention(QW^Q_i,\ KW^K_i,\ VW^V_i)$$

- 將 `input_dim` 分成 `h` 個 head，每個 head 的維度是 `head_dim`
- 每個 head 學習不同角度的注意力模式（例如：短距離語意、長距離依賴）
- 最後 Concat 所有 head 的輸出，再乘以 $W^O$ 投影回原始維度

### 實作方式

#### Single-Head 版本

```python
class myAttention(nn.Module):
    def __init__(self, input_dim, heads, head_dim):
        super().__init__()
        # 這裡先忽略 heads，只用 head_dim
        self.scale = head_dim ** -0.5
        self.to_q = nn.Linear(input_dim, head_dim, bias=False)
        self.to_k = nn.Linear(input_dim, head_dim, bias=False)
        self.to_v = nn.Linear(input_dim, head_dim, bias=False)
        self.to_out = nn.Linear(head_dim, input_dim)

    def forward(self, x):
        q = self.to_q(x)                             # (B, N, head_dim)
        k = self.to_k(x)                             # (B, N, head_dim)
        v = self.to_v(x)                             # (B, N, head_dim)

        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale  # (B, N, N)
        attn = dots.softmax(dim=-1)                  # softmax over key 維度
        out = torch.einsum('bij,bjd->bid', attn, v)  # (B, N, head_dim)

        return self.to_out(out)                      # (B, N, input_dim)
```

#### Multi-Head 版本（建議實作，拿 Bonus）

```python
class myAttention(nn.Module):
    def __init__(self, input_dim, heads, head_dim):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner_dim = heads * head_dim                 # 所有 head 合併的維度
        self.scale = head_dim ** -0.5

        # 一次投影所有 head
        self.to_qkv = nn.Linear(input_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, input_dim)

    def forward(self, x):
        b, n, _ = x.shape
        h = self.heads

        # 一次算出 Q, K, V 並切分成 3 份
        qkv = self.to_qkv(x)                        # (B, N, 3 * h * head_dim)
        qkv = qkv.chunk(3, dim=-1)                  # 3 個 (B, N, h * head_dim)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv
        )                                            # 各自 (B, h, N, head_dim)

        # Scaled dot-product attention
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale  # (B, h, N, N)
        attn = dots.softmax(dim=-1)

        # 加權求和
        out = torch.einsum('bhij,bhjd->bhid', attn, v)   # (B, h, N, head_dim)
        out = rearrange(out, 'b h n d -> b n (h d)')      # (B, N, h * head_dim)

        return self.to_out(out)                      # (B, N, input_dim)
```

### 設計建議

| 選項 | 說明 | 建議 |
|------|------|------|
| Attention Dropout | 在 `attn` 後加 Dropout | ✅ 可加，p=0.1 |
| bias=False | QKV projection 通常不加 bias | ✅ 遵循慣例 |
| heads 數量 | 通常是 4 或 8 | 4 比較省記憶體 |
| head_dim | 通常 32 或 64 | 64 效果較好 |

### ⚠️ 注意事項

- **禁止使用 `torch.nn.MultiheadAttention()`**（直接被扣分）
- softmax 的 `dim=-1` 要對 Key 的維度做（即 N 的方向），不要搞錯
- Multi-Head 版本：`inner_dim = heads * head_dim`，`to_out` 的輸入維度是 `inner_dim`，輸出是 `input_dim`
- 使用 `einops.rearrange` 來 reshape，比手動 `view` 更不容易出錯

---

## Part 7 — myTransformer

### 概念說明

Transformer Block 的標準結構（Pre-LayerNorm 版，現代常用）：

```
x → LayerNorm → Attention → + x  (殘差)
  → LayerNorm → FFN        → + x  (殘差)
```

- **殘差連接（Residual Connection）**：讓梯度直接流過，解決深層網路的梯度消失
- **LayerNorm**：穩定訓練，通常放在 Attention/FFN 之前（Pre-LN）
- 多個 Transformer Block 堆疊起來就是完整的 Transformer

### 實作方式

```python
class myTransformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim):
        super().__init__()

        # 可以堆疊多層，這裡示範 N 層
        num_layers = 6  # 可以在這調整，或改成從外部傳入

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                myAttention(dim, heads, dim_head),
                nn.LayerNorm(dim),
                myFFN(dim, mlp_dim),
            ]))

    def forward(self, x):
        for norm1, attn, norm2, ffn in self.layers:
            x = attn(norm1(x)) + x    # 殘差 + Attention
            x = ffn(norm2(x)) + x     # 殘差 + FFN
        return out
```

> **注意**：上面 `return out` 應改為 `return x`，這裡只是示意。

### 設計自由度

這是整個作業中**最自由**的部分，你可以自行決定：

| 設計項目 | 選項 | 建議 |
|----------|------|------|
| **層數** | 1～12 層 | 4～6 層是個好起點（平衡大小與準確度） |
| **LayerNorm 位置** | Pre-LN（前）或 Post-LN（後） | ✅ Pre-LN 訓練更穩定 |
| **殘差連接** | 要不要加 | ✅ 必加，不然很難訓練 |
| **Dropout** | 各處可加 | 建議 p=0.1，防止過擬合 |
| **額外 LayerNorm** | 最後加一個 Final LayerNorm | ✅ 建議加 |

### 完整建議實作

```python
class myTransformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, num_layers=6, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                myAttention(dim, heads, dim_head),
                nn.Dropout(dropout),
                nn.LayerNorm(dim),
                myFFN(dim, mlp_dim),
                nn.Dropout(dropout),
            ]))

        self.norm = nn.LayerNorm(dim)  # Final LayerNorm

    def forward(self, x):
        for norm1, attn, drop1, norm2, ffn, drop2 in self.layers:
            x = drop1(attn(norm1(x))) + x
            x = drop2(ffn(norm2(x))) + x
        return self.norm(x)
```

### ⚠️ 注意事項

- `return` 的值必須是 shape `(B, N+1, dim)` 的 tensor（+1 是 CLS token）
- 不要破壞 sequence 的長度，`myViT` 後面會取 `x[:, 0]` 當 CLS token 做分類
- `num_layers` 可以在這裡設定，**不用改 Part 9** 的 `myViT()` 呼叫（因為 `myViT` 只傳 `dim, heads, head_dim, mlp_dim`）
- 如果想讓 `num_layers` 從外部控制，需要修改 Part 9 的 `myViT(...)` 呼叫方式，但 Part 8 是 `DO NOT MODIFY`，**不能改 `myViT` 的定義**，所以層數要在 `myTransformer.__init__` 裡面寫死或給預設值

---

## Part 9 — 超參數設計

### 參數說明

```python
model = myViT(
    input_size = input_size,   # 固定 (32, 32, 3)，不要動
    patch_size = patch_size,   # 固定 4，不要動
    hidden_dim = 64,           # Token embedding 維度
    heads = 4,                 # Attention head 數量
    head_dim = 64,             # 每個 head 的維度
    mlp_dim = 64,              # FFN 中間層維度
    num_classes = num_classes  # 固定 10，不要動
)
```

### 模型大小估算

模型大小主要由以下決定（`L` = 層數）：

| 組件 | 參數量估算 |
|------|-----------|
| Tokenization + CLS + PosEmb | ~`hidden_dim² × patch²` |
| 每層 Attention (QKV + out) | ~`4 × hidden_dim × (heads × head_dim)` |
| 每層 FFN | ~`2 × hidden_dim × mlp_dim` |
| MLP Head | ~`hidden_dim × num_classes` |

**1MB ≈ 250K 個 float32 參數**（1M bytes / 4 bytes per float）

### 推薦配置

| 目標 | hidden_dim | heads | head_dim | mlp_dim | 層數 | 估計大小 |
|------|-----------|-------|---------|---------|------|---------|
| 最小（省大小）| 32 | 4 | 32 | 64 | 4 | ~0.3MB |
| 平衡 | 64 | 4 | 64 | 128 | 4 | ~0.7MB |
| 較高準確度 | 64 | 4 | 64 | 256 | 6 | ~1.5MB |

> 要拿 5% 的大小分數，模型必須 < **1MB**，建議以「平衡」配置為起點

### 快速確認模型大小的方法

```python
total_params = sum(p.numel() for p in model.parameters())
size_mb = total_params * 4 / (1024 ** 2)
print(f"參數量: {total_params:,}")
print(f"模型大小: {size_mb:.2f} MB")
```

### ⚠️ 注意事項

- `hidden_dim` 必須可以被 `heads` 整除（Multi-Head Attention 需要均分）
- `head_dim` 不一定要等於 `hidden_dim // heads`，但通常這樣設計最直覺
- 增加 `mlp_dim` 對準確度有幫助，但影響模型大小
- `num_epoch = 10` 是固定的，層數太深可能訓練不夠充分

---

## 模型大小 vs 準確度的取捨

### 分數結構回顧

| 項目 | 分數 | 條件 |
|------|------|------|
| 模型大小基本分 | 5% | < 1MB |
| 模型大小排名分 | 10% | 班上排名 |
| 準確度基本分 | 5% | > 57% |
| 準確度排名分 | 10% | 班上排名 |
| 另一個資料集排名 | 10% | 班上排名 |

### 策略建議

1. **首先確保 < 1MB**：先拿穩 5% 的基本分
2. **目標準確度 > 57%**：再拿 5% 的基本分
3. 基本分都拿到後，再透過**調整超參數**衝排名

### 提升準確度的技巧（在不超過 1MB 的前提下）

- 在 `train_transform` 加入資料增強（`RandomCrop`、`RandomHorizontalFlip`）— 但 Part 3 是 `DO NOT MODIFY`，**不能改 dataloader**
- 可以在 `myTransformer` 加 Dropout 防止 overfitting
- 適當增加層數（4～6 層通常比 1～2 層好很多）
- `mlp_dim` 設為 `hidden_dim` 的 2～4 倍效果較好

---

## 實作建議順序

```
1. 實作 myFFN          → 最簡單，確認概念
2. 實作 myAttention    → 先做 single-head，確認 shape 正確
3. 實作 myTransformer  → 組裝，確認可以 forward 不報錯
4. 跑一次訓練           → 確認流程通
5. 升級為 multi-head   → 拿 Bonus 20%
6. 調整 Part 9 超參數  → 優化大小與準確度
```

每完成一步，建議先跑一個 batch 確認 shape 沒問題：

```python
# 快速測試
x = torch.randn(2, 65, 64)  # (batch=2, tokens=64+1cls, dim=64)
model_test = myTransformer(dim=64, heads=4, dim_head=64, mlp_dim=128)
out = model_test(x)
print(out.shape)  # 應該是 (2, 65, 64)
```
