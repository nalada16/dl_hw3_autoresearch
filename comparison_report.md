# Autoresearch vs AutoML：比較分析報告

**任務背景：** HW3 Vision Transformer on CIFAR-10  
**作者：** r14725055  
**日期：** 2026-05-28

---

## 一、概念定義

### Autoresearch（本次使用）
由 AI Agent（Claude）驅動的**知識引導式**實驗迭代循環。每一輪實驗由 Agent 根據上一輪的結果，運用領域知識與推理，**手動設計下一個假設**，修改程式、執行訓練、評估結果、決定保留或捨棄，再進入下一輪。核心概念來自 Andrej Karpathy 的 `autoresearch`。

### AutoML（自動機器學習）
透過**演算法自動搜尋**最佳的模型架構與超參數組合，不需要人（或 Agent）逐一推理每個決策。常見形式包括：

| 類型 | 說明 | 代表工具 |
|---|---|---|
| 超參數最佳化（HPO） | 自動調整 learning rate、dropout、層數等 | Optuna、Ray Tune、SMAC |
| 神經架構搜尋（NAS） | 自動設計網路結構 | DARTS、ENAS、AutoKeras |
| 完整 Pipeline 自動化 | 含資料前處理、模型選擇、後處理 | Auto-Sklearn、H2O AutoML |

---

## 二、核心差異比較

### 2.1 搜尋策略

| 面向 | Autoresearch | AutoML |
|---|---|---|
| 搜尋方式 | 序列式、一次改一個變因 | 可平行化；Grid / Random / Bayesian / 演化 |
| 下一步決策依據 | Agent 的推理與領域知識 | 統計代理模型（如高斯過程）或隨機採樣 |
| 搜尋空間定義 | 隱式（由 Agent 臨時決定） | 顯式（事先宣告的超參數範圍） |
| 能否探索結構性變化 | ✅ 可以（如 Pre/Post-LN、Parallel Block） | ❌ HPO 不行；NAS 可以但實現複雜 |

**本次實例：** autoresearch 能嘗試「GPT-2 std=0.02 初始化」、「平行 attention+FFN 結構（GPT-J style）」、「LN 位置改變」——這些都是**離散的結構性決策**，純粹的 HPO-based AutoML 無法探索，因為它們不在任何預設的超參數空間內。

---

### 2.2 搜尋空間

```
Autoresearch 本次探索的空間（異質混合）：
─── 連續型：attn_dropout（0.0~0.25）、trans_dropout（0.0~0.1）
─── 離散型：num_layers（6/7/8/9）、heads（4/8/16/32）、mlp_dim（64~192）
─── 結構型：Pre-LN vs Post-LN、Sequential vs Parallel、
            Stochastic Depth、QKV bias、GPT-2 init、
            Asymmetric dropout、Graduated dropout

AutoML（HPO）通常只能處理：
─── 連續型 ✅
─── 離散型 ✅
─── 結構型 ❌（需要搭配 NAS）
```

在本次任務中，**最大的兩個突破**都來自結構性決策：
1. `attn_dropout` 從「沒有」到「有」（+1.42 pp）— HPO 可以調值，但發現「要不要加」需要推理
2. GPT-2 init（+0.38 pp）— 純粹是知識引導的想法，AutoML 完全無從得出

---

### 2.3 評估策略

| 面向 | Autoresearch（本次） | 標準 AutoML |
|---|---|---|
| 每次評估 | 完整訓練 10 個 epoch | 常用 Early Stopping / 學習曲線外插節省時間 |
| 評估集 | **測試集**（直接用 test accuracy 做 keep/discard） | 應使用**驗證集**，測試集只報最終結果 |
| 評估次數 | 59 次 | Bayesian Opt. 通常 50~200 次；Random Search 可能更多 |
| 時間成本 | 59 × ~5 分鐘 ≈ 約 5 小時（序列） | 若可平行：同預算可評估 10× 更多組合 |

> ⚠️ **重要注意：** 本次 autoresearch 使用測試集作為 keep/discard 的決策依據，這在嚴格的機器學習實驗設計中屬於「**測試集洩漏（test set leakage）**」。最終報告的 65.99% 有一定程度的樂觀偏差。正規 AutoML 會保留測試集到最後，用獨立驗證集做 HPO，避免此問題。

---

### 2.4 樣本效率

| | Autoresearch | Random Search | Bayesian Optimization |
|---|---|---|---|
| 找到 `attn_dropout=0.22` 需要幾次試驗 | ~8 次（有方向性） | 期望 ~15 次（1/0.05 範圍） | ~5~8 次 |
| 找到 `num_layers=7` 最優 | ~5 次 | 期望 ~4 次（1/4 個離散值） | ~3~4 次 |
| 找到結構性突破 | ✅ 能發現 | ❌ 不適用 | ❌ 不適用 |
| 整體效率 | **中等偏高**（推理有方向，但偶有錯誤假設） | 低 | 高（連續空間） |

---

### 2.5 可重現性與可解釋性

| | Autoresearch | AutoML |
|---|---|---|
| 每次實驗追蹤 | ✅ Git commit + results.tsv | 視工具而定（Optuna 有 DB、Ray Tune 有 log） |
| 決策可解釋性 | ✅ 高（每次有明確假設和推理） | ❌ 低（Bayesian 模型是黑盒） |
| 可重現性 | ✅ 高（固定 seed 可重現） | 通常 ✅（若記錄超參數組合） |
| 知識積累 | ✅ 發現的規律可遷移（如 attn_dropout 的重要性） | ❌ 只輸出最佳超參數，不解釋「為什麼」 |

---

## 三、本次實驗的 Autoresearch 流程圖

```
┌─────────────────────────────────────────────────────────────┐
│                   Autoresearch 迭代循環                      │
│                                                             │
│  讀取結果 ──→ Agent 推理下一個假設 ──→ 修改 test.py          │
│      ↑                                                ↓     │
│  評估結果                                        git commit  │
│  (accuracy + .pt size)                                ↓     │
│      ↑                                     uv run test.py   │
│  keep/discard 決定 ←──────────────────── 讀取 run.log        │
│      │                                                      │
│      ├─ keep: 保留 commit，以此為下一輪起點                  │
│      └─ discard: git reset --hard，回到上一個 keep           │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、本次各階段發現的分類

### 能被 AutoML（HPO）有效找到的
- ✅ `attn_dropout` 的最佳值（0.22）— 連續超參數
- ✅ `num_layers` 的最佳值（7）— 離散超參數
- ✅ `trans_dropout` 的最佳值（0.05）— 連續超參數
- ✅ `mlp_dim` 的最佳值（128）— 離散超參數
- ✅ `heads` 的最佳值（16）— 離散超參數

### 只有 Autoresearch（或人工）才能找到的
- ✅ **加入 `attn_dropout`** 這個設計決策（知識：ViT 注意力容易過擬合）
- ✅ **GPT-2 std=0.02 初始化**（知識：語言模型領域的技巧遷移）
- ✅ **FFN init 不能用 std=0.02**（知識：ReLU 應用 Kaiming，而非固定 std）
- ✅ **非對稱 dropout 設計**（attn_dropout >> trans_dropout）
- ✅ **排除 Post-LN 架構**（知識：Pre-LN 收斂更穩定）
- ✅ **排除 Parallel Attention+FFN**（知識：GPT-J 的設計不適合小模型）
- ✅ **attn dropout 與 heads 數量的交互效應**（16 heads 需要更高的 attn_dropout）

---

## 五、優缺點總結

### Autoresearch

| 優點 | 缺點 |
|---|---|
| 可探索離散結構性變化 | 序列執行，速度慢 |
| 每個決策有明確推理，可解釋 | 推理可能帶入錯誤偏見（confirmation bias）|
| 發現的知識可遷移、積累 | 測試集被用於決策（可能有洩漏問題） |
| 能整合跨領域知識（如 GPT-2 init） | 依賴 Agent 的知識品質 |
| 彈性高，可隨時調整搜尋方向 | 無法有效探索高維超參數空間 |

### AutoML

| 優點 | 缺點 |
|---|---|
| 可平行化，效率高 | 無法探索結構性/算法性創新 |
| 在定義好的搜尋空間內系統性覆蓋 | 需要事先定義搜尋空間（需要領域知識） |
| Bayesian Opt. 在連續空間非常有效 | 結果難以解釋（「為什麼」這組超參數好？）|
| 避免人為偏見 | 容易困在局部最優（特別是 Random Search）|
| 正確使用時不會有測試集洩漏 | 計算預算難以預測 |

---

## 六、延伸討論

### 6.1 AutoML 能設計「前後層不同」的架構嗎？

假設想讓 5 層 Transformer 中，前三層和後兩層採用不同的設計：

```
Layer 1 ─┐
Layer 2  ├─ 前三層：attn_dropout=0.3，heads=16
Layer 3 ─┘
Layer 4 ─┐
Layer 5  ┘─ 後兩層：attn_dropout=0.1，heads=8
```

**情況一：沒有預先定義 → AutoML 完全做不到**

HPO 預設所有層共用同一組超參數，它根本不知道「層之間可以不同」這件事：

```python
attn_dropout = trial.suggest_float("attn_dropout", 0.0, 0.3)
# 所有層套同一個值，無法自行問出「要不要讓每層獨立？」
```

**情況二：有預先定義 → AutoML 可以搜尋，但搜尋空間爆炸**

```python
for i in range(5):
    dropout_i = trial.suggest_float(f"attn_dropout_layer{i}", 0.0, 0.3)
```

即使這樣定義，搜尋空間變成原來的 5 次方，需要更多實驗才能收斂。而且「要這樣問」這個想法本身，仍然是人先想出來的。

**核心限制：AutoML 回答的是「在這個設計空間裡哪組數值最好」，但「設計空間的邊界應該畫在哪裡」，AutoML 無法自己決定。**

---

### 6.2 有 Model Size 約束時，AutoML 的根本弱點

本次作業有 1 MB 的硬性限制，這個場景完整暴露了 AutoML 的另一個根本弱點：

**AutoML 把約束當作事後濾網：**

```
NAS 流程：
  生成架構 A → 訓練 10 epoch → 測 size → 1.5MB ❌ discard（浪費）
  生成架構 B → 訓練 10 epoch → 測 size → 0.8MB ✅ 記錄
  生成架構 C → 訓練 10 epoch → 測 size → 1.2MB ❌ discard（浪費）
```

大量實驗資源浪費在「訓練完才發現超過限制」上，而且它不知道如何把剩餘 budget 用在刀口上。

**Autoresearch 把約束當作設計資源：**

```
本次實驗後期的推理過程：
  「目前 1,018,731 bytes，剩餘預算 29,845 bytes」
       ↓
  「head_dim=5 需要多 114,688 bytes → 超預算，直接排除，不浪費一次實驗」
       ↓
  「attn_dropout 是超參數，佔 0 bytes → 重點探索方向」
       ↓
  「第 8 層需要多 133,120 bytes → 超預算，排除」
       ↓
  「結論：剩餘預算只夠探索不佔 size 的旋鈕（dropout 值、init 方式）」
```

這種**在約束框架內主動推理最優路徑**的能力，AutoML 目前完全做不到。

| | NAS/AutoML | Autoresearch |
|---|---|---|
| 對待 size 限制 | 事後二元過濾（pass/fail） | 事前資源分配（budget planning） |
| 遇到超限 | discard，再重新隨機 | 計算「哪些改動在預算內」後才決定試什麼 |
| 搜尋方向引導 | 不受 size 影響 | **主動往不佔 size 的方向集中探索** |
| 對「免費改動」的認識 | 無 | ✅ 明確知道 dropout 值不佔任何 size |

**本次最後階段的突破（attn_dropout 0.20→0.22，+0.73 pp）就是這個邏輯的直接體現**：在預算耗盡的情況下，autoresearch 知道只能動「不花 size 的旋鈕」，因而集中火力把 attn_dropout 調到精確最優值。AutoML 在這個場景下，要麼撞牆（超 size）要麼隨機亂試，不會有這樣的推理。

---

## 七、結論

本次實驗說明 **autoresearch 與 AutoML 是互補而非替代的關係**：

- **AutoML** 適合在**已知設計**的前提下，高效搜尋連續/離散超參數（如確定要用 attn_dropout 之後，Optuna 可以快速找到 0.22 這個最佳值）。

- **Autoresearch** 適合**探索未知設計空間**——發現「要加什麼」、「架構應該長什麼樣子」，這是目前 AutoML 無法自動完成的部分。

在本次任務中，**最重要的兩個突破**（加入 attn_dropout、GPT-2 init）都是 AutoML 找不到的**設計層次的發現**。一旦這些設計決策確立，剩下的細部調參（dropout 值從 0.15 → 0.20 → 0.22）才是 AutoML 的強項。

> 理想的工作流程：**Autoresearch 定義設計** → **AutoML 精調超參數**

```
Autoresearch（本次）:  61.28% → 65.08%   ← 結構性突破，靠知識
AutoML-style 精調:    65.08% → 65.99%   ← 連續超參數搜尋
```

兩者結合，才能在有限預算內取得最佳結果。

---

*本報告基於 autoresearch/may28 分支上的 59 次實驗，最終最佳結果：65.99% test accuracy @ 0.972 MB。*
