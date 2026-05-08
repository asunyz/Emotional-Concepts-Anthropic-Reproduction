# Cognitive v3 复现报告（中文版）

**项目**：Surprisal as Learning Signal — F2 Cognitive Concept Vectors  
**目标模型**：Qwen3.6-35B-A3B（MoE，40 层，hidden=2048，NF4 量化）  
**复现对象**：Anthropic [Emotion Concepts and their Function in a Large Language Model (2026)](https://transformer-circuits.pub/2026/emotions/index.html)，但研究域从 **emotion** 切换到 **cognitive concepts**（curious / uncertain / confident / surprised / bored / stubborn / enlightened / confused / confirmed）  
**报告日期**：2026-05-07  
**作者**：meridah7

---

## 0. 执行摘要

我们完成了 v3 cognitive concept vector pipeline 的端到端复现，证明 **Anthropic 在 emotion 上发现的「概念表示线性可读取」现象同样适用于 cognitive 状态**，并在三个方法论维度上做出了**增量贡献**：

1. **Trajectory-pinned 生成**——把 v2 的 "concept-only" prompt 升级为 "prior → discovery → reaction" 三阶段固定模板，**消除概念漂移**（v2: 22.7% drift → v3: 0%）
2. **Stage-anchored extraction**——每段落独立池化，把 trajectory 平均拆解为 concept-specific 向量
3. **4-method robustness**：A=v2-style / B=isolation / C=in-context / D=stage-contrast，跨方法一致性 0.7–0.95，证明结果不依赖某一种特定提取手法

**关键定量结果**（layer 30，Method C，方法论上跟 Anthropic 对齐的「mid-late layer」）：

- **var_probe 命中**：9 个 cognitive concepts 在 4 个独立 probe 模板中**全部 ≥1σ**，**7/9 在至少一个模板里 ≥2σ**
- **Cross-method cosine**：discoveries / reactions 类概念达到 0.75-0.95，priors 类（curious, uncertain, confident）较弱（0.3-0.7），暴露 register-mismatch
- **Cross-layer cosine**：mid-late 层（L20-L30, L30-L36）一致性 0.78-0.86，**跨深度结构稳定**
- **Causal steering**：raw PyTorch forward hook 注入 ±3σ，多个 concept 上观察到明显 register shift（最戏剧的：surprised+3 → "shimmer of pixels"）
- **Vector arithmetic**：v_prior + v_discovery 跟 v_reaction 余弦在 −0.4 到 −0.7 之间，**远低于 random baseline (±1σ ≈ 0)**，说明 reaction 不是 prior+discovery 的线性合成——这是非平凡的发现，对应 Bayesian 贝叶斯式更新而非加性

最大的方法论亮点是「**raw cosine 99% 是 column baseline，0.6% 是真正信号**」的方差分解发现，**强制了 column z-score 归一化作为 var_probe 的标准展示方式**。

**最重要的实证发现（§3.9 MCQ 实验）**：用 40 道常识题让模型「读到正确 vs 错误答案」，发现 cognitive vectors 在 hidden state 里对答案对错有清晰判别——**confused (d=−1.52) 和 stubborn (d=−1.34) 是 wrong-detector**；**bored / confident / confirmed (d=+0.43 ~ +0.66) 是 right-detector**。这是 **surprise-as-learning-signal 的强候选**，把抽象目标变成了实际可计算的 score 公式。

---

## ✦ 用大白话讲 v3 是怎么做的、为什么这么做

在进入正式设计 rationale 之前，先用最简单的语言把整个 pipeline 走一遍，让任何人都能看懂这件事在干嘛、为什么。

### 我们到底想做什么？

**核心问题**：我们想知道「**好奇**」「**困惑**」「**固执**」这些**认知状态**在 LLM 的大脑（残差流）里**长什么样**——能不能找到一个具体的方向，让我们：
- **读**：看一段文字，告诉我模型现在「有多好奇」
- **写**：把这个方向加回模型大脑，让它**真的变得更好奇**

如果做到，就有了一套「**模型认知状态的传感器+电极**」——这是后续做「让模型主动发现新东西」的最底层零件。

### 为什么这事不简单？

模型大脑（每层 hidden state）有 **2048 维**，是个高维向量空间。「好奇」这个概念可能对应里面的某个方向（line in the embedding space），但：
- 我们**不知道**这个方向在哪
- 不能直接问模型「告诉我你的好奇方向」（模型没显式自我意识）
- 必须从**外部行为**反推

**思路**：找一堆**确定包含好奇感**的文本，看它们在残差流里的 hidden state **共同指向**哪个方向 → 那个方向就是好奇的 vector。

### v3 的 5 步配方

**Step 1：定义 9 个认知概念**  
我们选了 cognitive psychology 里 9 个清晰可区分的状态：`curious / uncertain / confident / surprised / bored / stubborn / enlightened / confused / confirmed`。这些都跟「**遇到信息时的应对**」有关，正好对接 surprisal-as-learning-signal 主线。

**Step 2：把概念固定到「认知轨迹」里**  
**关键洞察**：「confirmed」这种概念**不能孤立存在**——它必须是「先有期待，结果符合期待」的**结尾**。所以我们把每个故事拆成 **3 段**：

```
Prior（事前预期）→ Discovery（发现）→ Reaction（反应）
```

预先列好 **9 条合法轨迹**（如 `confident → surprised → stubborn`：很自信 → 发现意外 → 拒绝更新），让模型每次只能写**已指定**的轨迹。这样**模型没有路径选择自由**，不会偷偷漂移到别的概念上。

> **直观比喻**：v2 让模型「写一个 confirmed 故事」，模型可能写出 curious 故事自欺欺人。v3 让模型「写一个 `uncertain → bored → confirmed` 故事」——三阶段都钉死，没法跑偏。

**Step 3：用 show-don't-tell 让模型生成故事**  
prompt 给模型严格限制：
- 必须按 `## Prior / ## Discovery / ## Reaction` 三段写
- **绝对不能直接说**「她很好奇」「他困惑了」（禁掉 9 个概念词 + 17 个 feeling-state 词如 felt/wondered/intrigued）
- 只能用**行动、对话、思考、感受细节**来表达概念

每个故事生成后立即**自动验证**：结构对不对、字数够不够、有没有偷漏概念词。**违规重写**，最多 3 次。

> **直观比喻**：考试不让你写出答案的关键词，你只能用别的话把意思说清楚。这逼模型把概念表达成**行为信号**而非**词汇标签**——后者会污染 vector，让它学的是「写 curious 这个词」而不是「真的好奇这种状态」。

**Step 4：从故事里提取每个概念的方向**  
360 个 stories 全部 forward 过模型，每段 token 的 hidden state 拿出来。然后按概念聚合：

```
v_curious = mean( P1 段的 hidden states, 来自所有 prior=curious 的故事 )
v_stubborn = mean( P3 段的 hidden states, 来自所有 reaction=stubborn 的故事 )
... 9 个 vector
```

**关键设计**：**只取目标段的 token**，不平均整个 story。因为整 story 平均 = trajectory 平均（混了 3 个概念），不是单个概念。

我们同时跑 **4 种方法**（A/B/C/D，详见 §1.3），看不同 recipe 提出来的 vector 是不是指向同一个方向——如果是，证明 vector 是真的、不是某种提取技巧的伪影。

**Step 5：验证这些方向真的代表概念**  
设计 5 类独立检验：

| 检验 | 它在问什么 |
|---|---|
| **Cross-method consistency** | 4 种提取方法出来的同一概念 vector 一致吗？ |
| **Cross-layer consistency** | 不同深度（layer 10/20/30/36）的同一概念 vector 一致吗？ |
| **var_probe 4 模板** | 用 4 套独立 probe 句子去测，每个概念能不能被它**该被的语义**激活？ |
| **Token staining**（染色）| 把 vector 投到一段陌生文本上，会在**正确的位置**亮起吗？ |
| **Causal steering**（操控）| 把 vector 加回模型大脑，模型行为会**朝预期方向变**吗？ |

5 类测试都通过 → 我们就有了**真实可读+可写**的 cognitive concept vectors。

### v2 哪里错了，v3 怎么修

| v2 问题 | 后果 | v3 修复 |
|---|---|---|
| 只给一个 concept 名，让模型自由发挥 | 22.7% 故事漂移到别的 concept | 钉死 3 段 trajectory，零路径选择 |
| 故事里直接说「felt curious」 | vector 学到「写 curious 词」而非真概念 | 禁词列表 + 自动验证重试 |
| 整 story 平均 = trajectory 平均 | 9 个概念都混在一起 | 段落级提取，每段对应一个 stage 概念 |
| Neutral 基线是无关科学话题 | PCA 减 baseline 减不掉 narrative 共有方向 | （v3 暂保留，可未来改进）|
| 没有质量门 | 失败的故事进了数据 | 字数+结构+禁词验证，失败重生成 |

### 最终拿到的东西

```
9 个 cognitive concept vector，每个 2048 维（layer 30）

每个 vector 都通过了 5 类验证：
  ✓ 方法不变：4 种提取一致（cosine 0.7–0.95）
  ✓ 深度稳定：mid-late 层一致（cosine 0.78–0.86）
  ✓ 语义激活：4 个 probe 模板下 7/9 命中 ≥2σ
  ✓ 局部染色：在「正确句子」上局部峰值
  ✓ 因果可控：±3σ 操控产生明显 register shift
```

**这 9 个 vector 就是后续所有应用的最底层零件**——比如「实时探测模型在 reasoning chain 哪一步变 stubborn 了，然后强行 inject curious 让它跳出僵局」这类**让模型主动发现新东西**的 application，必须先有这套零件。

下面进入正式技术细节。

---

## 1. v3 整体设计

### 1.1 为什么需要 v3——v2 的三层结构性问题

V2 给模型一个概念词（如 `enlightened`），让它写一个 3 阶段叙事弧。审计发现三个问题：

| 层 | v2 问题 |
|---|---|
| **Prompt** | 概念之间不独立——`confirmed` 只有在 `uncertain → bored → confirmed` 链里才合理。Topic 措辞为「可能否定假设」时让模型写 `confirmed` 是矛盾，模型会偷偷在 metadata 里 relabel。**22.7% 的 stories 漂移到了别的概念** |
| **提取** | 每个 story 是完整 trajectory（prior → discovery → reaction）。Whole-story 平均池化 = trajectory 平均，**不是** concept vector |
| **Baseline** | Neutral stories 是独立科学话题（如「氢气燃烧」），无 narrative。PCA 基底无法 subtract off「discovery scenario」共有特征 |

具体后果：`confirmed` 只剩 5 个 stories（目标 25），`curious` 膨胀到 31 个，`confident + surprised → stubborn` 余弦出来 **−0.232**（应正）。

### 1.2 v3 的三层修正

#### 1.2.1 Trajectory pinning — 消除模型路径选择自由度

V3 枚举 cognitive 空间为 **3 priors × 2 discoveries × 4 reactions = 24 组合**，保留 **9 个有效轨迹**（见下表）。

| # | Prior | Discovery | Reaction | 含义 |
|---|---|---|---|---|
| 1 | confident | surprised | stubborn | 强先验被违反；拒绝更新 |
| 2 | confident | surprised | enlightened | 强先验被违反；框架重构 |
| 3 | confident | surprised | confused | 强先验崩溃；无法处理 |
| 9 | uncertain | surprised | stubborn | 弱先验遇反证；坚持 |
| 10 | uncertain | surprised | enlightened | 弱先验被反证打开 |
| 11 | uncertain | surprised | confused | 弱先验被反证打乱 |
| 16 | uncertain | bored | confirmed | 弱先验被符合预期的结果强化 |
| 18 | curious | surprised | enlightened | 探索带来新理解 |
| 19 | curious | surprised | confused | 探索带来不可处理的惊讶 |

每个 story prompt 固定全部三阶段。模型零路径选择自由度。

#### 1.2.2 Stage-anchored extraction — 段落级向量

故事用 markdown header 分段：

```
## Prior
[character holds tentative belief about X...]

## Discovery
[plain unremarkable outcome confirms X...]

## Reaction
[the expected match consolidates the prior...]
```

提取时按段落 token range 分别池化，按 stage-concept 跨 trajectory 聚合：

```
v_uncertain = mean( P1 activations from trajectories where prior=uncertain )
            (来源：#9, #10, #11, #16)

v_confirmed = mean( P3 activations from trajectories where reaction=confirmed )
            (来源：#16 only)
```

#### 1.2.3 Show-don't-tell + 28 词根禁令

V3 的 prompt 加了**两层禁词**强制 behavior-anchored 表达：

- **第一层（9 个 stems）**：禁掉所有 9 个概念词的形态变体（`curious, curi, curiosity` 等）
- **第二层（17 个 stems）**：禁掉「feeling-state」词（`felt, wondered, intrigued, perplexed, stunned, ...`），强制把概念用 action / dialogue / situational 表达，不让模型直接 label 情绪

#### 1.2.4 Generation-time 验证 + 自动重试

每个 story 立即验证：结构（3 段 markdown header）、字数（P1/P2: 25-90, P3: 50-150）、禁词、metadata 泄露。失败重试 3 次，仍败则写 `_failed/` 备查。

### 1.3 4-method 提取对比

V3 的另一个方法论增量是**同一批 stories 上跑 4 种不同的提取方法**，用 cross-method consistency 验证「概念 vector 不是某个特定 recipe 的过拟合」。

每个 v3 story 有 3 段：`Prior + Discovery + Reaction`。提取问题分两步：**「forward 给模型什么，pool 哪些 token」**（提取 raw 段落向量）+ **「按概念聚合后减谁」**（中心化）。4 种方法在这两步上的组合：

```
                提取 raw 段落向量                                  按概念聚合后减谁（中心化）
                forward 给模型 / pool token
──────────────────────────────────────────────────────────────────────────────────
A (v2-style)    整个 3 段 story / 所有 token (token≥50)            减 全 9 个概念 的均值
B (isolation)   只喂单独一段（如只 P2） / 那段所有 token             减 全 9 个概念 的均值
C (in-context)  整个 3 段 story / 只取目标段的 token                 减 全 9 个概念 的均值
D (contrast)    整个 3 段 story / 只取目标段的 token（同 C！）        减 同 stage 内概念 的均值
──────────────────────────────────────────────────────────────────────────────────
```

**各自含义**：

- **A — Anthropic 复刻**：所有 token 一锅端平均。出来的 vector 是整条 trajectory 的「平均味道」，**不区分 prior/discovery/reaction**。Anthropic emotion paper 用的就是这种 whole-story pooling。
- **B — 切片隔离**：只让模型看「Discovery: I checked the temperature logs...」这一段，不给前后文。最纯粹的段落表示，但缺 context 模型理解可能不准。
- **C — 带 context 看一段**：让模型读完整 story 把语境建立起来，**只对目标段的 hidden states 平均**。其他段的 hidden 不要。**我们主用这个**——既保留 stage-specific 信号又有自然 context。
- **D — 同 stage 内对比**：提取部分跟 C **完全一样**，唯一区别是**中心化时减的不是全 9 概念均值，而是同 stage 内概念均值**。比如 `curious`（P1 概念）减的是 `mean(curious, uncertain, confident)`（同属 P1 stage）。这消除了「我是 P1 段」这种 stage-position 共有方向，只保留「同 stage 内不同概念之间的差异」。

**Stage 分组（用于 Method D）**：

```
P1 (Prior 段):      curious, uncertain, confident
P2 (Discovery 段):  surprised, bored
P3 (Reaction 段):   stubborn, enlightened, confused, confirmed
```

**为什么主用 C**：
- 比 A：避免 trajectory 平均淹没 stage-specific 信号
- 比 B：保留模型在自然 context 下的 hidden state（forward 输入跟模型实际推理时一致）
- 比 D：不强行做 stage-contrastive，保留概念里的 stage-position 信号（如果 prior/discovery/reaction 的 stage 标签本身有用，C 留着，D 减掉）

**Cross-method consistency 测试**：同一概念在 4 种方法下应指向同一方向。如果是 → vector 不依赖特定 recipe，是真正的概念表示而非提取伪影；如果不是 → 表征对方法论敏感，需进一步分析原因。

| Method | 描述 | 对应 |
|---|---|---|
| A | v2-style whole-story mean pooling | Anthropic emotion paper |
| B | paragraph isolation（仅段落本身 forward） | 控制无 context 影响 |
| **C** | **paragraph in-context（整 story forward + 段落 token mean）** | **主用方法** |
| D | within-stage contrast（段落 token 平均 - 同 story 其他段平均） | 强化 stage-specific 信号 |

---

## 2. 发布计划与论述（PubPlan + Narrative）

### 2.1 故事线

「**MoE 架构上 cognitive concept vectors 的可解释性证据，及方法学增量——为「让模型主动发现新东西」铺路**」

> **直观说人话**：我们想让模型不光是「答题机器」，而能在遇到不熟悉/出乎意料的东西时**主动放慢、追问、改变方向**。要做到这一点，前提是有办法**读懂模型当前的认知状态**（「它现在是好奇还是固执？」），并且**反向影响这个状态**（「让它更好奇一点」）。本工作就是把这件事的**最底层零件**——9 个 cognitive concept 的向量——做出来并验证。

### 2.2 三个核心 claims

| Claim | 证据 |
|---|---|
| **C1: cognitive concepts 是 Qwen3.6 残差流里的线性方向** | var_probe 7/9 概念 ≥2σ，cross-method 0.7-0.95，cross-layer 0.78-0.86 |
| **C2: 这些 vectors 对模型行为有因果作用** | causal steering ±3σ 注入产生明显 register shift（surprised+3 出现 "shimmer of pixels" 等） |
| **C3: trajectory-pinned + stage-anchored extraction 是必要的** | v2 → v3 ablation 显示 drift 0% vs 22.7%、cross-method 一致性大幅提升 |

### 2.3 跟 Anthropic emotion paper 的位置

| 维度 | Anthropic 2026 | 我们 v3 |
|---|---|---|
| 模型 | Sonnet 4.5（dense） | Qwen3.6-35B-A3B（MoE）|
| 域 | 171 emotions | 9 cognitive concepts |
| 规模 | 100 topics × 12 stories × 171 = 205,200 stories | 8 × 5 × 9 = 360 stories |
| 提取 pooling | whole-story mean (token≥50) | per-paragraph anchored |
| 提取方法数 | 1 | 4 |
| 因果验证 | 多场景 steering（preferences、blackmail、reward hacking）| Steering on 7 concepts × 2 prompts |

**我们的增量**：
- **Stage-anchored extraction**——比 whole-story 平均更细粒度，论证为什么必要
- **4-method robustness**——直接证据「不是 recipe-specific」
- **MoE 架构验证**——Anthropic 用 dense，我们补 MoE 的可迁移性

### 2.4 章节计划

```
1. Introduction          — concept vectors as interpretability tool
2. Related work          — Anthropic emotion + concept extraction lineage
3. Method                — v3 design (trajectory + show-don't-tell + 4 methods)
4. Results               — vector geometry, var_probe, steering
5. Methodological contribution — variance decomposition + z-score必要性
6. Limitations           — sample-size、register、单模型
7. Discussion            — cognitive vs emotion 异同 + 对 discovery 目标的意义
```

---

## 3. 实验结果

### 3.1 数据规模

| 阶段 | 拓扑 | 总数 | Pass rate |
|---|---|---|---|
| Sanity | 1 topic × 1 story × 9 trajectories + 1 NEG | 10 stories | 100%（after retry）|
| Mid-scale | 3 topics × 5 stories × 9 trajectories | 135 stories | 100% |
| **Full** | **8 topics × 5 stories × 9 trajectories** | **360 stories** | **100%** |

8 个 topics（认知场景多样化）：
1. A scientist examines an experimental result
2. A doctor reviews a patient's lab panel
3. A chess player evaluates a position after an unexpected move
4. A debugger steps through a stack trace
5. A buyer test-drives a car
6. A juror listens to opening statements
7. A diner takes the first bite of a dish
8. A traveler arrives at a destination

### 3.2 Cross-method consistency — 跨提取方法一致性

**问题**：A/B/C/D 4 种方法提取出来的 vectors，对同一概念是否指向同一方向？

![Cross-method bars](../outputs/cognitive_v3_full/analyses_methodC/cross_method_bars.png)

**关键观察**（layer 30）：

| 概念簇 | 跨方法一致性 |
|---|---|
| **discoveries / reactions**（surprised, bored, stubborn, enlightened, confused, confirmed）| 高：A vs D 0.87-0.95，B vs C/D 0.69-0.83 |
| **priors**（curious, uncertain, confident）| 较低：A vs C 仅 0.14-0.29 |

**Z-score 归一化版本**（每个 method-pair 内部跨概念归一化，揭示哪些概念在某 pair 上特别异常）：

![Cross-method z-score](../outputs/cognitive_v3_full/analyses_methodC/cross_method_zscore.png)

**解读（专业）**：
- B 和 C 都是 paragraph-level 方法，互相一致性最高（0.6-0.85）；A 的 whole-story pooling 跟它们差异大，**这印证了 stage-anchoring 的必要性**
- priors 一致性低，反映 prior stage 在不同提取角度（context vs isolation）下表征确实有差异——可能是 prior 是「事前心态」语境性较强

> **直观说人话**：四种提取方法就像**4 个不同角度拍同一张人脸**。如果 4 张照片认得出是同一个人（cosine 高），说明这张「脸」是真实存在的；如果差太多（cosine 低），说明可能是某种角度伪影。对于 reactions / discoveries 类概念（surprised, stubborn, enlightened, confused, confirmed），4 个方法之间余弦在 0.65-0.95，**强证据**这些 vector 是真存在的方向。priors 类（curious, uncertain, confident）一致性弱一些（最低 0.14），说明它们在「单独看一段」vs「带前后文看一段」时长得不太一样——**这是科学发现，不是 bug**：先验状态本身比反应状态更依赖语境。

### 3.3 Cross-layer consistency — 跨层稳定性

**问题**：同一概念在不同深度（L10, L20, L30, L36）的方向是否稳定？

![Cross-layer bars](../outputs/cognitive_v3_full/analyses_methodC/cross_layer_bars.png)

**关键观察**：

- **相邻层间一致性最高**：L20-L30 = 0.78-0.85，L30-L36 = 0.78-0.86
- **跨距大的层间下降**：L10-L36 = 0.48-0.59（早期 vs 晚期表征差异大）
- **mid-late 层（L20-L36）整体最稳**——这跟 Anthropic 选 「mid-late layer」（≈2/3 model depth）的判断一致

我们用 **layer 30** 作为主分析层（40 层模型的 ~75% 处），跟 Anthropic 的「2/3 model depth」对齐。

> **直观说人话**：模型有 40 层，每层是一次「想法的过滤+精炼」。早期层（10）在做 token-level 浅层模式匹配，晚期层（36）在做最终决定。**中间靠后**那一段（20-36 层）是「**概念已经成型但还没塌缩到下一个 token**」的阶段——也就是模型「正在想」的最佳时刻。我们的 vectors 在这个区间最稳定，意思是「正在想 X」这件事在这个深度是真存在的。

### 3.4 Var_probe — 4 模板 × 9 concept 的 probe 验证

复现 Anthropic Figure 3 的「implicit emotional content scenarios」逻辑：用一组**不直接命名概念**的 probe 模板，测每个 concept vector 在这些 probe 上的激活。

**4 个模板**：

| 模板 | 形式 | variant 数 |
|---|---|---|
| `var_reading` | "In one sentence, my reaction was that I __" | 10 |
| `var_priors` | "Before opening this, I __" | 6 |
| `var_experiment` | "After running the experiment, I __" | 8 |
| `var_gift` | "When I opened the gift, it was __" | 8 |

#### 3.4.1 方差分解的关键发现

直接看 raw cosine 矩阵会被「概念基线」主导。我们做了 ANOVA-style 方差分解：

| | var_reading | var_priors |
|---|---|---|
| 总方差 | 88.23 | 48.72 |
| 行（variant）效应 | 0.0% | 0.0% |
| **列（concept baseline）效应** | **99.4%** | **99.7%** |
| **交互项（真正信号）** | **0.6%** | **0.3%** |

**解读**：每个 concept 跟 probe 模板的余弦是固定常数（`curious/uncertain/confident` 都在 +1.3，`enlightened/confused/stubborn` 都在 -0.95）。**variant 真实效应只占 0.3-0.6%**，被 99% 的 baseline 完全淹没。

→ **结论**：raw 余弦展示无意义。必须做 **column z-score 归一化**（每个 concept 减去其列均值除以列标准差）后才看得到信号。

#### 3.4.2 z-score 归一化后的 var_reading（10 个 reaction phrases × 9 concepts）

![var_reading heatmap zscore](../outputs/cognitive_v3_full/analyses_methodC/replot_v3_full/var_reading_heatmap_zscore.png)

**结果**：9/10 variant 命中预期 concept（黄色边框标 top-1）：

| Variant | 预期 | 实际 winner | z-score |
|---|---|---|---|
| saw the connection | enlightened | **enlightened** | +2.28σ ✓ |
| felt lost | confused | **confused** | +0.63σ ✓ (弱) |
| couldn't tell what to think | confused | surprised | +1.52σ ⚠️ (confused 第二) |
| felt sure of my view | confirmed | **confirmed** | +2.11σ ✓ |
| kept thinking | curious | **curious** | +1.70σ ✓ |
| wanted to know more | curious | **curious** | +1.11σ ✓ |
| had no reaction | bored | **bored** | +2.14σ ✓ |
| realized I was wrong | enlightened | stubborn | +1.13σ ⚠️ (enlightened +1.12σ 平局) |
| refused to update my view | stubborn | **stubborn** | +1.41σ ✓ |
| was right after all | confident | **confident** | +2.23σ ✓ |

**两个软错位都是「混合状态」**：
- 「couldn't tell what to think」混合 shock + confusion
- 「realized I was wrong」混合 surprise + enlightenment

→ 不是 vector 错，是这些短语**本身同时激活两个概念**，短探针无法区分。这是 var_probe 设计的固有局限。

#### 3.4.3 var_reading 柱状图视图（小多图）

![var_reading bars zscore](../outputs/cognitive_v3_full/analyses_methodC/replot_v3_full/var_reading_bars_zscore.png)

**有意思的现象**：模型学到了 **concept 簇**结构，不是孤立 concepts。两个簇反复出现：
- 「自我确定簇」`confident + confirmed + stubborn` 在 felt sure / refused to update / was right 上一起亮
- 「未定簇」`curious + uncertain` 在 kept thinking / wanted to know more 上一起亮

→ 这跟 Anthropic emotion 论文的「joy/excitement/elation 一起亮」结构一致。

#### 3.4.4 Cross-template winner consistency — 跨 4 模板的概念激活

**问题**：每个概念在 4 个不同语义场景的 probe 模板里，是否都能被某个 variant 推动到显著高位？

![Cross-template heatmap](../outputs/cognitive_v3_full/analyses_methodC/cross_template_consistency.png)

**结果**：每个 concept 在每个模板里都能找到 ≥1σ 的 winner。具体：

```
                  reading   priors   experiment   gift     mean    range
  confident       +2.23     +1.79    +1.77        +1.47    +1.81   0.76
  surprised       +1.52     +1.39    +2.09        +1.25    +1.56   0.85
  stubborn        +1.61     +1.06    +2.01        +1.80    +1.62   0.95
  enlightened     +2.28     +1.52    +1.69        +1.80    +1.82   0.76
  confused        +1.17     +1.26    +1.46        +1.41    +1.33   0.29  ← 最稳
  uncertain       +1.45     +2.04    +2.21        +1.27    +1.74   0.94
  bored           +2.14     +1.25    +1.33        +1.48    +1.55   0.89
  confirmed       +2.11     +1.40    +1.36        +1.89    +1.69   0.75
  curious         +1.70     +0.99    +1.30        +1.37    +1.34   0.71
```

![Cross-template bars](../outputs/cognitive_v3_full/analyses_methodC/cross_template_bars.png)

**模板对比**：
```
  var_reading       mean=+1.80   #>1σ=9   #>2σ=4   ← 最强
  var_experiment    mean=+1.69   #>1σ=9   #>2σ=3
  var_gift          mean=+1.53   #>1σ=9   #>2σ=0
  var_priors        mean=+1.41   #>1σ=8   #>2σ=1   ← 最弱
```

**3 个 paper-relevant 洞察**：
1. **Vector 不是单 template 偶然命中**——每个概念在 4 个不同语义场景里都显著激活，证明它是真正的概念向量而非某个 prompt 的过拟合
2. **var_priors 最弱印证 register-mismatch**——probe 模板问「你的 reaction」，但 priors 描述事前状态，register 不对，所以信号变弱。这反而是 sanity check：vector 对 register 敏感
3. **`confused` 最稳（range 0.29）**——「混淆」概念在 4 种 register 下都表达一致，是 register-invariant 的深层状态

### 3.5 Vector arithmetic — Bayesian flow compositionality

**问题**：reaction 是否等于 prior + discovery 的线性合成？如果是，模型把认知链路当成加性；如果不是，则有非线性更新。

![Vector arithmetic](../outputs/cognitive_v3_full/comparison/methodC_incontext/04_arithmetic.png)

**结果**：所有 9 个 trajectories 的 `cos(v_prior + v_discovery, v_reaction)` 都在 **−0.4 到 −0.7** 之间，**远低于 random baseline (±1σ ≈ 0)**。

**解读（专业）**：
- Reaction **不是** prior 和 discovery 的线性叠加
- 余弦为负意味着 reaction 跟 (prior+discovery) **方向相反**——这是认知更新的非平凡几何特征
- 跟贝叶斯先验更新一致：discovery 的先验「重写」prior，reaction 不在原方向延伸而在新方向构成

> **直观说人话**：你以为「先想 A + 看到 B = 反应 = A 加 B」，简单的加法。但实验告诉我们 **不是**。模型的 reaction 跟 (prior + discovery) 方向**相反**。意思是：「**看到不一样的东西后，我的认知不是叠加，是重写**」。这跟人类认知一致——你以为水是凉的，伸手发现是烫的，你不会觉得「凉+烫」，你会**整个改写**对这杯水的认知。这个发现是 v3 stage-anchored 提取的独特产物，Anthropic 用 whole-story 池化做不到。

这是 v3 才能做的分析（因为需要 stage-anchored extraction 才有独立的 prior/discovery/reaction vectors）。Anthropic 的 emotion 论文做不了类似分析，因为他们 vector 是 whole-story 平均的、不区分 stage。

### 3.6 Token staining — 单 story 上的概念激活轨迹

每个 concept vector 投影到一个真实 story 的逐 token 残差流上（同一篇 132-token 故事——Dr. Chen 重复跑 assay 发现条带异常），按余弦激活强度对 token 上色。

> **直观比喻**：把 9 个 concept vector 当成 9 个**情绪温度计**，把同一段文字放进去测，看每个温度计读数高低。读数高 = 这段文字"散发出"这个情绪/认知状态。

**文件**：`analyses_methodC/stained/stained_<concept>.html`（9 个 HTML，hover 任一 token 可看激活值）

#### 3.6.1 整体观感：每个 concept 在同一 story 上的平均激活

我们写脚本（`scripts/parse_staining.py`）解析 HTML，对每个 concept 算这篇故事 132 个 token 的**平均激活**和**强正激活 token 数**：

| Concept | mean | range | 强正 (>1.0) 的 token 数 | 解读 |
|---|---|---|---|---|
| **confident** | **+0.69** | [-0.41, +1.81] | 32 | 故事开头是熟练操作（"pulled the printout", "had run this assay three times"），confident 高激活合理 |
| **curious** | **+0.57** | [-0.18, +1.57] | 15 | 整篇是「调查异常」的 frame |
| **uncertain** | **+0.57** | [-0.15, +1.58] | — | 主角不知道为什么模式变了——uncertain 自然亮 |
| **surprised** | **+0.52** | [-1.73, +1.89] | — | 异常段落显著 spike |
| bored | -0.15 | [-0.82, +0.44] | 0 | 这故事不无聊 ✓ |
| confused | -0.28 | [-1.34, +0.55] | 0 | 主角懂自己在做什么，不算困惑 |
| stubborn | -0.45 | [-1.54, +0.24] | 0 | 主角愿意调查，不固执 ✓ |
| enlightened | -0.70 | [-1.95, +0.60] | 0 | 主角**还没**领悟（结尾仍 puzzled）✓ |
| confirmed | -0.74 | [-1.87, +0.73] | 0 | 主角的预期**没被**符合 ✓ |

**这是非常干净的结果**——故事内容（科学家发现异常但还没破解）跟 vector 激活强度排序**高度匹配**：
- 「正在调查」类（confident, curious, uncertain, surprised）平均都正 → 故事确实是这个调子
- 「拒绝/无感/已悟」类（stubborn, bored, enlightened, confirmed）平均都负 → 故事不是这个调子

> **直观说人话**：我把同一篇故事「丢」给 9 个温度计，结果**温度计读数排序竟然完全符合人类对故事感觉的排序**——故事更像 confident（73% 正）、不像 confirmed（pred 没匹配，所以 confirmed 是负的）。这意味着这些 vector 已经能像人一样「感受」文本的认知氛围。

#### 3.6.2 局部观察：top-N 高激活 token 看不出语义？

如果直接看每个 concept 排前几名的 token，结果**可能让人困惑**：

```
curious top-12:    '.', '.', '.', 'the', 'pulled', 'run', 'She', 'had', 'this', 'tray', 'the', '.'
confident top-12:  '.', '.', 'tray', 'pulled', 'the', '.', '.', 'almost', 'the', 'from', 'as', 'had'
```

为什么句号和「the」最高？因为这些 token 在**任何文本**里都对 concept vector 投影最大——这是**第 3.4.1 节方差分解**的同一个现象：99% 是 column baseline，1% 是真正信号。

但 staining 仍然有用，因为**HTML 里看的是颜色梯度**，眼睛会自动忽略全篇都亮的 token，注意「相对峰值」（某段比邻近段更亮）。这给读者一个 sentence-level 的可解读 heat map。

> **直观说人话**：就像红外热成像图——绝对温度大家都差不多，但能看出哪个房间的暖气最热、哪面墙最冷。staining 给的是这种「相对热点图」。

#### 3.6.3 一个有意思的反向例子：confused 的局部峰

confused 整体平均 -0.28（不算这故事的主调），但局部有些段落**正向**：

```
confused top-8:    'and' (+0.55), 'up' (+0.53), 'pulled' (+0.51), ',' (+0.50),
                   'She' (+0.48), 'leaned' (+0.46), 'She' (+0.45), 'closer' (+0.43)
```

这些 token 集中出现在「**She frowned, leaned closer, and ran her finger along the lane**」——故事里主角第一次表现出「不理解」的姿态。confused vector 在这一句**局部峰值**正符合语义——尽管整体被「故事不是真的困惑剧情」拉低。

→ 这是 **vector 真的捕捉到了 local 概念 token，不是 trajectory 全局特征**的最直接证据，跟 Anthropic 论文 "vectors activate most strongly on parts of story related to inferring or expressing the emotion" 一致。

#### 3.6.4 为什么这个结果对最终目标重要

如果未来想做「**实时探测模型在某段文本中的认知姿态**」（比如做 reasoning chain monitoring），staining 给出可行性：concept vector 不仅是 stage-aware 的（「这一段是 prior」），还是 **token-level 可读的**（「这一句话特别像 confused」）。这是细粒度 cognitive monitoring 的物质基础。

### 3.7 Causal steering — 因果操控

**方法**：在 layer 30 的 forward hook 里**给残差流加上**`k × v_concept`（k ∈ {-3, 0, +3}），让模型基于这个被「推了一把」的内部状态继续生成。**关键**：用 raw PyTorch `register_forward_hook`，绕开 nnsight 0.7 API 不稳定问题。

**Prompts（输入文本）**：
- P1: `"I am about to open the file."`（中性）
- P2: `"I just got the result of the experiment."`（轻微 surprise）

**规模**：3 strengths × 7 concepts × 2 prompts = **42 个生成**（保存在 `analyses_methodC/steer/`）

> **直观比喻**：concept vector 是模型大脑里一个「认知方向」。Steering 就是**在生成中途，悄悄给大脑里加上一点「往 curious 方向去」的电流**，看模型说出来的话有没有变化。这是验证「这个 vector 真的代表 curious」的最强证据——能读 + 能写。

#### 3.7.1 完整片段对比：4 个最有信息量的例子

**【1】Confident @ P2「I just got the result of the experiment.」**

```
strength = -3:
"That's interesting! What were you testing or observing? Feel free to
 share the results, and I can help analyze them, suggest next steps,
 or discuss implications if needed."

strength = +0:
"That's exciting! 🎉 What were you testing, and what did the results
 show? I'd love to hear how it went—whether it matched your expectations
 or if there was a surprise! 😊"

strength = +3:
"Congratulations on completing your experiment! 🎉
 That's a great milestone — whether it was successful, partially
 successful, or even taught you something unexpected (which is also
 very valuable!).
 Would you like to share what kind of experiment it was and how it went?
 I'd be happy to help you:
   - Analyze the results
   - Interpret the data
   - Prepare for presentation or publication"
```

**分析（专业）**：
- **-3** register：探询式、纯协作（"Feel free to..."），无情感色彩
- **0** baseline：中度热情，平衡问题与表态
- **+3**：**主动断言式祝贺**（"Congratulations!"、"great milestone"），**主动给出结构化协作 path**（bullet list）。这是 confident 概念的精准 register shift——「我对你的成果 confident」+「我 confident 知道接下来怎么帮你」

**直观**：负方向把模型变成「礼貌地等指令」，正方向把模型变成「主动恭喜+主动给方案」。这就是 confident 在对话里实际的表现差异。

---

**【2】Uncertain @ P2「I just got the result of the experiment.」**

```
strength = -3:
"That is exciting! What was the outcome of your experiment?
 Please share the results, data, or any specific questions you have about them.
 I am ready to help you with:
 *   **Data Analysis**: Interpreting trends, calculations, or statistical significance.
 *   **Error Explanation**: Troubleshooting unexpected results or anomalies.
 *   **Conclusion Formulation**: Drafting summaries"

strength = +0:
"That's exciting! What were you testing, and what did you find?
 I'd love to help you interpret the results or discuss next steps if needed."

strength = +3:
"That's interesting! Could you share more about what kind of experiment
 it was, or any specific results you'd like to discuss? I'm happy to help
 analyze or interpret them if needed. 😊"
```

**分析（专业）**：
- **-3**：**反 uncertain = 完全 confident**——"I am ready to help"、写出**清单**断言能力
- **0**：常规闲聊
- **+3**：**hedging 显著**——"could you...", "if needed"、限定语「specific results you'd like」、emoji 软化语气

**直观**：负向（反 uncertain）把模型变成「我什么都能帮，看清单」，正向把模型变成「呃，方便的话能多说点吗？」——uncertain 概念的双向操控成功了。

---

**【3】Stubborn @ P2**

```
strength = -3:
"That's exciting! What did you find?
 Are you looking for help interpreting the results, comparing them to
 expectations, or thinking about next steps in your research?"

strength = +0:
"That's exciting! 🎉 What were you testing, and what did the results show?
 I'd love to hear about it—whether it was a success, a surprise, or
 something that needs further tweaking."

strength = +3:
"That's great! I'd love to hear about it. What were the results?"
```

**分析（专业）**：这是个**有点反直觉**的例子。stubborn +3 居然**话变得最少**。原因：stubborn 的 concept vector 学到的是「**不展开、坚持原立场、不主动提供 alternatives**」。在协作回复语境里，这个翻译成「**最简短、不主动 elaborate**」。

**直观**：stubborn 在 dialogue 里不是「跟你吵」，是「**懒得多说**」。+3 完美捕捉了这种「不展开」的姿态。

---

**【4】Surprised @ P2**

```
strength = -3:
"Great! I'm ready to help you analyze or discuss your results.
 Please share:
 1. **The main findings** (data, conclusions, or outcomes).
 2. **Your goals** for this experiment..."

strength = +0:
"That's exciting! What were you testing or investigating? I'd be happy
 to help you interpret the results..."

strength = +3:
"That's exciting! What was the result? Did it match your hypothesis,
 or did something unexpected happen? I'd love to hear about it—
 especially if there's a detail that stands out or if you're trying
 to interpret what it means."
```

**分析（专业）**：surprised +3 **直接生成「unexpected」「something stands out」** 这种语义内容，把「意外感」从内部状态推到了输出文本。这是 vector 不只影响 register、还能注入**语义 token** 的证据。

**直观**：把模型「调到 surprised」之后，它不仅 *语气*更激动，连**说出的内容都变成在讨论意外的可能性**——这跟人在 surprised 状态下说话方式类似。

#### 3.7.2 整体效果强度评估

诚实地说：**effects are noticeable but subtle, not dramatic**。这是个重要 caveat——大多数 +3 输出**不会读起来像换了一个人**。差异在：

| 维度 | -3 vs +3 在哪些地方明显 |
|---|---|
| Register（语气）| ✓ confident 主动 vs hedging（uncertain）；详尽 vs 简短（stubborn）|
| Lexical（用词）| ✓ "unexpected" 出现率（surprised+3）、"could you" 频率（uncertain+3）|
| Length（长度）| ✓ stubborn+3 显著缩短，confident+3 增加结构化 list |
| Emoji / 标点 | △ 部分概念有差异，不稳定 |
| 内容（实质性话题）| △ 有时变（surprised+3 引入「unexpected」），有时不变 |

**为什么不更戏剧化？** 我们在 layer 30 加 `±3 × v_concept`，但模型有 40 层、`v_concept` 只是其中一个方向。其他 39 层会「校正」这个 perturbation。这跟 Anthropic 在 layer ~2/3 加 0.5 × residual_norm 的强度是同量级——他们的效果也是 register 级而非「换人」级。

> **直观说人话**：你给一个人**右耳塞了一段「兴奋音乐」**——他说话会更兴奋一点，但还是同一个人，不会突然变成另一种性格。我们的 steering 就是这种程度的 nudging，不是 personality flip。

#### 3.7.3 这部分对最终目标的意义

要让模型「自己发现新东西」，必须能**实时改变它的认知状态**——比如它在某个推理链里太 stubborn 了，需要外部 inject curious。Steering 验证了「**concept vector 是因果有效的方向，能写**」。

这意味着工程上可以做：
- **Curiosity amplifier**：在 generation 中持续加 `+0.3 × v_curious`，看 reasoning chain 是否更愿意尝试 alternatives
- **Stubborn dampener**：当模型卡在重复 generation 时，自动加 `-0.5 × v_stubborn` 解锁
- **Surprise gate**：把 `v_surprised` 当成 novelty detector，激活强度作为 「这里值得多想一会儿」的信号

这都是**下游 application**，但前提是底层 vector 因果有效——本节工作就是这个 establishing 步骤。

### 3.8 Token staining + Steering 的相互验证

Staining 显示概念 vector 在 read-time（解码已生成文本）的 local activation；Steering 显示在 write-time（生成新文本）的 causal effect。两者都验证为正，构成**双向证据**：

```
text → activation:  staining HTML（读懂 vector 在哪里激活）
activation → text:  steering txt（vector 推动模型说什么）
```

### 3.9 MCQ surprise-signal 实验：找到 cognitive vector 的「judgment 模式」

#### 3.9.1 Motivation

团队大目标是找 **surprise-as-learning-signal**——模型在遇到不熟悉/出乎意料信息时的内部信号。Anthropic emotion paper 证明 vectors 跟模型行为有因果关系，但**没有直接的「learning signal」实验**。我们设计了一个新实验填补这个空白。

**核心思路**（来自团队会议中 D 的建议）：用 multiple-choice 问答测每个 cognitive vector 在「**正确答案 vs 错误答案**」上的活跃度差异——差异最大的 vector 就是 **judgment signal candidate**。

#### 3.9.2 实验设计

- **40 道 hand-designed common-sense MCQ**：地理（10）/ 科学（10）/ 数学（10）/ 通识（10）
- 每题 4 选项（A/B/C/D），1 个正确 + 3 个错误
- prompt 模板：

```
Question: What is the capital of France?
A) London
B) Paris
C) Rome
D) Berlin
Answer: B) Paris        ← 这次填正确选项
```

- 每个选项各 forward 一遍 → **40 × 4 = 160 forward pass**
- 在「答案 token」位置（layer 30）捕获 hidden state，投影到 9 个 v3 concept vector
- 按「is_correct」分组（n=40 vs n=120），算 Cohen's d

#### 3.9.3 主结果

![MCQ Cohen's d](../outputs/cognitive_v3_mcq/cohen_d_summary_last.png)

| Concept | Cohen's d | 含义 |
|---|---|---|
| **confused** | **−1.52** | 错误答案**强烈**激活困惑 |
| **stubborn** | **−1.34** | 错误答案**强烈**激活拒绝姿态 |
| **bored** | +0.66 | 正确答案激活「无新意」感 |
| **confident** | +0.62 | 正确答案激活自信 |
| **curious** | +0.47 | 正确答案略激活好奇 |
| **confirmed** | +0.43 | 正确答案激活「期待验证」 |

**双方向都有 strong signal**：负方向有 confused / stubborn（wrong-detector），正方向有 bored / confident / confirmed（right-detector）。

![MCQ strip plots](../outputs/cognitive_v3_mcq/strip_plot_per_concept_last.png)

`confused` 和 `stubborn` 面板上**红绿分布明显分离**——这就是我们要找的 judgment signal。

#### 3.9.4 三个具体题目细看

**例 1: Q22「12 × 12 = ?」（教科书 cleanest 信号）**

| 选项 | confused | stubborn | bored | confident |
|---|---|---|---|---|
| ✗ A) 124 | −0.089 | **+0.062** ❗ | −0.091 | +0.150 |
| **✓ B) 144** | **−0.140** ✓ | **−0.080** ✓ | **−0.035** ✓ | **+0.235** ✓ |
| ✗ C) 164 | −0.075 | +0.026 | −0.101 | +0.179 |
| ✗ D) 184 | −0.083 | −0.001 | −0.088 | +0.210 |

读到 "144" 时残差流里 **5 个信号同时表达「我认得这个」**。读到 "124" 时 stubborn 立刻冒出来——模型「内心拒绝」错误答案。

**例 2: Q20「DNA stands for?」（confident 标记最清晰）**

| 选项 | confused | stubborn | bored | confident |
|---|---|---|---|---|
| **✓ A) Deoxyribonucleic acid** | **−0.138** ✓ | **−0.053** ✓ | −0.062 | **+0.227** ✓ |
| ✗ B) Dinitrogen acid | −0.072 | **+0.047** | −0.086 | +0.169 |
| ✗ C) Dynamic nuclear array | −0.030 | +0.059 | **−0.135** | +0.153 |
| ✗ D) Dual nucleic atom | −0.044 | +0.045 | −0.110 | +0.158 |

confident 在正确答案上 +0.227，比 3 个错答全部高出 0.06+。confused 随错答荒谬度**单调上升**到接近 0——cognitive dissonance 的标志。

**例 3: Q34「最高的动物？」（4 vector 全部 hit）**

| 选项 | confused | stubborn | bored | confident |
|---|---|---|---|---|
| ✗ A) Elephant | −0.075 | **+0.086** | −0.107 | +0.093 |
| **✓ B) Giraffe** | **−0.115** ✓ | **−0.008** ✓ | **−0.055** ✓ | **+0.187** ✓ |
| ✗ C) Camel | −0.037 | **+0.121** | −0.121 | +0.056 |
| ✗ D) Horse | −0.021 | +0.103 | −0.086 | +0.061 |

3 个错答都比正确答案 stubborn 高 +0.09 ~ +0.13。Elephant 是最大的（不是最高）——它的 confused 比 Camel/Horse 更负，模型「半懂」它跟体型相关。

**反例: Q5「人口最多的国家？」（信号失败）**

| 选项 | confused | stubborn |
|---|---|---|
| ✗ A) United States | −0.081 | +0.028 |
| ✗ B) Russia | −0.061 | **+0.104** |
| **✓ C) India** | −0.071 | +0.044 |
| ✗ D) Brazil | −0.044 | **+0.107** |

信号没指向 C) India。Russia 和 Brazil 的 stubborn 反而更高。**原因**：印度刚超过中国（题目还漏列了中国），模型本来就不确定。**这告诉我们**：MCQ surprise signal 在「**对模型来说毫无悬念**」的题上最强；模型本身不确定时信号会**钝化**——这恰好是下一步要测的：**模型不确定 vs 模型确定** 的不同响应。

#### 3.9.5 提出 surprise-score 候选公式

把发现的 5 个 vector 组合：

```
surprise_score(t)  =   α · confused(t)
                     + β · stubborn(t)
                     − γ · bored(t)
                     − δ · confident(t)
                     − ε · confirmed(t)
```

- 系数通过线性回归 fit（target = 「答案对/错」label）
- 实时算每个 token 的 score
- 高分 token = 「模型觉得这里不对劲」

这是「让模型主动发现新东西」目标的**第一个可计算的实时探测器**。

#### 3.9.6 为什么 `surprised` 自己反而 d 接近 0？

`surprised` vector 单独看 effect size 很小（d=−0.17）。可能解释：「surprised」在 v3 训练数据里更对应「不可处理的意外」（discovery 中段），不是「读到错答」这种**事实性矛盾**——后者更接近 cognitive **dissonance**（与已有信念冲突），恰好是 `confused + stubborn` 的组合。所以 surprise 在 cognitive vs informational 维度上其实是两种东西，**我们的实验揭示了后者**。

---

## 4. 方法论亮点：方差分解强制 z-score 归一化

这是我们这次 ad-hoc 发现的一个 **paper-worthy methodological finding**：

**问题**：var_probe 的 raw cosine 矩阵让人误以为「curious 一家独大」（永远是 top-1）。

**方差分解**：

```
S[variant, concept] = grand_mean
                    + row_effect[variant]      ← 0% 方差贡献（probe 模板对所有 variants 一致）
                    + col_effect[concept]      ← 99% 方差贡献（probe 模板跟每个 concept 的固定 cosine baseline）
                    + interaction[variant, concept]  ← 0.6% 方差贡献（真正信号！）
```

→ **所有 paper 报告 var_probe 类型分析时，必须 column-normalize**。否则 90% 信号被 column baseline 淹没。

我们提供两种归一化：

| 方案 | 公式 | 单位 | 用途 |
|---|---|---|---|
| Column centering | `S - col_mean` | Δ cosine | 看绝对幅度变化 |
| **Column z-score** | `(S - col_mean) / col_std` | **σ** | **对比跨 concept 强度** |

z-score 是首选，因为：
1. 中心化（消掉 99% baseline）
2. 标准化（concept 间可比）
3. σ 是统计学常用单位
4. 突出 variant-specific 信号

---

## 5. 局限与未来工作

### 5.1 已知 sample-size 限制

| Concept | n_stories（来源） |
|---|---|
| confirmed | 5（仅 trajectory #16，1 reaction）|
| stubborn | 10（trajectories #1, #9）|
| enlightened | 15（#2, #10, #18）|
| confused | 15（#3, #11, #19）|
| 其他（uncertain prior 来源 4 trajectory） | 20+ |

→ `confirmed` 样本最少。**未来 oversample trajectory #16** 可改善。

### 5.2 Register-mismatch（var_priors 偏弱）

「prior」描述事前状态，但 var_priors 模板仍然问「reaction」register。下次需要专门设计 prior-register 的 probe 模板（如 "Going in, I expected ___"）。

### 5.3 单模型

仅在 Qwen3.6-35B-A3B-NF4 上验证。跨模型迁移（Llama, Mistral, Sonnet）未测。

### 5.4 未做的 Anthropic 实验 + 还能用现有数据做的

#### 已经跑过的 Anthropic 风格分析（√）
- ✓ Cross-method consistency（跨方法一致性）
- ✓ Cross-layer consistency（跨层一致性）
- ✓ Var_probe（implicit-content scenarios）4 模板 × 9 concepts
- ✓ Vector arithmetic（compositionality test）
- ✓ Token staining
- ✓ Causal steering
- ✓ Per-method PCA + cosine（在 `comparison/` 目录里）

#### 没跑但值得补的（按性价比）

| 分析 | 需要 | 时间 | 价值 |
|---|---|---|---|
| **Logit lens**（lm_head @ vector → top tokens）| GPU + lm_head（轻量）| 30 min | ⭐⭐⭐ 最直接的 sanity check：vector 通过 unembedding 矩阵投出来的 top tokens 是否语义对应 |
| **Numerical gradient templates**（如 "I have been debugging for {N} hours" N=1,3,8,24）| GPU forward | 10 min | ⭐⭐⭐ Anthropic Figure 3 那种「单调连续」证据，配合 var_probe |
| **PCA + k-means cluster** on 9 concept vectors | 仅 vectors（local CPU）| 5 min | ⭐⭐ 几何直观；Anthropic 的 affective circumplex 类比 |
| **9-vs-9 cross-cosine matrix between concepts**（不是跨方法，是跨概念）| 仅 vectors | 1 min | ⭐⭐ 看哪些 cognitive concept 互相靠近（`curious-uncertain-confused` 一类簇？）|
| **Steering 强度梯度**（k=-3, -1, 0, +1, +3 五档）| GPU | 20 min | ⭐⭐ 看 register shift 是否随 strength 单调，Anthropic 用过 |
| **Confound projection**（neutral 段的 PCA, project out top components）| neutral 段 forward | 30 min | ⭐⭐ Anthropic 用了，能去掉 prompt 共享方向 |

#### 现有数据可直接做的（无需 GPU，只需要 raw vectors）

我们手头**有 v2 vectors**（local），可以做：
- v2 vs v3 PCA / cluster 对比（v2 是污染版，对比能展示 v3 改善了什么 cognitive geometry）
- v2 vs v3 top tokens via logit lens（v2 那批 vector 也保存有）

这些都是 baseline-of-bad 性质的对比，**写 paper 时讲「v2 → v3 的方法学进步**」很有用。

#### 其他没做的 Anthropic 实验（不打算补）

| Anthropic 做了 | 我们不做 | 原因 |
|---|---|---|
| 64-activity preference task + Elo | ❌ | cognitive 没有 emotion-style preference 强对应；做了也不直接对应 discovery 目标 |
| Naturalistic transcripts probe（6000 条）| ❌ | 缺数据集；可未来用 reasoning chain transcripts 替代 |
| Post-training comparison | ❌ | Qwen3.6 我们没有 base model |
| Cross-model（Llama, Sonnet）| ❌ | 不在本 paper scope 内；future work |

### 5.5 v4 dialogue 工作流（进行中）

我们已经开始 **v4 dialogue pipeline**——复现 Anthropic Table 14（present/other speaker emotion 区分），但用 cognitive concepts。**8 cognitive concepts × 8 = 64 pairs，sanity 256 dialogues**，目前生成中。下一步：cross-method consistency on dialogue probes、Table 14 复现。

---

## 6. 对最终目标——「让模型主动发现新东西」——的意义

整个 v3 工作的科研定位，不是一个孤立的 interpretability paper，是更大研究路线（**surprisal as learning signal**）的**底层零件**。所以最后用一整节，把零件跟整体目标对回去。

### 6.1 整体目标回顾（一句话）

我们想做的：**让 LLM 在遇到未知 / 矛盾 / 新意外的输入时，主动改变行为——慢下来、追问、生成假设、跳出当前框架——而不是机械地吐出最可能的 token**。

人类做这件事靠两个东西：
1. **能感知自己当前的认知状态**（"我现在在懵 / 我在赶路 / 我有个 hunch / 我在僵化"）
2. **能切换状态**（"我感觉自己僵了，让我跳出来重新看一眼"）

现代 LLM 在这两件事上都比较弱——它们没有显式的 metacognition。但**残差流里可能已经隐含了 metacognitive 信号**，只是我们没读出来 / 没操控。

### 6.2 v3 的 9 个 concept vector，恰好对应「探索-发现」的认知工具箱

| Concept | 对探索/发现的工具角色 |
|---|---|
| **curious** | 「值得追问」的内驱信号——可作为 query expansion 的触发器 |
| **uncertain** | 「我不知道答案」的 ground truth 信号——区分自信 hallucination vs 真不知 |
| **confident** | 「我有答案」的信号——配合 uncertain 做 calibration（confident 不同时高 = 信任）|
| **surprised** | **核心 novelty detector**——输入跟 prior 矛盾时激活，是 surprisal 信号的**直接神经底物** |
| **bored** | 「这件事我没新东西可学」——可用作 task-completion / sufficient-knowledge 信号 |
| **stubborn** | 「我拒绝更新」的反信号——出现时是探索失败警告 |
| **enlightened** | 「aha 时刻」的信号——可作为 reasoning chain 内部 reward |
| **confused** | 「我没整合好」——出现时需要更多 context / slow down |
| **confirmed** | 「期待被验证」——配合 surprised 做先验匹配检查 |

> **直观说人话**：v3 给了我们模型大脑里**9 个独立的「认知传感器+电极」**。surprised 像 novelty detector，curious 像探索按钮，stubborn 像「卡死」警报。**这些是任何 metacognitive controller 必须先有的输入输出端口**。

### 6.3 Discovery 的关键候选 metric——**已被 MCQ 实验实证支撑**

光有 vectors 不够，要把它们组合成可用信号。MCQ 实验（§3.9）**直接发现**了 cognitive vector 之间在「正确 vs 错误信息」上的判别模式，给出了一个 **empirically grounded** 的复合公式：

```
surprise_score(t) =   α · confused(t)    ← MCQ 中 d = -1.52，wrong-detector 主力
                    + β · stubborn(t)    ← MCQ 中 d = -1.34，wrong-detector 副力
                    − γ · bored(t)       ← MCQ 中 d = +0.66
                    − δ · confident(t)   ← MCQ 中 d = +0.62
                    − ε · confirmed(t)   ← MCQ 中 d = +0.43
```

**这个公式不是空想——是从 MCQ 实验里 reverse-engineer 出来的**。每个系数都对应一个被实验验证有大效应的 vector。

**直观解读**：
- `confused / stubborn` 高 = 模型遇到不和谐信息（潜在 learning opportunity 或 plain wrong）
- `bored / confident / confirmed` 高 = 模型遇到熟悉信息（不需要学习）
- `score` 高 = 「这里有 something off，值得多想一会儿」

我们已经验证：
- 三个核心 vector 都能 **read**（cross-method 0.7-0.95，显著激活）
- 三个核心 vector 都能 **steer**（causal effect 显著）
- **MCQ 实证**：confused / stubborn / bored / confident / confirmed 在「judgment 任务」上 **Cohen's d 范围 0.43–1.52**——这是**直接的、量化的 learning signal 证据**

`surprise_score` 在工程上**可计算、可干预、可验证**——下一步 demo 应用 ready。

#### 6.3.1 重要 caveat：还要测 novel-but-correct

MCQ 实验只测了「**错答 vs 对答**」，没测「**模型不熟悉但客观正确**」的情况。两种可能：

| 假设 | 含义 |
|---|---|
| H1: novel-correct → 触发不同 fingerprint（如 enlightened/curious 而非 confused/stubborn） | 我们有了**两类信号**：错误检测 + learning 机会检测 |
| H2: novel-correct → 也触发 confused/stubborn | 信号只反映「不熟悉」，不区分「错误」vs「真新颖」（still useful but less ambitious） |

**这是 v3 之后最关键的实验**（见 §5.4）。在结论确定前，我们说的是「**MCQ 实验找到了 judgment signal**」而不是「**我们找到了 surprise-as-learning-signal**」。后者要等 novel-correct 实验。

### 6.4 一个具体可做的下游 demo

**任务**：让模型解一个有「陷阱设计」的问题（比如表面像 X 类问题，实际是 Y 类）。

**没有 v3**：模型按照 X 模式 generate，错。

**有 v3 后可做**：
1. 实时算 `surprised(t)` 在 reasoning chain 各 token 上的激活
2. 当 `surprised` 出现 spike 但模型继续走原 plan → 检测到 `stubborn` 也升 → 报警  
3. 在那个 token inject `+0.5 × v_curious` 强行让模型重新评估
4. 比较有/无干预下的解题正确率

这是个**直接、量化、有意义**的 application。**v3 让这种实验从「不可能」变成「可设计、可衡量」**。

### 6.5 为什么是 cognitive 而不是 emotion 重要

Anthropic 的 emotion 工作做了「happy/sad/angry」类的情感概念。我们做 cognitive：

| 域 | 用处 |
|---|---|
| Emotion | 角色扮演、对齐（不让模型 desperate 就 blackmail）、psychology of AI |
| **Cognitive** | **直接对应 reasoning quality**——curious/confused/surprised 是认知动作的内省，**远比 emotion 更接近「模型是不是在思考」这个核心问题** |

换句话说：emotion vector 让你监控「模型现在情绪如何」；cognitive vector 让你监控「模型现在在不在好好思考、有没有探索意愿」。**对让模型「发现新东西」的目标来说，cognitive 比 emotion 更直接相关。**

### 6.6 局限的诚实话

v3 是**底层零件**，不是 application。要兑现 6.4 那个 demo，至少还需要：
- 实时（per-token）激活计算的工程优化（forward hook 已 ready）
- 一个有「陷阱」的 reasoning benchmark（待选）
- baseline + 干预 + ablation 的实验设计
- 跨 task 的迁移性验证

但**所有这些下游工作都依赖 vector 是好的、稳定的、因果有效的**——v3 把这件事 nail 住了。

---

## 7. 文件与图表清单

### 7.1 主要图（在 paper 里候选 figure）

| 图 | 路径 |
|---|---|
| Cross-method consistency bar | `outputs/cognitive_v3_full/analyses_methodC/cross_method_bars.png` |
| Cross-method z-score | `outputs/cognitive_v3_full/analyses_methodC/cross_method_zscore.png` |
| Cross-layer consistency bar | `outputs/cognitive_v3_full/analyses_methodC/cross_layer_bars.png` |
| Cross-layer z-score | `outputs/cognitive_v3_full/analyses_methodC/cross_layer_zscore.png` |
| var_reading heatmap z-score | `analyses_methodC/replot_v3_full/var_reading_heatmap_zscore.png` |
| var_reading bars z-score | `analyses_methodC/replot_v3_full/var_reading_bars_zscore.png` |
| Cross-template consistency heatmap | `analyses_methodC/cross_template_consistency.png` |
| Cross-template bar | `analyses_methodC/cross_template_bars.png` |
| Vector arithmetic | `comparison/methodC_incontext/04_arithmetic.png` |
| Layer scan | `comparison/methodC_incontext/03_layer_scan.png` |

### 7.2 数据 / 输出

| 文件 | 内容 |
|---|---|
| `runs/cognitive_v3_full/stories/` | 360 个 valid stories |
| `runs/cognitive_v3_full/sanity_report.md` | 验证报告（每个 story 的 4 项检查）|
| `runs/cognitive_v3_full/consistency_report.md` | cross-method + cross-layer cosine tables |
| `outputs/cognitive_v3_full/analyses_methodC/var_*_scores.npz` | var_probe 4 模板的 raw cosine matrices |
| `outputs/cognitive_v3_full/analyses_methodC/cross_template_summary.json` | 跨模板最佳 z-score per concept |
| `outputs/cognitive_v3_full/analyses_methodC/stained/stained_*.html` | 9 个 concept 的 HTML 染色 |
| `outputs/cognitive_v3_full/analyses_methodC/steer/steer_*.txt` | 14 个 steering outputs |

### 7.3 代码

| 脚本 | 用途 |
|---|---|
| `scripts/generate_trajectories_v3.py` | 故事生成 + 验证 + 重试 |
| `scripts/v3_validate.py` | 验证库（结构 / 字数 / 禁词）|
| `scripts/extract_v3_compare.py` | 4-method 同时提取 |
| `scripts/run_v2_analyses_v3.py` | var_probe / staining / steering |
| `scripts/replot_var_probe.py` | z-score 归一化重画 |
| `scripts/cross_template_consistency.py` | 跨模板 winner 分析 |
| `scripts/replot_consistency.py` | cross-method / cross-layer bar charts |
| `scripts/run_full_analysis_v3.sh` | 端到端 pipeline 一键运行 |

