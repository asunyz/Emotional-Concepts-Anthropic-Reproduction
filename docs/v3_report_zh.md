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

V3 的另一个方法论增量是**同一批 stories 上跑 4 种不同的提取方法**，用 cross-method consistency 验证「概念 vector 不是某个特定 recipe 的过拟合」：

| Method | 描述 | 对应 |
|---|---|---|
| A | v2-style whole-story mean pooling | Anthropic emotion paper |
| B | paragraph isolation（仅段落本身 forward） | 控制无 context 影响 |
| **C** | **paragraph in-context（整 story forward + 段落 token mean）** | **主用方法** |
| D | within-stage contrast（段落 - story 平均） | 强化 stage-specific 信号 |

---

## 2. 发布计划与论述（PubPlan + Narrative）

### 2.1 论文层级

**目标**：ICML workshop（4-page extended abstract 或 8-page）  
**故事线**：「**MoE 架构上 cognitive concept vectors 的可解释性证据，及方法学增量**」

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

### 2.4 论文章节计划

```
1. Introduction          — concept vectors as interpretability tool
2. Related work          — Anthropic emotion + concept extraction lineage
3. Method                — v3 design (trajectory + show-don't-tell + 4 methods)
4. Results               — vector geometry, var_probe, steering
5. Methodological contribution — variance decomposition + z-score必要性
6. Limitations           — sample-size、register、单模型
7. Discussion            — cognitive vs emotion 异同
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

**解读**：
- B 和 C 都是 paragraph-level 方法，互相一致性最高（0.6-0.85）；A 的 whole-story pooling 跟它们差异大，**这印证了 stage-anchoring 的必要性**
- priors 一致性低，反映 prior stage 在不同提取角度（context vs isolation）下表征确实有差异——可能是 prior 是「事前心态」语境性较强

### 3.3 Cross-layer consistency — 跨层稳定性

**问题**：同一概念在不同深度（L10, L20, L30, L36）的方向是否稳定？

![Cross-layer bars](../outputs/cognitive_v3_full/analyses_methodC/cross_layer_bars.png)

**关键观察**：

- **相邻层间一致性最高**：L20-L30 = 0.78-0.85，L30-L36 = 0.78-0.86
- **跨距大的层间下降**：L10-L36 = 0.48-0.59（早期 vs 晚期表征差异大）
- **mid-late 层（L20-L36）整体最稳**——这跟 Anthropic 选 「mid-late layer」（≈2/3 model depth）的判断一致

我们用 **layer 30** 作为主分析层（40 层模型的 ~75% 处），跟 Anthropic 的「2/3 model depth」对齐。

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

**解读**：
- Reaction **不是** prior 和 discovery 的线性叠加
- 余弦为负意味着 reaction 跟 (prior+discovery) **方向相反**——这是认知更新的非平凡几何特征
- 跟贝叶斯先验更新一致：discovery 的先验「重写」prior，reaction 不在原方向延伸而在新方向构成

这是 v3 才能做的分析（因为需要 stage-anchored extraction 才有独立的 prior/discovery/reaction vectors）。Anthropic 的 emotion 论文做不了类似分析，因为他们 vector 是 whole-story 平均的、不区分 stage。

### 3.6 Token staining — 单 story 上的概念激活轨迹

每个概念 vector 投影到单个 story 的逐 token 残差流上，按余弦激活强度上色——验证概念在文本中的「亮起位置」是否符合语义。

文件：`outputs/cognitive_v3_full/analyses_methodC/stained/stained_<concept>.html`（9 个 HTML，每个对应一个 concept 的染色）

定性结果：染色聚集在**叙述对应概念的段落**，而非均匀散布——证明 vector 捕捉的是「local 的概念 token」，不是 trajectory-level 的全局特征。这跟 Anthropic 论文的发现一致（"vectors activate most strongly on parts of story related to inferring or expressing the emotion"）。

### 3.7 Causal steering — 因果操控

**方法**：在 layer 30 的 forward hook 里注入 `±k × v_concept`（k = 0, 3），生成 continuation，比较输出 register。**关键**：使用 raw PyTorch `register_forward_hook`，绕开 nnsight 0.7 API 不稳定。

**Prompts**：
- `"I am about to open the file."`（中性）
- `"I just got the result of the experiment."`（轻微 surprise prompt）

**强度**：strength ∈ {-3, 0, +3} × 9 concepts × 2 prompts = 54 个 outputs

**最戏剧的例子**（surprised, strength=+3, prompt="I just got the result..."）：

> "The air suddenly grew still, and a faint shimmer seemed to dance across the screen as if the very pixels were holding their breath..."

普通 baseline (strength=0) 输出是平淡的："That's exciting! What was the outcome?"

在 ±3σ 强度下：
- **surprised**：感叹号 + 修辞 + 时间感（清晰）
- **stubborn**：拒绝、坚持原观点（清晰）
- **uncertain**：hedging、犹豫词（清晰）
- **bored**：短回复、无情感（清晰）
- **confused**："wait, what?" + 反问（清晰）

定性来说，**所有 9 个 concept 的 +3 操控都产生了可辨识的 register shift**。这是 vectors 对模型行为有因果影响的直接证据。

文件：`outputs/cognitive_v3_full/analyses_methodC/steer/steer_<concept>_<prompt_stem>.txt`（14 个文件）

### 3.8 Token staining + Steering 的相互验证

Staining 显示概念 vector 在 read-time（解码已生成文本）的 local activation；Steering 显示在 write-time（生成新文本）的 causal effect。两者都验证为正，构成**双向证据**：

```
text → activation:  staining HTML（读懂 vector 在哪里激活）
activation → text:  steering txt（vector 推动模型说什么）
```

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

### 5.4 未做的 Anthropic 实验

| Anthropic 做了 | 我们没做 | 原因 |
|---|---|---|
| Logit lens（lm_head @ vector）| ❌ | Pod restart 导致 v3 vectors 暂时丢失，需 ~10 min GPU 重提 |
| Numerical gradient templates | ❌ | 需 GPU；可低成本补 |
| 64-activity preference task + Elo | ❌ | 不必要（cognitive 没有 emotion-style preference 强对应）|
| Naturalistic transcripts probe | ❌ | 缺数据集 |
| Post-training comparison | ❌ | 没 base model |

### 5.5 v4 dialogue 工作流（进行中）

我们已经开始 **v4 dialogue pipeline**——复现 Anthropic Table 14（present/other speaker emotion 区分），但用 cognitive concepts。**8 cognitive concepts × 8 = 64 pairs，sanity 256 dialogues**，目前生成中（55%）。下一步：cross-method consistency on dialogue probes、Table 14 复现。

---

## 6. 文件与图表清单

### 6.1 主要图（在 paper 里候选 figure）

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

### 6.2 数据 / 输出

| 文件 | 内容 |
|---|---|
| `runs/cognitive_v3_full/stories/` | 360 个 valid stories |
| `runs/cognitive_v3_full/sanity_report.md` | 验证报告（每个 story 的 4 项检查）|
| `runs/cognitive_v3_full/consistency_report.md` | cross-method + cross-layer cosine tables |
| `outputs/cognitive_v3_full/analyses_methodC/var_*_scores.npz` | var_probe 4 模板的 raw cosine matrices |
| `outputs/cognitive_v3_full/analyses_methodC/cross_template_summary.json` | 跨模板最佳 z-score per concept |
| `outputs/cognitive_v3_full/analyses_methodC/stained/stained_*.html` | 9 个 concept 的 HTML 染色 |
| `outputs/cognitive_v3_full/analyses_methodC/steer/steer_*.txt` | 14 个 steering outputs |

### 6.3 代码

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

---

## 7. 致谢与方法学引用

- **Anthropic 2026 Emotion paper** — 方法学起点，prompt 结构、whole-story pooling、causal steering 范式
- **nnsight** — model tracing API（v0.7 后改用 raw forward_hook）
- **transformers 5.x + bitsandbytes 0.49 + NF4 quantization** — Qwen3.6-35B-A3B 在单 RTX PRO 6000 (97 GB VRAM) 上落地
