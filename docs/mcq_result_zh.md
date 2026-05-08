# MCQ 实验结果报告（中文版）

**实验名称**：Multiple-Choice Question Surprise-Signal 实验
**研究问题**：哪些 cognitive vector 在「模型读到正确答案 vs 错误答案」时**反应不一样**？这个差异就是我们要找的 **surprise as learning signal**。
**模型**：Qwen3.6-35B-A3B（NF4 量化）
**Probe vectors**：v3 cognitive concept vectors（9 个，layer 30，Method C）
**报告日期**：2026-05-08
**作者**：meridah7

---

## 0. 一句话结论

**找到了**。

把 40 道常识题的正确/错误答案分别塞进 prompt，让模型 forward 一遍，发现：

- **错误答案**显著激活 `confused` (Cohen's d = **−1.52**) 和 `stubborn` (d = **−1.34**)
- **正确答案**显著激活 `bored` (d = +0.66)、`confident` (d = +0.62)、`confirmed` (d = +0.43)

这是团队找了几个月的 **surprise-as-learning-signal 候选**——一个**量化的、可读取的**内部信号，能区分「模型见到熟悉信息」vs「模型见到不和谐信息」。

---

## 1. 实验设计（40 题 × 4 选项 = 160 forward pass）

### 1.1 MCQ 数据集

40 道**预期模型必答对**的常识题（按设计应有 ~95% accuracy）：

- 10 道地理（首都、海洋、大陆）
- 10 道科学（化学符号、行星数、光速）
- 10 道数学（7+8、12×12、平方根）
- 10 道通识（一周几天、原色是哪几种）

**为什么用简单题**：要验证「模型有这方面的知识」，否则模型把「错答」当成「正确」就乱了。简单题保证「正确就是正确，错误就是错误，模型分得清」。

### 1.2 Prompt 模板

每题塞 4 次 prompt（每个选项各塞一次），格式：

```
Question: What is the capital of France?
A) London
B) Paris
C) Rome
D) Berlin
Answer: B) Paris        ← 这次填正确选项
```

或：

```
Answer: A) London       ← 这次填错误选项
```

### 1.3 测什么

每次 forward 完，捕获 **layer 30 最后一个 token 的 hidden state**（即「Paris」/「London」最后位置的内部状态），把它投影到 9 个 v3 cognitive vector 上得到 9 个 cosine 值。

每题产出：
- 1 个「正确答案下的 9-d 投影」
- 3 个「错误答案下的 9-d 投影」

40 题 × 4 选项 = **160 个数据点**。

### 1.4 怎么分析

对每个 concept vector，把 160 个数据点分成两组：
- **correct 组**（n=40）：模型读到正确答案时的投影
- **incorrect 组**（n=120）：模型读到错误答案时的投影

算 **Cohen's d**（标准化的均值差）：

```
d > 0：vector 在正确答案上更激活
d < 0：vector 在错误答案上更激活
|d| ≥ 0.8：大效应
|d| ≥ 0.5：中效应
|d| ≥ 0.2：小效应
```

---

## 2. 结果

![Cohen's d ranking](../outputs/cognitive_v3_mcq/cohen_d_summary_last.png)

### 2.1 数值表

| Concept | Cohen's d | mean(正确) | mean(错误) | 解读 |
|---|---|---|---|---|
| **confused** | **−1.52** 🔴 | −0.098 | −0.062 | 错误答案**强烈**激活困惑 |
| **stubborn** | **−1.34** 🔴 | −0.002 | +0.046 | 错误答案**强烈**激活拒绝姿态 |
| **bored** | +0.66 🟢 | −0.070 | −0.085 | 正确答案更激活「无新意」感 |
| **confident** | +0.62 🟢 | +0.160 | +0.134 | 正确答案更激活自信 |
| **curious** | +0.47 🟢 | +0.042 | +0.026 | 正确答案略激活好奇 |
| **confirmed** | +0.43 🟢 | — | — | 正确答案激活「期待验证」 |
| uncertain | +0.26 | — | — | 微弱效应 |
| surprised | −0.17 | — | — | 几乎无效应 |
| enlightened | +0.04 | — | — | 无效应 |

### 2.2 散点分布

![Strip plots](../outputs/cognitive_v3_mcq/strip_plot_per_concept_last.png)

每个面板是一个 concept：
- 🟢 上半区绿点 = 正确答案（n=40）
- 🔴 下半区红点 = 错误答案（n=120）
- 虚线是各组均值

`confused` 和 `stubborn` 两个面板上**红绿明显分开**——这就是我们要的「judgment signal」。

---

## 3. 解读

### 3.1 模型像在做「内部判断」

我们没让模型回答，只让它**读** prompt。但 hidden state 已经显示出**对这个答案的内部反应**：

| 模型见到 | 内部激活什么 | 直观说人话 |
|---|---|---|
| 正确答案 | `confirmed` ↑、`bored` ↑、`confident` ↑ | 「对，就是这个，没新意，我早知道」 |
| 错误答案 | `confused` ↑、`stubborn` ↑ | 「这不对劲，我不接受」 |

**模型即使没生成 token，残差流里已经有「认知判断」的信号**。这是这个实验的核心价值。

### 3.2 提出 surprise-score 候选公式

把发现的这两组方向组合起来：

```
surprise_score(t)  =   α · confused(t)
                     + β · stubborn(t)
                     − γ · bored(t)
                     − δ · confident(t)
                     − ε · confirmed(t)
```

- 系数可以通过线性回归 fit（target = 「这个答案对/错」标签）
- 实时算每个 token 的 score
- 高分 token = 「模型觉得这里不对劲」的位置

这给「让模型主动发现新东西」目标提供了**第一个可计算的实时探测器**。

### 3.3 为什么 surprised 自己反而 d 接近 0？

有意思的发现——`surprised` vector 单独看 effect size 很小（d=−0.17）。

可能的解释：
- 「surprised」在 v3 训练数据里更对应「不可处理的意外」（discovery 中段），不是「读到错答」这种**事实性矛盾**
- 「事实性矛盾」更接近 cognitive **dissonance**（与已有信念冲突）—— 这恰好是 `confused + stubborn` 的组合
- 所以 surprise 在 cognitive vs informational 维度上其实是两种东西，我们的实验**揭示了后者**

---

## 4. 局限 + 下一步

### 4.1 局限

- **只测错误事实**：所有 incorrect 答案都是「客观错误」（伦敦不是法国首都）。**没测「正确但 novel」**——比如模型不知道的新事实
- **简单题**：40 道都是 model 应该熟悉的。**没测模型不熟悉的领域**——可能 confused/stubborn 反应不同
- **正确 vs 错误样本不均**：1:3（40 vs 120），但 Cohen's d 已经 normalize 这个偏差
- **没验证模型 accuracy**：我们假设这 40 题模型都答对，没实测。下次应跑一遍 baseline 让模型自己选，验证 accuracy ≥90% 再做 forward 实验

### 4.2 下一步实验（按优先级）

| 优先级 | 实验 | 预期发现 |
|---|---|---|
| ★★★ | **Novel-but-correct 测试**：用 word problem / 推理题，模型不会但答案确实对——测 `enlightened` 是否冒出来 | 找到「真正的 learning opportunity」signal |
| ★★ | 验证模型在 40 题上 accuracy（baseline）| 排除「模型没这知识」的混淆 |
| ★★ | 用 surprise_score 公式 fit 一个线性 classifier，测在 holdout 数据上的 AUC | 量化 signal 有多 reliable |
| ★ | 跨 layer 比较（layer 10/20/30/36）| 看哪一层信号最强 |
| ★ | 加 neutral projection（v3 缺失的）后重测 | 看效应是否更干净 |

---

## 5. 资源 + 文件

| | 路径 |
|---|---|
| 题库 | `inputs/cognitive_v3_mcq/questions.json` |
| 实验脚本 | `scripts/run_mcq_experiment.py` |
| 分析脚本 | `scripts/analyze_mcq.py` |
| 原始数据 | `outputs/cognitive_v3_mcq/raw_projections.json` |
| 主图 | `outputs/cognitive_v3_mcq/cohen_d_summary_last.png` |
| 散点图 | `outputs/cognitive_v3_mcq/strip_plot_per_concept_last.png` |

GPU 时间：~1 分钟（160 forward 在已 load 模型上）+ 模型加载 1 分钟 = ~2 分钟总计
