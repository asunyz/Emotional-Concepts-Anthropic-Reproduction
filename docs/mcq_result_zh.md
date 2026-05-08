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

### 2.3 三个具体题目细看（教科书样例 + 反例）

汇总数字漂亮，但需要看几道**具体题目**确认信号在 per-question 层面也成立。下面 3 道题展示三种情况。

#### 📐 例 1: Q22 数学题——「12 × 12 = ?」（教科书级 cleanest 信号）

每行是一个候选答案，每列是一个 vector 在「答案 token」位置的投影。

| 选项 | confused | stubborn | bored | confident |
|---|---|---|---|---|
| ✗ A) 124 | −0.089 | **+0.062** ❗ | −0.091 | +0.150 |
| **✓ B) 144** | **−0.140** ✓ | **−0.080** ✓ | **−0.035** ✓ | **+0.235** ✓ |
| ✗ C) 164 | −0.075 | +0.026 | −0.101 | +0.179 |
| ✗ D) 184 | −0.083 | −0.001 | −0.088 | +0.210 |

**怎么读这张表**：4 个 vector 全部指向**同一方向**——
- **confused**：正确答案 −0.140（最低，模型最不困惑）；3 个错答都比它高
- **stubborn**：正确答案 −0.080（最低，模型最不抗拒）；A) 124 高达 +0.062
- **bored**：正确答案 −0.035（最高，「这没什么新鲜的」）
- **confident**：正确答案 +0.235（最高，自信）

**直观解读**：当模型读到 "144"，残差流里 **5 个信号同时表达「我认得这个答案」**：不困惑、不固执、有点无聊（因为太熟了）、自信。读到 "124" 时，stubborn 立刻冒出来——模型「内心拒绝」这个错误。

#### 🧬 例 2: Q20 科学题——「DNA stands for?」（清晰的 confident 标记）

| 选项 | confused | stubborn | bored | confident |
|---|---|---|---|---|
| **✓ A) Deoxyribonucleic acid** | **−0.138** ✓ | **−0.053** ✓ | −0.062 | **+0.227** ✓ |
| ✗ B) Dinitrogen acid | −0.072 | **+0.047** | −0.086 | +0.169 |
| ✗ C) Dynamic nuclear array | −0.030 | +0.059 | **−0.135** | +0.153 |
| ✗ D) Dual nucleic atom | −0.044 | +0.045 | −0.110 | +0.158 |

**亮点**：confident 在正确答案上 +0.227，**比 3 个错答全部高出 0.06+**。模型像在说「**对，这就是 DNA**」。confused 在错答上 stagger 上升到 −0.030（接近 0），表示**越离谱越困惑**——这正是 cognitive dissonance 的样子。

#### 🦒 例 3: Q34 通识题——「世界最高的动物？」（4 个 vector 全 hit）

| 选项 | confused | stubborn | bored | confident |
|---|---|---|---|---|
| ✗ A) Elephant | −0.075 | **+0.086** | −0.107 | +0.093 |
| **✓ B) Giraffe** | **−0.115** ✓ | **−0.008** ✓ | **−0.055** ✓ | **+0.187** ✓ |
| ✗ C) Camel | −0.037 | **+0.121** | −0.121 | +0.056 |
| ✗ D) Horse | −0.021 | +0.103 | −0.086 | +0.061 |

**亮点**：所有 3 个错答都比正确答案 stubborn 高 +0.09 ~ +0.13。模型读到「Camel」/「Horse」时**显著抗拒**。还有个有意思的 nuance：A) Elephant 是最大的动物（但不是最高的）——它的 confused 是 −0.075（比 C/D 更负），说明模型「半懂」地判断它有点合理但不完全对。

#### ⚠️ 反例: Q5 地理题——「人口最多的国家？」（信号失效）

| 选项 | confused | stubborn |
|---|---|---|
| ✗ A) United States | −0.081 | +0.028 |
| ✗ B) Russia | −0.061 | **+0.104** |
| **✓ C) India** | −0.071 | +0.044 |
| ✗ D) Brazil | −0.044 | **+0.107** |

**问题**：在这道题上信号**没有像其他题那样指向 C) India**。confused 和 stubborn 在 4 个选项之间都差不多（confused −0.04 到 −0.08，stubborn −0.02 到 +0.10）。Russia 和 Brazil 的 stubborn 比 India 还高。

**为什么**？两种可能：
1. **模型不太确定**：「人口最多的国家」近年印度刚超过中国（题目里都没列中国），模型可能在 India 和 USA/Russia 之间犹豫——结果导致连「错答」也没有典型 dissonance 信号
2. **题目设计问题**：4 个选项都是「世界大国」，模型对每个都有相关 prior，错答不是「明显荒谬」（不像「12×12=124」这种）

**这告诉我们**：MCQ surprise signal 在「**对模型来说毫无悬念**」的题上最强；在「模型本来就不确定」的题上信号会**钝化**——这恰好是下一步要测的：**模型不确定 vs 模型确定** 的不同响应。

---

### 2.4 例子的统一模式

把 4 个例子摆一起看，浮出一个**通用的「正确答案 fingerprint」**：

```
正确答案的标志（5-vector pattern）：
  confused   ↓↓↓  最低（不困惑）
  stubborn   ↓↓↓  最低（不抗拒）
  confident  ↑↑↑  最高（自信）
  bored      ↑↑   较高（无新意）
  confirmed  ↑    略升（验证完毕）
```

而错误答案的 fingerprint 是相反方向（除了 confident 在错答上也常常较高，因为模型把任何答案都「显得自信」——这是 instruction-tuned 模型的本性）。

**最关键的「鉴别器」是 stubborn**——它的 d=−1.34 意味着 stubborn 在错答上**几乎从不为负**，在正确答案上**经常为负**。这是个 robust 的 wrong-detector。

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
