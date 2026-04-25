---
name: Top-conference MER ideas
overview: Define a top-conference-oriented idea portfolio for EEG+ECG multimodal emotion recognition, built on your current MVP pipeline and optimized for rapid hypothesis testing plus publishable rigor.
todos:
  - id: freeze-eval-protocol
    content: Define and freeze subject-level split, metrics, seeds, and statistical testing templates for all future ideas.
    status: pending
  - id: build-robustness-benchmark
    content: Add standardized stress tests for missing modalities, temporal lag perturbation, and noise injection.
    status: pending
  - id: prototype-idea-a
    content: Implement minimum viable version of missing-modality robust fusion and compare against early-fusion baseline.
    status: pending
  - id: prototype-idea-b
    content: Implement minimum viable cross-subject generalization mechanism (invariant encoder + lightweight subject adapter).
    status: pending
  - id: decide-mainline
    content: Select one lead direction based on effect size, stability, and reproducibility, then invest in full ablations and writing narrative.
    status: pending
isProject: false
---

# EEG+ECG 顶会导向 Idea 路线图

## 你的现状判断（基于当前代码）
- 你已经有了很好的“实验基础设施起点”：训练入口、配置、DREAMER 读取、窗口化、manifest 缓存。
- 当前短板不是“能不能跑”，而是“能不能形成审稿人认可的问题定义与证据链”。
- 你最该做的是把平台从 `baseline runner` 升级为 `hypothesis testing engine`（每个 idea 都能快速插拔、统一评估、自动产出证据）。

## 顶会更容易买单的 6 类 idea（按建议优先级）

### 1) 缺失模态鲁棒学习（最高优先）
- **问题痛点**：真实场景 ECG/EEG 常有掉线或噪声；大多数方法默认全模态可用。
- **idea 核心**：训练时随机模态丢弃 + 模态不确定性估计 + test-time 自适应融合。
- **为什么有机会**：从“精度竞赛”转成“可部署鲁棒性”，审稿人更容易认可价值。
- **关键指标**：完整模态性能、单模态缺失性能、随机缺失曲线下面积。

### 2) 跨被试泛化 + 个体差异建模
- **问题痛点**：MER 常被批评“同被试有效、跨人失效”。
- **idea 核心**：subject-invariant 表征 + subject-specific 轻量 adapter（或元学习初始化）。
- **为什么有机会**：这是该领域的核心挑战之一，容易形成“问题-方法-证据”闭环。
- **关键指标**：LOSO/跨被试协议、不同被试分布下稳定性。

### 3) 跨模态时序对齐学习（EEG/ECG 不同采样率与反应延迟）
- **问题痛点**：你现在做了工程对齐，但“生理反应时延可变”尚未建模。
- **idea 核心**：可学习时间偏移（soft alignment / cross-attention with lag prior）。
- **为什么有机会**：技术新意可写进方法章节，而不是仅靠预处理。
- **关键指标**：不同窗口长度、不同延迟扰动下的鲁棒性。

### 4) 因果/不变性驱动的情绪表征
- **问题痛点**：模型学到的是数据集偏差（被试习惯、设备特征）而非情绪因子。
- **idea 核心**：environment split（按被试/试次/刺激条件）+ invariant risk regularization。
- **为什么有机会**：叙事层级更高，从“分类器”升级为“泛化机制”。
- **关键指标**：跨环境外推性能、spurious correlation 干预实验。

### 5) 自监督预训练 + 小样本下游微调
- **问题痛点**：标签少、噪声大，监督学习不稳定。
- **idea 核心**：EEG/ECG 联合掩码重建或跨模态对比预训练，再做下游情绪分类。
- **为什么有机会**：如果能证明低标注优势，会明显增加论文含金量。
- **关键指标**：1%、5%、10%标签预算曲线；与纯监督对比。

### 6) 可解释融合（不是可视化装饰）
- **问题痛点**：MER 论文常被质疑“黑箱且不可验证”。
- **idea 核心**：通道/频段/心率片段级归因 + 干预验证（mask 后性能变化）。
- **为什么有机会**：能提升可信度，但通常作为加分项，不建议单独做主贡献。
- **关键指标**：归因一致性、反事实干预有效性。

## 平台层面必须先补的“顶会基础设施”
- **统一实验协议层**：固定切分（尤其 LOSO）、seed、指标、统计检验。
- **idea 插件接口**：把“融合策略/损失项/对齐模块”做成可配置模块，避免每次重写训练流程。
- **自动 ablation 模板**：每个 idea 默认输出主结果 + 3 个必要消融。
- **鲁棒性评测集**：系统生成模态缺失、噪声注入、时延扰动的 stress tests。

## 12 周建议节奏（研究收益最大化）
- 第 1-3 周：把评估协议与鲁棒性 benchmark 固化（这是未来所有论文的地基）。
- 第 4-7 周：主攻 `缺失模态鲁棒` 与 `跨被试泛化` 两条主线，各做最小可行方法。
- 第 8-10 周：选择更有增益的一条，深入做机制增强与完整消融。
- 第 11-12 周：补跨数据集/低标注实验，形成投稿叙事。

## 你现在最该做的方向选择
优先从以下组合中选一个作为“首篇顶会冲刺主线”：
- 组合 A（更稳）：缺失模态鲁棒 + 跨被试泛化。
- 组合 B（更新）：时序对齐学习 + 缺失模态鲁棒。
- 组合 C（更冒险）：因果不变性 + 跨被试泛化。

## 关键落点（结合当前仓库）
- 你已有数据层骨架可承载上述路线：`src/data_pipeline/dreamer/mat_io.py` 与 `src/data_pipeline/dreamer/tools/build_manifest.py`。
- 下一步重点不在“再造读取代码”，而在给 manifest 增加实验协议标签与扰动生成策略，并把训练入口从 toy 路由完全切到 MER 路由。