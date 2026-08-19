# RiskCal-TKG 论文大纲与当前完成状态

## 论文元信息

- 英文题目：*RiskCal-TKG: Drift-Aware Prequential Calibration for Temporal Knowledge Graph Forecasting under Controlled History Deletion*
- 作者：xinyu wang
- 当前版本日期：2026-08-12
- 当前证据等级：理论推导 + 软件验证 + 完整 ICEWS14 的低预算 CPU 诊断；尚无确认性服务器证据
- 建议投稿层级：目标仍可定位一区，但当前版本不是可投稿终稿；需要先完成方法实现、强基线、多数据集和五种子服务器实验

## 1. 核心研究问题

TKG forecasting 通常只报告 MRR/Hits@K，但排序质量不能回答候选答案集能否以目标概率覆盖记录答案，也不能说明为覆盖付出了多大的集合代价。本文研究严格时间外推、分布漂移和训练历史缺失共同作用时的预测集可靠性。

必须区分两个数学对象：

1. 已实现 MVP：在测试前选择一次全局 half-life，测试期间固定；不计算漂移特征，不按批次自适应。
2. 论文目标方法：根据 relation composition、score-margin、recurrence 和 batch size，在当前标签揭示前逐批选择 half-life；尚未实现。

当前稿件不把 MVP 的结果归因给自适应方法，也不宣称任意漂移下的分布无关保证。

## 2. 研究问题与唯一确认性主指标

- RQ1：静态阈值在严格时间外推下何时欠覆盖？
- RQ2：训练历史删除如何同时损害 scorer 排名与 calibration？
- RQ3：批级漂移自适应是否优于 Static、Rolling、Fixed Weighted、ACI 和 non-exchangeable 方法，并避免集合过度膨胀？
- RQ4：结论能否跨 backbone、数据集和缺失机制保持？

唯一主设置：ICEWS14、RE-GCN、30% entity-coverage-constrained uniform deletion、目标 coverage 0.90。

主估计量：

\[
\Delta_{\mathrm{under}}
=(0.90-\widehat{\mathrm{cov}}_{\mathrm{Static}})_+
-(0.90-\widehat{\mathrm{cov}}_{\mathrm{RiskCal}})_+.
\]

确认性 gate 同时要求点估计至少 0.03，且单侧 95% 置信下界大于 0。当前 CPU 诊断无资格检验该 gate。

## 3. Abstract

六句结构：背景、缺口、方法身份、严格评测协议、确认性结果占位、结论边界。服务器实验完成前，摘要只说明研究设计和方法状态，不填写任何本地诊断数值。

## 4. Introduction

1. TKG forecasting 的严格时间外推性质。
2. MRR/Hits 与 observed-label coverage、set size、selective risk 的差异。
3. 静态 KG answer-set calibration 的边界。
4. 时间漂移和历史缺失为何使固定阈值失准。
5. prequential timestamp-batched 协议。
6. 四项贡献，其中自适应 selector 必须标为 paper-facing specification，直到实现完成。

## 5. Related Work

### 5.1 TKG forecasting

Continuous-time DistMult、RE-GCN、TiRGN、Frequency、Repeat/Relaxed-Repeat、TGB 2.0。

### 5.2 KG calibration

Safavi calibration、KGCP、CondKGCP。

### 5.3 Temporal/non-exchangeable conformal methods

- CFEP，Findings of ACL 2026：最直接的同行评审 TKG 对手，但预测 event/relation types，不是全实体 link forecasting。
- NCPNET，KDD 2025：temporal graph conformal 方法，任务/数据不同。
- Lam et al.，SSRN 2026：ICEWS14 multi-hop path reasoning 的同期非同行评审预印本。
- ACI、weighted conformal、beyond-exchangeability、NExCRC。

创新性不能写成“首次把 conformal 用于 TKG”。目前仍可争取的差异是：全实体 subject/object forecasting、训练历史删除、scorer degradation 与 calibration drift 分离、批级 prequential 更新、coverage--set-size--ranking 联合审计。

## 6. Problem Formulation

1. TKG 四元组、subject/object 查询与逆关系。
2. closed-world transductive temporal extrapolation。
3. label-marginal coverage、query full-set coverage、partial answer recall 的区别。
4. margin nonconformity：

\[
a(x,y)=\max_{y'}f_\theta(x,y')-f_\theta(x,y),\quad
C_t(x)=\{y:a(x,y)\le q_t\}.
\]

5. 60/20/20 严格时间切分，以及确认性版本需要的四个互不重用 pre-test 角色。

## 7. Method and Theory

### 7.1 已实现方法

- Static finite-sample order statistic。
- Rolling，最近 1,000 个已揭示 score。
- Fixed weighted，候选 half-life \(\{7,14,30,\infty\}\)，测试前全局选择一次。
- timestamp batch 全部预测后再统一揭示标签。

### 7.2 已完成理论

- prediction set 随阈值嵌套；非负阈值下非空。
- observed-label inclusion 与 \(A_t\le q_t\) 完全等价。
- 含并列分数原子质量的确定性 coverage decomposition：

\[
1-\alpha-\delta_t-\epsilon_t
\le F_t(q_t)\le
1-\alpha+\delta_t+\epsilon_t+\eta_t.
\]

- \(N_{\mathrm{eff}}=(\sum_iw_i)^2/\sum_iw_i^2\) 及规则几何流闭式。
- \(h\to0\)、\(h\to\infty\) 的极限解释。
- 校准器不改变同一 scorer 的候选排序，因此不能提高 MRR/Hits。

该部分是确定性范围分析，不是 arbitrary-drift coverage theorem。

### 7.3 尚未实现的 paper-facing 方法

漂移特征、per-half-life coverage/size predictors、batch-adaptive selector、独立 tuning/selector/final-calibration 角色、冻结 tie rule 和 leakage tests。

## 8. Experimental Design

### 8.1 数据和缺失机制

- ICEWS14 已完成规范化与哈希：7,128 entities、230 relations、90,730 facts、365 timestamps。
- 服务器阶段：ICEWS05-15/GDELT 加一个结构明显不同的 temporal graph domain。
- 缺失：当前只有 entity-coverage-constrained deletion；仍需 independent、temporal-block、recent-biased、relation/source-family dropout。

### 8.2 Backbones/Baselines

- 主 backbone：RE-GCN；第二强模型建议 TiRGN。
- 简单 ranking baselines：Frequency、Repeat/Relaxed-Repeat。
- Calibration：Top-1、Static KGCP、CondKGCP、Rolling、Fixed Weighted、RiskCal adaptive、ACI、NExCRC/beyond-exchangeability、CFEP-derived comparison。

### 8.3 指标和推断

- Ranking：filtered MRR、Hits@1/3/10。
- Reliability：micro coverage、macro-time coverage、positive undercoverage、absolute gap、group/worst-window coverage。
- Efficiency：mean/median/P90 set size、normalized size、answer/abstention rate、selective risk、time/memory/storage。
- 五个 scorer seeds：17、29、43、59、71。
- 外层 seed、内层 contiguous timestamp block 的 paired moving-block inference；p 值必须 null-centered 或合法置换。

## 9. Results 组织

### 9.1 Ranking under deletion

先验证 neural scorer 是否优于 Frequency/Repeat；校准方法共享同一 scorer 时只报告一次 ranking。

### 9.2 Static calibration failure

同时展示 micro、macro-time、逐时间戳与最差窗口，避免高流量时间掩盖风险。

### 9.3 Adaptation vs strong baselines

确认性主表仅允许冻结的五种子服务器结果；本地数据不得填入。

### 9.4 Price of reliability

coverage 必须与 normalized set size、selective risk、answer rate 联合解释。

### 9.5 Group failures and cases

relation frequency、recurrence、direction、future-only entity、history count；成功/失败案例均需报告。

### 9.6 Ablation and efficiency

漂移特征、half-life、window、score function、feedback delay、timestamp-normalized weighting、exact/ANN scaling。

## 10. 已完成的本地证据

### 10.1 单元/烟雾验证

项目已有 62 个测试，覆盖 parsing、temporal split、calibration primitives、batching、artifact writing、论文导出链和 toy end-to-end。它们只证明软件行为，不证明论文效果。

### 10.2 早期子集 pilot

一个频率偏置的 ICEWS14 实体子集、单种子工程实验已保留在正文的非确认性小节。它没有达到目标 coverage，且 scorer 低于 Frequency；只作为失败可见性和管线记录。

### 10.3 完整 ICEWS14 CPU diagnostic

设置：完整 90,730 facts，3 seeds × 2 deletion rates，16-d DistMult，3 epochs，CPU；六个条件 414.2 秒完成。

主要诊断：

- Static micro coverage：0% deletion 为 0.8425，30% deletion 为 0.8518。
- Rolling/Fixed Weighted 的平均 coverage 约 0.90。
- 代价是平均预测集包含约 69%--74% 的 7,128 个实体，实用性很差。
- Fixed Weighted 在 49%--51% 的时间戳仍低于 0.90；最差时间戳 coverage 跨种子均值约 0.848/0.843。
- 五个条件选择 \(h=\infty\)，只有 seed 29 + 30% deletion 选择 \(h=7\)；没有证据表明固定衰减优于 Rolling。
- scorer MRR 从 0.1330 降至 0.1215；Frequency 约 0.0925。校准恢复平均 coverage，并没有恢复 ranking。

这些值只允许出现在标记为 non-confirmatory 的小节、数据文件和图中。

## 11. 当前不能完成的部分与原因

完整依赖矩阵见 COMPLETION_STATUS.md。主要缺口：

1. 缺方法实现：batch-adaptive selector、四角色无重用协议、额外缺失机制、feedback delay。
2. 缺 baseline code：RE-GCN/TiRGN adapter、Repeat、KGCP/CondKGCP、CFEP-derived、ACI、NExCRC。
3. 缺服务器计算：强 backbone、五种子、多数据集、完整消融、精确全实体 scaling。
4. 缺额外数据集：长时间事件域和至少一个结构不同的 temporal graph domain。
5. 缺确认性结果：主表、置信区间、主 gate、结果依赖型 Discussion/Conclusion/Abstract。
6. 缺作者元数据：affiliation、地址、ORCID、邮箱、funding、COI。
7. 缺期刊决定：官方模板、声明、版式和最新投稿要求。

## 12. 投稿前最低完成条件

1. 实现并测试 paper-facing adaptive method，冻结协议后再看服务器结果。
2. 完成唯一主设置的 5 seeds，以及其余预声明的 secondary settings。
3. 接入强 backbone、简单 heuristic 和直接 conformal competitors。
4. 至少 2--3 个数据域，不能全部是高度同质的事件抽取图。
5. 完成 paired block inference、消融、敏感性和资源报告。
6. 依据真实结果选择正结果、trade-off/负结果或 benchmark 定位，不事后扩张主张。
7. 更新作者信息、期刊模板、参考文献和 artifact URL。

## 13. 当前文件导航

- manuscript.tex：英文详细初稿。
- sections/theory_scope.tex：可严格成立的理论推导。
- COMPLETION_STATUS.md：证据等级、依赖矩阵和 2026 文献竞争边界。
- scripts/export_paper_diagnostic.py：从不可变运行目录重建论文 CSV、JSON 和图的唯一版本化导出入口。
- data/local_cpu_diagnostic_manifest.json：CPU 诊断证据哈希。
- data/local_cpu_diagnostic_aggregate.csv：诊断汇总。
- data/local_cpu_diagnostic_by_time.csv：内部时间 ID、原始时间标签和逐时刻覆盖率。
- data/local_cpu_diagnostic_macro_time_by_seed.csv：逐种子宏时间审计。
- figures/local_cpu_coverage_by_time.*：时间覆盖图。
- figures/local_cpu_coverage_size_tradeoff.*：coverage--set-size 图。
