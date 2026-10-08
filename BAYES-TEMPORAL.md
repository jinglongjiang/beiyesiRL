# 贝叶斯时序替换：三臂训练与闭环测评综合报告

更新日期：2026-10-08，最终完成核查与离线分析截至 21:32 KST。
**六组正式训练和六组完整评测全部完成：每组 10000 RL / 40000 次优化，每组 3000 个评测案例。**
全部 CSV 已独立重算：各有 3000 条唯一案例、6 场景各 500，成败计数与各组完成标记一致；162 份冻结源码与 13 份配置 SHA 全部一致。
本次只分析已保存结果、更新报告并上传 Git，没有改模型、重启训练、补跑评测或触碰 KDA。

统一名称：贝叶斯简单结构 = bayes_mean，滤波后只读取均值；贝叶斯复杂结构（修复版）= bayes，读取均值和 logvariance。
两者均在滤波内部使用方差；简单结构不是“不用贝叶斯”。修复版复杂结构不再执行 128 点积分，也不是精确期望价值计算。
第 17 节是本轮最终分析；第 15、16 节保留工程过程及历史运行快照；第 1--14 节保留修复前结果，不覆盖负结果。

## 最新结果总表：修复版，10000 次训练

SR/CR/TR = 成功率 / 原生碰撞率 / 超时率，单位 %；几何接触单独统计，不等同于 CR。

| 模型 | 在线 RL seed | 成功 / 碰撞 / 超时次数 | SR | CR | TR | 几何接触率 |
|---|---:|---|---:|---:|---:|---:|
| GRU | 42 | 2684 / 84 / 232 | 89.47 | 2.80 | 7.73 | 2.43 |
| 贝叶斯简单结构 | 42 | 2500 / 85 / 415 | 83.33 | 2.83 | 13.83 | 2.50 |
| 贝叶斯复杂结构（修复版） | 42 | 2693 / 93 / 214 | 89.77 | 3.10 | 7.13 | 2.63 |
| GRU | 43 | 2630 / 241 / 129 | 87.67 | 8.03 | 4.30 | 7.17 |
| 贝叶斯简单结构 | 43 | 2396 / 94 / 510 | 79.87 | 3.13 | 17.00 | 2.77 |
| 贝叶斯复杂结构（修复版） | 43 | 2431 / 59 / 510 | 81.03 | 1.97 | 17.00 | 1.70 |
| GRU，两 seed 合并描述 | 42 + 43 | 5314 / 325 / 361 | 88.57 | 5.42 | 6.02 | 4.80 |
| 贝叶斯简单结构，两 seed 合并描述 | 42 + 43 | 4896 / 179 / 925 | 81.60 | 2.98 | 15.42 | 2.63 |
| 贝叶斯复杂结构，两 seed 合并描述 | 42 + 43 | 5124 / 152 / 724 | 85.40 | 2.53 | 12.07 | 2.17 |

**结论：修复版复杂结构相对简单结构有闭环比较正信号，但没有稳定胜过 GRU，也未证明状态相关后验方差的独立因果价值。没有 METHOD_ENTRY_FOUND。**

- 复杂相对简单：SR 在 seed42 / seed43 分别提高 6.43 / 1.17 个百分点；合并点估计 +3.80 个百分点。但 seed42 接触略增，seed43 的 SR 配对区间跨 0，改善强度不稳定。
- 复杂相对 GRU：seed42 SR 仅 +0.30 个百分点，seed43 -6.63 个百分点；合并 -3.17 个百分点。较低碰撞伴随更多超时，不构成全面优势。
- 新简单结构 seed42 SR 从修复前 91.83% 降至 83.33%。旧、新训练初始化和 readout 形状不同，不能把当前复杂对简单的全部优势归因于方差信息。
- 两在线 RL seed 共用各臂本轮 IL seed42；评测重复同一 3000-case 池，6000 行不是 6000 个独立世界，也不是 fresh confirmation。

## 本轮完成与剩余证据

## 本轮完成与未完成总览

| 工作 | 状态 | 实际完成内容 | 尚缺什么 / 不能推出什么 |
|---|---|---|---|
| 4090 资产恢复 | 已完成 | 官方线本地代码、冻结配置、93.7 MB ORCA 数据恢复；非空 SHA 一致、版本 matched-native-orca-v1 | 没有下载或启用其他母体 |
| B0 旧积分偏置复核 | 已完成，首轮失败保留 | native 10696 状态：恒定比 99.410%、残差 1.555%、余弦中位 0.997830 | ORCA-prefix 首轮残差 2.230% 未过阈值；不是逐数重现原 15680 状态结果 |
| B1 sigma / 危险关联 | 已完成 | 192 原生 episode，按 episode 留出；4 步危险均值 AUC 0.724、逐维探针 0.864 | 几何对照 0.860；不证明独立信息、概率校准或导航收益；contact 测试折无正例 |
| B2 oracle 排序 | 部分完成，预算终止 | 预注册 12 root：8 个全测、第 9 个部分，找到 5 个无接触救回 | 3 个未测；不是总体 SR 上界，不证明可部署排序或 Bayesian 增量 |
| B3 模型修复 | 已完成 | 仅 production bayes_temporal.py：concat(mean, logvariance)，Mean 方差列置零；输出/head 维度不变 | 新 Full 是 belief feature embedding，不再是容积积分期望价值 |
| B3/B4 四项自检及 parity | 已完成 | 方差屏蔽/敏感性、同形状参数、普通与 legal-prefix、旧 Mean 等价映射误差 0；两设备各 18 项单测通过 | 工程通过不等于性能通过；旧 Bayes 权重不能直接 strict 加载新结构 |
| B4 评测入口 | 已完成 | 新 Full ep200 通过六场景入口，CSV 保留 3 success / 1 collision / 2 timeout | 六个案例只验收入口，不作 SR 或方法结论 |
| 新 IL | 已完成 | GRU/Mean/Full seed42 各 50 epoch，原生 best-IL 选择，strict 验收 | seed43 独立 IL 预算裁剪，未完成且不再排队；两个 RL seed 共用新 IL |
| B5 smoke | 已完成 | 三臂各 200 RL、800 optimizer steps；全部 PASS，完整成败记录见 15.7 | 是带探索训练，不是六场景正式 benchmark |
| B6/B7 正式启动与健康 | 已完成 | 六进程、正确 seed/config/GRID/data/IL；持续观察及全部 ep500 strict/finite/step2000 通过 | 启动验收不等于完成 10000 回合 |
| 正式 10000 RL | 六组全部完成 | training-results.json：各 10000 回合、40000 优化；全部 TRAINED | 训练累计 SR 不代替测试结果 |
| 六臂各 3000-case 评测 | 六组全部完成 | complete.json：全部 EVALUATED；18000 行完整成败数据重算一致 | 没有新训练种子/新场景的 fresh confirmation |
| 复杂 vs 简单闭环比较 | 已验证点估计与配对差值 | 两 seed SR 点估计均更高；分场景、安全代价及区间见 17 节 | 不等于方差校准、方差因果增量或稳定胜 GRU |
| 方差因果消融 / 简单臂退化归因 | 未完成 | 本轮没有固定/置乱方差或等价初始化对照 | 不提前认定修复后的所有差异来自有效后验信息 |

正式结果已完成；B2 的预算裁剪诊断仍保留为部分完成，不能将“正式跑全部完成”扩大成“所有历史诊断和归因都完成”。

## 修复前历史统一总表

SR/CR/TR 依次为成功率/原生碰撞率/超时率，单位 %。六场景行均为修复前优化版 10000 次训练结果，每场景 500 回合。

| 项目 | GRU | Bayes-Mean | Bayes-Full |
|---|---:|---:|---:|
| 旧 3000：SR / CR / TR | 72.57 / 3.63 / 23.80 | 80.83 / 2.33 / 16.83 | 85.90 / 2.83 / 11.27 |
| 旧 8000：SR / CR / TR | 82.60 / 2.47 / 14.93 | 88.47 / 1.93 / 9.60 | 84.53 / 1.30 / 14.17 |
| 本轮 10000：SR / CR / TR | 89.47 / 2.80 / 7.73 | 91.83 / 3.13 / 5.03 | 87.17 / 4.90 / 7.93 |
| 本轮成功 / 碰撞 / 超时次数 | 2684 / 84 / 232 | 2755 / 94 / 151 | 2615 / 147 / 238 |
| 基础圆形：SR / CR / TR | 100.00 / 0.00 / 0.00 | 99.60 / 0.20 / 0.20 | 98.00 / 1.40 / 0.60 |
| 基础方形：SR / CR / TR | 91.00 / 1.20 / 7.80 | 95.80 / 2.00 / 2.20 | 86.20 / 2.80 / 11.00 |
| 密集圆形：SR / CR / TR | 98.40 / 0.60 / 1.00 | 99.00 / 1.00 / 0.00 | 89.40 / 3.80 / 6.80 |
| 密集方形：SR / CR / TR | 65.60 / 7.40 / 27.00 | 78.80 / 7.60 / 13.60 | 65.80 / 13.20 / 21.00 |
| 大范围圆形：SR / CR / TR | 95.20 / 0.80 / 4.00 | 87.00 / 0.60 / 12.40 | 98.60 / 0.20 / 1.20 |
| 大范围方形：SR / CR / TR | 86.60 / 6.80 / 6.60 | 90.80 / 7.40 / 1.80 | 85.00 / 8.00 / 7.00 |
| 几何接触回合：旧 3000 / 8000 / 本轮 10000 | 93 / 66 / 73 | 63 / 52 / 81 | 77 / 34 / 131 |
| 本轮几何接触率 % | 2.43 | 2.70 | 4.37 |
| 本轮成功回合平均到达时间 s | 14.16 | 14.04 | 13.29 |
| 本轮成功回合平均路径 m | 12.26 | 11.49 | 11.27 |
| 本轮 discomfort 步数占比 % | 2.89 | 2.48 | 3.86 |
| 本轮平均停滞窗口占比 % | 10.42 | 10.32 | 10.84 |
| 本轮最小 clearance m | -0.259 | -0.187 | -0.185 |
| 本轮逐步加权推理延迟 ms | 10.51 | 12.86 | 12.99 |
| 注册参数 / 时序模块参数 | 1754857 / 1579520 | 290985 / 115648 | 290985 / 115648 |
| 本轮 RL 完成 / optimizer updates | 10000 / 40000 | 10000 / 40000 | 10000 / 40000 |
| 本轮训练墙钟 h | 5.16 | 7.30 | 7.61 |
| 本轮 3000 回合测评墙钟 h | 0.67 | 0.75 | 0.74 |
| IL 重训 | 否，复用原 IL | 否，复用原 IL | 否，复用原 IL |
| 本轮初始化 | IL 权重，RL 从头 | IL 权重，RL 从头 | IL 权重，RL 从头 |
| 科学结论 | 参照 | 成功/超时最好，安全并非全面占优 | 相对两臂 SR/CR/TR 均更差 |

表中“本轮”均指修复前的 2026-10-07 轮次，不指目前运行的显式方差修复版。修复前 10000 不是旧 8000 的续训，不能将两轮差异归因于仅多训练 2000 次。旧 8000 则是从 3000 权重 warm continuation。
仅训练 seed 42，测评复用同一案例池，没有 fresh confirmation。成功时间/路径只统计各自成功回合，不能凭 Full 更短就认定效率优势。
训练墙钟包含预填充、加载核验、validation 和保存；三臂共享 4090 并行，不是隔离硬件速度测试。
几何接触独立于原生 collision；负 clearance 表示半径范围重叠。停滞占比为逐回合窗口比例的均值。

**修复前 10000 的结论：没有 METHOD_ENTRY_FOUND。Mean 相对 GRU 多成功 71 回合，但多 10 次原生碰撞；Full 相对 Mean 少成功 140 回合、多碰撞 53 次、多超时 87 次。不能沿用旧 8000 的 Full 安全优势解释这组结果，也不能将其作为当前修复版结论。**

本轮原始 CSV、checkpoint、训练计时与完成记录：

/home/abc/workspace/CrowdNav(20270731_backup2)/CrowdNav/crowd_nav/runs/bayes-optimized-20261007/accepted-run

## 1. 历史 8000 阶段结论（不是最新结论）

**目前有单训练 seed 的贝叶斯时序替换正信号，但没有 METHOD_ENTRY_FOUND。**

- Bayes-Mean 在 8000 次训练后达到 SR 88.47%，比 GRU 高 5.87 个百分点；超时更少，原生碰撞点估计也更低。
- Bayes-Full 达到 SR 84.53%、CR 1.30%；碰撞和几何接触最少，但成功率低于 Mean，超时和绕行更多。
- Full 相对 Mean 的成功率优势从 3000 次时的 +5.07 个百分点，变为 8000 次时的 -3.93 个百分点。不能说“训练越久，完整 posterior 消费越好”。
- Mean 本身仍是高斯滤波器：方差参与 Kalman gain 和状态更新。它不是完全不使用不确定性的普通确定性网络。
- 当前结果支持“这种低参数量滤波时序结构值得继续研究”，尚不支持“显式期望价值消费已经形成独立、稳定、可发表的新增能力”。
- Full 的安全改善和通行损失同时存在。没有预先冻结的综合效用或可接受 trade-off 阈值，不替用户宣布它全面优于 Mean。

## 2. 实际做了什么

保留原关系空间编码、Top-5 输入、80 个候选动作、解析后继状态、价值 lookahead、ORCA value pretraining 和 online MC-return value refinement。
只比较三种时序模块；没有新奖励、新动作、新预测器、新人类模型或外部 Bayesian planner。

```text
合法历史窗口 [B,24,8,13]
  -> 原关系空间编码 [B,24,256]
  -> GRU 或 64 维高斯时序滤波
  -> 256 维时序价值特征
  -> 原标量 Value Head
  -> 原 80 动作 successor-value lookahead
  -> 原测试执行规则
```

| 臂 | 时序状态 | 价值特征读取 | 当前定位 |
|---|---|---|---|
| GRU | 四层、宽度 256 的 GRU hidden | 最后有效时刻特征 | 常规时序参照 |
| Bayes-Mean | 64 维对角 Gaussian 的均值和方差 | readout(mu) | 强简单 Bayesian 对照 |
| Bayes-Full | 同一参数化的 Gaussian | 128 点 cubature 近似 E[readout(z)] | 完整方差消费参考 V1 |

两个 Bayesian 臂参数数量相同，但分别训练；其权重、在线轨迹和 replay 后期内容不相同。
因此 Full–Mean 是训练后结构比较，不是对同一已训练状态的一次纯读出干预。

### 滤波与学习边界

```text
a = sigmoid(retention_logits)
m_prior = a * m_previous + drift
P_prior = a^2 * P_previous + softplus(Q)
K = P_prior / (P_prior + R(x))
m = m_prior + K * (measurement(x) - m_prior)
P = P_prior * R(x) / (P_prior + R(x))
```

首个真实帧直接更新初始 prior，不额外推进一次 transition；正方差参数化有数值下限。
每次输入窗口从 prior 重算，不把假想后继帧写回真实历史。

所有滤波参数仍只受原 value MSE 监督，没有 NLL、ELBO、预测似然或校准损失。
latent 是学习到的特征，不是已识别的人类目标或物理交互类型。
CV successor 是反事实价值输入，不是真实未来观测，也不是完整 observation-distribution integration。
本版不声称 calibrated Bayesian uncertainty、active information gain 或 Bayes-optimal planning。
递归 Gaussian 更新和 cubature 本身不是本研究的原创机制；当前尚未建立方法级 novelty 差分。

## 3. 冻结协议与公平性

| 项目 | 实际执行 |
|---|---|
| 训练 seed | 仅 42；没有完成第二个训练 seed |
| 官方训练母体 | 用户提供的本地 CrowdNav，未切换外部项目 |
| 平台 | 三臂均在 RTX 4090、Python 3.10.19、Torch 2.9.1+cu128 / CUDA 12.8 |
| 共享 ORCA 数据 | 5056 次尝试；保留 5037 条成功和 3 条碰撞轨迹，共 235980 帧 |
| 未保留尝试 | 16 次，沿用原过滤逻辑；不能擅称全部是 timeout |
| IL | 每臂实际完成 50 epoch，batch 256，同一数据与 epoch shuffle 规则 |
| IL 初始化 checkpoint | 继承同一 quick-validation selection 规则；GRU 选 epoch 5，两个 Bayesian 臂选 epoch 35 |
| RL | 原 MC-return value regression；每 episode 4 次更新，batch 256，lr 1e-5，gamma 0.99 |
| 探索 | 原 epsilon 0.3 -> 0.05，1500 episode 衰减 |
| Replay | 200000 transition 容量，T=24；不将分叉的 on-policy 轨迹冒充相同数据 |
| 奖励与动作 | 原奖励不变；5 速度 x 16 方向，不增加 stop 动作 |
| 时间限制 | 训练 50 s，正式测试 25 s，dt=0.25 s |
| 训练期 validation | 每 1000 episode 50 回合，不是六场景 benchmark |
| 正式测评 | 6 个原生场景，每场景 500 回合，每臂每阶段 3000 回合 |
| 测评案例 | base seed 20261006；固定 native case block，与早期 IL quick-selection 案例分开 |
| 固定 checkpoint | 第 3000 / 8000 次，不按这两次正式测试成绩挑 best |
| 参数量 | Full 与 Mean 相同；与 GRU 不相同 |
| 执行链 | 三臂共用原 safety filter / smoothing 与测试规则，没有为 Bayesian 放松安全条件 |

共同修正了合法历史 padding、探索分支 observation append 和 replay episode boundary。
合成的零前缀不作为真实时间证据输入时序编码；每控制步只追加一次真实当前观测。
这些是共享工程修正，不是 Ours 的贡献。当前科学源码 hash 与冻结 protocol 相符。

### 必须明确的续训限制

原 checkpoint 保存 policy、target network、optimizer 和 statistics，但没有在线 replay、RNG state 或 environment case counter。
本次在 3000 处停止、完成测评后，三臂统一从第 3001 次恢复，额外训练 5000 次：

- 原权重、目标网络和 optimizer 均核验恢复成功，没有从头训练，没有重跑 IL。
- 三臂都用同一 ORCA corpus 的 5000 条轨迹重建 replay，skipped=0，buffer_size=200000。
- 使用原 matched RNG reset；环境训练案例可能重复早期序列。
- 因此这是统一 warm continuation，**不是完全无缝的独立 8000 次连续训练复现**。
- GRU 原进程停止前跑到约 3814，但恢复使用其 3000 checkpoint；后约 814 次只算额外耗费，不计入模型优势。
- 8000 阶段再次使用同一测试案例，不是 fresh confirmation；两个阶段不是 6000 个新的独立场景。
- CUDA deterministic mode 在原训练配置中关闭；相同训练 seed 也不等于 bitwise deterministic reproduction。

## 4. 总体结果

每行完整包含 3000 个案例，没有删除失败轨迹。SR/CR/TR 为原生终止标签。
几何接触列是独立端点检查，不与原生 collision 混为一个指标。

| 阶段 | 模型 | 成功 / 碰撞 / 超时 | SR | CR | TR | 几何接触 episode |
|---|---|---|---:|---:|---:|---:|
| 3000 | GRU | 2177 / 109 / 714 | 72.57% | 3.63% | 23.80% | 93 |
| 3000 | Mean | 2425 / 70 / 505 | 80.83% | 2.33% | 16.83% | 63 |
| 3000 | Full | 2577 / 85 / 338 | 85.90% | 2.83% | 11.27% | 77 |
| 8000 | GRU | 2478 / 74 / 448 | 82.60% | 2.47% | 14.93% | 66 |
| 8000 | Mean | 2654 / 58 / 288 | **88.47%** | 1.93% | **9.60%** | 52 |
| 8000 | Full | 2536 / 39 / 425 | 84.53% | **1.30%** | 14.17% | **34** |

| 模型 | 3000 -> 8000 的 SR 变化 | CR 变化 | TR 变化 | 几何接触 episode 变化 |
|---|---:|---:|---:|---:|
| GRU | +10.03 pp | -1.17 pp | -8.87 pp | -27 |
| Mean | +7.63 pp | -0.40 pp | -7.23 pp | -11 |
| Full | -1.37 pp | -1.53 pp | +2.90 pp | -43 |

Full 追加训练后安全点估计改善，但通行表现没有同步提升。
不能把这解释成方向失败；也不能只取其 3000 高 SR 和 8000 低 CR 拼成同一模型的成绩。

## 5. 分场景结果

表内均为“成功 / 原生碰撞 / 超时”的完整人数，每格合计 500。

| 原生场景 | 环境 | GRU 8000 | Mean 8000 | Full 8000 |
|---|---|---|---|---|
| baseline_circle | 5 人，圆半径 4 m | 497 / 3 / 0 | 498 / 1 / 1 | 500 / 0 / 0 |
| baseline_square | 10 人，宽 10 m | 442 / 9 / 49 | 471 / 5 / 24 | 423 / 2 / 75 |
| dense_circle | 10 人，圆半径 4 m | 492 / 7 / 1 | 497 / 1 / 2 | 490 / 2 / 8 |
| dense_square | 20 人，宽 10 m | 245 / 18 / 237 | 292 / 19 / 189 | 283 / 15 / 202 |
| large_circle | 12 人，圆半径 6 m | 447 / 8 / 45 | 490 / 7 / 3 | 446 / 0 / 54 |
| large_square | 20 人，宽 14 m | 355 / 29 / 116 | 406 / 25 / 69 | 394 / 20 / 86 |

3000 阶段分场景 SR 保留，用于判断方向变化，不进行 checkpoint selection：

| 场景 | GRU 3000 | Mean 3000 | Full 3000 |
|---|---:|---:|---:|
| baseline_circle | 98.4% | 99.4% | 95.6% |
| baseline_square | 72.4% | 83.4% | 88.8% |
| dense_circle | 97.8% | 96.4% | 93.6% |
| dense_square | 28.2% | 48.6% | 60.6% |
| large_circle | 85.2% | 88.2% | 90.6% |
| large_square | 53.4% | 69.0% | 86.2% |

主要定位：

- Mean 8000 在六个场景的成功数都高于 GRU；dense_square 碰撞却是 19 vs 18，不能声称每个场景安全也都更好。
- Full 相对 Mean 少的 118 个成功中，baseline_square 少 48 个、large_circle 少 44 个；这两类占净缺口 92/118。
- large_circle：Full 0 碰撞、54 超时；Mean 7 碰撞、3 超时。安全与通过能力的折中最清楚。
- baseline_square：Full 2 碰撞、75 超时；Mean 5 碰撞、24 超时，仍是类似折中。
- dense_square 对所有方法仍难，最高 SR 只有 Mean 的 58.4%；不能声称密集人群问题已经解决。
- Full 从 3000 到 8000 的 large_square 成功数由 431 降至 394，dense_square 由 303 降至 283；不是全场景一致收敛。

## 6. 同案例配对比较

按 scenario、episode、seed 对齐，已验证每份 CSV 3000 个唯一键且两臂键一致。
配对指相同初始案例，不代表动作分叉后仍处于同一 physical state。

使用固定分析 seed 20261007，10000 次场景内 paired-case bootstrap，给出探索性 95% 区间。
它只描述固定模型、固定六类测试案例的抽样变化；不包含训练 seed 方差、模型选择不确定性或场景外泛化，也未作多重比较校正。

| 比较（左减右），8000 | SR 差及区间（pp） | CR 差及区间（pp） | 左独有成功 / 右独有成功 |
|---|---|---|---|
| Mean - GRU | +5.87 [4.43, 7.30] | -0.53 [-1.20, 0.13] | 330 / 154 |
| Full - GRU | +1.93 [0.37, 3.47] | -1.17 [-1.83, -0.50] | 311 / 253 |
| Full - Mean | -3.93 [-5.37, -2.47] | -0.63 [-1.23, -0.03] | 196 / 314 |

Full - Mean 的几何接触率差为 -0.60 pp，探索性区间 [-1.17, -0.03]。
3000 阶段 Full - Mean 的 SR 差为 +5.07 pp，区间 [3.43, 6.70]；两阶段的排名反转不能忽略。

**关键限制：区间即使不跨 0，也不能升级为跨训练 seed 的稳定方法收益。**

进一步看 8000 的 Full / Mean 终止类别配对：

| Full 结果，行；Mean 结果，列 | Mean 成功 | Mean 碰撞 | Mean 超时 |
|---|---:|---:|---:|
| Full 成功 | 2340 | 39 | 157 |
| Full 碰撞 | 24 | 6 | 9 |
| Full 超时 | 290 | 13 | 122 |

这证明不是 Full 在每个案例都更加安全或更差：
它在 196 个 Mean 未成功的案例成功，但也在 314 个 Mean 成功的案例失败。
尤其有 290 个“Mean 成功、Full 超时”的案例，应优先用于之后的失败分析，不能只展示 Full 救回的轨迹。

## 7. 安全、通行与行为指标

以下为 8000 阶段。路径和行为平均包含全部 outcome，避免只统计成功幸存者。
成功时间单列为条件指标；不能忽略各臂成功案例集合不同。

| 指标 | GRU | Mean | Full |
|---|---:|---:|---:|
| 端点几何接触 episode / 3000 | 66 | 52 | 34 |
| 接触采样步数 | 73 | 55 | 36 |
| 接触采样累计时间（s） | 18.25 | 13.75 | 9.00 |
| 全数据最小 clearance（m） | -0.2045 | -0.1615 | -0.1664 |
| Discomfort 步数 / 总步数 | 7650 / 189699 | 5466 / 186924 | 3419 / 190665 |
| Discomfort 步数比例 | 4.03% | 2.92% | 1.79% |
| 成功 episode 平均用时（s） | 14.412 | 14.757 | 14.495 |
| 全 outcome 平均路径（m） | 12.774 | 11.936 | 13.314 |
| 全 outcome episode 平均停滞窗口比例 | 18.26% | 12.82% | 15.40% |
| 全 outcome episode 平均后退比例 | 15.23% | 12.12% | 15.51% |
| 全 outcome episode 平均 progress efficiency | 0.6438 | 0.6918 | 0.6262 |

clearance 为中心距离减双方半径；负值表示端点重叠。接触按 clearance<0、dt=0.25 s 统计，不是连续 swept-contact 检测。
因此原生碰撞数与几何接触数不同不能直接当成漏报或方法优势。
Discomfort 使用原配置 0.2 m 阈值；比例按总采样步数加权。
停滞窗口沿用测试代码：2 s 滑窗内目标方向进展小于 0.2 m；不是“停车次数”。
Progress efficiency 是目标距离减少量 / 路径长度，不是导航成功率。

为控制成功案例选择偏差，另看两臂均成功的配对案例：

| 比较（左减右） | 共同成功数 | 平均用时差（s） | 平均路径差（m） |
|---|---:|---:|---:|
| Mean - GRU | 2324 | +0.138 | -0.340 |
| Full - GRU | 2225 | -0.015 | +0.498 |
| Full - Mean | 2340 | -0.162 | +0.833 |

Full 不是简单的“所有成功任务都更慢”；共同成功任务略快，但路径更长、总体超时更多。
“posterior 导致过度保守”目前只是解释假设，不是已证实的因果机制。

## 8. 学习过程与参数量

| 项目 | GRU | Mean | Full |
|---|---:|---:|---:|
| IL epoch-50 记录 value MSE | 0.0014 | 0.0017 | 0.0016 |
| 实际继承 best IL epoch | 5 | 35 | 35 |
| RL episode-3000 最后一条 value MSE | 0.0196 | 0.0166 | 0.0229 |
| RL episode-8000 最后一条 value MSE | 0.0083 | 0.0068 | 0.0105 |
| 注册参数 | 1754857 | 290985 | 290985 |
| 时序模块参数 | 1579520 | 115648 | 115648 |
| 有 optimizer 更新状态的参数数量 | 1698305 | 234433 | 234433 |
| 最终 optimizer state 数 | 38 | 33 | 33 |
| 每个已更新参数的累计 AdamW step | 32000 | 32000 | 32000 |

全体注册参数中各有 56552 个旧兼容 head 参数，本轮 value 路径不更新它们。
Bayesian 注册参数比 GRU 少约 83.4%，但容量不匹配，所以“同容量 superiority”不成立。

训练 loss 是各臂自己采集的轨迹与监督分布上的量，最后一个 minibatch MSE 不是可比较的测试价值误差。
3000 -> 8000 loss 下降不能证明 benchmark 会单调变好；Full 已提供反例。
每 1000 次的 50 回合 validation 中，Full 基本全为 100% SR；Mean 的 4000/5000 为 98%，其他近 100%；GRU 3000 为 96%、其余为 100%。
这与六场景 benchmark 的差距表明单一小 validation 不能替代完整测试，更不能凭 validation 满分提前宣称解决问题。

## 9. 实际计算与运行验收

| 项目 | GRU | Mean | Full |
|---|---:|---:|---:|
| 额外 5000 次续训墙钟（h） | 3.551 | 4.660 | 4.852 |
| 8000 checkpoint 的 3000 回合测评墙钟（h） | 1.826 | 1.808 | 1.865 |
| 测试决策步加权平均计时（ms） | 31.565 | 31.807 | 32.188 |
| 平均 episode 内 p95 计时（ms） | 38.324 | 32.604 | 32.974 |
| 最终完整训练 checkpoint 大小（MiB） | 27.93 | 5.65 | 5.65 |

墙钟包含恢复、训练期 validation、保存等；服务器同时有其他 Codex 的任务，负载随阶段变化。
计时围绕策略决策、包含 CUDA 同步，但不含环境 step 及计时之后的部分测试后处理。
“episode 内 p95 的平均”不是将全部 step 合并后的总体 p95。
现有证据不能声称 Bayesian 在线更快；参数更少也没有换来本轮明显的决策计时改善。
Full 每次读出含 128 个 cubature 点，是额外计算来源；没有 operator profile，不能将全部墙钟差异归因于它。

本次重新核验：

- 六份正式 CSV 共 18000 条 episode 记录，各自 3000 唯一案例；两个阶段使用重复案例，不是 18000 个独立世界。
- 三个 8000 checkpoint SHA 与完成记录一致，按本目录配置 strict load 成功，policy 和 target tensors 有限。
- 三臂各有 32000 optimizer step，符合有效 8000 x 4 更新，而非只改文件名或 episode 元数据。
- 模型/训练/测试源码 hash 与冻结 protocol 一致；没有在出结果之后修改奖励或架构。
- 19 个既有工程测试通过，不把单测通过当作科学收益。
- 本地测试仍有 NumPy ABI / h5py 弃用警告；测试通过不代表本地依赖完全无警告，正式训练使用的是单独的服务器环境。
- 每个 episode 的 fallback/override 次数及逐帧 posterior、value ranking 没有完整写入测评 CSV；不能编造这些归因指标。
- 测评 CSV 没有逐步原生 reward，因此不报告未经记录的 benchmark accumulated return。训练日志 Avg Reward 不能替代它。

3000 阶段 Mean 的父控制器曾暂停以避免 Full 重复启动；其父进程墙钟包含额外等待。
这一历史计时污染保留，不能用于架构效率论证；8000 测评没有这一父进程暂停问题。

## 10. 已知、未知与下一步

| 问题 | 当前证据状态 |
|---|---|
| 贝叶斯替换是否真正训练并进入控制？ | 是；训练状态、strict load、native closed loop 和结果齐全 |
| 是否比 GRU 有任务收益？ | 固定单 seed/cohort 中有，Mean 尤其明显；尚无跨训练 seed 确认 |
| Full 是否全面优于 Mean？ | 否；8000 时少碰撞，但多超时、少成功 |
| sigma 是否代表真实交互不确定性？ | 未证实；没有 calibration / predictive-likelihood 证据 |
| Full 的安全改善是否由方差消费直接造成？ | 未隔离；分别训练后的 dynamics、权重和在线数据均可参与解释 |
| 隐藏目标、遮挡或 actor continuity 是否得到解决？ | 未测试这些特定机制，不能从六个 nominal 场景推广 |
| 是否已构成新方法论文？ | 尚未；机制差分、跨 seed、fresh confirmation 均未完成 |

**唯一优先诊断：解释 Full–Mean 在 baseline_square / large_circle 的“少碰撞、多超时”。**
保留 Mean-8000 为强简单 Bayesian 对照，不因 Full 更复杂就默认它是 Ours。

建议后续固定现有 checkpoint 做同一合法窗口上的读出干预，记录 posterior、候选价值差和执行 trace，区分：
transition/filter 不足，方差消费改变动作偏好，或原执行接口的共同作用。
读出互换属于训练分布外诊断，不能直接当作公平重训对照；更不能仅凭动作变化宣布不确定性收益。
只有观察到具体原因，才设计一次针对性改动并冻结新的 matched protocol；不从这份报告自动启动 V2。

若要把当前 Mean 正信号升级为论文证据，需要多训练 seeds、完整相同预算基线和预注册 fresh cases。
若要以 Full 的安全能力为贡献，需要预先冻结 safety/progress 目标和可接受代价，再确认优势未被 Mean 或简单安全调节吸收。
本轮未做新参数扫描、新场景筛选、新外部母体搜索或额外训练。

## 11. 历史负结果与工程记录（压缩附录）

| 记录 | 保留的事实与解释 |
|---|---|
| 初始 smoke 中断 | 停在 post-IL 验证，未完成在线阶段；不是科学负结果 |
| full-short smoke | action index 接口导致四条在线 replay 全部插入失败；工程无效，不证明 RL 成功 |
| verified smoke | 8 demonstrations、2 IL epoch、4 RL episode；pre-RL 2/5 成功、3 timeout，final validation 2/2，仅 wiring test |
| 完整训练首次本地启动错误 | TrainConfig/parser 接口错在正式首个 optimizer step 前；修正并重新冻结，不计性能结果 |
| ORCA 收集两次启动阻塞 | TensorBoard 与 sibling env.config 缺失；原日志和 commands 保留 |
| 旧 checkpoint 类型误认风险 | 泛用 rl_model.pth 是 SARL 参数，不能冒充相应时序模型；本轮三臂均严格核验 |
| Mamba 两臂取消 | 用户取消，不是模型失败；不复跑、不纳入正式成绩、不借用未复现论文数字 |
| 3000 的 GRU 超预算 | 停止时约 3814；只用 3000 snapshot 评价及恢复，额外计算不掩盖 |
| 续训非无缝 | replay/RNG/case counter 缺失，统一重建，限制如第 3 节 |
| 本次分析脚本首跑失败 | 本地 editable 包将 import 指向另一副本；在 checkpoint 审核前失败，无结果输出；固定本目录优先及路径断言后全量重算 |

历史 smoke、负结果、协议修正和取消任务的原始文件均保留。正文压缩重复进度文字，不删除原始实验。

## 12. 文件、来源与复算

唯一综合报告就是本文件。下列相对路径均位于本项目，目录根为：

/home/abc/workspace/CrowdNav(20270731_backup2)/CrowdNav

| 内容 | 文件 |
|---|---|
| 时序实现 | crowd_nav/policy/bayes_temporal.py |
| 合法历史、价值及候选消费 | crowd_nav/policy/mamba_rl.py；此名称为旧入口名，本轮只构造 GRU / Bayesian 模块 |
| 原始协议与数据校验 | crowd_nav/runs/bayes-matched-v1/protocol.json、dataset.json |
| 预算变化 | 同目录 budget-3000-amendment.json、budget-8000-amendment.json |
| 调度事件与总体完成状态 | 同目录 budget-8000-events.jsonl、complete-8000-all.json |
| 3000 原始 CSV / 完成记录 | 同目录 {gru,bayes_mean,bayes}-seed42/benchmark-3000.csv、complete-3000.json |
| 8000 原始 CSV / 完成记录 | 同目录 {gru,bayes_mean,bayes}-seed42-to8000/benchmark-8000.csv、complete-8000.json |
| 训练日志 / 续训计算 | 对应目录 train.log、stdout.log、training-8000.json |
| 可复算综合统计 | crowd_nav/runs/bayes-matched-v1/comprehensive-analysis.json |
| 只读分析脚本 | scripts/summarize-bayes-results.py |

数据 SHA256：
8ed0a8d9e6d5676d2887ad5ccd359f2148d51cbfbb82015ba1d1fb32d24208ae

8000 checkpoint SHA256：

- GRU：f4ead5b02a7ba9b49afdd25abaed4bc5fa3e2063f75fab1041d628fe29918fa6
- Mean：41d17f11c8965c9e527decdffaaefc6fcd9da97a91f81bb2ddc8c1e2f935985e
- Full：70ff6a8d34db103a620df36f7a9b3d78c60cb784381e7d7b9dbf107157d82452

在项目根目录复算已有数据（不启动训练或测评）：

```bash
.venv-bayes/bin/python scripts/summarize-bayes-results.py
```

本目录不是 Git worktree，本次没有初始化仓库或上传到其他 Bayesian 项目。

## 13. 2026-10-07 合并优化验收与 10000 次 RL 重跑

状态：工程验收通过，三臂各 10000 次正式 RL 和 3000 回合 benchmark 全部完成；最新统计见开头统一总表。

本轮验收对象是实际合并代码，不是临时 monkey patch。旧 predict 的函数体与
20261007_162833 备份通过 AST 对照；候选构造采用向量化实现，SARL 使用 MC replay。
Bayesian filter、cubature、空间编码、reward、80 动作、T=24 和 MC 目标没有更改。

### 验收结果

- 原有 19 项测试通过，保留本地 numpy/h5py 环境警告。
- 本地及 4090 均完成 parity：GRU、Mean、Full 各 128 个 native 状态，覆盖 5/20 人。
  最终动作不一致均为 0；候选窗口和价值输出最大差均为 0。
- Bayesian parity 覆盖 train/eval、融合 dropout、epsilon exploration、history 与 RNG。
  GRU 仅做 eval parity；其 cuDNN 内部 dropout 状态不能靠公共 RNG snapshot 完整回放，
  不宣称已验证 GRU train-mode bitwise parity。其真实训练路径另由 200 episode smoke 验收。
- Replay 的 legal-prefix/legacy、n-step 1/3、PER 开/关共 8 组对照通过，包含环形覆盖与配对采样。

| Arm | 完成 RL episodes | Optimizer steps | Smoke wall time（含启动/预填充/验证） | IL 重训 |
|---|---:|---:|---:|---|
| GRU | 200 | 800 | 400.34 s | 否 |
| Bayes-Mean | 200 | 800 | 550.46 s | 否 |
| Bayes-Full | 200 | 800 | 590.50 s | 否 |

三臂 checkpoint hash 在本地重新核验，policy、target 和 optimizer moments 有限。
IL 各自严格加载，ORCA replay 预填充均为 5000 条轨迹、skipped=0；没有 IL epoch 循环。
这些是工程 smoke，不是性能 benchmark，也不证明 Bayesian 收益或端到端提速比例。

### 修正与无效尝试

1. 新 MC replay 原先无条件写 priority 且未更新 max_priority；恢复与旧 replay 相同的
   PER 条件与 max 更新。当前正式 SARL 不使用 PER，该修正不算方法贡献。
2. 验收 fixture 首次误读 IL 的 policy key；实际文件使用 value，纠正测试读取方式后严格加载。
3. GRU train-mode RNG 对照受 cuDNN 内部状态干扰；收缩上述 parity 声明，不将其包装成科学失败。
4. 第一次 server smoke 启动目录缺 sibling env.config，在任何 RL 更新前退出。
   启动器补齐配置，并分离 stdout.log 与 native train.log；失败目录、日志和 error.json 保留。

### 正式运行协议

三臂在同一 4090 上并行，各自从原匹配实验的 IL 权重重新开始 10000 次在线 RL。
不复用 smoke 权重，也不从旧 8000 RL 权重续训；不运行 Mamba。
batch=256、每 episode 4 updates、seed=42、gamma/reward/action/history 保持原协议。
每 500 次保存，每 1000 次做 50-episode validation；完成后自动做原六场景各 500 次测评。
测评复用开发测试池，不叫 fresh confirmation；仅一个训练 seed，论文级结论仍不成立。

server 使用独立源码快照及已存在的 IL/data 资产，没有重新下载模型或安装环境。
协议记录 source/data/IL hashes，监督器检测非法加载、replay 跳过与磁盘风险；本地每分钟同步。
本地完整验收、配置、checkpoint、训练及后续测评记录：

/home/abc/workspace/CrowdNav(20270731_backup2)/CrowdNav/crowd_nav/runs/bayes-optimized-20261007/accepted-run

初次无效启动的本地同步记录保留在其上一级目录；server 另保留 optimized-10000 失败目录。
当前进度以各 formal 目录的 state.json、stdout.log 和最终 complete.json 为准。

## 14. 为什么本轮 Full 的成功率低于 Mean：源码与结果分析

日期：2026-10-08。本节以正式源码、冻结配置、训练日志和已经完成的 3000 回合 CSV 为依据。
不修改正式模型，不重训，不把未完成的新诊断当成证据。

### 14.1 先给判断，而不是给一个未经证明的唯一原因

**当前最有依据的解释是：Full 在同一个学习型滤波器上增加了“方差参与非线性价值读出”的路径，
但训练目标没有约束这个方差的独立不确定性含义，也没有直接监督候选动作间的价值排序。
该路径能拟合训练回报，却没有形成稳定的跨场景导航增益。**

这里要区分三种证据强度：

- 已确认的事实：Full 的 SR/CR/TR 更差；训练后期自身 replay 的 value MSE 更低；验证无法区分两臂。
- 已确认的结构薄弱点：方差只有 value MSE 监督，Full 的非线性期望存在曲率与数值近似依赖。
- 尚未确认的因果分配：140 个净失败中，多少来自方差读出、多少来自两臂已经不同的均值/编码权重，
  多少来自训练数据分叉或积分误差。现有正式日志没有逐动作 posterior 和候选分数，不能给出这些比例。

因此，以下不是“已经证明 cubature 导致了 140 次失败”，也不是“Full 永远不如 Mean”。
它是对当前 V1 最关键失配的定位及证据边界。

### 14.2 架构逐层展开

```text
共同输入：24 帧合法历史，每帧 8 x 13 token
  robot token -> Linear(13,64) -> ReLU -> LayerNorm
  5 human tokens + 每人 8 维关系量
      -> Linear(21,64) -> ReLU -> LayerNorm
      -> 4-head self-attention -> max pooling
  robot 64 + human-global 64
      -> Dropout(0.1)
      -> Linear(128,256) -> LayerNorm -> ReLU -> Linear(256,256)
  得到每帧 scene embedding x_t，维度 256
      |
      v
  共同 Gaussian filter，latent 64
      observation = Linear(256,64)(x_t)
      R_t = softplus(Linear(256,64)(x_t)) + epsilon
      A = diagonal sigmoid(retention_logits)
      Q = softplus(process_logits) + epsilon
      mu_prior = A * mu_previous + drift
      P_prior = A^2 * P_previous + Q
      K = P_prior / (P_prior + R_t)
      mu_t = mu_prior + K * (observation - mu_prior)
      P_t = P_prior * R_t / (P_prior + R_t)
      |
      +-- Mean: Linear(64,256) -> SiLU -> Linear(256,256), 输入 mu_t
      |
      +-- Full: 对 mu_t +/- 8 sqrt(P_t[j]) e_j 共 128 个点
                分别过相同 readout，再平均
      |
      v
  Linear(256,1) -> scalar V
  80 候选解析后继 -> R(s,a) + 0.99 V(H + successor)
  共同 clearance filter / risk margin / action smoothing -> 执行
```

源码位置：bayes_temporal.py 第 24-41、59-79、81-94 行；mamba_rl.py 第 35-132、430-471、780-934 行。

关键澄清：

1. Mean 也维护 P，P 会通过 K 改变均值更新；Mean 不是无不确定性的普通网络。
2. Full 不包含更强的时序 transition、额外行人输入或额外历史。两臂注册参数均为 290985。
3. 配置中的 n_layers=4 没有给 Bayesian 模块建立四层滤波器；构造器只接收 d_model 和 latent_dim。
4. 最终 value head 是线性的，因此“先平均 feature 再过 value head”确实等于平均各点的 value。
   这里没有把期望放错层的直接算术错误。
5. 两臂分别训练，空间编码器、mu/P 参数、readout 和 value head 已不同。不能把两臂的 SR 差全部
   解释成在同一已训练网络上打开一次方差开关的影响。

### 14.3 最优先的结构问题：方差的意义没有被训练目标识别

实际更新是 train.py 第 554-585 行：

```text
values_pred = policy.forward_value(states)
loss = MSE(values_pred, Monte Carlo returns)
loss.backward()
```

IL 的主目标也只有 Monte Carlo value regression。没有独立 predictive likelihood、innovation likelihood、
NLL、ELBO 或 calibration 目标；也没有真实 latent 标签。

这意味着 R、Q、P 只需帮助 scalar V 拟合 return，并不需要对“真实人群运动有多不确定”负责。
递归公式具有 Gaussian filtering 形式，不等于方差已被验证为环境中的概率不确定性。

Mean 的方差主要调节“新特征与旧特征如何融合”；Full 还让方差直接改变 nonlinear readout。
Full 因而增加了一个额外的 value 拟合自由度，而不一定增加可靠的导航信息。

更具体地，若在固定 mu 下考虑 Full 的 P 梯度，它受以下项驱动：

```text
d loss / d P_j = 2 (V_full - return) * d V_full / d P_j
```

这个梯度奖励的是减少 scalar value 残差，不是让概率密度匹配观测分布。
方差可能用于补偿 readout 的形状或编码误差。这是源码允许的机制，不是已经测得的方差失准。

**所以当前不能使用的推理是：Full 多用了真实 posterior 信息，因此理论上应更好。**
当前首先缺少“多用的 P 确实对应可验证的不确定性”的证据。

### 14.4 Full 实际增加的是曲率修正，不是自动安全机制

令共享 readout 与最终线性头组成 f。对同一组固定权重：

```text
V_mean = f(mu)
V_full approximately E[f(z)], z ~ N(mu, diag(P))
V_full - V_mean approximately 0.5 sum_j P_j * d^2 f(mu) / d mu_j^2
```

最后一式只是局部二阶解释，不能代替实际数值诊断。
SiLU 加带正负权重的线性投影没有全局固定曲率符号，因此该修正可以提高价值，也可以降低价值。
代码没有“方差越大，必须减分”或 posterior collision-risk constraint。

更重要的是，影响决策的是候选间修正的差，而不是所有候选共同提高了多少：

```text
Delta_a = V_full(a) - V_mean(a)
Q_full(a) - Q_full(b)
  = Q_mean(a) - Q_mean(b) + gamma * (Delta_a - Delta_b)
```

如果 Delta 对所有动作近似相同，排序不变；如果 Delta_a - Delta_b 足以改变小的 action margin，
排序就会改变。改善 scalar MSE 不保证这些排序改变是正确的。

**当前最值得怀疑的是 action-dependent value curvature correction，而不是简单的“Full 太保守”。**
但没有同权重消融和候选分数记录，还不能声称该修正已经被证明是碰撞的直接原因。

### 14.5 128 点 cubature 是另一处可定位风险，不是已坐实的主犯

代码使用 64 维轴向规则：mu +/- sqrt(64) sqrt(P_j) e_j，各权重 1/128。
它不是 128 条实际行人未来，也不是 128 个带独立行为似然的 intent hypotheses。

根据点的定义，可以直接算出它匹配均值和对角二阶矩，但不匹配一般 Gaussian 四阶矩：

```text
单坐标四阶矩：cubature = 64 P_j^2；Gaussian = 3 P_j^2
两个不同坐标的平方乘积：cubature = 0；Gaussian = P_i P_j
```

因此它对非线性 readout 的积分可能出现高阶近似误差。是否重要取决于实际权重、latent 尺度和
激活区间；不能因为某个坐标偏移为 8 个标准差，就把整个规则简单称为“极端未来主导”。
在 64 维空间，规则的半径与单坐标含义不同，投影后的偏移也取决于 readout 权重。

这里还存在一个适合分析的简化：最终 f 是一层 SiLU hidden 加线性输出。
每个 hidden preactivation 都是一个一维 Gaussian，期望值可分解为各 hidden 的一维积分再线性组合。
所以将来核查积分误差，不必训练更大模型或增加 latent modes。
本轮没有执行完成这种积分对照，不把它当作已得到的改善。

### 14.6 候选后继的“不确定性更新”也是代理计算，不是真实新证据

mamba_rl.py 第 876-934 行以当前速度推进行人，以候选机器人动作构造后继 token，
再将这个假想帧放进 history 送入滤波器。

该帧不是新获得的真实观测。不同机器人候选改变 scene feature，也可能改变学到的 R_t 和 P_t。
Full 因而能直接通过“这个候选产生的代理方差”调整其价值；Mean 也受 gain/均值影响，但没有
同样的直接期望读出路径。

这种 surrogate successor-value evaluation 是当前继承框架的定义，没有把假想帧写回真实 history，
也没有使用隐藏未来真值。它不是已经发现的 execution bug。
但是，不能将其解释为正确积分了未来观测分布的 Bayesian belief lookahead。
若要给 P 一个严格的观测概率含义，这个接口也必须有相应语义，目前尚未建立。

### 14.7 正式结果支持“泛化与决策质量失配”，不支持“没收敛”

最后 1000 episode 中保存的 value-loss 日志均值约为：

| 项目 | Mean | Full | 解释 |
|---|---:|---:|---|
| 后期自身 replay 的 value MSE | 0.004215 | 0.003215 | Full 更低，但两臂 replay 不同，不是共同测试集误差 |
| 十次 quick validation 的 SR | 每次 100% | 每次 100% | 验证指标饱和，无法发现正式测试差距 |
| 正式 3000 回合 SR | 91.83% | 87.17% | Full 少成功 140 回合 |
| 原生碰撞次数 | 94 | 147 | 不符合“只因更保守而少成功”的解释 |
| 超时次数 | 151 | 238 | 同时存在安全和通行失败 |

实际 RL/validation 使用 5 人、半径 4 m circle，环境 time_limit=50 s；正式测试包含六场景，
5-20 人，time_limit=25 s。配置虽有 eval_envs 六场景条目，训练期实际调用的是同一个 explorer 的
run_k_episodes(phase='val')；它没有在此处遍历正式六场景。

依据：formal-bayes/config.ini、train.log 的 SOT 与 EVAL 行、train.py 第 2974-3003 行、
crowd_sim/envs/crowd_sim.py 第 270-275 行。两个 Bayesian 臂的训练配置均如此。

**这是已确认的 validation blind spot，不是已确认的 Full 独有病因。**
它解释了为什么训练过程一直显示“正常/100%”，却无法发现 Full 在分布外场景中的退化。
这种差距可以放大不受约束的方差读出风险，但放大程度尚未归因。

### 14.8 140 个净失败具体来自哪里

同一 3000 案例按 Mean -> Full 结果配对：

- 两者都成功：2448。
- Mean 成功、Full 碰撞：105。
- Mean 成功、Full 超时：202。
- Mean 失败、Full 成功：167，其中 Mean 碰撞 56、Mean 超时 111。
- 所以 Full 丢掉 307 个 Mean 的成功，也救回 167 个 Mean 的失败，净少 140。

| 场景 | Full 相对 Mean 的净成功变化 | 主要失分结构 |
|---|---:|---|
| 基础圆形 | -8 | 已见任务类型也有少量额外碰撞/超时 |
| 基础方形 | -48 | Mean 成功的 55 个案例在 Full 下超时，另 11 个碰撞 |
| 密集圆形 | -48 | Mean 成功的 34 个案例超时，另 16 个碰撞 |
| 密集方形 | -65 | Mean 成功的 77 个案例超时，另 41 个碰撞；Full 同时有 53 个 rescue |
| 大范围圆形 | +58 | Full 救回 Mean 的 61 个超时和 3 个碰撞，不能说 Full 在所有场景都差 |
| 大范围方形 | -29 | 额外失败与 rescue 同时存在，净效果仍负 |

行为统计也不支持简单的“Full 更抖”解释：

- 全部回合平均 action-switch frequency：Full 2.180/s，Mean 2.444/s。
- 平均 turn-reversal frequency：Full 1.018/s，Mean 1.196/s。
- 超时回合中 stalled-window fraction：Full 47.77%，Mean 42.24%。
- 超时回合中 oscillatory-stall fraction：Full 32.59%，Mean 38.57%。

Full 的超时更像包含更多无有效进展的窗口，而不能统一叫作更严重的来回抖动。
在 2448 个共同成功案例上，Full 平均快约 0.80 s、路径短约 0.31 m，但最小 clearance 平均小约
0.013 m。它不是所有动作都慢，而是跨案例的失败选择更差。
这些是行为关联，不能从 aggregate 推断某一 root 的具体碰撞原因。

### 14.9 可以排除的流行解释，以及尚不能排除的解释

| 解释 | 当前判断 | 理由 |
|---|---|---|
| Full 没训练完、更新次数少 | 排除 | 两臂都是 10000 episode / 40000 optimizer updates |
| Full 参数更多，所以输 | 排除 | 两臂参数数量相同；计算更多不等于参数更多 |
| Full 只是太保守 | 排除作为总解释 | Full 碰撞与超时均更多；共同成功案例反而更快 |
| Full 更抖所以必然差 | 不支持作为总解释 | Full 的全体 switch/reversal 和超时 oscillatory-stall 均更低 |
| Full 的 padding 或向量化单独出错 | 当前无证据 | 合法历史契约共享，合并前后既有 parity 通过；不等于排除所有潜在 bug |
| Gaussian 方差学成了 value-fitting 自由度 | 机制上有依据，未量化因果 | 只有 value MSE，Full 多一条直接方差读出路径 |
| cubature 误差改变关键排序 | 未验证 | 数学近似风险存在，但没有完成同权重积分对照 |
| Full 已训练的 mu/空间编码本身更差 | 未排除 | 两臂独立 IL/RL，不能从不同模型的结果分离这个因素 |
| 高密度/方形泛化失配 | 有强关联证据 | 场景损失集中，validation 不覆盖正式分布；不是唯一原因证明 |
| Bayesian 方法总体无效 | 不成立 | Mean 仍优于 GRU 的 SR，且 Full 在大圆形有明显正差；这里只能否定当前收益主张 |

Top-5 截断、scene-first aggregation、对角 Gaussian、简单 retention transition、训练/测试 smoothing
差异等都值得记录，但大多为两臂共有。不能把它们直接指定为 Full 输 Mean 的独有原因。

### 14.10 架构优化的分析建议，不在本轮实施

1. 以 Mean 作为当前干净参照。它保留递归均值/方差滤波，去掉 128 点期望读出，已有更好的 SR 证据。
   但 Mean 的 collision 比 GRU 略高，不能宣称其安全也全面胜出。
2. 不加 mixture、world model、双记忆或新 safety penalty 来解释当前负结果。首先定义 P 究竟应该
   表示哪一种可验证误差；若没有定义，Full 只是对任意 latent 尺度进行价值平均。
3. 未来若获准实验，优先区分“Full 学到的整个表示不好”和“Full 的期望消费不好”：固定同一 checkpoint，
   再做 consumer-only 对照。直接关掉方差也可能使依赖方差训练的网络失配，不能仅凭该消融判因。
4. 再单独核查数值积分，不改变 filter、reward 或 action space；当前单层 SiLU readout 可以利用
   一维 Gaussian marginal 分解，没必要先增加模型复杂度。
5. 若后续需要重新训练，先让 validation 真正覆盖冻结的六场景与 25 s 限制，否则新的“100% validation”
   仍可能只代表容易的 5 人 circle。这是评估可信度问题，不是给 Full 调参找赢家。

**最终结论：当前 Full 多了一条未被概率学习目标约束的非线性方差消费路径，并在有限训练分布下
没有表现出稳定泛化能力；这是最有根据的架构诊断。尚不能把具体失败全部归给 sigma 过大、积分错误
或某个参数。没有足够证据时，继续补结构不是合理的“修复”。**

### 本轮操作边界说明

在用户再次明确“只分析”前，误新增一个诊断脚本并启动小规模复跑。已中断该进程并删除新增脚本。
首次入口错误、第二次序列化错误及第三次未完成的输出均保留在 runs/bayes-optimized-20261007 下的
full-mean-diagnostic-20261008、对应 -r2/-r3 目录，不纳入本节科学结论。
正式 bayes_temporal.py、mamba_rl.py、train.py、ppo_buffer.py 的 SHA256 与已完成训练 protocol 一致；
未修改正式算法、未启动新训练。只补充本综合报告。

## 15. 2026-10-08 posterior-consumption repair night

本节记录用户随后明确授权的修复和训练任务，替代上一节的“本轮只分析”操作边界。
工作预算：03:27:10--11:27:10 KST。只改变 bayes_temporal.py 的 posterior readout；
不运行 Mamba backbone，不修改 KDA，不改变 reward、状态、动作、T、LR、depth 或安全配置。
当前为中间记录，正式训练和新 benchmark 结果尚未产生，不能写成方法收益。

### 15.1 B0: 首轮不通过，调查后原生复核通过

| 数据口径 | 真实状态数 | 常数偏置占比 | 状态残差/mean 特征 | 余弦中位 | 冻结判据 |
|---|---:|---:|---:|---:|---|
| 512 条 ORCA 轨迹各自首 24 帧 | 12,288 | 99.507% | 2.230% | 0.99767 | 失败：残差 >2% |
| 原 B1 已预注册的 192 条 native Full rollout | 10,696 | 99.410% | 1.555% | 0.99783 | 通过 |

首轮失败数据和判据未修改。原生复核另行冻结 follow-up protocol，使用整个已注册 B1 cohort，
没有按收益或结果过滤；192 条重放的 outcome 和 steps 全部与旧 benchmark 一致。
ORCA 初始帧 sigma 中位 0.783，13--24 帧约 0.536；偏置模长从 0.369 降至 0.178。
这支持“首 24 帧过度代表滤波启动段”的采样解释，但不能声称精确重现用户原来那组 15,680 状态。
进一步对同一份 native 数据按 24 种 history 长度等权重算，残差变为 2.288%，接近 ORCA 的 2.230%。
native 的完整 24 帧窗口占 59.21%，ORCA 首窗口 cohort 中仅占 4.17%；这实证支持启动段权重解释。
等权分析是保留全部状态的事后差异诊断，不是另换通过标准；B0 的通过只适用于记录的 native 频率口径。
原生复核的 cosine p5 为 0.9553，也说明存在非恒定尾部；“99.4% 主导偏置”不是“所有输入严格常数”。
线性 head 能吸收真正恒定的分量，不能精确消掉剩余状态相关残差。
以上 B0 数字均为本轮已验证；旧 F3 积分误差结论没有重新运行，不以积分错误作为修复理由。

### 15.2 B1: sigma 有风险相关性，但独立信息价值未成立

训练/测试按 episode 划分：144/48，不随机拆 frame；以下是 held-out AUC，区间按 episode bootstrap。

| 事后危险标签 | sigma 均值 | sigma 最大值 | 64 维线性探针 | 当前净距简单对照 |
|---|---|---|---|---|
| 未来 4 步净距 <0.2 | 0.724 [0.660,0.784] | 0.608 [0.524,0.677] | 0.864 [0.796,0.908] | 0.860 [0.826,0.894] |
| 未来 8 步净距 <0.2 | 0.711 [0.640,0.771] | 0.604 [0.521,0.678] | 0.828 [0.742,0.886] | 0.817 [0.776,0.858] |
| 未来 4/8 步几何 contact | 未能估计 | 未能估计 | 未能估计 | 测试折没有正例 |
| episode 最终 native collision | 0.726 | 0.725 | 0.936 | 0.795 |

**已验证：** sigma 与近距离事件存在 held-out 预测相关性，不能把它归为完全无状态信息。
**未验证：** sigma 是否增加当前几何/历史之外的独立信息；它是否是校准后验；Full 是否有闭环增益。
collision probe 的测试折只有一个碰撞 episode，重复 frame 不构成独立碰撞证据；0.936 不作可靠泛化主张。
当前净距已经几乎达到 sigma 线性探针的 AUC，因此不能仅从 B1 宣布“Bayesian 信息已经足够”。

### 15.3 B3/B4: 改动和验收

新结构：原 Gaussian filter -> concat(mean, logvariance) -> 128--256--256 readout -> 原 256--1 head。
Mean 臂的 64 个 logvariance 通道精确置零，Full 保留；两臂 state_dict key/shape 及初始化相同。
删除新模型内的 128 点积分 buffer 和调用；旧 encoder 源码、checkpoint 和结果全部保留。
这称为显式 posterior feature embedding，不再称为 cubature expected value。

| 自检 | 结果 |
|---|---|
| Mean 改 variance 后输出逐位相同 | 通过 |
| Full 改 variance 后输出变化 | 通过；初始化 fixture 最大变化 0.4456，不是性能指标 |
| Mean/Full state_dict key、shape 和同 seed 初始化一致 | 通过 |
| 普通 forward 和 legal-prefix 混合长度分组、forward_last、反向传播 | 通过 |
| 三臂 value_head | 均为 weight [1,256]，257 参数 |
| 旧训练权重的 mean readout 等价映射 | 通过；旧权重填入前 64 个 mean 列，variance 列为 0，最大输出误差 0 |
| 本地与 4090 单元测试 | 各 18 通过，排除 Mamba CUDA backbone 测试 |
| GRID | 显式初始化并硬断言 80，5x16/exponential/v_min=0.05/no stop |
| 原生六场景评测入口 | 通过；新 Full 的 ep200 权重实际完成六行 CSV，3 success / 1 collision / 2 timeout；仅入口验收，不是性能估计 |

新 Bayes 读出参数形状改变，因此不能 strict 加载旧 Bayes IL；没有静默 partial load。
三臂统一使用已恢复的 matched-native-orca-v1 数据重做 50 epoch IL，按原生 best-IL 规则选权重。
正式 RL 将复用本轮匹配 IL，不从 200-episode smoke RL 权重继续训练。
数据 SHA256：8ed0a8d9e6d5676d2887ad5ccd359f2148d51cbfbb82015ba1d1fb32d24208ae。
新 encoder SHA256：7740d10e654c961cceb4248ff7ca4a8401dee8ae1c5d327592aa4f4d6f27c2ac。

### 15.4 运行队列和尚未完成的证据

4090 独立目录为 /root/bayes-fix-20261008；数据单独放在 /root/bayes-assets。
本轮不下载新依赖或 checkpoint。原始 seed43 模板的显存上限为 0.8，生成配置统一改为 0.15；
移除 backbone/seed 和已统一资源项后，六臂科学配置完全一致。
原队列：seed42 三臂 IL -> 三臂 200 RL smoke -> seed43 三臂 IL -> 六进程各 10000 RL。
04:17:42 KST 在新 smoke/正式导航结果出现前冻结 il-reuse-amendment：Bayes-Mean 首轮 IL 实测 394.85 s，
顺序再做第二个 50-epoch IL block 会超出启动预算；本地 15 GiB RAM、可用 8.4 GiB、swap 已用 3.9 GiB，
也不适合承担三臂 IL。修订队列保留三臂本轮完整 50-epoch seed42 IL 与 200 RL smoke，
验收后复用于正式 RL seed42/43，各 10000 episode，然后各 3000-case benchmark。
若旧控制器已经启动 seed43 IL 预处理，该部分作为预算中止的工程工作保留，不作为方法负结果。
**准确范围是两个配对在线 RL seed、每臂共用一个新 IL 初始化；不是两个独立完整 IL+RL 初始化 seed。**
没有减少三臂已冻结的 IL 50 轮、正式 RL 10000 回合或 smoke 200 回合预算，也不复用 smoke RL 权重。
所有 smoke 未通过时不会放行正式队列；错误/partial load/GRID 异常/数据错误/磁盘或内存风险触发自身 worker 停止，
不会终止其他 Codex 的进程。PID 使用 /proc starttime 检查，自身存活和不存在 PID 两种状态均检查。

| 项目 | 当前证据状态 |
|---|---|
| B2 oracle 排序 SR headroom | 30 分钟 CPU 确定性重放完成；预注册 12 个失败 root，8 个完整、第 9 个部分、3 个未测，找到 5 个无接触救回；详见下文，不能称总体 SR 上界 |
| B5 三臂 200 RL smoke SR/CR/TO | 已验证：三臂全部通过，各 800 optimizer steps；计数见 15.7 |
| B6 六个 10000 RL 正式进程 | 已启动：09:50:40 KST；三臂 x seed42/43，全部从各臂新 IL seed42 初始化 |
| B7 正式进程健康确认 | 已验证：持续观察超过 20 分钟，六臂 ep500 权重 strict/finite/optimizer step2000 全部通过 |
| 三臂两 RL seed 的正式性能 | 未验证；旧单 seed 数字不可代替新结果，IL 初始化随机性尚未重复 |

本地原始证据：crowd_nav/runs/bayes-fix-20261008 下的 protocol、b0、b0-followup-protocol、
b0-native、b1、对应 npz 和 self-tests；remote-run 自动镜像本轮远端日志、配置和 checkpoint。
唯一下一优先级是完成匹配训练，检验 Full vs Mean，而不是立即加入 KL 或排序监督。
B1 暂不支持“sigma 完全没学出来所以必须加 KL”；KL 本身也不自动定义导航危险度。
若新 Full 没有独立收益，再区分方差学习与方差消费，不从当前 AUC 推断论文结论。
即便新 Full 优于 Mean，仍需后续固定/打乱方差剖面的机制消融；仅比较显式方差与零通道，
不能自动排除固定剖面或优化路径差异。3000-case benchmark 也不能替代更多独立训练初始化。

### 15.5 B2: 单次首动作改选的局部空间

本项在查看新结果前注册 12 个旧 Full 失败 root，固定首个 24-frame 窗口附近的一个控制时刻。
每个候选均从原 seed reset 后重放到同一 root；不 deepcopy RVO2，不变更动作集合、
原一步安全 mask、0.3 action smoothing 或后续原 Full 策略。原动作重放必须恢复同一 root hash、
执行动作、结局和步数；全部已测试候选通过此执行一致性检查。
30 分钟固定预算结束时，前 8 个 root 完成全部允许候选，第 9 个完成 61/80，最后 3 个未运行。
状态为 BUDGET_LIMITED_PARTIAL，不把未测 root 记为无 headroom。

| root | 原结局 | 测试候选/允许候选 | 无接触救回 | 最佳救回步数 | 最小净距 m |
|---|---|---|---|---|---|
| baseline-circle / case21 / t20 | collision | 80/80 | 否 | -- | -- |
| baseline-square / case0 / t23 | timeout | 77/77 | 否 | -- | -- |
| baseline-square / case2 / t23 | timeout | 80/80 | 是 | 83 | 0.1935 |
| baseline-square / case4 / t23 | timeout | 2/2 | 否 | -- | -- |
| baseline-square / case10 / t23 | timeout | 80/80 | 是 | 61 | 0.0708 |
| baseline-square / case12 / t23 | timeout | 76/76 | 是 | 52 | 0.2049 |
| baseline-square / case20 / t23 | timeout | 80/80 | 否 | -- | -- |
| dense-circle / case3 / t23 | timeout | 77/77 | 是 | 64 | 0.1839 |
| dense-circle / case7 / t23 | timeout | 61/80，部分 | 是 | 55 | 0.1652 |

已验证：已测试的 9 个失败 root 中，5 个可通过一次合法改选、随后恢复原策略无接触到达。
若仅把这 5 个已找到的 known-future 干预放回原 192-case cohort，其余案例保持原轨迹，
对应描述性 +2.604pp；这是离线局部可达改善，不是总体“完美排序 SR 上界”。
无接触也不等于维持 0.2m discomfort margin，表内多个救回轨迹仍进入 discomfort 区域。
未验证：可部署排序器能否预测这些救回动作；方差是否提供所需信息；是否在其他时刻有更大空间。
本结果不支持现在同时加入排序监督，更不改变正式 matched 训练方案。
原始证据：b2-protocol.json、b2-trials.jsonl、b2-progress.json、b2.json；保留所有失败候选。

### 15.6 调度工程事件，非科学结果

06:55 KST 检查发现本地 handoff 监控已因一次 SSH handshake 断连退出；
远端三臂 IL 没有被中断，双分钟 rsync 镜像仍然正常，正式或 smoke 结果当时尚未出现。
保留原 handoff.log，加入只读状态请求最多四次重试后重新启动监控。
fixture 已确认只读请求失败可重试，而 retire/launch 等写操作不重试，避免丢失回复后重复启动。
这是调度可靠性修复，不修改模型、随机流或科学协议，不计为任何臂的训练失败。

### 15.7 B5: smoke 实际结果

| 臂 | seed | RL episode | success / collision / timeout | SR / CR / TR % | optimizer steps | 启动至结束 s | 验收 |
|---|---|---|---|---|---|---|---|
| GRU | 42 | 200 | 171 / 25 / 4 | 85.5 / 12.5 / 2.0 | 800 | 405.4 | PASS |
| Mean | 42 | 200 | 151 / 46 / 3 | 75.5 / 23.0 / 1.5 | 800 | 600.6 | PASS |
| Full | 42 | 200 | 158 / 33 / 9 | 79.0 / 16.5 / 4.5 | 800 | 630.6 | PASS |

已验证：这些是 trajectory.log 中完整 200 回合的训练结局，不是滚动窗口或小验证集峰值。
三臂均 strict 加载匹配的新 IL；无在线 IL 回退；GRID locked/Built 均为 80；
数据版本 matched-native-orca-v1；新 checkpoint 参数有限且 optimizer steps=800；无 traceback。
未验证：这些高探索碰撞率能否在正式训练后消退；Full 是否有独立净收益。
smoke 带 epsilon-greedy，不能等同确定性 3000-case benchmark；不因 Full 暂高于 Mean 宣布成功。
checkpoint-check、smoke-results.json、各臂 result.json 与完整 trajectory.log 均保留。

### 15.8 B6/B7: 正式运行与持续验收

启动：2026-10-08 09:50:40 KST；六进程在 4090 同时运行，各 train_episodes=10000。
本轮实际完成的新 IL 均为 50 轮；原生 best-IL 选择的 GRU/Mean/Full epoch 分别为 5/50/40。
复用 IL SHA 分别为：
GRU c9b32750bef715095056f7ae615bb350fef7cb4563d4057623d011004f7ceeee；
Mean f652414c8ec089f28673a18511f6a6617a9528d73ac3feabdf4f1602b5777f4e；
Full 86fef5260a4d2115d6cd4960c56d7bfacd7b315695f609772116eaa61079af16。
未加载 smoke 的 RL 权重或 optimizer；在线 RL seed42/43 从对应同一个新 IL checkpoint 启动。
旧控制器已封存，其刚开始准备的三个 seed43 IL 进程精确识别后停止，原日志保留，
归类 BUDGET_ABORTED_UNUSED_SECOND_IL，不是科学负结果；未停止 KDA 或其他项目进程。

10:16:50 KST 的已验证运行快照：

| 臂 | RL seed | PID | 末次 RL 日志 ep / 已记录结局数 | 已记录 success / collision / timeout | ep500 验收 |
|---|---|---|---|---|---|
| GRU | 42 | 2231388 | 902 / 903 | 795 / 97 / 11 | strict、finite、step2000 |
| GRU | 43 | 2231390 | 852 / 853 | 736 / 97 / 20 | strict、finite、step2000 |
| Mean | 42 | 2231392 | 588 / 589 | 461 / 116 / 12 | strict、finite、step2000 |
| Mean | 43 | 2231393 | 592 / 593 | 458 / 121 / 14 | strict、finite、step2000 |
| Full | 42 | 2231394 | 576 / 577 | 479 / 88 / 10 | strict、finite、step2000 |
| Full | 43 | 2231391 | 594 / 595 | 499 / 92 / 4 | strict、finite、step2000 |

活跃日志/trajectory 分别读取，相差一个回合是非原子快照；完整训练结束时会严格检查恰好 10000 条结局。
当前各臂训练回合数不同且仍探索，表内数字不构成 matched-budget 性能比较。
六个 PID 均按 /proc starttime 验证身份；正确 seed、algo=sarl、legal-prefix、GRID=80、
新 IL strict loading、offline 数据版本均已核实；未发现权重加载或离线数据回退、重训练 IL 或 traceback。
原策略在无安全候选等情况下的决策 fallback 没有新增逐次计数，不能声称该类 fallback 全为 0；其逻辑保持不变。
每臂 ep500 的 optimizer 都已实际完成 2000 次更新，严格恢复到正确新架构，全部参数有限。
Full 两 seed 的 readout 第一层方差列 Adam exp_avg 最大绝对值分别为 0.0009103 / 0.0006431；
Mean 两 seed 为精确 0。AdamW 的 weight decay 可使零梯度列权重微动，故不拿权重差代替通道梯度证据。
已验证：显式方差通道进入 Full 优化；未验证：它表达的是校准后验，或能改善最终导航。

快照磁盘剩余 6.79 GiB，可用内存约 23.5 GiB；早前六进程 GPU 总占用约 6.3 GiB。
控制器每 15 秒检查自身进程/日志和资源；磁盘 <1 GiB、可用内存 <2 GiB、traceback 等会工程停止并保存原因。
两分钟 rsync 镜像仍在运行；没有下载新依赖，也没有改 KDA。训练可能长于本轮八小时，按预期留在后台。
源代码中继承的 Mamba/DoubleQ/IQL 日志标签未清理；实际 checkpoint stage 为 rl_training_sarl，
backbone 仅 gru/bayes_mean/bayes，没有构造或训练 Mamba 模块。
远端日志使用 CST/UTC+8，本报告统一将 UNIX 时间转换成 KST/UTC+9，不能直接混用日志小时。

自动后续：六个 10000-episode checkpoint 全部通过验收后，逐臂执行原六场景各 500 回合，
共六份 3000-case CSV；保留所有成功、碰撞、超时与独立 geometric contact，不用 192-case 替代。
完成后生成 training-results.json、benchmark-result.json、complete.json；工程失败写 error.json。
并行 GPU 计时含共享负载，不自动当作独占部署延迟优势。
原始证据还包括 formal-version.json、controller-retirement.json、startup-health-final.json、
unit-tests-local.xml、remote-run/unit-tests-remote.xml、b4-eval-entry.json/csv 与完整运行镜像。

### 15.9 B8: 下一轮优先级与结论边界

已验证：旧积分有主导恒定分量；显式 logvariance 替换通过接口/梯度/IL/RL smoke 验收；
三臂 x 两在线 RL seed 已正式运行，且不是只创建空进程或只跑 smoke。
已确认未通过的检查：最初 ORCA-prefix B0 的 2.230% 残差超出 2% 阈值，原数据保留且已解释采样差异。
未验证：Full vs Mean 最终净收益、方差的独立因果价值、概率校准、完整训练初始化稳定性。
下一轮先完成这套冻结训练和 3000-case 评测，本轮暂不加 KL、预测监督或排序监督。
B1 并非 AUC 近 0.5；但 geometry 对照接近其效果，不能直接判为“信息充足，只缺消费”。
B2 有局部动作空间，但没有可部署排序器或 Bayesian-specific 归因，不足以优先上排序监督。
若 Full 未获独立收益，再基于固定/打乱方差剖面消融区分学习不足与消费不足；
不因 smoke 正负或某个单 seed 结果自动更改冻结模型。

## 16. 历史快照：11:31 时已跑完、在跑、尚未跑

更新时间口径：2026-10-08 11:31:12 KST 的远端只读快照。
本节是历史进度记录，已由第 17 节最终结果更新；下面的“在跑/未完成”不代表当前状态。
本次只整理已有数据和运行状态，没有修改模型、配置、训练预算或评测协议，也没有重启任何训练。
第 15 节的 10:16 快照保留为启动验收证据，不与下面的新进度混用。

### 16.1 仍在运行的正式训练

| 臂 | 在线 RL seed | PID | 已记录训练结局 / 10000 | success / collision / timeout 计数 | 最新已保存 checkpoint | 运行状态 |
|---|---|---|---|---|---|---|
| GRU | 42 | 2231388 | 3440 / 10000 | 3255 / 143 / 42 | ep3000 | 身份核实，运行中 |
| GRU | 43 | 2231390 | 3509 / 10000 | 3284 / 191 / 34 | ep3500 | 身份核实，运行中 |
| Mean | 42 | 2231392 | 2203 / 10000 | 2014 / 156 / 33 | ep2000 | 身份核实，运行中 |
| Mean | 43 | 2231393 | 2216 / 10000 | 2014 / 162 / 40 | ep2000 | 身份核实，运行中 |
| Full | 42 | 2231394 | 2247 / 10000 | 2074 / 158 / 15 | ep2000 | 身份核实，运行中 |
| Full | 43 | 2231391 | 2282 / 10000 | 2115 / 159 / 8 | ep2000 | 身份核实，运行中 |

这些计数来自各臂 trajectory.log 的训练结局，包含失败，不是 evaluation CSV。
GRU 已跑到更高预算，不能用当前累计比例比较三臂优劣；探索率、策略和在线采样分布也随训练变化。
快照中六个进程的 /proc 身份均匹配，GRID=80、offline 数据版本、strict IL loading 均确认；
六份 result.json 尚未生成，training-results.json / complete.json 不存在，controller 仍为 FORMAL。
没有 error.json，六份 stdout 均未发现 traceback。这里只列最新 checkpoint 文件存在；
此前明确做过的 strict/finite/optimizer 验收是六臂 ep500，不能把文件存在冒充重新验收所有中途权重。

资源快照：磁盘余量 6.32 GiB，可用内存约 23.74 GiB，GPU 总占用约 8.47 GiB、利用率 92%。
自动镜像截至 11:29:56 KST 仍正常，新的远端快照也已同步本地。
资源和工程保护仍由原控制器运行；此次整理不取消保护、不改队列。

### 16.2 已完成的同预算小 validation

六臂都已完成 ep2000 时的原生 validation：epsilon=0、k=50。
它与第 15 节的 200 回合探索 smoke 是两种不同记录，更不是完整六场景 benchmark。

| 臂 | seed | RL budget | validation 数 | 日志 SR / CR / TR % | 日志 nav s | 日志平均路径 m |
|---|---|---|---|---|---|---|
| GRU | 42 | 2000 | 50 | 100 / 0 / 0 | 15.54 | 10.08 |
| GRU | 43 | 2000 | 50 | 100 / 0 / 0 | 14.34 | 9.61 |
| Mean | 42 | 2000 | 50 | 100 / 0 / 0 | 15.42 | 9.53 |
| Mean | 43 | 2000 | 50 | 100 / 0 / 0 | 16.85 | 10.52 |
| Full | 42 | 2000 | 50 | 100 / 0 / 0 | 14.56 | 10.29 |
| Full | 43 | 2000 | 50 | 100 / 0 / 0 | 15.08 | 9.72 |

已验证：小 validation 确实执行完成，模型不是始终无法导航。
未验证：六场景泛化、安全净收益或 Full 的独立机制价值。全为 100% 也意味着该小验证的 SR 不能区分三臂。
这里的 CR 是原生日志口径；没有据此断言独立几何 contact 为零。
nav/path 的差异只作过程描述，不单独宣称方法提升，不据此选模型、调参数或提前停止训练。
来源：各 formal 目录 formal.stdout.log 中 ep=2000 的 EVAL / EVAL-EXTENDED。

### 16.3 没跑完和明确没做的内容

| 项目 | 状态 | 原因 / 后续处理 |
|---|---|---|
| 六臂各 10000 RL | 正在运行，均未完成 | 继续原冻结预算，不用中途比例作最终判断 |
| 六臂各 3000-case benchmark | 尚未开始 | 队列等待所有训练验收完成；每臂 6 场景各 500，保留失败 |
| 最终 SR/CR/TR、Time、discomfort、minimum separation | 修复版尚无完整结果 | 必须由新 benchmark 计算，不能套修复前数据 |
| Full vs Mean、Mean vs GRU 的两 RL seed 结论 | 尚未完成 | 对齐完整预算后分别报告各 seed 与合并描述，不把 episode 当独立训练 seed |
| B2 剩余 3 root 和第 9 root 剩余候选 | 未运行，预算裁剪 | 原 partial 结果保留；没有安排自动补跑，不记成负结果 |
| 独立 seed43 新 IL | 未完成，已预算中止 | 两个 RL seed 共用各臂 IL seed42；不再排队，不冒充两组完整初始化 |
| 方差的 calibration / likelihood / KL 监督 | 本轮未做 | 按本轮限制没有加辅助损失；AUC 或非零梯度不等于校准后验 |
| 固定/打乱方差剖面消融 | 未做，属后续机制验证 | 即便 Full 赢 Mean，也还不能排除固定剖面、初始化或优化路径解释 |
| 排序监督 | 本轮未做 | B2 仅知未来的局部诊断，不足以支持同时改训练目标 |
| fresh confirmation、更多独立 IL+RL seeds | 本轮未做 | 当前是共享 IL 的两在线 RL seed 对照，不是论文级最终确认 |

### 16.4 已发现的问题与不能掩盖的限制

1. 最初 ORCA-prefix B0 未过残差阈值；已保留失败和采样解释，没有删除或改成通过。
2. B1 的风险关联几乎被当前净距解释，不能只报高 AUC 宣称 Bayesian 信息有独立价值；collision 留出折只有一个碰撞 episode。
3. B2 是失败筛选 cohort、已知未来、单次改选，且部分完成；5 个无接触救回也不全部满足 0.2m discomfort margin。
4. 新结构保证方差能输入 readout，实际优化动量也非零；但近固定方差剖面仍可能被当作偏置使用，尚未完成训练后因果消融。
5. 三臂 smoke 有碰撞和超时，小 validation 全成功，两者不矛盾：前者带探索，后者小且口径不同；不能选好看的数字代替正式评测。
6. 只有两在线 RL seed、共用 IL 初始化，容量也不与 GRU 完全相等；结论范围必须限制，不能直接宣布论文方法成立。
7. 本地 handoff 曾因 SSH 断连退出，已修复只读重试；seed43 IL 的预算中止保留为工程记录，没有影响正式队列。
8. 原策略 fallback 逐次计数、独立校准和独占部署延迟尚不完整；未记录的指标不填 0、不推断为无问题。

### 16.5 报告依据与唯一后续

最新进度证据：crowd_nav/runs/bayes-fix-20261008/remote-run/report-progress-snapshot.json。
已完成修复证据：同实验目录中的 b0.json、b0-native.json、b1.json、b2.json、self-tests.json、
smoke-results.json、startup-health-final.json、IL result.json、单测 XML 和完整原始日志。
运行中的 checkpoint、配置、训练曲线和 trajectory 均保存在 remote-run/runs/formal-*，镜像继续更新。

唯一已排队后续仍是：完成六份冻结的 10000 RL -> 逐份验收 -> 六份 3000-case benchmark -> 汇总新结果。
此次没有新算法开发或复跑历史实验。最终 Full 是否优于 Mean，当前仍是未验证问题。

## 17. 最终完成结果与分析：显式方差修复版

### 17.1 完成验收与实验口径

本节分析来自六份最终 CSV，不是启动快照、smoke 或小 validation。独立分析脚本只读结果，不加载模型、不启动训练。
六组均完成 10000 次在线 RL、40000 次优化；六份 benchmark 各 3000 行，完整保留成功、碰撞、超时。
脚本验证所有案例唯一、数值有限、六场景各 500、六模型的 scenario/episode/execution-seed 对齐，且计数与各组 benchmark-result.json、全局 complete.json 一致。
global complete.json 为 COMPLETE，没有 error.json；162 份冻结源码和 13 份冻结配置 SHA 与本地镜像全部匹配。

保持原空间编码、80 动作、T=24、奖励、解析后继状态、value head、ORCA 数据和 MC-return RL。
新复杂结构为 64 维高斯滤波 -> concat(mu, logvariance) -> 256 维特征 -> 原标量 value head。
简单结构相同参数形状，但显式 logvariance 通道置零；滤波内部仍使用方差。
两个在线 RL seed 共享各臂 IL seed42。使用原生 best-IL checkpoint，未把 smoke 的 RL 权重用于正式初始化。
测试为六个已有原生场景各 500 个案例、25 秒上限；这不是全新测试池或两组完整独立 IL+RL 初始化。

工程上的“方差现在能够影响输出”已经通过；科学上的“模型利用了有价值的状态相关后验方差”仍未归因验证。

### 17.2 逐场景原始成败：不只看总均值

各单元依次为成功 / 原生碰撞 / 超时 / 独立几何接触回合数。每个场景 500 回合，前三项和为 500；接触不额外加进这个总数。

| RL seed | 原生场景 | GRU | 贝叶斯简单结构 | 贝叶斯复杂结构（修复版） |
|---:|---|---|---|---|
| 42 | 基础圆形 | 500 / 0 / 0 / 0 | 466 / 0 / 34 / 0 | 499 / 0 / 1 / 0 |
| 42 | 基础方形 | 455 / 6 / 39 / 5 | 457 / 9 / 34 / 7 | 452 / 12 / 36 / 11 |
| 42 | 密集圆形 | 492 / 3 / 5 / 2 | 469 / 1 / 30 / 1 | 494 / 3 / 3 / 3 |
| 42 | 密集方形 | 328 / 37 / 135 / 34 | 284 / 36 / 180 / 31 | 334 / 28 / 138 / 21 |
| 42 | 大范围圆形 | 476 / 4 / 20 / 2 | 483 / 1 / 16 / 1 | 491 / 2 / 7 / 1 |
| 42 | 大范围方形 | 433 / 34 / 33 / 30 | 341 / 38 / 121 / 35 | 423 / 48 / 29 / 43 |
| 43 | 基础圆形 | 492 / 5 / 3 / 3 | 497 / 1 / 2 / 1 | 498 / 0 / 2 / 0 |
| 43 | 基础方形 | 474 / 14 / 12 / 14 | 425 / 10 / 65 / 9 | 409 / 8 / 83 / 8 |
| 43 | 密集圆形 | 405 / 91 / 4 / 87 | 471 / 5 / 24 / 4 | 493 / 0 / 7 / 0 |
| 43 | 密集方形 | 345 / 64 / 91 / 55 | 224 / 35 / 241 / 31 | 216 / 18 / 266 / 15 |
| 43 | 大范围圆形 | 482 / 16 / 2 / 14 | 402 / 3 / 95 / 3 | 477 / 0 / 23 / 0 |
| 43 | 大范围方形 | 432 / 51 / 17 / 42 | 377 / 40 / 83 / 35 | 338 / 33 / 129 / 28 |

已验证的局部行为问题：

- 复杂结构 seed43 的密集方形有 266/500 超时，SR 43.20%，比 GRU 的 69.00% 少 25.80 个百分点。它减少接触，但通行失败非常明显。
- 复杂结构 seed43 的基础/大范围方形 SR 分别为 81.80% / 67.60%，均低于简单结构和 GRU；圆形上的优势不能掩盖这一点。
- 复杂结构 seed42 大范围方形 CR 9.60%，高于简单结构的 7.60% 和 GRU 的 6.80%；不是“修复后安全全面改善”。
- GRU 也不稳定：seed43 密集圆形原生碰撞 91/500、几何接触 87/500；其总 SR 较高不表示安全性最好。
- 复杂相对简单的 SR：seed42 在 5/6 场景更高、1/6 更低；seed43 在 3/6 更高、3/6 更低。总 SR 正差并非各场景一致改善。

### 17.3 配对差值和不确定性

以下差值均为左侧模型减右侧模型，单位百分点。SR 正值更好，CR/TR/接触负值更好。
区间采用 10000 次场景内配对案例 bootstrap，固定随机种子 20261008。
它只衡量这几个已训练模型的测试案例不确定性，不衡量新训练 seed 的总体不确定性，也未作多重比较校正。

| 比较 | RL seed | SR 差 | SR 探索性 95% 区间 | CR 差 | TR 差 | 接触差 |
|---|---:|---:|---|---:|---:|---:|
| 复杂 - 简单 | 42 | +6.43 | [+4.87, +7.97] | +0.27 | -6.70 | +0.13 |
| 复杂 - 简单 | 43 | +1.17 | [-0.47, +2.80] | -1.17 | 0.00 | -1.07 |
| 复杂 - GRU | 42 | +0.30 | [-0.97, +1.57] | +0.30 | -0.60 | +0.20 |
| 复杂 - GRU | 43 | -6.63 | [-8.17, -5.13] | -6.07 | +12.70 | -5.47 |
| 简单 - GRU | 42 | -6.13 | [-7.70, -4.57] | +0.03 | +6.10 | +0.07 |
| 简单 - GRU | 43 | -7.80 | [-9.43, -6.17] | -4.90 | +12.70 | -4.40 |

复杂对简单的配对成功转移：seed42 复杂独成功 391、简单独成功 198，净多成功 193；seed43 分别 329 / 294，净多成功仅 35。
这不是把两臂成功次数相减后假装案例独立；具体成败转移全部保存在 final-analysis.json。

两 seed 合并只作描述：复杂对简单 SR +3.80，条件案例区间 [+2.60, +4.98]；复杂对 GRU SR -3.17，区间 [-4.15, -2.22]。
合并 bootstrap 对同一个案例在两个 seed 上使用相同重采样索引，避免把 6000 行错当 6000 个独立世界。
即使合并区间不跨零，也不能据此称跨训练种子稳定、已 fresh confirmation 或已达到论文级方法结论。

### 17.4 时间、停滞、安全及计算

成功到达时间和成功路径仅基于各自成功回合；它们受成功案例集合变化影响。discomfort 是所有回合的逐步加权比例，停滞是逐回合窗口比例均值。

| 模型 | seed | 成功时间 s | 成功路径 m | 停滞窗口 % | discomfort 步数 % | 最小净距 m | 逐步加权推理 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| GRU | 42 | 14.16 | 12.26 | 10.42 | 2.89 | -0.259 | 11.66 |
| 简单 | 42 | 14.75 | 11.59 | 12.35 | 3.06 | -0.162 | 13.99 |
| 复杂 | 42 | 14.46 | 11.51 | 10.91 | 3.62 | -0.203 | 13.96 |
| GRU | 43 | 14.04 | 12.02 | 12.11 | 5.02 | -0.235 | 11.62 |
| 简单 | 43 | 14.43 | 11.72 | 13.26 | 2.85 | -0.168 | 14.16 |
| 复杂 | 43 | 15.47 | 12.12 | 17.09 | 2.14 | -0.141 | 13.92 |

共同成功案例再配对，避免单纯比较不同幸存集合：

- 复杂对简单 seed42：2302 个共同成功案例，时间 -0.341 s、路径 -0.134 m、最小净距 -0.0043 m。更快/短，但净距略小。
- seed43：2102 个共同成功案例，时间 +1.013 s、路径 +0.378 m、最小净距 +0.0071 m。更保守，但慢/长。
- 复杂对 GRU 的共同成功时间差为 seed42 +0.306 s、seed43 +1.484 s，不能把较短路径直接说成更高通行效率。

| 模型 | seed | RL / 优化次数 | 训练 S / C / T | 训练墙钟 h | 3000-case 测评墙钟 h |
|---|---:|---|---|---:|---:|
| GRU | 42 | 10000 / 40000 | 9745 / 193 / 62 | 4.82 | 0.73 |
| 简单 | 42 | 10000 / 40000 | 9757 / 197 / 46 | 7.32 | 0.90 |
| 复杂 | 42 | 10000 / 40000 | 9766 / 205 / 29 | 7.28 | 0.85 |
| GRU | 43 | 10000 / 40000 | 9690 / 260 / 50 | 4.83 | 0.68 |
| 简单 | 43 | 10000 / 40000 | 9733 / 207 / 60 | 7.37 | 0.92 |
| 复杂 | 43 | 10000 / 40000 | 9752 / 214 / 34 | 7.34 | 0.96 |

删除 cubature 后，复杂和简单的实测推理几乎同量级；不能再说新复杂版必须逐次执行 128 点积分。
但贝叶斯两臂训练仍约为 GRU 的 1.5 倍，推理约慢 20%；更少参数不等于更快执行。
这些是共享 4090 和同期外部负载下的观测，不是隔离设备 benchmark，不直接归因到单个算子。
所有模型训练 SR 约 97%，却在六场景评测中明显下降；小 validation 的 100% 不能代表泛化已经解决。

### 17.5 与修复前 10000 版相比：不能只报告好消息

仅作历史性能对照，两轮初始化、IL 选择和 readout 参数化不同，不是只开关方差通道的同权重实验。

| seed42 模型 | 修复前 SR / CR / TR % | 修复后 SR / CR / TR % | SR 差 pp |
|---|---|---|---:|
| GRU | 89.47 / 2.80 / 7.73 | 89.47 / 2.80 / 7.73 | 0.00 |
| 贝叶斯简单结构 | 91.83 / 3.13 / 5.03 | 83.33 / 2.83 / 13.83 | -8.50 |
| 贝叶斯复杂结构 | 87.17 / 4.90 / 7.93 | 89.77 / 3.10 / 7.13 | +2.60 |

复杂结构 seed42 相比旧版更好是实际结果；但简单结构大幅退步同样是实际结果。
“复杂胜简单”同时包含复杂提升和简单退步，不能全部解释成发现有效 posterior uncertainty。
旧简单 readout 是 64 输入，新简单是 128 输入且后半置零；两者函数类可以等价映射，但随机初始化尺度、随机数消耗和新 IL 训练路径不自动等价。
本轮 parity 证明存在零误差等价映射，正式训练没有采用旧模型等价初始化。因此“初始化/优化路径造成退步”是合理待检验解释，不是已确认根因。
GRU 总体计数相同仅是总体计数吻合，不自动证明两轮全部逐步动作、轨迹或权重完全相同。

### 17.6 已知、未知与最终判断

| 问题 | 当前证据 | 判断 |
|---|---|---|
| 训练和完整测评是否完成 | 六组完成标记、CSV 计数、冻结 SHA 一致 | 已验证 |
| 显式方差是否可以影响模型输出 | 方差敏感性、自检、梯度和训练接通 | 工程上已验证 |
| 复杂是否优于本轮简单 | 两 seed SR 点估计正差，seed42 较强、seed43 较弱；安全方向不一致 | 有比较正信号，非稳定全面优势 |
| 是否胜过强 GRU | seed42 基本平手、seed43 SR 明显更低且超时更多 | 未达到；不能宣布方法胜出 |
| 增益是否来自状态相关方差 | 没有固定/置乱方差对照，简单臂自身退步 | 未验证 |
| 方差是否是真实概率校准 | 没有 likelihood/KL/预测监督及正式校准检验 | 未验证；不能把 logvariance 特征作用等同校准后验价值 |
| 跨完整初始化是否稳定 | 只有两个在线 RL seed，共享 IL | 证据不足 |
| 是否有 fresh confirmation | 复用开发中已有 3000-case 池 | 没有 |
| 是否已发现新 fallback/放松约束收益 | 本轮未改约束；CSV 无逐决策 fallback 计数 | 不声称已完成完整 fallback 归因，也不填 0 |

最终判断：**显式后验特征的比较信号成立，但当前版本未稳定超过 GRU，状态相关不确定性的因果贡献尚未成立。没有 METHOD_ENTRY_FOUND。**
不能再把旧 cubature 的近常数行为照搬成新结构结论；也不能因新结构读取方差就宣布旧问题已科学解决。
目前最需要避免的误判是：弱化后的简单基线 + 少数种子差异，被包装成后验机制成功。

唯一最高优先级后续：**先做方差因果消融与简单臂退化归因，验证信息本身，而不是立即加 KL、改奖励或再跑一套复杂架构。**
具体先冻结现有权重，比较合法均值相同下原方差、固定逐维方差、跨状态置乱方差对候选排序与闭环的影响；这种干预只能是诊断，必须标明分布外风险。
随后若确需重训，只做旧简单结构等价初始化的受控对照，分开检验初始化/优化路径。未完成这些之前不声称新增 Bayesian 决策能力。
此建议不是本次自动启动的实验；本次只交完整分析和保存结果。

### 17.7 可复核交付与 Git

单一综合报告继续更新本文件，没有另建大量独立 MD。

原始六组结果与冻结协议：
/home/abc/workspace/CrowdNav(20270731_backup2)/CrowdNav/crowd_nav/runs/bayes-fix-20261008/remote-run

完整统计、各场景、配对转移、置信区间、计算量及原始文件 SHA：
/home/abc/workspace/CrowdNav(20270731_backup2)/CrowdNav/crowd_nav/runs/bayes-fix-20261008/final-analysis.json

可重复离线分析入口：scripts/analyze-bayes-fix-results.py。只需 Python + NumPy，不需要 GPU、checkpoint 或 ORCA 数据；固定 bootstrap 随机数。
Git 目标：https://github.com/jinglongjiang/beiyesiRL.git 。同步报告、分析脚本、冻结协议、最终 CSV/JSON、日志和已有负结果，不上传模型/优化器权重、离线示范数据、虚拟环境或凭据。
SYNC-MANIFEST.json 记录本次提交快照的文件清单、SHA、大小与完整 benchmark 状态。提交号在 Git 历史中查验，避免在同一次提交内容里制造自引用 hash。
