# 贝叶斯方差决策层拆解

版本：bayes-decomp-v1-20261009

## 现象

本轮已完成两个训练 seed × 六格 ×100 episode，共1200条自然轨迹、77352个决策步。
仅本地3060推理与CPU统计；零训练、零4090访问、原项目未修改。只有运行期 readout 输入干预。

最终判断：REAL_DIRECTION_NOT_ESTABLISHED。下一步唯一分支：INFORMATION_LAYER。

已验证通过：方差直接读出通道参与决策。未验证：它是否稳定改善任务。fixed的净距方向跨seed翻转；候选内置换时real净距胜率两seed均低于50%，但real平均即时reward较高，奖励频率与幅度方向冲突。后文的事后分解显示平均reward优势主要来自终点奖励差，不能写成稳定安全收益。

这不是完整贝叶斯与均值臂的比较，也不是新方法或长期闭环收益确认。

## 问题与冻结口径

- fixed：每个训练seed独立使用六格case400–415的全部候选logvar逐维均值；正式测试case0–99，完全不重叠。
- shuffle_within：每个决策置换80个完整logvar向量，mean不变；ctrl_shuffle_mean只置换mean作为阳性对照。
- real真实执行；干预只评估同一root影子动作，不写历史。所有reward、安全过滤、80动作、T=24、25s时限、0.3动作平滑保持原设置。
- 主几何指标：沿用同一前一执行动作和平滑系数后，对全体行人实际下一位置计算终点净距。人的动作在机器人本步动作执行前生成，原robot.visible=false，因此同一root影子动作可共享实际人的下一位置。
- 原生即时reward：使用母体的扫掠几何与timeout/collision/goal优先级，逐步与真实env.step奖励校验；另外保存候选CV终点dmins，三者不混称。
- 翻转步二项检验按要求保留，但同episode步相关，不能当独立样本；同时给出2000次episode聚类bootstrap。8项主比较另报Bonferroni二项p。分格与终止窗口为描述性分析。

## 对照验收

| 检查 | 状态 | 证据 |
|---|---|---|
| GRID=80、导入来源、权重严格加载 | 已验证通过 | 每个环境与模型均断言；SHA启动和结束复验 |
| 6格×3ep×2seed smoke | 已验证通过 | controls.json与smoke逐episode原始记录 |
| real完整重放两遍 | 已验证通过 | 1200/1200；值、输入、动作、位置、reward、净距逐位一致，翻转0 |
| 既有跨环境benchmark结局/步数 | 已确认失败：步数非完全复现 | 结局1200/1200一致；步数1196/1200一致，4条只差1–2步，均不删除 |
| 阳性对照 | 已验证通过 | 两seed各自mean-shuffle翻转≥1% |
| 替换实际生效 | 已验证通过 | 非替换通道逐位不变；置换逐维边际完全相同；首次前后张量摘要及逐步value变化已保存 |
| 原生奖励与执行语义 | 已验证通过 | 实际平滑动作一致；counterfactual真实动作reward误差<1e-10 |

## 全部步结果

| seed | 干预 | 翻转/步 | 翻转率 | 净距real胜/干预胜/平局 | real胜率(非平局) | episode区间 | reward real胜/干预胜/平局 |
|---|---|---:|---:|---|---:|---|---|
| 42 | fixed | 9441/36169 | 26.102% | 5273/4168/0 | 0.5585 | [0.5435, 0.5726] | 165/162/9114 |
| 42 | shuffle_within | 23245/36169 | 64.268% | 9441/13804/0 | 0.4062 | [0.3976, 0.4150] | 406/606/22233 |
| 42 | ctrl_shuffle_mean | 35124/36169 | 97.111% | 13508/21616/0 | 0.3846 | [0.3715, 0.3971] | 559/849/33716 |
| 43 | fixed | 13962/41183 | 33.902% | 6595/7367/0 | 0.4724 | [0.4572, 0.4886] | 88/120/13754 |
| 43 | shuffle_within | 27903/41183 | 67.754% | 11543/16360/0 | 0.4137 | [0.4049, 0.4223] | 190/513/27200 |
| 43 | ctrl_shuffle_mean | 39861/41183 | 96.790% | 20937/18924/0 | 0.5253 | [0.5159, 0.5350] | 317/746/38798 |

### 即时收益统计

| seed | 干预 | 指标 | real−干预平均差(翻转步) | episode区间 | 二项p(描述性) | Bonferroni8 p |
|---|---|---|---:|---|---:|---:|
| 42 | fixed | clearance | 0.005070 | [0.0028, 0.0073] | 0.0000000 | 0.0000000 |
| 42 | fixed | reward | 0.001004 | [0.0002, 0.0019] | 0.9119555 | 1.0000000 |
| 42 | shuffle_within | clearance | -0.012875 | [-0.0142, -0.0115] | 0.0000000 | 0.0000000 |
| 42 | shuffle_within | reward | 0.000840 | [0.0005, 0.0013] | 0.0000000 | 0.0000000 |
| 42 | ctrl_shuffle_mean | clearance | -0.040024 | [-0.0446, -0.0357] | 0.0000000 | 0.0000000 |
| 42 | ctrl_shuffle_mean | reward | 0.001020 | [0.0006, 0.0014] | 0.0000000 | 0.0000000 |
| 43 | fixed | clearance | -0.007680 | [-0.0110, -0.0042] | 0.0000000 | 0.0000000 |
| 43 | fixed | reward | -0.000147 | [-0.0006, 0.0004] | 0.0313499 | 0.2507996 |
| 43 | shuffle_within | clearance | -0.018689 | [-0.0204, -0.0169] | 0.0000000 | 0.0000000 |
| 43 | shuffle_within | reward | 0.000364 | [0.0000, 0.0007] | 0.0000000 | 0.0000000 |
| 43 | ctrl_shuffle_mean | clearance | 0.009860 | [0.0069, 0.0127] | 0.0000000 | 0.0000000 |
| 43 | ctrl_shuffle_mean | reward | 0.000527 | [0.0003, 0.0008] | 0.0000000 | 0.0000000 |

### 按场景

| 分层 | 干预 | 步数 | 翻转 | 翻转率 | 净距real胜率 | reward real胜率 |
|---|---|---:|---:|---:|---:|---:|
| seed42/scene/baseline_circle | fixed | 4661 | 1189 | 0.2551 | 0.5845 | 0.8095 |
| seed42/scene/baseline_circle | shuffle_within | 4661 | 3035 | 0.6511 | 0.3552 | 0.4915 |
| seed42/scene/baseline_circle | ctrl_shuffle_mean | 4661 | 4562 | 0.9788 | 0.2854 | 0.4189 |
| seed42/scene/baseline_square | fixed | 5527 | 1490 | 0.2696 | 0.5362 | 0.4426 |
| seed42/scene/baseline_square | shuffle_within | 5527 | 3586 | 0.6488 | 0.4389 | 0.3750 |
| seed42/scene/baseline_square | ctrl_shuffle_mean | 5527 | 5361 | 0.9700 | 0.4710 | 0.4191 |
| seed42/scene/dense_circle | fixed | 5472 | 1470 | 0.2686 | 0.5673 | 0.5000 |
| seed42/scene/dense_circle | shuffle_within | 5472 | 3645 | 0.6661 | 0.3462 | 0.2267 |
| seed42/scene/dense_circle | ctrl_shuffle_mean | 5472 | 5380 | 0.9832 | 0.3110 | 0.2736 |
| seed42/scene/dense_square | fixed | 7509 | 2154 | 0.2869 | 0.5608 | 0.4694 |
| seed42/scene/dense_square | shuffle_within | 7509 | 5175 | 0.6892 | 0.4390 | 0.3904 |
| seed42/scene/dense_square | ctrl_shuffle_mean | 7509 | 7280 | 0.9695 | 0.4479 | 0.3657 |
| seed42/scene/large_circle | fixed | 6766 | 1531 | 0.2263 | 0.5781 | 0.4545 |
| seed42/scene/large_circle | shuffle_within | 6766 | 3906 | 0.5773 | 0.3971 | 0.2708 |
| seed42/scene/large_circle | ctrl_shuffle_mean | 6766 | 6653 | 0.9833 | 0.2835 | 0.3427 |
| seed42/scene/large_square | fixed | 6234 | 1607 | 0.2578 | 0.5302 | 0.5349 |
| seed42/scene/large_square | shuffle_within | 6234 | 3898 | 0.6253 | 0.4371 | 0.4904 |
| seed42/scene/large_square | ctrl_shuffle_mean | 6234 | 5888 | 0.9445 | 0.4859 | 0.4570 |
| seed43/scene/baseline_circle | fixed | 4743 | 2112 | 0.4453 | 0.4252 | 0.3810 |
| seed43/scene/baseline_circle | shuffle_within | 4743 | 3683 | 0.7765 | 0.3557 | 0.3167 |
| seed43/scene/baseline_circle | ctrl_shuffle_mean | 4743 | 4566 | 0.9627 | 0.5241 | 0.3514 |
| seed43/scene/baseline_square | fixed | 6440 | 2001 | 0.3107 | 0.4908 | 0.5122 |
| seed43/scene/baseline_square | shuffle_within | 6440 | 4420 | 0.6863 | 0.4351 | 0.2946 |
| seed43/scene/baseline_square | ctrl_shuffle_mean | 6440 | 6295 | 0.9775 | 0.5530 | 0.2680 |
| seed43/scene/dense_circle | fixed | 5941 | 2163 | 0.3641 | 0.4198 | 0.3542 |
| seed43/scene/dense_circle | shuffle_within | 5941 | 4479 | 0.7539 | 0.3755 | 0.1696 |
| seed43/scene/dense_circle | ctrl_shuffle_mean | 5941 | 5748 | 0.9675 | 0.5638 | 0.3421 |
| seed43/scene/dense_square | fixed | 8655 | 2246 | 0.2595 | 0.5632 | 0.4615 |
| seed43/scene/dense_square | shuffle_within | 8655 | 5223 | 0.6035 | 0.4607 | 0.3071 |
| seed43/scene/dense_square | ctrl_shuffle_mean | 8655 | 8442 | 0.9754 | 0.5280 | 0.2724 |
| seed43/scene/large_circle | fixed | 7602 | 3287 | 0.4324 | 0.4335 | 0.3929 |
| seed43/scene/large_circle | shuffle_within | 7602 | 5249 | 0.6905 | 0.3835 | 0.2427 |
| seed43/scene/large_circle | ctrl_shuffle_mean | 7602 | 7269 | 0.9562 | 0.4420 | 0.2910 |
| seed43/scene/large_square | fixed | 7802 | 2153 | 0.2760 | 0.5188 | 0.4194 |
| seed43/scene/large_square | shuffle_within | 7802 | 4849 | 0.6215 | 0.4556 | 0.2898 |
| seed43/scene/large_square | ctrl_shuffle_mean | 7802 | 7541 | 0.9665 | 0.5506 | 0.3026 |

### 按人数

| 分层 | 干预 | 步数 | 翻转 | 翻转率 | 净距real胜率 | reward real胜率 |
|---|---|---:|---:|---:|---:|---:|
| seed42/humans/5 | fixed | 4661 | 1189 | 0.2551 | 0.5845 | 0.8095 |
| seed42/humans/5 | shuffle_within | 4661 | 3035 | 0.6511 | 0.3552 | 0.4915 |
| seed42/humans/5 | ctrl_shuffle_mean | 4661 | 4562 | 0.9788 | 0.2854 | 0.4189 |
| seed42/humans/10 | fixed | 10999 | 2960 | 0.2691 | 0.5517 | 0.4607 |
| seed42/humans/10 | shuffle_within | 10999 | 7231 | 0.6574 | 0.3922 | 0.3307 |
| seed42/humans/10 | ctrl_shuffle_mean | 10999 | 10741 | 0.9765 | 0.3908 | 0.3746 |
| seed42/humans/12 | fixed | 6766 | 1531 | 0.2263 | 0.5781 | 0.4545 |
| seed42/humans/12 | shuffle_within | 6766 | 3906 | 0.5773 | 0.3971 | 0.2708 |
| seed42/humans/12 | ctrl_shuffle_mean | 6766 | 6653 | 0.9833 | 0.2835 | 0.3427 |
| seed42/humans/20 | fixed | 13743 | 3761 | 0.2737 | 0.5477 | 0.5000 |
| seed42/humans/20 | shuffle_within | 13743 | 9073 | 0.6602 | 0.4382 | 0.4422 |
| seed42/humans/20 | ctrl_shuffle_mean | 13743 | 13168 | 0.9582 | 0.4649 | 0.4135 |
| seed43/humans/5 | fixed | 4743 | 2112 | 0.4453 | 0.4252 | 0.3810 |
| seed43/humans/5 | shuffle_within | 4743 | 3683 | 0.7765 | 0.3557 | 0.3167 |
| seed43/humans/5 | ctrl_shuffle_mean | 4743 | 4566 | 0.9627 | 0.5241 | 0.3514 |
| seed43/humans/10 | fixed | 12381 | 4164 | 0.3363 | 0.4539 | 0.4270 |
| seed43/humans/10 | shuffle_within | 12381 | 8899 | 0.7188 | 0.4051 | 0.2321 |
| seed43/humans/10 | ctrl_shuffle_mean | 12381 | 12043 | 0.9727 | 0.5582 | 0.3049 |
| seed43/humans/12 | fixed | 7602 | 3287 | 0.4324 | 0.4335 | 0.3929 |
| seed43/humans/12 | shuffle_within | 7602 | 5249 | 0.6905 | 0.3835 | 0.2427 |
| seed43/humans/12 | ctrl_shuffle_mean | 7602 | 7269 | 0.9562 | 0.4420 | 0.2910 |
| seed43/humans/20 | fixed | 16457 | 4399 | 0.2673 | 0.5415 | 0.4429 |
| seed43/humans/20 | shuffle_within | 16457 | 10072 | 0.6120 | 0.4582 | 0.2975 |
| seed43/humans/20 | ctrl_shuffle_mean | 16457 | 15983 | 0.9712 | 0.5386 | 0.2891 |

### 失败前窗口与成功参照

| 分层 | 干预 | 步数 | 翻转 | 翻转率 | 净距real胜率 | reward real胜率 |
|---|---|---:|---:|---:|---:|---:|
| seed42/all | fixed | 36169 | 9441 | 0.2610 | 0.5585 | 0.5046 |
| seed42/all | shuffle_within | 36169 | 23245 | 0.6427 | 0.4062 | 0.4012 |
| seed42/all | ctrl_shuffle_mean | 36169 | 35124 | 0.9711 | 0.3846 | 0.3970 |
| seed42/collision_pre4 | fixed | 67 | 19 | 0.2836 | 0.3158 | 0.1111 |
| seed42/collision_pre4 | shuffle_within | 67 | 29 | 0.4328 | 0.4828 | 0.5625 |
| seed42/collision_pre4 | ctrl_shuffle_mean | 67 | 52 | 0.7761 | 0.4808 | 0.4688 |
| seed42/collision_pre8 | fixed | 122 | 31 | 0.2541 | 0.3871 | 0.2000 |
| seed42/collision_pre8 | shuffle_within | 122 | 63 | 0.5164 | 0.4921 | 0.5833 |
| seed42/collision_pre8 | ctrl_shuffle_mean | 122 | 102 | 0.8361 | 0.4510 | 0.4250 |
| seed42/timeout_pre4 | fixed | 172 | 48 | 0.2791 | 0.6042 | 1.0000 |
| seed42/timeout_pre4 | shuffle_within | 172 | 127 | 0.7384 | 0.4724 | 0.6667 |
| seed42/timeout_pre4 | ctrl_shuffle_mean | 172 | 170 | 0.9884 | 0.4941 | 0.5000 |
| seed42/timeout_pre8 | fixed | 344 | 91 | 0.2645 | 0.5824 | 1.0000 |
| seed42/timeout_pre8 | shuffle_within | 344 | 246 | 0.7151 | 0.4472 | 0.5000 |
| seed42/timeout_pre8 | ctrl_shuffle_mean | 344 | 339 | 0.9855 | 0.5103 | 0.5714 |
| seed42/success_all | fixed | 31478 | 7931 | 0.2520 | 0.5495 | 0.5076 |
| seed42/success_all | shuffle_within | 31478 | 19670 | 0.6249 | 0.4011 | 0.3960 |
| seed42/success_all | ctrl_shuffle_mean | 31478 | 30583 | 0.9716 | 0.3706 | 0.3904 |
| seed42/success_last8 | fixed | 4320 | 842 | 0.1949 | 0.4335 | 0.7895 |
| seed42/success_last8 | shuffle_within | 4320 | 1880 | 0.4352 | 0.4537 | 0.7000 |
| seed42/success_last8 | ctrl_shuffle_mean | 4320 | 4074 | 0.9431 | 0.4816 | 0.7000 |
| seed43/all | fixed | 41183 | 13962 | 0.3390 | 0.4724 | 0.4231 |
| seed43/all | shuffle_within | 41183 | 27903 | 0.6775 | 0.4137 | 0.2703 |
| seed43/all | ctrl_shuffle_mean | 41183 | 39861 | 0.9679 | 0.5253 | 0.2982 |
| seed43/collision_pre4 | fixed | 24 | 3 | 0.1250 | 0.6667 | 0.6667 |
| seed43/collision_pre4 | shuffle_within | 24 | 12 | 0.5000 | 0.8333 | 0.8571 |
| seed43/collision_pre4 | ctrl_shuffle_mean | 24 | 17 | 0.7083 | 0.4706 | 0.4545 |
| seed43/collision_pre8 | fixed | 48 | 5 | 0.1042 | 0.6000 | 0.6667 |
| seed43/collision_pre8 | shuffle_within | 48 | 23 | 0.4792 | 0.6087 | 0.8571 |
| seed43/collision_pre8 | ctrl_shuffle_mean | 48 | 40 | 0.8333 | 0.4250 | 0.3750 |
| seed43/timeout_pre4 | fixed | 428 | 102 | 0.2383 | 0.7059 | 不可算 |
| seed43/timeout_pre4 | shuffle_within | 428 | 263 | 0.6145 | 0.4449 | 0.0000 |
| seed43/timeout_pre4 | ctrl_shuffle_mean | 428 | 416 | 0.9720 | 0.4760 | 0.0000 |
| seed43/timeout_pre8 | fixed | 856 | 210 | 0.2453 | 0.7000 | 不可算 |
| seed43/timeout_pre8 | shuffle_within | 856 | 502 | 0.5864 | 0.4661 | 0.0455 |
| seed43/timeout_pre8 | ctrl_shuffle_mean | 856 | 831 | 0.9708 | 0.4633 | 0.0286 |
| seed43/success_all | fixed | 30260 | 10989 | 0.3632 | 0.4387 | 0.4262 |
| seed43/success_all | shuffle_within | 30260 | 21065 | 0.6961 | 0.3973 | 0.2652 |
| seed43/success_all | ctrl_shuffle_mean | 30260 | 29242 | 0.9664 | 0.5184 | 0.3302 |
| seed43/success_last8 | fixed | 3896 | 1293 | 0.3319 | 0.4316 | 0.3846 |
| seed43/success_last8 | shuffle_within | 3896 | 2627 | 0.6743 | 0.4507 | 0.3636 |
| seed43/success_last8 | ctrl_shuffle_mean | 3896 | 3699 | 0.9494 | 0.5710 | 0.5506 |

## 证据限制与工程记录

- 已验证通过：以上均是冻结Full权重内的同状态输入干预；两seed单独报告，不用合并数字掩盖方向不一致。
- 未验证：后验校准、方差独立于全部几何信息、长时域任务价值、碰撞率/SR增益、Full vs Mean、跨新训练seed外推。
- 干预不是“关掉方差学习”：固定值同时移除跨状态和候选内变化；置换保留边际，却破坏mean/logvar/候选的联合对应，可能分布外。不能自动等同于不确定性因果效益。
- 净距更大不等于任务更好；原reward的progress/time系数为0，大量reward平局是该冻结任务的属性，不应强行制造额外奖励来区分。
- 碰撞/超时前窗口按real结局分层，干预没有真实闭环结局；成功轨迹全保留，不据失败子集宣布总体效果。
- 已确认失败：首次校准包装器无条件调用不存在的set_env，在首条轨迹前停止，无科学数据；按母体既有可选接口检查修复，旧代码/日志/manifest保留。
- 已确认失败：保存修复冻结时遇到Python3.8相对__file__键导致KeyError，在改写manifest前停止；路径规范化后修复，无科学数据。
- 已确认失败：首次正式采集在seed42/large_circle/case34触发额外跨环境benchmark步数断言而停止。该case本地原policy及埋点policy都为success/67步、动作与位置逐位一致，旧benchmark为66步。此断言不等于本地real确定性门失效；跨环境精确步数改为记录，原始中断版完整保留，新版从头采集不拼接。具体跨环境差异原因未验证。
- 旧模块导入会打印缺失mamba_ssm警告；实际backbone硬断言bayes，没有运行Mamba。

## 可选方案与推荐

推荐唯一下一步：信息层检验，不进入闭环收益确认、不重训。
本轮未满足预注册的跨seed方向有利标准；这不等于证明方差没有信息。需区分候选差异是否有新增信息与消费者如何利用。
既有b1-rollout.npz来自修复前滤波器；旧/修复后嵌套检验已经完成，应先引用保存结果，不重复旧数据检验冒充新证据。

## 资产

protocol.json / frozen-manifest.json / frozen-manifest-initial.json
controls.json / fixed-seed42.json / fixed-seed43.json
stepwise/summary.json / stepwise/seed*-scene*-case*.json / 同名.npz
calibration.log / calibration-retry.log / smoke.log / full.log
engineering-preflight.json / engineering-manifest-key.json
权重只读引用原备份，不上传、不重新复制到4090。

实际计时：校准seed42 73.7s、seed43 86.0s；smoke 102.2s；正式双重重放 3583.2s。

## 信息层已有证据：明确不是本轮新实验

不重复运行已经完成的嵌套检验。旧b1来自修复前，两个修复后seed来自此前192个episode采集；以下只是引用保存结果。

| 方差来源 | 目标 | ΔAUC | 95% CI | 结论 |
|---|---|---:|---|---|
| old | future-4-clearance-0.2 | +0.0375 | [+0.0157, +0.0574] | 已验证通过：两描述子线性基线之上的增量关联 |
| old | future-8-clearance-0.2 | +0.0321 | [-0.0130, +0.0708] | 未验证：尚未证明有增量 |
| seed42 | future-4-clearance-0.2 | +0.0411 | [+0.0141, +0.0681] | 已验证通过：两描述子线性基线之上的增量关联 |
| seed42 | future-8-clearance-0.2 | +0.0639 | [+0.0156, +0.1110] | 已验证通过：两描述子线性基线之上的增量关联 |
| seed43 | future-4-clearance-0.2 | +0.0334 | [+0.0087, +0.0575] | 已验证通过：两描述子线性基线之上的增量关联 |
| seed43 | future-8-clearance-0.2 | +0.0709 | [+0.0233, +0.1193] | 已验证通过：两描述子线性基线之上的增量关联 |

这些检验说明某些当前状态sigma有增量关联，但不回答80候选之间logvar差异是否携带动作相关增量，也不证明消费者的方向有利。若本轮进入信息层，优先厘清这一剩余问题，不把旧b1重复计算包装成新进展。

本轮只拆方差直接读出通道；上游滤波依然用variance计算Kalman gain并更新mean，所有干预都保留这条路径。不能把直接通道的结果推广成整个方差机制或完整贝叶斯的价值结论。

## 最终保存验收

- 已验证通过：3项独立合约单测；1200份逐episode JSON与NPZ完整，77352步、所有数组有限、动作索引与收益记录一致。
- 已验证通过：原模型/环境/配置/权重源SHA全部复验；分析代码与预注册hash一致。
- 已验证通过：PID 180652在运行时由精确argv与/proc确认；完成后/proc已不存在，无pgrep自匹配。
- 已确认失败：一次独立的进程监控内联命令发生换行转义SyntaxError，没有接触实验进程或数据；已保留engineering-monitor.json并用独立精确argv检查替换。
- 本轮工程错误均单独记录，不作为科学样本；重启后的完整正式采集没有traceback、断言失败或非零退出。初次中断版未拼接进最终结果。

## 结果解读：参与决策不等于稳定占优

- 已验证通过：fixed翻转26.10%/33.90%；候选内logvar置换翻转64.27%/67.75%，阳性对照翻转97.11%/96.79%。方差的直接读出通道并非未被消费。
- 已确认失败（限定指标）：候选内置换时，real即时净距胜率40.62%/41.37%，episode区间均低于50%；平均净距差为−12.9/−18.7mm。
- 未验证：稳定净收益。fixed的净距方向跨seed翻转；shuffle中real平均即时reward较高，但非平局reward胜率仅40.12%/27.03%，存在频率与损失幅度的冲突，不能写成总体导航获益或总体导航受损。
- 以下按已冻结原reward中的终止/碰撞/不适事件做事后解释，不参与改写预注册结论。较少的大额正差可以压过较多的小额负差；终点奖励差也不等于后续整段成功率差。
- shuffle对照下，终点奖励差累计贡献+19/+10，全部即时reward差累计约+19.53/+10.15。因此平均reward正差主要来自终点奖励，而不是足以确认的碰撞改善。
- 表格四舍五入后的0.0000不代表原始值为零；seed43/shuffle的平均reward差区间实际为[0.00004221, 0.00070882]。完整精度和二项p值保存在summary.json中。

| seed/干预 | 奖励差类别 | real胜 | 干预胜 | 累计real−干预reward |
|---|---|---:|---:|---:|
| seed42/fixed | terminal_goal_difference | 13 | 3 | +10.000000 |
| seed42/fixed | native_collision_difference | 0 | 1 | -0.481289 |
| seed42/fixed | discomfort_difference | 152 | 158 | -0.039931 |
| seed42/shuffle_within | terminal_goal_difference | 21 | 2 | +19.000000 |
| seed42/shuffle_within | native_collision_difference | 2 | 1 | +0.483575 |
| seed42/shuffle_within | discomfort_difference | 383 | 603 | +0.043985 |
| seed43/fixed | terminal_goal_difference | 5 | 7 | -2.000000 |
| seed43/fixed | native_collision_difference | 0 | 0 | +0.000000 |
| seed43/fixed | discomfort_difference | 83 | 113 | -0.057048 |
| seed43/shuffle_within | terminal_goal_difference | 16 | 6 | +10.000000 |
| seed43/shuffle_within | native_collision_difference | 1 | 0 | +0.477169 |
| seed43/shuffle_within | discomfort_difference | 173 | 507 | -0.324829 |

推荐仍为信息层，不直接启动闭环确认或匹配重训。已有当前状态sigma嵌套检验有正增量关联；剩余问题是候选间logvar差异是否携带与动作后果相关的增量，以及消费者为何呈现上述权衡。本轮没有检验这个剩余问题，也没有对Full vs Mean下结论。
