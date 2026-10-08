"""Render one evidence-led report from the frozen diagnostics, no inference."""

import csv
import datetime
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'crowd_nav/runs/bayes-diagnostic-20261009'
TAGS = ('old', 'seed42', 'seed43')
NAMES = {'old': '修复前', 'seed42': '修复后 seed42', 'seed43': '修复后 seed43'}
ARMS = {'gru': 'GRU', 'bayes_mean': '贝叶斯简单结构', 'bayes': '贝叶斯复杂结构'}
TARGETS = {'future-4-clearance-0.0': '未来4步净距<0m', 'future-4-clearance-0.2': '未来4步净距<0.2m',
           'future-8-clearance-0.0': '未来8步净距<0m', 'future-8-clearance-0.2': '未来8步净距<0.2m',
           'episode-collision': 'episode原生碰撞'}
STATUS = {'INCREMENTAL_ASSOCIATION_CONFIRMED': '已验证：正增量关联',
          'PREDICTION_DEGRADATION_CONFIRMED': '已确认失败：预测退步',
          'INCREMENT_NOT_ESTABLISHED': '尚未证明有增量',
          'EVIDENCE_INSUFFICIENT_SINGLE_CLASS': '尚未证明：单类别，AUC不可算'}


def read(path):
    return json.loads(path.read_text())


def number(value):
    return '不可算' if value is None else '%.4f' % value


def quantile(value):
    return '无数据' if value is None else '%.3f [%.3f, %.3f]' % (value['median'], value['p25'], value['p75'])


def pack_traces(seed, diagnostic):
    directory = BASE / ('seed%d-episodes' % seed)
    keys = ('sigma', 'descriptors', 'clearance', 'actions', 'positions')
    values = {k: [] for k in keys}
    metadata = []
    offsets = [0]
    position_offsets = [0]
    for episode in range(192):
        metadata.append(read(directory / ('episode-%03d.json' % episode)))
        with np.load(directory / ('episode-%03d.npz' % episode)) as record:
            for k in keys:
                values[k].append(record[k])
            offsets.append(offsets[-1] + len(record['sigma']))
            position_offsets.append(position_offsets[-1] + len(record['positions']))
    arrays = {k: np.concatenate(v) for k, v in values.items()}
    arrays['episode_offsets'] = np.array(offsets, dtype=np.int64)
    arrays['position_offsets'] = np.array(position_offsets, dtype=np.int64)
    packed = BASE / ('episode-traces-seed%d.npz' % seed)
    np.savez_compressed(packed, **arrays)
    with np.load(packed) as check:
        for episode in range(192):
            for k in keys:
                boundaries = position_offsets if k == 'positions' else offsets
                assert np.array_equal(check[k][boundaries[episode]:boundaries[episode+1]], values[k][episode])
    diagnostic.save(BASE / ('episode-traces-seed%d.json' % seed), {'status': 'LOSSLESS_VERIFIED',
         'metadata': metadata, 'npz_sha256': diagnostic.sha(packed), 'episodes': 192,
         'offset_schema': 'sigma/descriptors/clearance/actions use episode_offsets; positions use position_offsets',
         'cache': 'Original per-episode cache remains local; only lossless packed equivalents uploaded.'})


def main():
    spec = importlib.util.spec_from_file_location('diagnostic', ROOT / 'scripts/diagnose-bayes-variance.py')
    diagnostic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostic)
    protocol = diagnostic.pinned()
    nested = {tag: read(BASE / ('nested-' + tag + '.json')) for tag in TAGS}
    samples = {seed: read(BASE / ('sample-seed%d.json' % seed)) for seed in (42, 43)}
    assert all(s['status'] == 'COMPLETE' and s['episodes'] == 192 and s['outcome_and_step_parity'] for s in samples.values())
    parity = read(BASE / 'label-descriptor-parity.json')
    assert parity['labels_equal'] and parity['valid_equal'] and parity['descriptors_max_abs_error'] == 0
    for seed in (42, 43):
        pack_traces(seed, diagnostic)
    timeout = read(BASE / 'timeout-analysis.json')
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
    lines = ['# 贝叶斯方差增量与超时性质：3060 诊断报告', '', '更新时间：' + now, '',
        '## 结论与范围', '',
        '本轮三项均完成。只有只读推理与离线统计，没有算法改动、训练协议改动或任何 RL/IL 重训。',
        '权重、配置和六份最终 CSV 已在本地；整个任务没有连接 4090。统计使用宿主 Python 3.8 / sklearn 1.0.2，重采用本地 RTX 3060。',
        '以下方差增量仅相对于 current_clearance + history_age 两列的 L2 逻辑回归基线，不是相对于全部几何信息、任意非线性几何模型或强策略基线。',
        'AUC 是关联指标；本轮不检验后验校准、方差的因果动作价值或新算法性能，不产生 METHOD_ENTRY_FOUND。', '',
        '**修复后主要结果：**',
        '- 未来4步/0.2m：seed42 ΔAUC=+0.0411，95% CI [+0.0141,+0.0681]；seed43 +0.0334，[+0.0087,+0.0575]。',
        '- 未来8步/0.2m：seed42 +0.0639，[+0.0156,+0.1110]；seed43 +0.0709，[+0.0233,+0.1193]。两个目标均在两seed上为正增量关联。',
        '- 实际接触/原生碰撞目标事件不足，不能确认或否定相应增量。不要把危险净距预测提升替换成碰撞预测能力。',
        '- 复杂结构seed43的510次超时中，347次属于预注册卡住/低效组；其主体现象不是“正常推进但仅被25秒截断”。', '',
        '## 一、冻结统计口径', '',
        '- 每个数据集沿用同一 192 个自然案例、144 train / 48 test episode；case_index % 4 == 0 为 test，全部 train/test episode IDs 在统计 JSON 中保存。',
        '- 各目标仅使用原 valid 掩码：不足 k 步的安全结尾删失，已出现危险的不足 k 步窗口保留为正例。标签使用动作执行后的净距，描述子使用动作前当前净距和真实 history 长度。',
        '- A 输入两列描述子，B 输入同两列 + 64 维 sigma。StandardScaler 仅在对应训练折拟合。',
        '- C ∈ {0.001, 0.01, 0.1, 1, 10}，在训练集内部进行 5 折 StratifiedGroupKFold，按 episode 隔离；各模型独立按平均折 AUC 选 C，不接触测试标签选择正则化。',
        '- 95% 区间来自 2000 次测试 episode 配对 bootstrap；A/B 使用相同重采样，固定随机数 20261009。单类别重采样剔除并报告次数，不能补作零 AUC。',
        '- 这是多个探索性目标，没有多重比较校正；bootstrap 未重复整个训练/C选择流程，也不代表新训练种子的总体区间。', '',
        '## 二、任务1：修复前方差的嵌套检验', '',
        '本节只适用于旧 Full 滤波器的 sigma。旧数据包含 10696 步，不代表修复后模型。', '',
        '| 目标 | AUC(A) | AUC(B) | ΔAUC | ΔAUC 95% CI | 判断 |', '|---|---:|---:|---:|---|---|']
    def table(tag):
        for target, r in nested[tag]['results'].items():
            ci = r.get('delta_auc_95_ci')
            interval = '不可算' if ci is None else '[%+.4f, %+.4f]' % tuple(ci)
            lines.append('| %s | %s | %s | %s | %s | %s |' % (TARGETS[target], number(r['auc_A']), number(r['auc_B']),
                         '不可算' if r['delta_auc'] is None else '%+.4f' % r['delta_auc'], interval, STATUS[r['status']]))
    table('old')
    lines.extend(['', '原 split/valid 计数全部复现；future4/0.2 恰为 train 7573 / test 2581，正例 657 / 225。',
        '原五个边际 AUC 按原 probe 代码重新计算，与 b1.json 中可计算目标的数值差均小于 1e-8。',
        '修复前两个净距<0m目标均无测试正例，不能算 AUC。碰撞测试集只有 2 个正例 episode（不是早期报告文字中的1个）；其区间对极少事件高度敏感。',
        '原碰撞 bootstrap 有 1747/2000 次可用、253 次单类别；正增量关联不能当作可靠的碰撞预测能力确认。', '',
        '## 三、任务2：修复后方差，两个 RL seed 分开', ''])
    for seed in (42, 43):
        tag = 'seed%d' % seed
        lines.extend(['### ' + NAMES[tag], '', '| 目标 | AUC(A) | AUC(B) | ΔAUC | ΔAUC 95% CI | 判断 |',
                      '|---|---:|---:|---:|---|---|'])
        table(tag)
        lines.append('')
    lines.extend(['### 两 seed 的方向检验', '', '| 目标 | seed42 ΔAUC | seed43 ΔAUC | 跨 seed 判断 |', '|---|---:|---:|---|'])
    agreement = {}
    for target in TARGETS:
        a, b = [nested['seed%d' % seed]['results'][target] for seed in (42, 43)]
        if a['delta_auc'] is None or b['delta_auc'] is None:
            claim = '尚未证明：至少一组无法检验，不判断一致性'
        elif a['delta_auc'] * b['delta_auc'] < 0:
            claim = '已验证：方向不稳定；禁止合并点估计代替结论'
        elif a['status'] == b['status'] == 'INCREMENTAL_ASSOCIATION_CONFIRMED':
            claim = '已验证：两组均正增量关联，仅限这两个模型/案例'
        else:
            claim = '点估计同向；至少一组未确认增量'
        agreement[target] = claim
        lines.append('| %s | %s | %s | %s |' % (TARGETS[target], number(a['delta_auc']), number(b['delta_auc']), claim))
    lines.extend(['', '没有合并两 seed 的 ΔAUC。相同起始案例不意味着相同轨迹；两个策略执行后采样的状态、危险标签和轨迹长度均可能不同。',
        '因此旧/新 AUC 差异不直接归因于方差训练变化；需要固定观测序列的对照才能隔离这一点。', '',
        '### 三项强制自检', '', '| 检查 | seed42 | seed43 | 依据 |', '|---|---|---|---|'])
    geom = {tag: nested[tag]['marginals']['future-4-clearance-0.2']['current_clearance']['test_auc'] for tag in TAGS}
    lines.append('| GRID及副本 | 通过：80 / 指定项目 | 通过：80 / 指定项目 | 每次 configure 后 init_grid_from_cfg，CUDA设备名硬断言 RTX3060 |')
    lines.append('| 结局与步数复现 | 通过：192/192 | 通过：192/192 | 同 case/seed 对照各权重 benchmark-10000.csv；不是用户猜测的 benchmark-3000.csv |')
    lines.append('| 标签、描述子、边际口径 | 通过 | 通过 | 原 B1 碰撞case确定性回放：labels/valid完全一致，descriptors误差0；新NPZ键与dtype和旧NPZ一致 |')
    lines.append('')
    lines.append('几何边际 AUC：旧 %.4f，修复后 seed42 %.4f（差 %+.4f），seed43 %.4f（差 %+.4f）。' %
                 (geom['old'], geom['seed42'], geom['seed42']-geom['old'], geom['seed43'], geom['seed43']-geom['old']))
    assert all(abs(geom[tag]-geom['old']) <= .10 for tag in ('seed42', 'seed43')), 'Large geometry shift requires alignment audit'
    lines.extend(['均未触发事前定义的绝对变化>0.10复核标记；旧边际精确复现和标签fixture也支持口径一致。变化非零可来自策略导致的轨迹分布变化，不能自动判成代码错。', '',
        '| 数据 | 步数 | 成功/碰撞/超时（192案例） | 完整重采墙钟 s |', '|---|---:|---|---:|'])
    for seed, s in samples.items():
        c = s['outcome_counts']
        lines.append('| 修复后 seed%d | %d | %d / %d / %d | %.2f |' % (seed, s['steps'], c.get('success',0), c.get('collision',0), c.get('timeout',0), s['seconds']))
    lines.extend(['', '这些192案例的结局仅用于采样验收，不是新增独立 SR benchmark。', '',
        '### 五个边际读法并列', '', '这里只报告边际 AUC，不拿它代替上面的嵌套比较。方向仍只在训练集确定；逐维线性探针保持原实现。', '',
        '| 目标 | 边际读法 | 修复前 | 修复后42 | 修复后43 |', '|---|---|---:|---:|---:|'])
    for target in TARGETS:
        for metric in ('sigma_mean','sigma_max','sigma_linear','current_clearance','history_age'):
            values = [nested[tag]['marginals'][target].get(metric, {}).get('test_auc') for tag in TAGS]
            lines.append('| %s | %s | %s | %s | %s |' % (TARGETS[target], metric, *[number(v) for v in values]))
    lines.extend(['', '### 有效样本与正则化选择', '', '| 数据/目标 | train/test步数 | train/test正例步 | train/test正例episode | 选定C：A/B | 可用bootstrap/2000 |', '|---|---|---|---|---|---|'])
    for tag in TAGS:
        for target, r in nested[tag]['results'].items():
            fitting = r.get('fitting')
            cs = '不可选/检验不足' if not fitting else '%g / %g' % (fitting['A']['selected_C'], fitting['B']['selected_C'])
            lines.append('| %s / %s | %d / %d | %d / %d | %d / %d | %s | %s |' %
                (NAMES[tag], TARGETS[target], r['train_steps'], r['test_steps'], r['train_positives'], r['test_positives'],
                 r['train_positive_episodes'], r['test_positive_episodes'], cs, r.get('valid_draws', '不可算')))
    lines.extend(['', '## 四、任务3：超时性质，修复后三臂×两seed', '',
        '使用已有六份最终3000-case CSV，不使用修复前 codex3000 替代。所有失败保留；没有延长时限或重跑超时案例。',
        '训练配置 time_limit=50，测试配置=25 的差异已核实；原生在计时边界判定，CSV 的超时一般为101步/25.25秒。本轮不改变该规则。',
        'goal_progress_efficiency=净目标距离缩短量/实际路径；stall为2秒窗口沿初始目标方向前进<0.2m；它不是完全静止的比例。',
        '事前分组：卡住/低效=效率≤0.2或stall≥0.5或后退≥0.4；慢推进=效率≥0.5、stall≤0.2、后退≤0.2、且路径≥同模型同seed同场景成功路径中位数；其余混合/未定。',
        '这些阈值是粗描述，不是已校准的行为分类器。慢推进比例不是50秒可救回上界，卡住比例也不是“延长时间必然无效”。', '',
        '### 全部超时的类别比例', '', '| 模型 | seed | 超时n | 卡住/低效n (%) | 慢推进n (%) | 混合/未定n (%) |', '|---|---:|---:|---|---|---|'])
    for key, result in timeout['results'].items():
        arm, seed = key.split('-seed')
        g = result['groups']['all']
        bins = [g['timeout_bins'][k] for k in ('stuck_like','forward_like','mixed_unresolved')]
        lines.append('| %s | %s | %d | %s | %s | %s |' % (ARMS[arm], seed, g['by_outcome']['timeout']['n'],
                      *['%d (%.2f%%)' % (v['n'], 100*v['fraction']) for v in bins]))
    lines.extend(['', '复杂seed43：347/510（68.04%）为卡住/低效，2/510（0.39%）为慢推进，161/510（31.57%）混合/未定。',
        '不能把其大量超时统一解释成“走得正常，仅25秒不够”；也不能根据这些汇总指标证明未来50秒必然救不回来。', '',
        '### success/timeout整体字段分布', '', '格式：中位数 [p25, p75]。S=success，T=timeout；各自案例集合不同，只作描述。', '',
        '| 模型/seed/结局/n | 进度效率 | stall | 振荡stall | 后退 | 路径m | 时长s | 步数 | 最小净距m | 平均净距m |',
        '|---|---|---|---|---|---|---|---|---|---|'])
    def distribution_rows(scope):
        for key, result in timeout['results'].items():
            arm, seed = key.split('-seed')
            for o, label in (('success','S'), ('timeout','T')):
                row = result['groups'][scope]['by_outcome'][o]
                lines.append('| %s/%s/%s/%d | %s |' % (ARMS[arm], seed, label, row['n'],
                             ' | '.join(quantile(row['fields'][k]) for k in diagnostic.FIELDS)))
    distribution_rows('all')
    lines.extend(['', '### 六场景分解', '', '下表的效率、stall、后退均为timeout中位数；其对应success中位数一并展示，卡住/慢推进/混合为预注册分类计数。', '',
                  '| 模型/seed | 场景 | S/T n | 效率 S/T | stall S/T | 后退 S/T | T类别：卡住/慢推进/混合 |', '|---|---|---|---|---|---|---|'])
    for key, result in timeout['results'].items():
        arm, seed = key.split('-seed')
        for scene, g in result['groups'].items():
            if scene == 'all':
                continue
            ss, tt = [g['by_outcome'][o] for o in ('success','timeout')]
            medians = ['%s / %s' % tuple('无数据' if x['fields'][f] is None else '%.3f' % x['fields'][f]['median'] for x in (ss,tt))
                       for f in ('goal_progress_efficiency','stalled_window_ratio','backtracking_ratio')]
            counts = [g['timeout_bins'][k]['n'] for k in ('stuck_like','forward_like','mixed_unresolved')]
            lines.append('| %s/%s | %s | %d/%d | %s | %s | %s | %d/%d/%d |' %
                         (ARMS[arm],seed,scene,ss['n'],tt['n'],*medians,*counts))
    for scene in ('dense_square','large_square'):
        lines.extend(['', '### ' + scene + '：全部字段四分位数', '',
              '| 模型/seed/结局/n | 进度效率 | stall | 振荡stall | 后退 | 路径m | 时长s | 步数 | 最小净距m | 平均净距m |',
              '|---|---|---|---|---|---|---|---|---|---|'])
        distribution_rows(scene)
    lines.extend(['', '时限假设判断：**仅凭“训练50秒/测试25秒”解释整个退步，证据不支持；大规模延时重跑没有被这轮描述数据充分支持。**',
        '尤其复杂seed43的dense_square、large_square超时分别有186/266、85/129为卡住/低效；两格慢推进计数均为0，不能拿较长路径自动说是在有效赶路。',
        '混合组仍留有不确定性，不能把跨阈值规则当成因果排除。是否存在少量25秒截断但50秒能成功的案例，本轮没有验证。', '',
        '## 五、证据状态与工程记录', '',
        '- 已验证：全部样本/配置/来源约束、两组192回合结局和步数复现、标签和描述子fixture、5项诊断单测、旧边际AUC复现、全部离线分布。',
        '- 尚未证明：无法计算目标的增量、CI跨零目标的增量、概率校准、因果动作收益、全部当前几何之外的新信息、50秒下可恢复成功率。',
        '- 已确认失败（工程而非科学）：seed42统计曾早于NPZ写出而启动，FileNotFoundError在拟合前发生；原日志保留，采样完成后用同一冻结脚本运行成功，不计为科学观察。',
        '- 宿主SciPy较旧，出现NumPy别名/ABI警告；没有计算异常或收敛失败。旧边际复现和独立ROC配对自检通过，不把警告隐藏成无风险。',
        '- inherited mamba_rl文件有可选mamba_ssm导入警告；实际硬断言backbone=bayes，没有运行Mamba或安装新依赖。',
        '- 原B1 test碰撞正例episode为2，本轮按数组核实，修正此前“只有1个”的文字判断；不改任何原始结果。', '',
        '## 六、下一轮只需要的对照，不提出算法改动', '',
        '| 待分辨问题 | 所需对照 | 本轮不能替代的证据 |', '|---|---|---|',
        '| 新方差学习还是不同轨迹分布 | 固定同一合法观测序列，用旧/新滤波器分别读出sigma，沿用同一label/split | 同case不同策略的闭环AUC不能隔离学习变化 |',
        '| σ增量是否只是两列几何的非线性表达 | 在相同episode划分、训练内选正则化下，加入强非线性几何基线与容量匹配对照 | 当前A只有两个线性描述子，不能代表所有simple explanation |',
        '| 状态相关方差还是固定剖面 | 保留模型权重/均值的固定逐维方差、合法分层置乱方差对照，单独记录分布外风险 | AUC正增量不等于价值头实际利用它改善动作 |',
        '| 极少碰撞事件是否可靠 | 独立预注册、事件充分的自然测试案例，不改变当前划分来补正例 | 单类别/2个正例episode不能靠frame数量扩大功效 |',
        '| 25秒截断能否恢复少量案例 | 仅若明天决定验证，固定现有模型及case做25/50秒成对续跑，0–25秒prefix必须一致并保留全部失败 | 本轮CSV不含final_distance/return，不能推算恢复SR |', '',
        '本轮未提出或执行 KL、预测监督、readout初始化修复、reward/filter/smoothing/动作网格/T 改动；不训练任何臂。', '',
        '## 七、可复核文件', '',
        '冻结协议与全部原始/统计结果：', str(BASE), '',
        '诊断：scripts/diagnose-bayes-variance.py；报告生成：scripts/report-bayes-diagnostics.py；单测：tests/test_bayes_diagnostics.py。',
        '冻结提交：cc995bf。报告和结果的最终提交号以Git历史为准。protocol.json包含全部输入SHA，统计JSON保存C选择、折AUC、split IDs和bootstrap可用次数。',
        '逐回合原始sigma、描述子、净距、动作与位置保存为episode-traces-seed42/43.npz，并有对应JSON；384回合逐数组核对，无损合并。原缓存仍保留本地，不另上传重复缓存。',
        'timeout-analysis.json包含三结局×六场景×全部九个字段的完整四分位数；timeout-distributions.csv提供扁平表，不遗漏非重点场景或collision。', ''])
    path = ROOT / 'BAYES-VARIANCE-TIMEOUT-DIAGNOSTIC.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    diagnostic.save(BASE / 'diagnostic-summary.json', {'status': 'COMPLETE', 'at_kst': now,
         'two_seed_agreement': agreement, 'geometry_marginal_auc': geom,
         'self_checks': {'GRID': 80, 'episode_parity': {'42': 192, '43': 192},
                         'old_label_descriptor_fixture': parity, 'statistical_tests': '5 passed'},
         'algorithm_changes': False, 'training_protocol_changes': False, 'training_runs': 0, 'remote_4090_access': False})
    with (BASE / 'timeout-distributions.csv').open('w', newline='') as stream:
        fields = ['model', 'scope', 'outcome', 'n'] + [f + '_' + q for f in diagnostic.FIELDS for q in ('p25','median','p75')]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, result in timeout['results'].items():
            for scope, group in result['groups'].items():
                for outcome, z in group['by_outcome'].items():
                    row = {'model': key, 'scope': scope, 'outcome': outcome, 'n': z['n']}
                    row.update({f + '_' + q: None if z['fields'][f] is None else z['fields'][f][q]
                                for f in diagnostic.FIELDS for q in ('p25','median','p75')})
                    writer.writerow(row)
    print(path)


if __name__ == '__main__':
    main()
