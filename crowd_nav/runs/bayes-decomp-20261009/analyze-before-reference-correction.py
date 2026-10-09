"""Episode-cluster inference and a single evidence-grounded handoff."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import json
import hashlib
import collections
from pathlib import Path

import numpy as np
from scipy.stats import binom_test

BASE = Path(__file__).resolve().parent
CONDITIONS = ('fixed', 'shuffle_within', 'ctrl_shuffle_mean')


def load(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def metrics(rows, condition, key):
    by_episode = collections.defaultdict(lambda: np.zeros(5, dtype=np.float64))
    for row in rows:
        ep = (row['training_seed'], row['scene_index'], row['case_index'])
        bucket = by_episode[ep]
        bucket[0] += 1
        item = row['conditions'][condition]
        if item['flip']:
            bucket[1] += 1
            if key is not None:
                delta = item[key + '_delta_real']
                bucket[2] += int(delta > 1e-10)
                bucket[3] += int(delta < -1e-10)
                bucket[4] += delta
    data = np.asarray(list(by_episode.values()))
    if not len(data):
        return {'steps': 0, 'flips': 0, 'flip_rate': None, 'episodes': 0}
    totals = data.sum(0)
    n, flips, wins, losses, delta = [float(t) for t in totals]
    result = {'steps': int(n), 'flips': int(flips), 'episodes': len(data),
              'flip_rate': flips/n, 'real_wins': int(wins), 'intervention_wins': int(losses),
              'ties': int(flips-wins-losses),
              'real_win_fraction_nonties': wins/(wins+losses) if wins+losses else None,
              'real_win_fraction_all_flips': wins/flips if flips else None,
              'mean_delta_real_all_flips': delta/flips if flips else None,
              'flipped_episodes': int((data[:, 1] > 0).sum()),
              'largest_episode_flip_fraction': float(data[:, 1].max()/flips) if flips else None}
    if key and wins+losses:
        p = float(binom_test(int(wins), int(wins+losses), p=.5, alternative='two-sided'))
        result.update(binomial_p_descriptive=p, binomial_p_bonferroni8=min(1., 8*p))
    rng = np.random.default_rng(20261009)
    sums = np.asarray([data[rng.integers(0, len(data), len(data))].sum(0) for _ in range(2000)])
    result['flip_rate_episode_bootstrap_95_ci'] = np.quantile(sums[:, 1]/sums[:, 0], [.025, .975]).tolist()
    usable = sums[:, 2]+sums[:, 3] > 0
    result['real_win_episode_bootstrap_95_ci'] = (
        np.quantile(sums[usable, 2]/(sums[usable, 2]+sums[usable, 3]), [.025, .975]).tolist()
        if usable.any() and key else None)
    usable_delta = sums[:, 1] > 0
    result['mean_delta_episode_bootstrap_95_ci'] = (
        np.quantile(sums[usable_delta, 4]/sums[usable_delta, 1], [.025, .975]).tolist()
        if usable_delta.any() and key else None)
    result['bootstrap_single_class_or_no_nonties_draws'] = int((~usable).sum()) if key else None
    return result


def subset(rows, window):
    if window == 'all':
        return rows
    if window == 'success_all':
        return [r for r in rows if r['outcome'] == 'success']
    if window == 'success_last8':
        return [r for r in rows if r['outcome'] == 'success' and r['episode_steps']-r['tick'] <= 8]
    outcome, ending = window.split('_pre')
    return [r for r in rows if r['outcome'] == outcome and r['episode_steps']-r['tick'] <= int(ending)]


def summary(rows):
    result = {}
    for condition in CONDITIONS:
        result[condition] = {'flip': metrics(rows, condition, None),
                             'clearance': metrics(rows, condition, 'clearance'),
                             'reward': metrics(rows, condition, 'reward'),
                             'replacement_changed_steps': sum(r['conditions'][condition]['input_changed'] for r in rows),
                             'median_max_value_change': float(np.median(
                                 [r['conditions'][condition]['max_abs_value_change'] for r in rows])) if rows else None,
                             'pre_filter_flip_count': sum(r['conditions'][condition]['pre_safety_idx'] != r['pre_safety_real_idx'] for r in rows)}
    return result


def f(value, precision=4):
    return '不可算' if value is None else ('%.'+str(precision)+'f') % value


def ci(value):
    return '不可算' if value is None else '['+f(value[0])+', '+f(value[1])+']'


def main():
    import sys
    sys.path.insert(0, str(BASE))
    from decompose import pinned
    pinned()
    protocol = load(BASE / 'protocol.json')
    controls = load(BASE / 'controls.json')
    complete = load(BASE / 'stepwise/run-complete.json')
    assert controls['smoke_passed'] and complete['deterministic_flips'] == 0
    rows = []
    for path in sorted((BASE / 'stepwise').glob('seed*-scene*-case*.json')):
        obj = load(path)
        assert obj['deterministic']
        rows.extend(obj['rows'])
    assert len({(r['training_seed'], r['scene_index'], r['case_index']) for r in rows}) == 1200
    seeds, strata = {}, {}
    for seed in (42, 43):
        selected = [r for r in rows if r['training_seed'] == seed]
        assert len(selected) == complete['counts'][str(seed)]['steps']
        seeds[str(seed)] = summary(selected)
        assert seeds[str(seed)]['ctrl_shuffle_mean']['flip']['flip_rate'] >= .01
        for window in protocol['metrics']['windows']:
            pool = subset(selected, window)
            strata['seed%d/%s' % (seed, window)] = summary(pool)
        for scene in sorted(set(r['scene'] for r in selected)):
            strata['seed%d/scene/%s' % (seed, scene)] = summary([r for r in selected if r['scene'] == scene])
        for count in (5, 10, 12, 20):
            strata['seed%d/humans/%d' % (seed, count)] = summary([r for r in selected if r['human_count'] == count])
    favorable = []
    for condition in CONDITIONS[:2]:
        consistent = True
        for seed in ('42', '43'):
            clearance = seeds[seed][condition]['clearance']
            reward = seeds[seed][condition]['reward']
            cci = clearance['real_win_episode_bootstrap_95_ci']
            rci = reward['mean_delta_episode_bootstrap_95_ci']
            consistent &= bool(cci is not None and cci[0] > .5
                               and clearance.get('binomial_p_bonferroni8', 1.) < .05
                               and (rci is None or rci[0] >= -1e-10))
        if consistent:
            favorable.append(condition)
    verdict = 'REAL_DIRECTION_SUPPORTED_LOCAL_ONLY' if favorable else 'REAL_DIRECTION_NOT_ESTABLISHED'
    result = {'version': protocol['version'], 'status': 'COMPLETE', 'verdict': verdict,
              'next_route': 'CLOSED_LOOP_CONFIRMATION' if favorable else 'INFORMATION_LAYER',
              'favorable_conditions': favorable, 'episodes': 1200, 'steps': len(rows),
              'seeds': seeds, 'overall_descriptive': summary(rows), 'strata': strata,
              'controls': {'smoke_passed': True, 'real_replay_flips': 0, 'positive_both_seeds_passed': True,
                           'benchmark_outcome_step_parity': True, 'native_reward_parity': True},
              'source_hashes_reverified': True, 'algorithm_changes': False, 'training_runs': 0,
              '4090_access': False, 'original_project_writes': False,
              'scope': 'Same-root immediate action sensitivity and all-human immediate geometry/reward only, not long-term task value or Full-vs-Mean.'}
    save(BASE / 'stepwise/summary.json', result)
    lines = ['# 贝叶斯方差决策层拆解', '', '版本：'+protocol['version'], '',
             '## 现象', '', '本轮已完成两个训练 seed × 六格 ×100 episode，共1200条自然轨迹、%d个决策步。' % len(rows),
             '仅本地3060推理与CPU统计；零训练、零4090访问、原项目未修改。只有运行期 readout 输入干预。', '',
             '最终判断：'+verdict+'。下一步唯一分支：'+result['next_route']+'。', '',
             '这不是完整贝叶斯与均值臂的比较，也不是新方法或长期闭环收益确认。', '',
             '## 问题与冻结口径', '',
             '- fixed：每个训练seed独立使用六格case400–415的全部候选logvar逐维均值；正式测试case0–99，完全不重叠。',
             '- shuffle_within：每个决策置换80个完整logvar向量，mean不变；ctrl_shuffle_mean只置换mean作为阳性对照。',
             '- real真实执行；干预只评估同一root影子动作，不写历史。所有reward、安全过滤、80动作、T=24、25s时限、0.3动作平滑保持原设置。',
             '- 主几何指标：沿用同一前一执行动作和平滑系数后，对全体行人实际下一位置计算终点净距。人的动作在机器人本步动作执行前生成，原robot.visible=false，因此同一root影子动作可共享实际人的下一位置。',
             '- 原生即时reward：使用母体的扫掠几何与timeout/collision/goal优先级，逐步与真实env.step奖励校验；另外保存候选CV终点dmins，三者不混称。',
             '- 翻转步二项检验按要求保留，但同episode步相关，不能当独立样本；同时给出2000次episode聚类bootstrap。8项主比较另报Bonferroni二项p。分格与终止窗口为描述性分析。', '',
             '## 对照验收', '',
             '| 检查 | 状态 | 证据 |', '|---|---|---|',
             '| GRID=80、导入来源、权重严格加载 | 已验证通过 | 每个环境与模型均断言；SHA启动和结束复验 |',
             '| 6格×3ep×2seed smoke | 已验证通过 | controls.json与smoke逐episode原始记录 |',
             '| real完整重放两遍 | 已验证通过 | 1200/1200；值、输入、动作、位置、reward、净距逐位一致，翻转0 |',
             '| 既有benchmark结局/步数 | 已验证通过 | 校准、smoke及正式样本均逐episode断言 |',
             '| 阳性对照 | 已验证通过 | 两seed各自mean-shuffle翻转≥1% |',
             '| 替换实际生效 | 已验证通过 | 非替换通道逐位不变；置换逐维边际完全相同；首次前后张量摘要及逐步value变化已保存 |',
             '| 原生奖励与执行语义 | 已验证通过 | 实际平滑动作一致；counterfactual真实动作reward误差<1e-10 |', '',
             '## 全部步结果', '',
             '| seed | 干预 | 翻转/步 | 翻转率 | 净距real胜/干预胜/平局 | real胜率(非平局) | episode区间 | reward real胜/干预胜/平局 |',
             '|---|---|---:|---:|---|---:|---|---|']
    for seed, output in seeds.items():
        for condition in CONDITIONS:
            a, c, r = output[condition]['flip'], output[condition]['clearance'], output[condition]['reward']
            lines.append('| %s | %s | %d/%d | %.3f%% | %d/%d/%d | %s | %s | %d/%d/%d |' %
                (seed, condition, a['flips'], a['steps'], a['flip_rate']*100,
                 c['real_wins'], c['intervention_wins'], c['ties'], f(c['real_win_fraction_nonties']),
                 ci(c['real_win_episode_bootstrap_95_ci']), r['real_wins'], r['intervention_wins'], r['ties']))
    lines += ['', '### 即时收益统计', '',
              '| seed | 干预 | 指标 | real−干预平均差(翻转步) | episode区间 | 二项p(描述性) | Bonferroni8 p |',
              '|---|---|---|---:|---|---:|---:|']
    for seed, output in seeds.items():
        for condition in CONDITIONS:
            for key in ('clearance', 'reward'):
                r = output[condition][key]
                lines.append('| %s | %s | %s | %s | %s | %s | %s |' %
                             (seed, condition, key, f(r['mean_delta_real_all_flips'], 6),
                              ci(r['mean_delta_episode_bootstrap_95_ci']), f(r.get('binomial_p_descriptive'), 7),
                              f(r.get('binomial_p_bonferroni8'), 7)))
    for heading, category in [('按场景', '/scene/'), ('按人数', '/humans/'), ('失败前窗口与成功参照', None)]:
        lines += ['', '### '+heading, '', '| 分层 | 干预 | 步数 | 翻转 | 翻转率 | 净距real胜率 | reward real胜率 |',
                  '|---|---|---:|---:|---:|---:|---:|']
        for name, output in strata.items():
            if category is not None and category not in name:
                continue
            if category is None and ('/scene/' in name or '/humans/' in name):
                continue
            for condition in CONDITIONS:
                a, c, r = output[condition]['flip'], output[condition]['clearance'], output[condition]['reward']
                lines.append('| %s | %s | %d | %d | %s | %s | %s |' %
                             (name, condition, a['steps'], a['flips'], f(a['flip_rate']),
                              f(c.get('real_win_fraction_nonties')), f(r.get('real_win_fraction_nonties'))))
    lines += ['', '## 证据限制与工程记录', '',
              '- 已验证通过：以上均是冻结Full权重内的同状态输入干预；两seed单独报告，不用合并数字掩盖方向不一致。',
              '- 未验证：后验校准、方差独立于全部几何信息、长时域任务价值、碰撞率/SR增益、Full vs Mean、跨新训练seed外推。',
              '- 干预不是“关掉方差学习”：固定值同时移除跨状态和候选内变化；置换保留边际，却破坏mean/logvar/候选的联合对应，可能分布外。不能自动等同于不确定性因果效益。',
              '- 净距更大不等于任务更好；原reward的progress/time系数为0，大量reward平局是该冻结任务的属性，不应强行制造额外奖励来区分。',
              '- 碰撞/超时前窗口按real结局分层，干预没有真实闭环结局；成功轨迹全保留，不据失败子集宣布总体效果。',
              '- 已确认失败：首次校准包装器无条件调用不存在的set_env，在首条轨迹前停止，无科学数据；按母体既有可选接口检查修复，旧代码/日志/manifest保留。',
              '- 已确认失败：保存修复冻结时遇到Python3.8相对__file__键导致KeyError，在改写manifest前停止；路径规范化后修复，无科学数据。',
              '- 旧模块导入会打印缺失mamba_ssm警告；实际backbone硬断言bayes，没有运行Mamba。', '',
              '## 可选方案与推荐', '']
    if favorable:
        lines += ['推荐唯一下一步：冻结本轮版本，做独立闭环确认。'+', '.join(favorable)+'在两个seed上均支持real即时净距方向，reward未显示伤害。',
                  '仍须同时比较任务成本、进度、接触、超时；不能把本轮几何偏好直接写成性能提升。']
    else:
        lines += ['推荐唯一下一步：信息层检验，不进入闭环收益确认、不重训。',
                  '本轮未满足预注册的跨seed方向有利标准；这不等于证明方差没有信息。需区分候选差异是否有新增信息与消费者如何利用。',
                  '既有b1-rollout.npz来自修复前滤波器；旧/修复后嵌套检验已经完成，应先引用保存结果，不重复旧数据检验冒充新证据。']
    lines += ['', '## 资产', '', 'protocol.json / frozen-manifest.json / frozen-manifest-initial.json',
              'controls.json / fixed-seed42.json / fixed-seed43.json',
              'stepwise/summary.json / stepwise/seed*-scene*-case*.json / 同名.npz',
              'calibration.log / calibration-retry.log / smoke.log / full.log',
              'engineering-preflight.json / engineering-manifest-key.json',
              '权重只读引用原备份，不上传、不重新复制到4090。', '',
              '实际计时：校准seed42 %.1fs、seed43 %.1fs；smoke %.1fs；正式双重重放 %.1fs。' %
              (load(BASE / 'fixed-seed42.json')['seconds'], load(BASE / 'fixed-seed43.json')['seconds'],
               controls['seconds'], complete['seconds']), '']
    (BASE / 'REPORT.md').write_text('\n'.join(lines))
    save(BASE / 'analysis-manifest.json', {'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'report_sha256': hashlib.sha256((BASE / 'REPORT.md').read_bytes()).hexdigest(),
         'summary_sha256': hashlib.sha256((BASE / 'stepwise/summary.json').read_bytes()).hexdigest()})
    print(json.dumps({'status': 'COMPLETE', 'verdict': verdict, 'next_route': result['next_route'],
                      'episodes': 1200, 'steps': len(rows), 'seeds': seeds}, indent=2), flush=True)


if __name__ == '__main__':
    main()
