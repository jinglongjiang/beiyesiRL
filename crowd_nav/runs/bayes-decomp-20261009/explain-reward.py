"""Post-hoc reward-magnitude explanation; does not change any verdict or metric."""
import collections
import hashlib
import json
from pathlib import Path

base = Path(__file__).resolve().parent
result = {}
for seed in (42, 43):
    for condition in ('fixed', 'shuffle_within'):
        contributions = collections.defaultdict(lambda: {'n': 0, 'positive': 0, 'negative': 0, 'sum_delta': 0.})
        flips = 0
        for path in sorted((base / 'stepwise').glob('seed%d-scene*-case*.json' % seed)):
            for row in json.loads(path.read_text())['rows']:
                item = row['conditions'][condition]
                if not item['flip']:
                    continue
                flips += 1
                delta = item['reward_delta_real']
                if abs(delta) <= 1e-10:
                    group = 'tie'
                elif row['real_reward'] == 1 or item['reward'] == 1:
                    group = 'terminal_goal_difference'
                elif row['real_swept_clearance'] < 0 or item['swept_clearance'] < 0:
                    group = 'native_collision_difference'
                else:
                    group = 'discomfort_difference'
                bucket = contributions[group]
                bucket['n'] += 1
                bucket['positive'] += int(delta > 1e-10)
                bucket['negative'] += int(delta < -1e-10)
                bucket['sum_delta'] += delta
        result['seed%d/%s' % (seed, condition)] = {'flips': flips, 'groups': dict(contributions)}
summary = json.loads((base / 'stepwise/summary.json').read_text())
for key, row in result.items():
    seed, condition = key.replace('seed', '').split('/')
    observed = sum(g['sum_delta'] for g in row['groups'].values()) / row['flips']
    assert abs(observed-summary['seeds'][seed][condition]['reward']['mean_delta_real_all_flips']) < 1e-12
data = {'status': 'POST_HOC_DESCRIPTIVE_ONLY', 'results': result,
        'verdict_unchanged': summary['verdict'], 'purpose': 'Explain sign-frequency versus magnitude disagreement, not a new success criterion'}
(base / 'reward-explanation.json').write_text(json.dumps(data, indent=2)+'\n')
report = base / 'REPORT.md'
lines = ['\n## 结果解读：参与决策不等于稳定占优\n',
         '- 已验证通过：fixed翻转26.10%/33.90%；候选内logvar置换翻转64.27%/67.75%，阳性对照翻转97.11%/96.79%。方差的直接读出通道并非未被消费。',
         '- 已确认失败（限定指标）：候选内置换时，real即时净距胜率40.62%/41.37%，episode区间均低于50%；平均净距差为−12.9/−18.7mm。',
         '- 未验证：稳定净收益。fixed的净距方向跨seed翻转；shuffle中real平均即时reward较高，但非平局reward胜率仅40.12%/27.03%，存在频率与损失幅度的冲突，不能写成总体导航获益或总体导航受损。',
         '- 以下按已冻结原reward中的终止/碰撞/不适事件做事后解释，不参与改写预注册结论。较少的大额正差可以压过较多的小额负差；终点奖励差也不等于后续整段成功率差。\n',
         '| seed/干预 | 奖励差类别 | real胜 | 干预胜 | 累计real−干预reward |', '|---|---|---:|---:|---:|']
for key, row in result.items():
    for group in ('terminal_goal_difference', 'native_collision_difference', 'discomfort_difference'):
        bucket = row['groups'].get(group, {'positive': 0, 'negative': 0, 'sum_delta': 0})
        lines.append('| %s | %s | %d | %d | %+.6f |' %
                     (key, group, bucket['positive'], bucket['negative'], bucket['sum_delta']))
lines += ['', '推荐仍为信息层，不直接启动闭环确认或匹配重训。已有当前状态sigma嵌套检验有正增量关联；剩余问题是候选间logvar差异是否携带与动作后果相关的增量，以及消费者为何呈现上述权衡。'
          '本轮没有检验这个剩余问题，也没有对Full vs Mean下结论。', '']
assert '## 结果解读：参与决策不等于稳定占优' not in report.read_text()
with report.open('a') as stream:
    stream.write('\n'.join(lines))
manifest_path = base / 'analysis-manifest.json'
manifest = json.loads(manifest_path.read_text())
manifest['report_before_posthoc_sha256'] = manifest['report_sha256']
manifest['report_sha256'] = hashlib.sha256(report.read_bytes()).hexdigest()
manifest['posthoc_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
final_path = base / 'final-manifest.json'
final = json.loads(final_path.read_text())
final['files'] = {str(p.relative_to(base)): {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
                  for p in sorted(base.rglob('*')) if p.is_file() and p != final_path}
final_path.write_text(json.dumps(final, indent=2)+'\n')
print(json.dumps(data, indent=2))
