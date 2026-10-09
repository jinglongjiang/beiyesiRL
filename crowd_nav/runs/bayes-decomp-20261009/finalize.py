"""Verify complete artifacts and append reused evidence, without changing inference."""
import sys
sys.dont_write_bytecode = True
import hashlib
import json
from pathlib import Path
import numpy as np

base = Path(__file__).resolve().parent
sys.path.insert(0, str(base))
from decompose import pinned

pinned()
analysis = json.loads((base / 'analysis-freeze.json').read_text())
assert hashlib.sha256((base / analysis['file']).read_bytes()).hexdigest() == analysis['sha256']
summary = json.loads((base / 'stepwise/summary.json').read_text())
assert summary['status'] == 'COMPLETE' and summary['episodes'] == 1200
assert (base / 'REPORT.md').exists()
steps, episodes = 0, 0
for path in sorted((base / 'stepwise').glob('seed*-scene*-case*.npz')):
    data = np.load(path)
    meta = json.loads(path.with_suffix('.json').read_text())
    n = len(meta['rows'])
    assert data['values'].shape == (n, 4, 80)
    for name in data.files:
        assert np.isfinite(data[name]).all(), (path, name)
    assert n == meta['meta']['steps'] and meta['deterministic']
    for tick, row in enumerate(meta['rows']):
        assert 0 <= row['real_idx'] < 80
        assert abs(row['real_reward'] - data['all_candidate_reward'][tick, row['real_idx']]) < 1e-10
        for condition in ('fixed', 'shuffle_within', 'ctrl_shuffle_mean'):
            item = row['conditions'][condition]
            assert item['flip'] == (item['idx'] != row['real_idx'])
            assert abs(item['clearance'] - data['all_candidate_clearance'][tick, item['idx']]) < 1e-10
            if item['permutation'] is not None:
                assert sorted(item['permutation']) == list(range(80))
    steps += n
    episodes += 1
assert episodes == 1200 and steps == summary['steps']
pid = int((base / 'pids.txt').read_text().strip())
directory = Path('/proc/%d' % pid)
assert not directory.exists(), 'Experiment process still alive; do not finalize'
unitlog = (base / 'unit-tests.log').read_text()
assert 'Ran 3 tests' in unitlog and '\nOK\n' in unitlog
prior = json.loads((base / 'information-layer-existing.json').read_text())
extra = ['\n## 信息层已有证据：明确不是本轮新实验\n',
         '不重复运行已经完成的嵌套检验。旧b1来自修复前，两个修复后seed来自此前192个episode采集；以下只是引用保存结果。\n',
         '| 方差来源 | 目标 | ΔAUC | 95% CI | 结论 |', '|---|---|---:|---|---|']
for tag in ('old', 'seed42', 'seed43'):
    obj = prior['data'][tag]
    assert hashlib.sha256(Path(obj['source']).read_bytes()).hexdigest() == obj['source_sha256']
    for key in ('future-4-clearance-0.2', 'future-8-clearance-0.2'):
        r = obj['results'][key]
        lo, hi = r['delta_auc_95_ci']
        status = '已验证通过：两描述子线性基线之上的增量关联' if lo > 0 else '未验证：尚未证明有增量'
        extra.append('| %s | %s | %+.4f | [%+.4f, %+.4f] | %s |' % (tag, key, r['delta_auc'], lo, hi, status))
extra += ['', '这些检验说明某些当前状态sigma有增量关联，但不回答80候选之间logvar差异是否携带动作相关增量，也不证明消费者的方向有利。'
          '若本轮进入信息层，优先厘清这一剩余问题，不把旧b1重复计算包装成新进展。', '',
          '本轮只拆方差直接读出通道；上游滤波依然用variance计算Kalman gain并更新mean，所有干预都保留这条路径。不能把直接通道的结果推广成整个方差机制或完整贝叶斯的价值结论。', '',
          '## 最终保存验收', '',
          '- 已验证通过：3项独立合约单测；1200份逐episode JSON与NPZ完整，%d步、所有数组有限、动作索引与收益记录一致。' % steps,
          '- 已验证通过：原模型/环境/配置/权重源SHA全部复验；分析代码与预注册hash一致。',
          '- 已验证通过：PID %d在运行时由精确argv与/proc确认；完成后/proc已不存在，无pgrep自匹配。' % pid,
          '- 已确认失败：一次独立的进程监控内联命令发生换行转义SyntaxError，没有接触实验进程或数据；已保留engineering-monitor.json并用独立精确argv检查替换。',
          '- 本轮工程错误均单独记录，不作为科学样本；重启后的完整正式采集没有traceback、断言失败或非零退出。初次中断版未拼接进最终结果。', '']
report = base / 'REPORT.md'
assert '## 信息层已有证据' not in report.read_text(), 'Do not duplicate report append'
with report.open('a') as stream:
    stream.write('\n'.join(extra))
analysis_manifest = base / 'analysis-manifest.json'
stage = json.loads(analysis_manifest.read_text())
stage['report_before_append_sha256'] = stage['report_sha256']
stage['report_sha256'] = hashlib.sha256(report.read_bytes()).hexdigest()
stage['append_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
analysis_manifest.write_text(json.dumps(stage, indent=2) + '\n')
paths = sorted(p for p in base.rglob('*') if p.is_file() and p.name != 'final-manifest.json')
final = {'status': 'VERIFIED_COMPLETE', 'episodes': episodes, 'steps': steps,
         'running_full_process': False, 'weights_uploaded': False,
         'files': {str(p.relative_to(base)): {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                                            'bytes': p.stat().st_size} for p in paths}}
(base / 'final-manifest.json').write_text(json.dumps(final, indent=2) + '\n')
print(json.dumps({'status': final['status'], 'episodes': episodes, 'steps': steps,
                  'verdict': summary['verdict'], 'next_route': summary['next_route'], 'report': str(report)}))
