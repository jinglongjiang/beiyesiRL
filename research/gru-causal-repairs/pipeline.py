"""Sequential bounded evaluator. Stop on any nonzero child; never sweep parameters."""
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = sys.executable
ARMS = ['nominal', 'behavior-mc', 'deploy-mc', 'progress', 'commit8', 'adaptive']


def execute(script, *args):
    label = '-'.join([Path(script).stem] + list(args)).replace('--', '')
    start = time.perf_counter()
    print('START', label, flush=True)
    with (HERE / (label + '.log')).open('w') as handle:
        subprocess.run([PY, str(HERE / script), *args], cwd=ROOT, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)
    stages.append({'stage': label, 'seconds': time.perf_counter() - start, 'exit_code': 0})
    (HERE / 'pipeline-state.json').write_text(json.dumps(stages, indent=2) + '\n')
    print('DONE', label, round(stages[-1]['seconds'], 1), flush=True)


stages = []
if __name__ == '__main__':
    try:
        assert (HERE / 'training.json').exists(), 'Training must finish successfully first'
        execute('run.py', 'probe')
        execute('analyze.py', 'ranking')
        for cohort in ['roots', 'fresh']:
            for arm in ARMS:
                execute('run.py', 'evaluate', '--cohort', cohort, '--arm', arm)
            execute('analyze.py', cohort)
        summary = json.loads((HERE / 'fresh-summary.json').read_text())['overall']
        ranking = json.loads((HERE / 'ranking-summary.json').read_text())
        a_pass = (summary['deploy-mc']['success'] > max(summary['behavior-mc']['success'], summary['nominal']['success'])
                  and summary['deploy-mc']['contact'] <= min(summary['behavior-mc']['contact'], summary['nominal']['contact'])
                  and ranking['deploy-mc']['A10']['first_then_original_safe_success'] > ranking['behavior-mc']['A10']['first_then_original_safe_success'])
        (HERE / 'A-source-repair-gate.json').write_text(json.dumps({'passed': a_pass,
            'conditional_coverage_required': not a_pass,
            'meaning': 'Development repair gate, not method-entry confirmation or original historical-cause proof'}, indent=2) + '\n')
        if not a_pass:
            execute('coverage.py', 'train')
            for cohort in ['roots', 'fresh']:
                arms = ['coverage-control', 'targeted-head', 'targeted-full']
                if cohort == 'fresh':
                    arms = ['nominal'] + arms
                for arm in arms:
                    execute('coverage.py', 'evaluate', '--cohort', cohort, '--arm', arm)
        (HERE / 'pipeline-complete.json').write_text(json.dumps({'status': 'COMPLETED', 'stages': stages}, indent=2) + '\n')
    except Exception as exc:
        (HERE / 'pipeline-failure.json').write_text(json.dumps({'status': 'STOPPED', 'error': repr(exc), 'completed_stages': stages}, indent=2) + '\n')
        raise
