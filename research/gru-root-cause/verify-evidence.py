"""Check completed evidence contracts and write a reproducible hash manifest."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol=read(HERE/'protocol.json')
    old=HERE.parent/'recovery-duration'
    assert sha(old/'artifacts'/'nominal.pth')==protocol['nominal_sha256']
    baseline=read(HERE/'repaired-baseline.json')
    contrasts=read(HERE/'value-contrasts.json')
    assert len(baseline)==240 and len({(r['scene'],r['case']) for r in baseline})==240
    assert len(contrasts)==164 and all(r['repaired_policy_reaches_same_root'] for r in contrasts)
    assert read(HERE/'preflight.json')['passed']
    for variant in ['deadline50','no-smoothing','no-filter','explore']:
        data=read(HERE/f'upstream-{variant}.json')
        assert len(data)==164*(4 if variant=='explore' else 1)
        assert len({(r['scene'],r['case']) for r in data})==164
    manifests=[]
    for arm in ['control','matched-smoothing']:
        trained=HERE/'artifacts'/'causal-training'/arm
        manifest=read(trained/'manifest.json')
        assert manifest['complete'] and manifest['episodes']==2000 and manifest['updates']==8000
        assert sha(trained/'nominal.pth')==manifest['final_sha256']
        assert len(list((trained/'episodes').glob('*.npz')))==2000
        assert len(read(trained/'curves.json'))==2000
        manifests.append(manifest)
        directory=HERE/'artifacts'/'causal-evaluation'/arm
        assert read(directory/'complete.json')['episodes']==240
        assert len(read(directory/'baseline.json'))==240
        rows=read(directory/'contrasts.json')
        assert len(rows)==164
        for row in rows:
            scene=protocol['baseline']['scenes'].index(row['scene'])
            for label in ['selected','forward']:
                actual=np.load(directory/'contrasts'/f"{scene}-{row['case']}-{label}.npz")['records']
                original=np.load(HERE/'artifacts'/'contrasts'/f"{scene}-{row['case']}-{label}.npz")['records']
                np.testing.assert_array_equal(actual[0,:5],original[0,:5])
    assert manifests[0]['initial_sha256']==manifests[1]['initial_sha256']
    assert manifests[0]['demo_sha256']==manifests[1]['demo_sha256']
    assert manifests[0]['protocol']==manifests[1]['protocol']
    assert subprocess.check_output(['git','diff','--','crowd_nav','crowd_sim'],cwd=ROOT,text=True)==''
    files={}
    for path in sorted(HERE.rglob('*')):
        if not path.is_file() or path.name=='evidence-manifest.json' or '__pycache__' in path.parts:
            continue
        files[str(path.relative_to(ROOT))]={'sha256':sha(path),'bytes':path.stat().st_size}
    for relative in ['crowd_nav/policy/mamba_rl.py','crowd_nav/contracts.py',
                     'crowd_sim/envs/crowd_sim.py','research/recovery-duration/run.py',
                     'research/recovery-duration/parent-config.ini',
                     'research/navigation-findings/counterfactual-protocol.json']:
        files[relative]={'sha256':sha(ROOT/relative),'bytes':(ROOT/relative).stat().st_size}
    result={'verified':True,'episodes':240,'roots':164,'causal_training_episodes':4000,
            'same_first_executable_actions_verified':True,
            'nominal_checkpoint_unchanged':True,'source_worktree_unmodified':True,
            'original_project_written':False,'commit_at_verification':subprocess.check_output(
                ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'files':files}
    (HERE/'evidence-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print('VERIFIED',len(files),'files',sum(v['bytes'] for v in files.values()),'bytes')


if __name__=='__main__':
    main()
