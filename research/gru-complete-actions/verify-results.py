"""Validate complete registered evidence and create a local reproducibility manifest."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'artifacts'/'v2-50s'
PROTOCOL=json.loads((HERE/'protocol.json').read_text())
groups=['difficult','rescues','damages','successful_controls']
roots=[tuple(v) for g in groups for v in PROTOCOL[g]]
counts={'legal_actions':0,'all_branches':0,'legal_branches':0,'trace_steps':0}
for scene,case,step in roots:
    folder=OUT/'catalogue'/f'{scene}-{case}'
    data=np.load(folder/'inputs.npz')
    assert data['safe'].shape==(80,) and data['windows'].shape==(80,24,8,13)
    assert all(np.isfinite(data[k]).all() for k in data.files)
    counts['legal_actions']+=int(data['safe'].sum())
    for index in range(80):
        files=[folder/f'action-{index:02d}-{policy}.json' for policy in ['A_deploy','B_raw_greedy']]
        if data['safe'][index]:
            assert all(p.exists() for p in files), f'Missing legal branch {scene}/{case}/{index}'
        if not any(p.exists() for p in files):
            continue
        assert all(p.exists() for p in files), 'Unpaired continuation outputs'
        pair=[]
        for p in files:
            row=json.loads(p.read_text())
            a=np.load(p.with_suffix('.npz'))['records']
            assert len(a)>0 and np.isfinite(a).all()
            assert int(a[0,0])==step and int(a[0,5])==index
            assert int(a[-1,0])<=200
            np.testing.assert_array_equal(np.diff(a[:,0]),np.ones(max(len(a)-1,0)))
            np.testing.assert_allclose(a[0,3:5],row['first_command'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(a[0,7],data['first'][index,0],rtol=0,atol=1e-9)
            result=row['result']
            np.testing.assert_allclose(a[:,7]@(.99**np.arange(len(a))),result['return'],rtol=0,atol=1e-9)
            assert int((a[:,8]<0).any())==result['contact']
            assert len(a)==result['steps'] and len(a)*.25==result['duration']
            assert result['duration']<=50.25-step*.25+1e-9
            pair.append(row)
            counts['all_branches']+=1
            counts['legal_branches']+=int(row['legal_first_action'])
            counts['trace_steps']+=len(a)
        np.testing.assert_array_equal(pair[0]['first_command'],pair[1]['first_command'])
for p in (OUT/'prefixes').glob('*.npz'):
    a=np.load(p)['records']
    assert np.isfinite(a).all()
prefixes=json.loads((OUT/'prefixes.json').read_text())
for p in prefixes:
    assert all(v['predicted_clearance']>=.2 for v in p['legality'])
cp=json.loads((OUT/'checkpoint-pairs.json').read_text())
if cp['executed']:
    assert len(cp['rows'])==38*5*2
    assert len({(v['stage'],v['policy'],v['root']['scene'],v['root']['case']) for v in cp['rows']})==38*5*2
    scenes=['baseline_circle','baseline_square','dense_circle','dense_square','large_circle','large_square']
    for row in cp['rows']:
        if row['stage']!='RL-2000':
            continue
        root=row['root']
        data=np.load(OUT/'catalogue'/f"{scenes.index(root['scene'])}-{root['case']}"/'inputs.npz')
        np.testing.assert_allclose(row['actual_next_q'],data['B_q'][row['indices']],rtol=0,atol=2e-6)
head=json.loads((OUT/'head-fit.json').read_text())
if head['executed']:
    data=np.load(OUT/'head-fit-inputs.npz')
    assert data['features'].shape==(counts['legal_actions'],256)
    assert data['labels'].shape==(counts['legal_actions'],2)
    assert int(data['fit_cases'].sum())==head['fit_case_count']
    ids_fit={v['root']['case'] for v in head['cases'] if v['root']['case']<120}
    ids_hold={v['root']['case'] for v in head['cases'] if v['root']['case']>=120}
    assert not ids_fit&ids_hold
parent=HERE.parent/'recovery-duration'/'artifacts'/'nominal.pth'
assert hashlib.sha256(parent.read_bytes()).hexdigest()==PROTOCOL['nominal_sha256']
files=[]
for p in sorted(HERE.rglob('*')):
    if p.is_file() and p.name!='evidence-manifest.json' and '__pycache__' not in p.parts:
        files.append({'path':str(p.relative_to(HERE)),'bytes':p.stat().st_size,
                      'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
manifest={'verified':True,'counts':counts,'files':files,
          'nominal_sha256':PROTOCOL['nominal_sha256'],
          'M4_verified':False,'historical_RL_replay_recovered':False,
          'limits':'Verifies numerical/provenance consistency, not algorithmic novelty or historical training causality'}
(HERE/'evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'verified':True,'counts':counts,'files':len(files),
                  'bytes':sum(v['bytes'] for v in files)},indent=2))
