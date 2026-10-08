"""Read-only inference and offline variance/timeout diagnostics, never training."""

import argparse
import collections
import configparser
import csv
import hashlib
import importlib.util
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'crowd_nav/runs/bayes-diagnostic-20261009'
OLD = ROOT / 'crowd_nav/runs/bayes-fix-20261008'
REMOTE = OLD / 'remote-run'
TARGETS = ('future-4-clearance-0.0', 'future-4-clearance-0.2',
           'future-8-clearance-0.0', 'future-8-clearance-0.2', 'episode-collision')
FIELDS = ('goal_progress_efficiency', 'stalled_window_ratio', 'oscillatory_stall_window_ratio',
          'backtracking_ratio', 'path_length_m', 'duration_s', 'steps', 'min_clearance_m', 'mean_clearance_m')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text())


def audit_module():
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location('original_b1_audit', ROOT / 'scripts/audit-bayes-fix.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def freeze():
    assert not (BASE / 'protocol.json').exists(), 'Do not overwrite frozen protocol'
    original = read(OLD / 'b1.json')
    inputs = [OLD / 'b1-rollout.npz', OLD / 'b1.json', ROOT / 'scripts/audit-bayes-fix.py',
              ROOT / 'scripts/diagnose-bayes-variance.py']
    version = read(REMOTE / 'formal-version.json')
    for name, expected in version['source_sha256'].items():
        assert sha(ROOT / name) == expected, name
        inputs.append(ROOT / name)
    for seed in (42, 43):
        directory = REMOTE / 'runs' / ('formal-bayes-seed' + str(seed))
        inputs.extend((directory / 'rl_model_ep10000.pth', directory / 'config.ini',
                       directory / 'benchmark-10000.csv', REMOTE / 'configs' / ('bayes-seed%d.eval.ini' % seed)))
    for arm in ('gru', 'bayes_mean', 'bayes'):
        for seed in (42, 43):
            inputs.append(REMOTE / 'runs' / ('formal-%s-seed%d' % (arm, seed)) / 'benchmark-10000.csv')
    metadata = original['episode_metadata']
    assert len(metadata) == 192
    save(BASE / 'protocol.json', {
        'created_unix': time.time(), 'purpose': 'Hypothesis sharpening only; no algorithm or training changes',
        'device': 'local RTX 3060 only; all assets already local; no SSH/4090 access',
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in inputs},
        'cohort': metadata, 'split': 'case_index % 4 == 0 is test, otherwise train; 144/48 episodes',
        'targets': TARGETS, 'future_label': 'min(post-action clearance[t:t+k]) < threshold; valid if full window or any danger',
        'predictors': {'A': 'current_clearance + history_age', 'B': 'A + 64 sigma dimensions'},
        'statistics': {
            'C_grid': [0.001, 0.01, 0.1, 1.0, 10.0], 'cv_folds': 5, 'cv_seed': 20261009,
            'cv': 'StratifiedGroupKFold by episode within training only, maximizing mean held-out fold AUC',
            'fit': 'StandardScaler fitted per CV training fold; L2 LogisticRegression lbfgs, max_iter=2000, tol=1e-7, no class weights',
            'tie_rule': 'smallest C among equal mean CV AUC',
            'bootstrap': '2000 paired draws of 48 held-out EPISODES; identical draw indices for A/B',
            'bootstrap_seed': 20261009, 'single_class': 'AUC/CI null, not zero or evidence of no information',
            'marginals': 'Reuse original audit.probe exactly (including its training-only direction and fixed linear probe)',
            'multiple_testing': 'Five exploratory endpoints per dataset, no multiplicity correction; not method confirmation',
        },
        'sampling': {'seeds': [42, 43], 'episodes_per_scenario': 32, 'grid': 80,
                     'original_label_source': 'scripts/audit-bayes-fix.py b1',
                     'acceptance': 'exact matching benchmark outcomes AND steps; identical schema/dtypes; sigma nonnegative finite',
                     'geometry_auc_flag': 'absolute change > 0.10 flags definition/trajectory audit, not automatic proof of definition error'},
        'timeout_rule': {
            'forward_like': 'efficiency >= 0.5, stall <= 0.2, backtrack <= 0.2, path >= same arm/seed/scenario successful-path median',
            'stuck_like': 'efficiency <= 0.2 OR stall >= 0.5 OR backtrack >= 0.4',
            'otherwise': 'mixed/unresolved; do not count as recoverable at 50s',
            'scope': 'Descriptive coarse bins only, no causal attribution or predicted 50s success',
        },
        'stops': ['No upstream edits or training', 'Any changed pinned source/input stops execution',
                  'Any replay outcome/step mismatch stops that seed and preserves invalid artifacts',
                  'Any split/schema mismatch stops inference', 'No changing CV grid or timeout bins after outcomes'],
    })
    print('Frozen protocol:', BASE / 'protocol.json', flush=True)


def pinned():
    protocol = read(BASE / 'protocol.json')
    for name, expected in protocol['source_sha256'].items():
        assert expected and sha(ROOT / name) == expected, 'Pinned input changed: ' + name
    return protocol


def paired_bootstrap(y, a, b, episode, protocol):
    from sklearn.metrics import roc_auc_score
    ids = sorted(set(episode.tolist()))
    groups = {e: np.flatnonzero(episode == e) for e in ids}
    rng = np.random.default_rng(protocol['statistics']['bootstrap_seed'])
    values = []
    invalid = 0
    for _ in range(2000):
        indices = np.concatenate([groups[e] for e in rng.choice(ids, len(ids), replace=True)])
        if np.unique(y[indices]).size < 2:
            invalid += 1
            continue
        aa = roc_auc_score(y[indices], a[indices])
        bb = roc_auc_score(y[indices], b[indices])
        values.append((aa, bb, bb - aa))
    limits = np.quantile(values, [0.025, 0.975], axis=0).T.tolist() if values else [None] * 3
    return dict(zip(('auc_A_95_ci', 'auc_B_95_ci', 'delta_auc_95_ci'), limits),
                valid_draws=len(values), invalid_single_class_draws=invalid, episode_clusters=len(ids))


def select_fit(x, y, groups, protocol):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    settings = protocol['statistics']
    folds = list(StratifiedGroupKFold(n_splits=settings['cv_folds'], shuffle=True,
                 random_state=settings['cv_seed']).split(x, y, groups))
    scores = []
    warnings.simplefilter('error', ConvergenceWarning)
    def estimator(c):
        return make_pipeline(StandardScaler(), LogisticRegression(C=c, penalty='l2', solver='lbfgs',
                             max_iter=2000, tol=1e-7, random_state=settings['cv_seed']))
    for c in settings['C_grid']:
        fold_values = []
        for train, valid in folds:
            assert set(groups[train]).isdisjoint(groups[valid])
            if len(np.unique(y[train])) < 2 or len(np.unique(y[valid])) < 2:
                fold_values.append(None)
                continue
            model = estimator(c).fit(x[train], y[train])
            fold_values.append(float(roc_auc_score(y[valid], model.decision_function(x[valid]))))
        usable = [v for v in fold_values if v is not None]
        assert len(usable) >= 2, 'Insufficient two-class internal CV folds'
        scores.append({'C': c, 'fold_auc': fold_values, 'mean_auc': float(np.mean(usable))})
    best = max(scores, key=lambda row: (row['mean_auc'], -row['C']))
    return estimator(best['C']).fit(x, y), {'selected_C': best['C'], 'cv_scores': scores}


def nested(dataset, tag):
    import torch
    from sklearn.metrics import roc_auc_score
    torch.set_num_threads(1)
    protocol = pinned()
    with np.load(dataset) as source:
        data = {k: source[k] for k in source.files}
    original = np.load(OLD / 'b1-rollout.npz')
    assert set(data) == set(original.files)
    for name, value in data.items():
        assert value.dtype == original[name].dtype, name
    episodes, cases = data['episode'], data['case_index']
    assert sorted(set(episodes.tolist())) == list(range(192))
    for e in range(192):
        assert set(cases[episodes == e]) == {protocol['cohort'][e]['episode']}
    holdout = cases % 4 == 0
    train_ids = sorted(set(episodes[~holdout].tolist()))
    test_ids = sorted(set(episodes[holdout].tolist()))
    assert len(train_ids) == 144 and len(test_ids) == 48 and set(train_ids).isdisjoint(test_ids)
    module = audit_module()
    results, marginals = {}, {}
    for target in TARGETS:
        y, valid = data['label-' + target], data['valid-' + target]
        train, test = valid & ~holdout, valid & holdout
        counts = {'train_steps': int(train.sum()), 'test_steps': int(test.sum()),
                  'train_episodes': len(set(episodes[train])), 'test_episodes': len(set(episodes[test])),
                  'train_positives': int(y[train].sum()), 'test_positives': int(y[test].sum()),
                  'train_positive_episodes': len(set(episodes[train & y])),
                  'test_positive_episodes': len(set(episodes[test & y]))}
        if tag == 'old':
            previous = read(OLD / 'b1.json')['results'][target]
            assert all(counts[k] == previous[k] for k in ('train_steps', 'test_steps', 'train_episodes', 'test_episodes', 'train_positives', 'test_positives'))
        marginals[target] = module.probe(data['sigma'], y, valid, episodes, cases, data['descriptors'])
        if tag == 'old' and 'sigma_linear' in marginals[target]:
            previous = read(OLD / 'b1.json')['results'][target]
            for name in ('sigma_mean', 'sigma_max', 'sigma_linear', 'current_clearance', 'history_age'):
                assert abs(marginals[target][name]['test_auc'] - previous[name]['test_auc']) < 1e-8, name
        if np.unique(y[train]).size < 2 or np.unique(y[test]).size < 2:
            results[target] = dict(counts, status='EVIDENCE_INSUFFICIENT_SINGLE_CLASS',
                                   auc_A=None, auc_B=None, delta_auc=None, delta_auc_95_ci=None)
        else:
            xa = data['descriptors'].astype(np.float64)
            xb = np.concatenate((xa, data['sigma'].astype(np.float64)), axis=1)
            scores, fitting = [], {}
            for label, x in (('A', xa), ('B', xb)):
                model, info = select_fit(x[train], y[train], episodes[train], protocol)
                scores.append(model.decision_function(x[test]))
                fitting[label] = info
            a, b = [float(roc_auc_score(y[test], score)) for score in scores]
            boot = paired_bootstrap(y[test], *scores, episodes[test], protocol)
            low, high = boot['delta_auc_95_ci']
            status = ('INCREMENTAL_ASSOCIATION_CONFIRMED' if low > 0 else
                      'PREDICTION_DEGRADATION_CONFIRMED' if high < 0 else 'INCREMENT_NOT_ESTABLISHED')
            results[target] = dict(counts, status=status, auc_A=a, auc_B=b, delta_auc=b-a,
                                   fitting=fitting, **boot)
        print(json.dumps({'stage': 'nested', 'tag': tag, 'target': target, 'result': results[target]}), flush=True)
    save(BASE / ('nested-' + tag + '.json'), {'dataset_sha256': sha(dataset), 'tag': tag,
         'train_episode_ids': train_ids, 'test_episode_ids': test_ids, 'results': results,
         'marginals': marginals, 'limitation': 'Association beyond two descriptors only; not calibrated Bayes, causal navigation gain, or generalization to new training seeds.'})


def build_labels(records):
    # This loop is copied verbatim in meaning from original b1(), including
    # post-action clearance and its incomplete-window positive exception.
    labels, validity = {}, {}
    for horizon in (4, 8):
        for threshold in (0., .2):
            name = 'future-{}-clearance-{}'.format(horizon, threshold)
            ys, vs = [], []
            for record in records:
                clearance = record['clearance']
                for tick in range(len(clearance)):
                    future = clearance[tick:tick+horizon]
                    dangerous = bool(future.min() < threshold)
                    ys.append(dangerous)
                    vs.append(len(future) == horizon or dangerous)
            labels[name] = np.asarray(ys, dtype=bool)
            validity[name] = np.asarray(vs, dtype=bool)
    labels['episode-collision'] = np.concatenate([np.full(len(r['sigma']), r['outcome'] == 'collision') for r in records])
    validity['episode-collision'] = np.ones(sum(len(r['sigma']) for r in records), dtype=bool)
    return labels, validity


def sample(seed, limit=None):
    import torch
    protocol = pinned()
    torch.set_num_threads(2)
    assert torch.cuda.is_available() and '3060' in torch.cuda.get_device_name(0), 'Local RTX3060 required'
    module = audit_module()
    for imported in (module.crowd_nav, module.crowd_sim, module.policy_module):
        Path(imported.__file__).resolve().relative_to(ROOT)
    directory = REMOTE / 'runs' / ('formal-bayes-seed%d' % seed)
    cfg_path = REMOTE / 'configs' / ('bayes-seed%d.eval.ini' % seed)
    cfg = configparser.RawConfigParser(inline_comment_prefixes=('#', ';'))
    assert cfg.read(str(cfg_path))
    module.contracts.init_grid_from_cfg(cfg)
    assert module.contracts.grid_action_dim(module.contracts.GRID) == 80
    assert cfg.get('mamba', 'temporal_backbone') == 'bayes'
    assert cfg.getfloat('env', 'time_limit') == 25 and cfg.getfloat('env', 'time_step') == .25
    saved = torch.load(directory / 'rl_model_ep10000.pth', map_location='cpu', weights_only=False)
    assert saved['episode'] == 10000 and saved['algo'] == 'sarl'
    state_dict = {k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k: v for k, v in saved['policy_state'].items()}
    policy = module.policy_module.MambaRLPolicy(cfg, device='cuda').eval()
    assert policy.temporal_encoder.readout[0].in_features == 128
    assert not policy.temporal_encoder.mean_only
    policy.load_state_dict(state_dict, strict=True)
    policy.use_sarl_predict = True
    policy.set_phase('test')
    policy.epsilon = 0
    references = {(r['scenario'], int(r['episode'])): r for r in csv.DictReader((directory / 'benchmark-10000.csv').open())}
    records = []
    started = time.time()
    output = BASE / ('seed%d-episodes' % seed)
    output.mkdir(parents=True, exist_ok=True)
    scenarios = {s[0]: s for s in module.SCENARIOS}
    with torch.no_grad():
        for e, entry in enumerate(protocol['cohort']):
            if limit is not None and e >= limit:
                break
            desc, sim, count, dimension, size = scenarios[entry['scene']]
            episode = entry['episode']
            reference = references[desc, episode]
            assert int(reference['seed']) == entry['seed']
            cfg = configparser.RawConfigParser(inline_comment_prefixes=('#', ';'))
            assert cfg.read(str(cfg_path))
            module.contracts.init_grid_from_cfg(cfg)
            assert module.contracts.grid_action_dim(module.contracts.GRID) == 80
            cfg.set('sim', 'test_sim', sim)
            cfg.set('sim', 'human_num', str(count))
            cfg.set('sim', dimension, str(size))
            env = module.CrowdSim()
            env.configure(cfg)
            env.phase = 'test'
            robot = module.Robot(cfg, 'robot')
            robot.set_policy(policy)
            robot.env = env
            env.set_robot(robot)
            if hasattr(policy, 'set_env'):
                policy.set_env(env)
            policy.set_env_dt(.25)
            policy._test_args = SimpleNamespace(gating=False, discrete_search=False, mamba_bias=False,
                mamba_rescue=False, behavior_profile='nominal', measure_latency=False)
            path = output / ('episode-%03d.npz' % e)
            meta_path = output / ('episode-%03d.json' % e)
            if path.exists() and meta_path.exists():
                meta = read(meta_path)
                assert meta['valid'] and meta['checkpoint_sha256'] == sha(directory / 'rl_model_ep10000.pth')
                values = np.load(path)
                record = {'sigma': values['sigma'], 'descriptors': values['descriptors'],
                          'clearance': values['clearance'], 'outcome': meta['outcome'], 'episode': episode}
            else:
                original_predict = policy.predict
                current, actions = [], []
                def observe(state):
                    raw = policy._build_joint_state_34(state.self_state, state.human_states)
                    token = module.contracts._batch_joint34_to_tokens_vectorized(raw.reshape(1, -1))[0]
                    history = (list(policy._history) + [token])[-24:]
                    tensor = torch.tensor(np.asarray(history), dtype=torch.float32, device='cuda')[None]
                    _, variance = policy.temporal_encoder.posterior(policy.spatial_encoder(tensor))
                    clearance = min(np.hypot(robot.px-h.px, robot.py-h.py)-robot.radius-h.radius for h in env.humans)
                    current.append((variance[0, -1].sqrt().cpu().numpy(), float(clearance), len(history)))
                    action = original_predict(state)
                    actions.append((float(action.vx), float(action.vy)))
                    return action
                policy.predict = observe
                try:
                    outcome, steps, positions, clearances, _, _, _ = module.test_episode(env, robot, policy, entry['seed'], case_desc=desc)
                finally:
                    policy.predict = original_predict
                assert len(current) == steps
                record = {'sigma': np.asarray([r[0] for r in current], dtype=np.float32),
                          'descriptors': np.asarray([[r[1], r[2]] for r in current], dtype=np.float64),
                          'clearance': np.asarray(clearances), 'outcome': outcome, 'episode': episode}
                valid = outcome == reference['outcome'] and steps == int(reference['steps'])
                np.savez_compressed(path, sigma=record['sigma'], descriptors=record['descriptors'], clearance=record['clearance'],
                                    actions=np.asarray(actions), positions=np.asarray(positions))
                save(meta_path, {'valid': valid, 'scene': desc, 'case_index': episode, 'episode_id': e,
                     'seed': entry['seed'], 'outcome': outcome, 'steps': steps,
                     'reference_outcome': reference['outcome'], 'reference_steps': int(reference['steps']),
                     'checkpoint_sha256': sha(directory / 'rl_model_ep10000.pth')})
                if not valid:
                    save(BASE / ('invalid-seed%d.json' % seed), read(meta_path))
                    raise RuntimeError('Outcome/step parity failed; preserved invalid episode ' + str(e))
            records.append(record)
            print(json.dumps({'stage': 'sample', 'seed': seed, 'episodes': len(records), 'outcome': record['outcome'],
                              'seconds': time.time()-started, 'grid': 80, 'parity': True}), flush=True)
    if len(records) != 192:
        print('Preflight only, not a complete cohort', flush=True)
        return
    sigma = np.concatenate([r['sigma'] for r in records])
    descriptors = np.concatenate([r['descriptors'] for r in records])
    episodes = np.concatenate([np.full(len(r['sigma']), i) for i, r in enumerate(records)])
    cases = np.concatenate([np.full(len(r['sigma']), r['episode']) for r in records])
    labels, validity = build_labels(records)
    assert np.isfinite(sigma).all() and np.all(sigma >= 0) and np.isfinite(descriptors).all()
    np.savez_compressed(BASE / ('b1-seed%d.npz' % seed), sigma=sigma, descriptors=descriptors, episode=episodes, case_index=cases,
                        **{'label-' + k: v for k, v in labels.items()}, **{'valid-' + k: v for k, v in validity.items()})
    save(BASE / ('sample-seed%d.json' % seed), {'status': 'COMPLETE', 'grid': 80, 'import_root': str(ROOT),
         'episodes': 192, 'steps': len(sigma), 'outcome_and_step_parity': True,
         'seconds': time.time()-started, 'device': torch.cuda.get_device_name(0),
         'checkpoint_sha256': sha(directory / 'rl_model_ep10000.pth'),
         'config_sha256': sha(cfg_path), 'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
         'outcome_counts': dict(collections.Counter(r['outcome'] for r in records)),
         'label_contract': 'exact original post-action future window and censoring; current descriptors before action'})


def distribution(rows):
    return {'n': len(rows), 'fields': {name: dict(zip(('p25', 'median', 'p75'), np.quantile(
            [float(r[name]) for r in rows], [.25, .5, .75]).tolist())) if rows else None for name in FIELDS}}


def timeout():
    pinned()
    results = {}
    for seed in (42, 43):
        for arm in ('gru', 'bayes_mean', 'bayes'):
            path = REMOTE / 'runs' / ('formal-%s-seed%d' % (arm, seed)) / 'benchmark-10000.csv'
            rows = list(csv.DictReader(path.open()))
            assert len(rows) == 3000 and len({(r['scenario'], r['episode']) for r in rows}) == 3000
            groups = {'all': rows}
            groups.update({s: [r for r in rows if r['scenario'] == s] for s in sorted({r['scenario'] for r in rows})})
            summaries = {}
            for scope, subset in groups.items():
                bins = collections.Counter()
                by_scene = {s: [r for r in rows if r['scenario'] == s and r['outcome'] == 'success'] for s in {r['scenario'] for r in rows}}
                for row in subset:
                    if row['outcome'] != 'timeout':
                        continue
                    efficiency, stall, backtrack, path_length = [float(row[k]) for k in
                        ('goal_progress_efficiency', 'stalled_window_ratio', 'backtracking_ratio', 'path_length_m')]
                    reference = by_scene[row['scenario']]
                    path_median = float(np.median([float(r['path_length_m']) for r in reference])) if reference else None
                    if efficiency <= .2 or stall >= .5 or backtrack >= .4:
                        category = 'stuck_like'
                    elif efficiency >= .5 and stall <= .2 and backtrack <= .2 and path_median is not None and path_length >= path_median:
                        category = 'forward_like'
                    else:
                        category = 'mixed_unresolved'
                    bins[category] += 1
                n = sum(bins.values())
                summaries[scope] = {
                    'by_outcome': {o: distribution([r for r in subset if r['outcome'] == o]) for o in ('success', 'timeout', 'collision')},
                    'timeout_bins': {k: {'n': bins[k], 'fraction': bins[k]/n if n else None} for k in ('stuck_like', 'forward_like', 'mixed_unresolved')},
                }
            results['%s-seed%d' % (arm, seed)] = {'csv_sha256': sha(path), 'groups': summaries}
    save(BASE / 'timeout-analysis.json', {'status': 'COMPLETE', 'source': 'repaired six final 3000-case CSVs',
          'rule': read(BASE / 'protocol.json')['timeout_rule'], 'results': results,
          'limits': 'No final_distance or return; no 50s replay; bins are descriptive, not recoverable success rates or causal time-limit effects.'})
    print('Timeout analysis saved', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=('freeze', 'nested', 'sample', 'timeout'))
    parser.add_argument('--seed', type=int, choices=(42, 43))
    parser.add_argument('--limit', type=int)
    parser.add_argument('--tag')
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if args.task == 'freeze':
        freeze()
    elif args.task == 'nested':
        assert args.tag in ('old', 'seed42', 'seed43')
        dataset = OLD / 'b1-rollout.npz' if args.tag == 'old' else BASE / ('b1-' + args.tag + '.npz')
        nested(dataset, args.tag)
    elif args.task == 'sample':
        assert args.seed is not None
        sample(args.seed, args.limit)
    else:
        timeout()
