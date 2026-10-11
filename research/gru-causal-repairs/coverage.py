"""Pre-registered conditional coverage and feature-trainability controls."""
import argparse
import copy
import json
import time

import numpy as np
import torch

import run as a

NAMES = ['coverage-control', 'targeted-head', 'targeted-full']


def catalogue():
    xs, ys, cases, meta = [], [], [], []
    stats = json.loads((a.OLD / 'statistics.json').read_text())['cases']
    active = {a.q.key(v) for v in stats if v['category'] != 'NO_CONTACT_FREE_RESCUE_IN_TESTED_SINGLE_OR_PREFIX_ACTIONS'}
    for root in a.q.roots():
        if a.q.key(root) not in active:
            continue
        source = a.OLD / 'catalogue' / a.q.key(root)
        data = np.load(source / 'inputs.npz')
        indices = np.flatnonzero(data['safe'] & (data['first'][:, 1] == 0))
        for i in indices:
            branch = json.loads((source / f'action-{i:02d}-A_deploy.json').read_text())
            xs.append(data['windows'][i])
            ys.append((branch['result']['return'] - data['first'][i, 0]) / .99)
            cases.append(root['case'])
            meta.append({'root': root, 'action': int(i)})
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32), np.asarray(cases), meta


def train():
    cfg, p = a.setup()
    initial = copy.deepcopy(p.state_dict())
    common = np.load(a.OUT / 'matched-data.npz')
    x, y, cases, meta = catalogue()
    a.save('coverage-data.json', {'samples': meta, 'split': 'whole numeric even-case fitting, odd-case heldout',
                                'same_frozen_deploy_policy': True, 'no_new_branch_simulation': True})
    results = {}
    for name in NAMES:
        p.load_state_dict(initial)
        for parameter in p.parameters():
            parameter.requires_grad_(name != 'targeted-head')
        if name == 'targeted-head':
            for parameter in p.value_head.parameters():
                parameter.requires_grad_(True)
        xx, yy = (common['windows'], common['deploy']) if name == 'coverage-control' else (x, y)
        fit = common['case'] < 2012 if name == 'coverage-control' else cases % 2 == 0
        xt, yt = torch.as_tensor(xx, device=p.device), torch.as_tensor(yy, device=p.device)
        optimizer = torch.optim.Adam([v for v in p.parameters() if v.requires_grad], lr=1e-5)
        rng = np.random.default_rng(121012)
        batches = rng.choice(np.flatnonzero(fit), size=(1000, 128))
        np.save(a.OUT / f'{name}-batch-indices.npy', batches)
        curves = []
        start = time.perf_counter()
        for j, index in enumerate(batches):
            torch.manual_seed(121012 + j)
            p.train()
            optimizer.zero_grad(set_to_none=True)
            loss = (p.forward_value(xt[index]) - yt[index]).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([v for v in p.parameters() if v.requires_grad], 10.)
            assert torch.isfinite(norm)
            optimizer.step()
            if (j + 1) % 100 == 0:
                p.eval()
                with torch.no_grad():
                    pred = torch.cat([p.forward_value(xt[k:k + 128]) for k in range(0, len(xt), 128)]).cpu().numpy()
                curves.append({'update': j + 1, 'MSE': float(loss),
                               'fit_MAE': float(np.abs(pred[fit] - yy[fit]).mean()),
                               'heldout_MAE': float(np.abs(pred[~fit] - yy[~fit]).mean())})
                print('COVERAGE TRAIN', name, j + 1, 'MSE', float(loss), flush=True)
        torch.save(p.state_dict(), a.OUT / f'{name}.pth')
        p.eval()
        tx = torch.as_tensor(x, device=p.device)
        with torch.no_grad():
            pred = torch.cat([p.forward_value(tx[k:k + 128]) for k in range(0, len(tx), 128)]).cpu().numpy()
        np.savez_compressed(a.OUT / f'{name}-catalogue-predictions.npz', prediction=pred, target=y, cases=cases)
        results[name] = {'seconds': time.perf_counter() - start, 'updates': 1000, 'curves': curves,
                         'sha256': a.sha(a.OUT / f'{name}.pth'),
                         'catalogue_even_MAE': float(np.abs(pred[cases % 2 == 0] - y[cases % 2 == 0]).mean()),
                         'catalogue_odd_MAE': float(np.abs(pred[cases % 2 != 0] - y[cases % 2 != 0]).mean())}
        a.save('coverage-training.json', results)
    a.probe(NAMES, 'coverage-ranking-probes.json')


def evaluate(arm, cohort):
    cfg, p = a.setup()
    source = a.r.OLD / 'artifacts/nominal.pth' if arm == 'nominal' else a.OUT / f'{arm}.pth'
    p.load_state_dict(torch.load(source, map_location=p.device))
    p.eval()
    p.set_phase('test')
    rows = []
    if cohort == 'roots':
        stats = json.loads((a.OLD / 'statistics.json').read_text())['cases']
        for item in stats:
            if item['category'] == 'NO_CONTACT_FREE_RESCUE_IN_TESTED_SINGLE_OR_PREFIX_ACTIONS':
                continue
            root = {k: item[k] for k in ('scene', 'case', 'step', 'group', 'partition')}
            humans = a.old_tape(cfg, p, root)
            _, snap = a.r.frozen_root(cfg, p, root, humans)
            row = a.run_episode(cfg, p, a.m.SCENES.index(root['scene']), root['case'], humans,
                                arm, snap, f'coverage-roots/{arm}/{a.q.key(root)}')
            row.update({'root': root, 'category': item['category']})
            rows.append(row)
            a.save(f'coverage-roots-{arm}.json', {'results': rows})
    else:
        if arm == 'nominal':
            a.weight(p, 'nominal')
        for scene in range(6):
            for case in range(320, 340):
                humans = a.tape(cfg, p, scene, case)
                rows.append(a.run_episode(cfg, p, scene, case, humans, arm,
                                          output=f'coverage-fresh/{arm}/{scene}-{case}'))
                a.save(f'coverage-fresh-{arm}.json', {'results': rows})
                if len(rows) % 10 == 0:
                    print('COVERAGE EVAL', arm, len(rows), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['train', 'evaluate'])
    parser.add_argument('--arm', choices=NAMES + ['nominal'], default='coverage-control')
    parser.add_argument('--cohort', choices=['roots', 'fresh'], default='roots')
    args = parser.parse_args()
    if args.stage == 'train':
        train()
    else:
        evaluate(args.arm, args.cohort)
