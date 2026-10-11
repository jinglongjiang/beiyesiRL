"""One-factor matched causal training, not recovery or a new algorithm."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import types

import numpy as np
import torch

import run as r

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / 'causal-training-protocol.json').read_text())


def train(arm):
    cfg, policy = r.setup()
    out = HERE / 'artifacts' / 'causal-training' / arm
    assert not out.exists(), 'Do not overwrite a completed or interrupted scientific run'
    out.mkdir(parents=True)
    alpha = SPEC['arms'][arm]
    init = r.OLD / 'artifacts' / 'gru-il.pth'
    demos = r.OLD / 'artifacts' / 'corrected-orca.pth'
    policy.load_state_dict(torch.load(init, map_location=policy.device))
    policy.set_phase('train')
    original_candidates = policy.executable_candidates

    def training_candidates(self):
        commands = original_candidates()
        if self._phase == 'train' and alpha > 0 and self._last_action is not None:
            commands = [r.m.ActionXY(alpha*self._last_action.vx+(1-alpha)*a.vx,
                                     alpha*self._last_action.vy+(1-alpha)*a.vy)
                        for a in commands]
        return commands

    policy.executable_candidates = types.MethodType(training_candidates, policy)
    memory = r.m.EpisodeMemory(seed=61)
    memory.episodes.extend(torch.load(demos, map_location='cpu'))
    assert len(memory.episodes) == 512
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-5)
    rng = np.random.default_rng(61)
    env = r.m.make_env(cfg, policy, phase='train')
    policy.set_env(env)
    assert env.time_limit == 50 and env.human_num == 5
    metadata = {'arm': arm, 'smoothing': alpha, 'seed':61,
                'initial_sha256': hashlib.sha256(init.read_bytes()).hexdigest(),
                'demo_sha256': hashlib.sha256(demos.read_bytes()).hexdigest(),
                'protocol': SPEC, 'started_unix': time.time()}
    r.m.save_json(out / 'manifest.json', metadata)
    curves = []
    started = time.perf_counter()
    for ep in range(2000):
        env.reset(options={'test_case':10000+ep})
        policy.reset_episode_stats()
        ts, rewards, commands_log = [], [], []
        epsilon = .3 + (.05-.3)*min(ep/1500.,1.)
        for step in range(202):
            if rng.random() < epsilon:
                tok = r.m.token(policy, r.m.state(env))
                index = int(rng.integers(80))
                action = policy.executable_candidates()[index]
            else:
                part = r.decompose(policy, env)
                assert not part['risk'].any()
                tok, index = part['token'], part['best']
                action = part['commands'][index]
            ts.append(tok)
            commands_log.append([action.vx, action.vy, index])
            r.m.commit_action(policy, tok, action, index)
            _, reward, term, trunc, info = env.step(action)
            rewards.append(reward)
            if term or trunc:
                break
        assert term or trunc
        memory.add(ts, rewards)
        losses = [r.m.update(policy, memory, optimizer) for _ in range(4)]
        curves.append({'episode':ep+1, 'outcome':info['event'], 'steps':len(rewards),
                       'return':float(sum(rewards)), 'discounted_return':float(memory.episodes[-1][1][0]),
                       'loss':float(np.mean(losses)), 'epsilon':epsilon})
        trajectory = out / 'episodes' / f'{ep+1:04d}.npz'
        trajectory.parent.mkdir(exist_ok=True)
        np.savez_compressed(trajectory, tokens=np.asarray(ts,np.float32),
                            rewards=np.asarray(rewards,np.float32),
                            commands=np.asarray(commands_log,np.float32),
                            returns=memory.episodes[-1][1])
        if (ep+1)%25 == 0:
            r.m.save_json(out / 'curves.json', curves)
            recent = curves[-25:]
            print(arm, ep+1, 'SR',np.mean([v['outcome']=='reach_goal' for v in recent]),
                  'loss',np.mean(losses),'seconds',time.perf_counter()-started,flush=True)
        if (ep+1)%500 == 0:
            torch.save(policy.state_dict(),out / f'rl-{ep+1}.pth')
    final = out / 'nominal.pth'
    torch.save(policy.state_dict(), final)
    metadata.update({'elapsed_seconds':time.perf_counter()-started,
                     'final_sha256':hashlib.sha256(final.read_bytes()).hexdigest(),
                     'episodes':len(curves),'updates':8000,'complete':True})
    r.m.save_json(out / 'manifest.json',metadata)
    print('COMPLETE',arm,json.dumps(metadata),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('arm',choices=list(SPEC['arms']))
    args = parser.parse_args()
    train(args.arm)
