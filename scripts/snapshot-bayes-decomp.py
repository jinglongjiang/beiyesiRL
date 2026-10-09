"""Build a publication inventory without importing models or launching runs."""
import hashlib
import json
from pathlib import Path


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / 'crowd_nav/runs/bayes-decomp-20261009'
    target = base / 'sync-manifest.json'
    summary = json.loads((base / 'stepwise/summary.json').read_text())
    assert summary['status'] == 'COMPLETE'
    assert summary['episodes'] == 1200 and summary['steps'] == 77352
    files = []
    for path in sorted(base.rglob('*')):
        if not path.is_file() or path == target:
            continue
        assert not path.is_symlink(), path
        assert path.suffix not in {'.pth', '.pt', '.ckpt', '.pem', '.key'}, path
        assert path.stat().st_size < 100 * 1024 * 1024, path
        files.append({'path': str(path.relative_to(base)),
                      'bytes': path.stat().st_size, 'sha256': digest(path)})
    manifest = {
        'repository': 'https://github.com/jinglongjiang/beiyesiRL.git',
        'scope': 'Completed decision-layer diagnostic, including invalid evidence and diagrams',
        'source_directory': '/home/abc/workspace/bayes-decomp-20261009',
        'weights_uploaded': False,
        'training_launched': False,
        'episodes': summary['episodes'], 'steps': summary['steps'],
        'verdict': summary['verdict'], 'next_route': summary['next_route'],
        'file_count_excluding_manifest': len(files),
        'bytes_excluding_manifest': sum(item['bytes'] for item in files),
        'files': files,
    }
    target.write_text(json.dumps(manifest, indent=2) + '\n')
    assert all(digest(base / item['path']) == item['sha256'] for item in files)
    print(json.dumps({key: value for key, value in manifest.items() if key != 'files'}))


if __name__ == '__main__':
    main()
