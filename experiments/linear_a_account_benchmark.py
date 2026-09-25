"""Run: python -m experiments.linear_a_account_benchmark freeze|verify|run."""
import collections
import json
import sys
from pathlib import Path
from linear_a import audit, account_benchmark

OUT = Path('experiments/linear-a-account-benchmark')

def main(command):
    if command == 'freeze':
        files = ['linear_a/account_benchmark.py', 'linear_a/audit.py', __file__,
                 'tests/test_linear_a_account_benchmark.py']
        files += [OUT / name for name in ('PROTOCOL.md','inputs.json','labels.json','source-review.json','sources.json')]
        # Use repository-relative paths in the manifest.
        files = [Path(p).relative_to(Path.cwd()) if Path(p).is_absolute() else Path(p) for p in files]
        audit.freeze(OUT, files, [s['path'] for s in json.loads((OUT/'sources.json').read_text())])
    elif command in ('verify', 'run'):
        audit.verify(OUT)
        if command == 'run':
            inputs = json.loads((OUT/'inputs.json').read_text())
            rows = [account_benchmark.inspect(a) for a in inputs]
            audit.save(OUT/'results.json', {'scope':'source-annotated development arithmetic; not function recovery',
                'objects':len({a['object'] for a in inputs}), 'accounts':len(inputs),
                'strict_counts':dict(collections.Counter(r['strict']['status'] for r in rows)),
                'transcribed_counts':dict(collections.Counter(r['transcribed']['status'] for r in rows)),
                'rows':rows, 'linear_a_scored':False, 'control_gate_passed':False})
            print(json.dumps(rows,indent=2))
    else:
        raise ValueError(command)

if __name__ == '__main__':
    main(sys.argv[1])
