#!/usr/bin/env python3
"""Partition a matlab.unittest suite into shards for parallel `matlab -batch` runs.

Used by run_mmacos_tests.sh when -j N > 1.  Two sub-commands:

  partition --names NAMES --tests-dir DIR --timings DIR --jobs N --out DIR [--split "tA tB"]
      NAMES holds one test name per line ("tClass/test_method(...)"), as MATLAB
      enumerated the suite.  Writes out/shard_<k>.txt (one names file per shard)
      and out/plan.txt (what went where and the predicted time per shard).

  summarize --out DIR [--label TEXT]
      Reads out/results_<k>.csv written by each shard's MATLAB run and prints the
      totals, the slowest classes, and every failure with its shard's log.
      Exits 1 if any test failed or any shard produced no results file.

Rules:
  * A class never spans model sizes: each class's model size is read from its
    file (macos.init(NNN) or a ModelSize = NNN property; default 128), and the
    shards are allotted per size in proportion to that size's predicted time, so
    no MATLAB process ever calls init() with two sizes (the init-reinit heap bug
    the serial runner also dodges).
  * A class is one unit (its TestClassSetup fixtures are built once), except the
    classes named in --split, whose tests are scheduled individually; name only
    classes whose setup is cheap (init + a path).
  * Units are assigned longest-first to the least-loaded shard (LPT).  Durations
    come from the timings CSVs (Name,Duration); an unknown test counts 1 s.
"""
import argparse, collections, csv, glob, os, re, sys


def read_timings(tdir):
    t = {}
    for f in glob.glob(os.path.join(tdir, '*.csv')):
        with open(f, newline='') as fh:
            for row in csv.DictReader(fh):
                try:
                    t[row['Name']] = float(row['Duration'])
                except (KeyError, ValueError):
                    pass
    return t


def model_size(tests_dir, cls):
    path = os.path.join(tests_dir, cls + '.m')
    try:
        src = open(path, errors='ignore').read()
    except OSError:
        return 128
    m = re.search(r'init\(\s*(\d+)\s*\)', src)
    if m:
        return int(m.group(1))
    m = re.search(r'ModelSize\s*=\s*(\d+)', src)
    if m:
        return int(m.group(1))
    return 128


def partition(a):
    names = [l.strip() for l in open(a.names) if l.strip()]
    timings = read_timings(a.timings)
    split = set(a.split.split())
    # units: (key, size, duration, [names])
    by_cls = collections.OrderedDict()
    for n in names:
        cls = n.split('/', 1)[0]
        by_cls.setdefault(cls, []).append(n)
    units = []
    for cls, tests in by_cls.items():
        size = model_size(a.tests_dir, cls)
        if cls in split:
            for n in tests:
                units.append((n, size, timings.get(n, 1.0), [n]))
        else:
            units.append((cls, size, sum(timings.get(n, 1.0) for n in tests), tests))
    # shards per size, proportional to predicted time, at least one each
    per_size = collections.defaultdict(float)
    for _, size, d, _ in units:
        per_size[size] += d
    total = sum(per_size.values()) or 1.0
    sizes = sorted(per_size)
    jobs = max(a.jobs, len(sizes))
    alloc = {s: 1 for s in sizes}
    for _ in range(jobs - len(sizes)):
        # give the next shard to the size with the largest predicted time per shard
        s = max(sizes, key=lambda s: per_size[s] / alloc[s])
        alloc[s] += 1
    shards = []  # (size, load, [names], [unit keys])
    for s in sizes:
        for _ in range(alloc[s]):
            shards.append([s, 0.0, [], []])
    for key, size, d, tests in sorted(units, key=lambda u: -u[2]):
        cand = [sh for sh in shards if sh[0] == size]
        sh = min(cand, key=lambda x: x[1])
        sh[1] += d
        sh[2].extend(tests)
        sh[3].append(f'{key} ({d:.0f} s)')
    # number the shards by descending load: the runner launches them in order,
    # and when memory clamps the parallelism the longest must start first
    shards.sort(key=lambda sh: -sh[1])
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'plan.txt'), 'w') as plan:
        plan.write(f'{len(names)} tests, {len(by_cls)} classes, predicted {total:.0f} s serial; '
                   f'{len(shards)} shards\n')
        for k, (size, load, tests, keys) in enumerate(shards):
            with open(os.path.join(a.out, f'shard_{k}.txt'), 'w') as f:
                f.write('\n'.join(tests) + '\n')
            plan.write(f'shard {k}: model {size}, predicted {load:.0f} s, {len(tests)} tests: '
                       + ', '.join(keys) + '\n')
    longest = max(sh[1] for sh in shards) if shards else 0
    print(f'{len(shards)} shards; predicted wall {longest:.0f} s vs {total:.0f} s serial '
          f'(plan: {os.path.join(a.out, "plan.txt")})')


def summarize(a):
    files = sorted(glob.glob(os.path.join(a.out, 'shard_*.txt')))
    rows, missing, failed = [], [], []
    for f in files:
        k = re.search(r'shard_(\d+)\.txt$', f).group(1)
        rf = os.path.join(a.out, f'results_{k}.csv')
        if not os.path.exists(rf):
            missing.append(k)
            continue
        with open(rf, newline='') as fh:
            for row in csv.DictReader(fh):
                row['shard'] = k
                rows.append(row)
    def truthy(v):
        return str(v).strip().lower() in ('1', 'true')
    n_pass = sum(truthy(r['Passed']) for r in rows)
    n_fail = sum(truthy(r['Failed']) for r in rows)
    n_inc = sum(truthy(r['Incomplete']) for r in rows)
    by_cls = collections.defaultdict(float)
    for r in rows:
        by_cls[r['Name'].split('/', 1)[0]] += float(r['Duration'])
    label = a.label or 'sharded'
    if a.t0:
        walls = []
        for f in files:
            k = re.search(r'shard_(\d+)\.txt$', f).group(1)
            lf = os.path.join(a.out, f'shard_{k}.log')
            if os.path.exists(lf):
                walls.append(f'{k}:{int(os.path.getmtime(lf) - a.t0)}s')
        print('shard wall times since launch (s): ' + ' '.join(walls))
    print(f'=== {label}: {n_pass} pass, {n_fail} fail, {n_inc} incomplete, {len(rows)} tests over '
          f'{len(files)} shards, {sum(by_cls.values()):.0f} s of test time ===')
    for cls, d in sorted(by_cls.items(), key=lambda x: -x[1])[:8]:
        print(f'  {cls:28s} {d:8.1f} s')
    for r in rows:
        if truthy(r['Failed']) or truthy(r['Incomplete']):
            print(f'  FAIL {r["Name"]}  (shard {r["shard"]}: {os.path.join(a.out, "shard_" + r["shard"] + ".log")})')
    for k in missing:
        print(f'  NO RESULTS from shard {k}: {os.path.join(a.out, "shard_" + k + ".log")}')
    sys.exit(1 if (n_fail or n_inc or missing) else 0)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('partition')
    q.add_argument('--names', required=True)
    q.add_argument('--tests-dir', required=True)
    q.add_argument('--timings', required=True)
    q.add_argument('--jobs', type=int, required=True)
    q.add_argument('--out', required=True)
    q.add_argument('--split', default='')
    q.set_defaults(fn=partition)
    s = sub.add_parser('summarize')
    s.add_argument('--out', required=True)
    s.add_argument('--label', default='')
    s.add_argument('--t0', type=float, default=0.0, help='launch epoch; prints each shard\'s wall time')
    s.set_defaults(fn=summarize)
    a = p.parse_args()
    a.fn(a)
