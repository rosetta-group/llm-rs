"""Report and archive the completed, separately frozen development comparisons."""
import json
import math
from pathlib import Path
import tarfile

from experiments import rejection_followups_v2 as development

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/rejection-followups'
STATE = ROOT / 'artifacts/rejection-followups'
read, seal = development.read, development.seal


def report():
    development.verify()
    benchmark = read(OUT/'benchmark.json')
    controls = read(OUT/'control-results.json')
    calibration = read(OUT/'calibration-results.json')
    groups = {}
    for row in calibration['rows']:
        g=groups.setdefault(row['kind'],dict(n=0,original=0,candidate=0,inconclusive=0))
        g['n']+=1;g['original']+=row['original']['accepted'] is not None
        g['candidate']+=row['candidate']['accepted'] is not None
        g['inconclusive']+=row['candidate']['inconclusive']
    positive=groups['positive']
    new=groups.get('frequency_copy',dict(n=0,original=0,candidate=0,inconclusive=0))
    lines=['# Three rejection follow-ups: development results','',
      '## What was done','',
      '- Compared the original rule with a transfer-centered candidate on released cases.',
      '- Fitted independent keys under five frozen language priors for each frequency-preserving copying control, then transferred sealed keys.',
      '- Implemented and benchmarked chunked and incremental refinement in a new module; earlier frozen code remains unchanged.','',
      '## Why','',
      'The original fit ceiling rejected genuine English even though its key transferred. Mutation controls also introduced unfamiliar pieces, while exhaustive full-text swap scoring made repeated controls expensive.','',
      '**Transfer:** decoding a separate passage with the learned key fixed.',
      '**Development:** released examples used to design a rule; they cannot independently confirm that rule.','',
      '## 1. Decision calibration','',
      f"The candidate accepts **{positive['candidate']}/{positive['n']}** released genuine ciphers, versus **{positive['original']}/{positive['n']}** for the original rule. "
      'It removes only the fit-excess ceiling: agreement of language winners, both 0.25 margins, 0.50 transfer ceiling, 95% coverage and cap checks remain.','',
      '| Input | Cases | Original accepted | Candidate accepted | Inconclusive |',
      '|---|---:|---:|---:|---:|']
    for kind,g in groups.items():
        lines.append(f"| {kind} | {g['n']} | {g['original']} | {g['candidate']} | {g['inconclusive']} |")
    lines += ['', 'Acceptance is desirable for positives and an error for the negative classes. '
      'The original Latin mutation case remains inconclusive under both rules. All examples here share only three released source/key blocks. '
      'The apparent improvement is development evidence, not a fresh sensitivity or false-positive estimate.', '',
      'The [sensitivity table](calibration-results.json) records 21 transfer ceilings from 0.00 to 1.00. '
      'No threshold was selected on new sealed text; the 0.50 candidate was declared before the stronger-control fits.', '',
      '## 2. Frequency-preserving copying control','',
      'Each copied passage consumes exactly its source token bag. Local copying changes order while preserving token count, frequencies, '
      'types and glyph inventory. Each new control gets a full search under all five priors; the positive cipher’s fitted key is not reused.', '',
      '| Source block | Fit tokens / types | Recent-repeat rate, original → copied | Transfer coverage | Transfer winner / excess | Candidate |',
      '|---|---:|---:|---:|---|---|']
    for row in controls['outcomes']:
        d=row['diagnostics']['fit'];w=row['candidate']['transfer']['winner'];s=row['scores'][w]
        result='inconclusive' if row['candidate']['inconclusive'] else row['candidate']['accepted'] or 'rejected'
        lines.append(f"| {row['source_language']} | {d['original']['tokens']} / {d['original']['types']} | "
          f"{d['original']['recent_repeat_rate']:.1%} → {d['copied']['recent_repeat_rate']:.1%} | {s['coverage']:.2%} | "
          f"{w} / {s['transfer_excess']:.3f} | {result} |")
    lines += ['', 'Recent-repeat rate is the share of tokens already seen within the preceding 50 positions. '
      'This generator is a controlled stress test, not an implementation of Timm’s generator and not a model of all meaningless text.', '',
      '## 3. Refiner cost and equivalence','',
      '| Fixture | Legacy swaps (s) | Chunked (s) | Incremental (s) | Speedup, legacy / incremental | Peak RSS, legacy / incremental (MiB) |',
      '|---|---:|---:|---:|---:|---:|']
    names=list(dict.fromkeys(r['name'] for r in benchmark['rows']))
    for name in names:
        b={r['backend']:r for r in benchmark['rows'] if r['name']==name}
        a,c,i=b['legacy'],b['chunked'],b['incremental']
        lines.append(f"| {name} | {a['median_seconds']:.4f} | {c['median_seconds']:.4f} | {i['median_seconds']:.4f} | "
           f"{a['median_seconds']/i['median_seconds']:.1f}× | {a['peak_rss_bytes']/2**20:.1f} / {i['peak_rss_bytes']/2**20:.1f} |")
    error=max(r['score_audit_max_error'] for r in benchmark['rows'])
    lines += ['', 'Times are medians of two warmed exhaustive pair-swap batches, with two threads. '
      'Every backend and repetition chose the same earliest winning pair and exact full score. '
      f'The largest sampled incremental-score difference was {error:.3g} bits; close minima are rescored with the original objective. '
      'Small complete-refinement tests also match the legacy mapping, recovered text, score, evaluation count and kicks; two-letter keys use the chunked full scorer.', '',
      'Chunking alone reduced memory but did not improve runtime on these fixtures. Incremental scoring changes only the n-grams touched by a proposal and its homophone cost. '
      'The 31× copying-fixture result is a search-step speedup, not an end-to-end decoder speedup. The initial EM and repair stages are unchanged.', '',
      'With U units, N decoded one-letter positions and fixed n-gram order, the old pair sweep scans O(U² N) positions and builds O(U³) candidate-key storage. '
      'Incremental proposal work visits O(U N) affected-window occurrences across the sweep; near-tie full rescoring can increase runtime and has the old worst-case scoring order. '
      'Candidate batches bound key storage to O(batch_size × U), with linear input/position-index storage.','',
      '## Resource and provenance record','']
    all_scores=[s for row in controls['outcomes'] for s in row['scores'].values()]
    refiners=[s['refiner'] for s in all_scores]
    lines += [f"- Stronger controls: {len(controls['outcomes'])}/3 pairs, {len(all_scores)} fits, "
              f"{controls['fit_worker_seconds']/3600:.3f} aggregate fit-worker hours; status: `{controls['stop_reason'] or 'completed'}`.",
              f"- Refiner time within those fits: {sum(r['seconds'] for r in refiners)/3600:.3f} aggregate hours; "
              f"{sum(s['cap_hit'] for s in all_scores)} fitted candidates reached a limit.",
              '- The first follow-up freeze was preserved but not run. Preflight increased the sweep guard from 50 to 200 to accommodate the inherited 30 kicks; time and evaluation budgets stayed fixed. '
              'See [the recorded correction](CONTROL_RESOURCE_REPAIR.md).',
              '- An unsupported synthetic alphabet symbol was caught before any benchmark timing; the fixture now reads the pinned prior alphabet. '
              'See [setup correction](benchmark-setup-correction.json).',
              '- CPU only; five workers with two threads each. No manuscript text or unused sealed answers were opened.','',
              '## Decision and limits','',
              'Use the incremental backend for further development with the declared numerical checks and work limits. '
              'Keep the transfer-centered rule as a candidate for fresh confirmation; its development results do not replace the original screen. '
              'Keep frequency-preserving copying in the negative-control set alongside shuffle and mutation controls. '
              'The existing 1% character-error / 10% word-error manuscript gate remains unchanged.','',
              'A [costed confirmation plan](CONFIRMATION_PLAN.md) is a proposal only. No fresh confirmation was run as part of these three development follow-ups.','',
              '## Records and reproduction','',
              '[Protocol](PROTOCOL.md) · [executed freeze](freeze-v2.json) · [benchmark](benchmark.json) · '
              '[controls](control-results.json) · [calibration](calibration-results.json) · [verification](verification.json) · '
              '[released development records](evaluated-records.tar.gz)','',
              '```sh',
              '.venv/bin/python -m experiments.rejection_followups_v2 verify',
              '.venv/bin/python -m experiments.audit_rejection_followups',
              'NUMBA_NUM_THREADS=2 .venv/bin/python -m unittest tests.test_rejection_followups -v',
              '```','',
              'Working sources/priors must be restored at the paths in the freeze. The development archive preserves its input plan, '
              'public texts, fit records, key seals, transfer records, graded decisions and benchmark fixtures. Creation stages refuse overwrites.','']
    with (OUT/'REPORT.md').open('x') as f:f.write('\n'.join(lines))
    with (OUT/'evaluated-records.tar.gz').open('xb') as raw, tarfile.open(fileobj=raw,mode='w:gz') as tar:
        tar.add(STATE/'control-plan.json',arcname='control-plan.json')
        for directory in ('public','fit','transfer','sealed-keys','graded','benchmark','benchmark-inputs'):
            for path in sorted((STATE/directory).glob('*.json')):
                tar.add(path,arcname=f'{directory}/{path.name}')
    mean=controls['fit_worker_seconds']/max(1,len(all_scores))
    cost=dict(blocks=90,candidate_languages=5,fitted_input_kinds=['positive','shuffle','copy_mutate','frequency_copy'],
              paired_absent_language_requires_extra_fits=False,fits=90*5*4,mean_measured_fit_seconds=mean,
              same_cost_projection_worker_hours=90*5*4*mean/3600,
              preliminary_budget_worker_hours=2*90*5*4*mean/3600+5,
              not_a_guaranteed_runtime=True,fresh_confirmation_run=False)
    seal(OUT/'confirmation-cost.json',cost)
    mutation=next(r for r in benchmark['rows'] if r['name']=='english-copy' and r['backend']=='incremental')
    proposals_per_sweep=mutation['pairs']+23*mutation['units']
    alpha=.05/5;n=90
    failure_gate=max(k for k in range(n+1) if sum(math.comb(n,j)*.1**j*.9**(n-j) for j in range(k+1))<=alpha)
    text=f'''# Proposed fresh confirmation: not executed

The three follow-ups are development on three released source/key blocks. A larger
study requires a new protocol, new passages, new keys and a source-availability audit.

## Proposed endpoints and sample size

Plan 90 independent source/key blocks covering all five languages and multiple
sources/genres. The intended population is the predeclared language/source mixture;
these counts do not establish a separate error bound for each individual language.
Each block includes a supported positive, a shuffled input, copy/mutate input,
frequency-preserving copying input, and an absent-language decision reusing the
positive's wrong-prior fits. Fit every input under all five priors; transfer sealed keys.

A conservative reference calculation splits alpha=0.05 over five endpoints (positive
sensitivity and four negative classes), alpha=0.01 each. Under independent Bernoulli
trials, zero acceptances in 90 negatives gives a one-sided upper bound
1 - 0.01^(1/90) = {1-alpha**(1/n):.4%}. A sensitivity gate of at least
{n-failure_gate}/90 correctly accepted positives rejects sensitivity <=90% at alpha=0.01
by the binomial tail. Source clustering or nonidentical sampling can invalidate that
simple calculation: define the sampling population and statistical analysis before
freezing, and retain per-language/source results. Paired negative classes are never
pooled to dilute a failure.

## Cost estimate and remaining uncertainty

The completed stronger controls averaged {mean:.1f} fit-worker seconds per fit.
Ninety blocks × four fitted inputs × five priors = **1,800 fits**. Applying the same
cost to all inputs gives **{cost['same_cost_projection_worker_hours']:.1f} fit-worker hours**,
or at least {cost['same_cost_projection_worker_hours']/5:.1f} wall hours with five continuously
busy workers, excluding overhead. A preliminary two-times allowance plus one reserved
five-hour batch is **{cost['preliminary_budget_worker_hours']:.1f} fit-worker hours**.

This is a planning estimate, not an agreed run budget or measured cost for the whole
study. The estimate comes from frequency-preserving controls on only three languages;
mutation controls, other sources and harder keys can cost more. Measure representative
released examples of every fitted input class before selecting the final budget.
Per-fit caps and work limits must remain explicit, with cap outcomes inconclusive.

The released mutation fixture has {mutation['units']} used units, or
{proposals_per_sweep:,} proposals per complete sweep (all 23 letter choices and all
pair swaps). A 20-million-proposal allowance permits only
{20_000_000//proposals_per_sweep} such sweeps. The smaller frequency-preserving controls
can require many more sweeps to complete the inherited 30 kicks. Therefore do not
carry this work allowance into fresh confirmation without first checking convergence
on released mutation examples. No completed follow-up limit was changed in response
to its outcomes.

## Preconditions

```text
Audit enough independent fresh sources and exclude all consumed source IDs
Check projected cost on every development input class
Declare language/source sampling, endpoints, multiplicity and stopping rules
Freeze and commit the candidate rule and selected decoder
Generate new keys and passages only after that freeze
Run the confirmation once and report every failure, cap and omitted case
```

The existing manuscript recovery gate is unchanged. Even a rejection-study pass would
not establish that Voynich belongs to the tested cipher family or identify a word.
'''
    with (OUT/'CONFIRMATION_PLAN.md').open('x') as f:f.write(text)
    print(OUT/'REPORT.md')


if __name__=='__main__':
    report()
