# Linear A closeout — 2026-09-24

This is the final handoff for the Linear A audit and its Linear B controls on `codex/linear-a-audit`.
**Work is stopped at the user's request; no further experiment or source search is queued.**
This decision concerns this track, not concurrent Etruscan or Voynich work.

**Gate:** a declared requirement before semantic transfer, such as 18/20 successful controls.
**Freeze:** committed hashes of code, settings and source inputs; it preserves a result, not its truth.
**Family reference:** an expression about relatives; it need not identify both ends of a parent–child relation.

## What was done

- Audited the original five rounds, repaired duplicate-sensitive profiles and two unreachable
  statistical tests, and tested a sign-only entry-ending model.
- Reviewed spelling correspondences, account arithmetic, possible personal-name slots,
  `qi-tu-ne`, name/designation controls, son/daughter expressions and Theban *65/FAR readings.
- Preserved protocols, source disagreements, uncertainty, failed controls and generated results.
  The last image-access record is `442c69d`; the last computational freeze is `aeb16d9`.
- Updated the repository's summaries, plan, result index, file map and reproduction instructions.
  Earlier experiment reports and frozen inputs remain historical records.

## Why it was done

The original failures included repairable software/statistical defects, so they did not justify
a general impossibility claim. The repaired tests and follow-ups still support no new Linear A
translation, language identification, parentage edge or gender assignment.

## The unreachable-test repair

The pattern is **the Monte Carlo floor**: with the plus-one correction, the smallest attainable
p-value is \(1/(N+1)\), where \(N\) is the number of null draws.

```text
Compute the smallest attainable p-value from the draw count
Reject settings whose floor cannot pass the declared threshold
Run the fixed comparison with sufficient draws
Keep the original threshold and report the result, including failures
```

1. **Correspondence probe:** 100 draws gave a floor of 1/101 ≈ 0.0099, above 0.05/7 ≈ 0.00714.
   The repair used 9,999 draws. The discovered rule still failed its held-out test (p = 0.1653).
2. **Trade control:** 50 draws gave a floor of 1/51 ≈ 0.0196. With 999 draws, 3/20 samples passed,
   below the required 18/20. Resolution guards now catch impossible settings.
3. **Exploration is not confirmation:** the already-inspected 12-pair spelling pattern reached
   p = 0.0001. Source review retained six pairs, and their onset-conditioned test failed
   (p = 0.1605). More simulation did not supply independent linguistic evidence.

These were experimental significance tests, not a unit test whose expected result was weakened.
See the [repair report](../experiments/linear-a-audit/REPORT.md) for code changes and exact counts.

## Evidence ledger

The corpus counts are 696 readable Linear A **word types**, not 696 running words, and 5,932
DĀMOS Linear B documents. Later manually labelled controls are curated development material;
object holdout does not make the source selection independently blinded.

| Stage | Result and limit | Record |
|---|---|---|
| Original rounds 1–5 | Lexical Greek recovery 10%, 0%, 0% versus 90%; Hittite profile also matches shuffled syllables | [history](LINEAR_A.md) |
| Repair audit | Luwian self-control 19/20, Palaic 20/20; real and shuffled B both Greek 20/20, so profile specificity fails | [report](../experiments/linear-a-audit/REPORT.md) |
| Entry endings | 4/20 size-matched B samples pass versus 18 required; shuffled 0/20; no A scoring | [report](../experiments/linear-a-structure-v2/REPORT.md) |
| Correspondence source audit | 12 pairs reduced to 6; length-conditioned p = .0011 but onset-conditioned p = .1605; specific adaptation lead archived | [report](../experiments/linear-a-correspondence/REPORT.md) |
| Accounting pilot | 7 eligible KN objects versus 33 A objects; not evaluable, no real-data fits | [report](../experiments/linear-a-ledger/REPORT.md) |
| Account benchmark | 9 accounts on 8 objects: 2 strict balances, 6 uncertain/incomplete, 1 incompatible-dimension control | [report](../experiments/linear-a-account-benchmark/REPORT.md) |
| Kinship feasibility | 5 B fixtures; 48 graphic rows on 22 A objects; 2 pair-shaped hits contain non-name signs | [report](../experiments/linear-a-kinship/REPORT.md) |
| Person slots | HT85/117: 43 rows, including 32 counted words and 3 counted logograms; no parentage formula | [report](../experiments/linear-a-person-slots/REPORT.md) |
| Qi-tu-ne | HT7b counted occurrence and HT87/117 headings verified; both person and category interpretations fit | [report](../experiments/linear-a-qi-tu-ne/REPORT.md) |
| Name/designation control | 38 cases on 19 objects; balanced recall 37.4%, coverage 39.4%; deterministic feature ceiling 84.9% below 90% gate | [report](../experiments/linear-b-person-role/REPORT.md) |
| Relational control | 19 expressions on 16 objects; form-only family recall 50%; primary exact-frame model abstains on all 19 | [report](../experiments/linear-b-relations/REPORT.md) |
| Theban *65/FAR | 13 selected objects; 3/6 published conversions disagree with visible components; no eligible closed account | [report](../experiments/linear-b-theban-65/REPORT.md) |
| Image-access review | No full-context target images obtained; Gp124 glyph crop lacks neighbours; no spacing measurement or new semantic label | [report](../experiments/linear-b-theban-images/REPORT.md) |

## What the son/daughter work established

1. **Recognition and assignment differ.** Three intact `i-*65` expressions transfer in the form-only
   B diagnostic. Daughter forms do not. MY Oe106 permits conflicting parent/child assignments;
   four supplied family cases leave the child unnamed. No directed family edges were inferred.
2. **Missing support is not a successful negative control.** The primary relational model makes
   no calls on 19 real cases or any of 196 evaluable label swaps. Zero negative passes cannot
   validate an always-abstaining method. The 59 retrieved marker hits are not a semantic census.
3. **Quantities do not settle the Theban reading.** James's published versus conditional-component
   amounts are Fq252 186/178 Z, Fq254[+]255 169/86 Z and Fq277 525/641 Z. Both columns stay visible.
   These are not restored totals; damaged scope prevents adjudicating “son” versus commodity.
4. **Source disagreements remain explicit.** `te-ja-re` is a provisional edition reading against
   frozen SigLA `te-*56-re`. Fq236's reported hand reassignment and competing spacing arguments
   are literature findings, not a new inspection of the tablet. An Etruscan analogy supplies a
   question, not evidence for the same Linear A formula.

## Stopping point and conditional reopening

The pattern is **an evidence-gated pause**. These conditions document what would change the
assessment; they are not an active work queue or authorization to continue.

```text
Keep this track stopped
If the user explicitly reopens it with new evidence or a distinct falsifiable method:
    Review source identity, damage, scope and competing interpretations
    Define and freeze a new known-language control before semantic scoring
    Apply it to Linear A only if its declared gates pass
Otherwise retain the existing results and failed gates
```

- The parked image lead needs full-context Fq236, Gp124, Gp227 and Fq254[+]255 surfaces and
  the direct AGS reply. TOP I pp94–95 are verified targets for Fq236; other page numbers are
  not established. The [unsent acquisition packet](../experiments/linear-b-theban-images/ACQUISITION.md)
  records exact object IDs and bibliographic details. No external request was sent.
- Even with scans, damage may prevent a verdict. Transcription whitespace and an isolated
  glyph cannot establish sign attachment. Repeat searches of the same failed routes are parked.
- Broader overlap, individual loans, arithmetic inference and `qi-tu-ne`'s semantic class remain
  unresolved. Failed gates do not prove that Linear A lacks usable information, and Naibbe's
  experimental passage lengths are not a universal decipherment threshold.

## Preservation and verification

The closing check runs **74 focused tests**, verifies **11 follow-up freezes**, and checks the
**8 local image-review source hashes**. These are software/provenance checks, not linguistic
validation. Commands and archive requirements are in [REPRODUCE.md](REPRODUCE.md#linear-a-closeout-verification).
The [file map](REPO_MAP.md#linear-a-library-linear_a) locates the implementation; the
[chronological log](../RESEARCH_LOG.md) retains decisions made before later results superseded them.

Raw corpora, scans and HTML snapshots live in git-ignored `artifacts/linear-a-sources/`.
Preserve their pinned bytes: a fresh web download may differ or be unavailable. Duhoux and
Pierini material read only through web extraction has manual notes, not a claimed local PDF.
The image review has a source manifest but no computational freeze. Earlier frozen code,
protocols, inputs and results must not be edited to turn a failed gate into a pass.
