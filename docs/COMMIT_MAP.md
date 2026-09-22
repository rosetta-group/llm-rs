# Commit hash map after the 2026-09-22 history rewrite

On 2026-09-22 the local history was rewritten once, before any of these commits had been pushed,
to remove a Hugging Face access token that commit `fea997006b03302e580927d129c987de5667b9a3`
had placed on line 1 of `config.py`. The token was removed from the working tree in the very next
commit, so exactly one tree in the history contained it. The token itself was revoked.

**What changed:** in that one commit the `config.py` blob was replaced by the same line with the
token replaced by the placeholder `REDACTED-HF-TOKEN`. Nothing else in any tree changed. Every
commit before it keeps its hash, except two whose GPG signature header was dropped.
Every commit after it has a new hash because its parent changed.

**Why it matters:** frozen protocols and reports cite the commit at which a method was frozen.
Those citations were updated to the new hashes in the Markdown reports and the research log.
The JSON records under `experiments/*/results.json` still carry the old hashes; they are records
of what the driver saw when it ran and were not edited. Use this table to translate.

The rewrite was done with `git filter-branch --index-filter`, replacing one blob by its object id;
the working tree, the tests and all three `verify` commands were checked before and after.

| Old commit | New commit | Subject | Note |
|---|---|---|---|
| `0498ff0f61276e6dfb98408d82bcee7f19aa562c` | `5387165bc39a446acb71082201f58b884c4606df` | Add script to debug dataset gen | content unchanged; the commit lost its GPG signature header in the rewrite |
| `72ef7c1761fc97d10cccaefa0f1cd4aebba70fa4` | `04df9822dc56fd0373307892975036a042c4c608` | add optional scrambling | content unchanged; the commit lost its GPG signature header in the rewrite |
| `fea997006b03302e580927d129c987de5667b9a3` | `85f05daaba1af6c6ff2c3eab4bf4b1b01b099e75` | config | the rewritten commit: `config.py` line 1 now reads a placeholder |
| `6c1b34478c3e6cd773b65b5ac6bea7b7261c6d6f` | `56cf754327fc95bc63d0e3b53a8b61d3afb595ca` | Implement Voynich research pipeline and record completed experiments | content unchanged; hash changed because an ancestor changed |
| `28ceb16a5b17cbfdbc251baa3ecf36bdd6db91bc` | `49a4522fea1461e1ec38fbf5dbb02782f1ec0522` | Record frozen decipherment benchmarks and image-association studies | content unchanged; hash changed because an ancestor changed |
| `146fa75ce524c5e20ed18f1308cb9fc1ca65f2fe` | `752c3fa5ca752d3d6806ac0cf4814b351072ed14` | Freeze MDL selection, published cipher comparators and historical prose prior | content unchanged; hash changed because an ancestor changed |
| `eea51e8b7ddb4ad1c871f6da6e5229e50a1344fe` | `582f18ae0dda84f16fa9121ec104de2c1a574842` | Record fresh MDL and standard-cipher results with reproducible evaluation archives | content unchanged; hash changed because an ancestor changed |
| `b46ae8a8b9b4204ee0c9c34e78686f25cd87034b` | `7fde3dc88ec0aed3aa573d7ad8527a94695a98b3` | Archive Voynich folio scans and version pixel layout descriptions | content unchanged; hash changed because an ancestor changed |
| `64d37f4d083b0ca477621fbccba4a0ed76f0fb12` | `983542cdf186b631c20e424dd2e8abbae7c9e6f0` | Add joint segmentation-and-EM decoder for codebook-free Naibbe; record development results and freeze fresh protocol | content unchanged; hash changed because an ancestor changed |
| `4f8655120962342b1e835f8b40329ea5d94bb770` | `5bae4789f24931db3f536df0c021bafdbc4485a1` | Record fresh codebook-free Naibbe evaluation: 12.5% CER modern, 33.5% Dante; gate not met | content unchanged; hash changed because an ancestor changed |
| `cae7c8602fae3310b9749f5244a108a44229cbec` | `699ab78e15618ce1a9d8e852627ef867a1dc0aef` | Round two of codebook-free Naibbe: usage pruning, Petrarca verse prior, frozen fresh protocol | content unchanged; hash changed because an ancestor changed |
| `e19566cd2c649115261a620f8c7753c945362a0f` | `982bdd163093aecfc82e5f78b995597c4312ae17` | Record round-two fresh evaluation: 9.5% CER modern, 10.3% Dante; gate not met | content unchanged; hash changed because an ancestor changed |
| `6dd9c28235c87f1fdc064ecf6f5bd63be83af170` | `57bb12975e10e21f4f43199862880d6dba8cbb29` | Round three of codebook-free Naibbe: lexical polish, error attribution, frozen fresh protocol | content unchanged; hash changed because an ancestor changed |
| `fbc149617657323923fdfaf4337d1fdeeaed994b` | `2221e8312b9e8afc3ce53970bbf20f2c64b28684` | Round three: draw fresh passages from the whole corpus; re-freeze before any passage exists | content unchanged; hash changed because an ancestor changed |
| `d11998f67ba9cb22c4d93ebdce49045f8e4f83ff` | `1ac9d5c5b3b2f90b8fac963cbab3055a5f192bfb` | Record round-three fresh evaluation: 8.8% CER modern, 10.5% Dante; gate not met | content unchanged; hash changed because an ancestor changed |
| `ca090d58c121ae72b886a004fb4bdc9493119b21` | `9e00dfaa4d5ba02ea8f7374bc768bf54df5faf27` | Documentation: plain-English overview, results table, repo map, glossary, conventions and reproduction guide | documentation branch tip, `docs/readable-repo` |

Two hashes cited in records do not appear above:

- `e5186d77f296db4203215bf899d562d8445988cd` in `experiments/joint-development-v3/results.json`
  was the first version of the documentation commit, on which the round-three development driver
  happened to run. That commit was rebased before the rewrite and no longer exists; the code and
  inputs it hashed are identical to those of `6dd9c28…` (now `57bb129…`), which the round-three
  freeze records by content hash.
- The commits in `experiments/standard-decipherment/SOURCE_LICENSE.md` belong to external source
  repositories, not to this one.

Short forms used in prose: `fea9970 → 85f05da`, `6c1b344 → 56cf754`, `28ceb16 → 49a4522`,
`146fa75 → 752c3fa`, `64d37f4 → 983542c`, `cae7c86 → 699ab78`, `fbc1496 → 2221e83`.
