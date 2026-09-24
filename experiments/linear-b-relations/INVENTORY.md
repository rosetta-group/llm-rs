# Reviewed relational expressions

Source labels and spans are supplied development annotations, not model predictions.
N means an onomastic span; W codes preserve only non-name word equality; Q marks supplied terminal -qe.
No case or gender is a model input. FAMILY includes patronymic affiliation, not necessarily an immediate CHILD_OF edge.

| Case | Object | Supplied fragment | Gold | Rival/source kind | Anonymous sequence | Direction status |
|---|---|---|---|---|---|---|
| r001 | MY V-(-) 659 | `o-to-wo-wi-je` / `tu-ka-te-qe` | FAMILY | kinship | N W001 Q | unnamed_child |
| r002 | MY V-(-) 659 | `a-ne-a2` / `tu-ka-te-qe` | FAMILY | kinship | N W001 Q | unnamed_child |
| r003 | MY Oe(-) 106 | `o-te-ra` / `tu-ka-te-re` | FAMILY | kinship | N W002 | disputed_binding |
| r004 | PY Ae(-) 344 | `pi-ṛọ-ẉọ-na` / `wi-do-ẉọ-i-jo` / `i-*65` | FAMILY | kinship | N N W003 | disputed_binding |
| r005 | PY Aq(-) 218 | `qo-te-wo` / `i-*65` | FAMILY | kinship | N W003 | unnamed_child |
| r006 | PY Jn(1) 725 | `wa-ti-ko-ro` / `i-*65-qe` | FAMILY | kinship | N W003 Q | unnamed_child |
| r007 | PY An(3) 654 | `a-re-ku-tu-ru-wo` / `e-te-wo-ke-re-we-i-jo` | FAMILY | patronymic | N N | lineage_not_exact_edge |
| r008 | KN Ai(-) 824 | `a-pi-qo-i-ta` / `do-e-ra` | OTHER | service | N W004 | not_applicable |
| r009 | PY Eb(-) 1187 | `e-ni-to-wo` / `a-pi-me-de-o` / `do-e-ro` | OTHER | service | N N W005 | not_applicable |
| r010 | PY Jn(1) 310 | `do-e-ro` / `ke-we-to-jo` | OTHER | service | W005 N | not_applicable |
| r011 | PY Jn(1) 605 | `do-e-ro` / `pe-re-qo-no-jo` | OTHER | service | W005 N | not_applicable |
| r012 | PY Jn(1) 750 | `e-u-we-to-ro` / `do-e-ro` | OTHER | service | N W005 | not_applicable |
| r013 | KN Am(2) 821 | `ko-pe-re-u` / `e-qe-ta` | OTHER | occupation | N W006 | not_applicable |
| r014 | PY Ae(1) 134 | `ke-ro-wo` / `po-me` | OTHER | occupation | N W007 | not_applicable |
| r015 | PY Ae(1) 108 | `qo-te-ro` / `a3-ki-pa-ta` | OTHER | occupation | N W008 | not_applicable |
| r016 | PY Ae(1) 264 | `pi-ra-jo` / `a3-ki-pa-ta` | OTHER | occupation | N W008 | not_applicable |
| r017 | MY V-(-) 659 | `ri-su-ra` / `qo-ta-qe` | OTHER | pair | N N Q | not_applicable |
| r018 | MY Au(-) 102 | `te-ra-wo` / `ka-ri-se-u-qe` | OTHER | pair | N N Q | not_applicable |
| r019 | PY Jn(1) 725 | `ko-ma-do-ro` / `po-so-ra-ko` | OTHER | pair | N N | not_applicable |

## Unresolved cases excluded from scoring

| Case | Object | Why excluded |
|---|---|---|
| r020 | MY Au(-) 102 | Son versus personal name Ion. Neither reading is silently promoted to gold. |
| r021 | KN Vs(2) 1523 | Hiller accepts participle going; Duhoux argues son. Three intact i-jo occurrences are not three independent objects. |
| r022 | PY Aq(-) 64 | Published son interpretation but doubtful relation-marker sign in pinned corpus; excluded from strict scoring. |
| r023 | PY Jn(1) 431 | Published son interpretation, damaged intervening context and doubtful marker onset; excluded from strict scoring. |
| r024 | MY Oe(-) 112 | Doubtful collective daughters reading, without a secure individual binding. |
| r025 | TH Gp(-) 227 | Attached -u-jo is missed by a bare-word search; parent name begins in a break. Candidate only. |
| r026 | MY Oe(-) 121 | Paper prints a continuous expression where DĀMOS separates words; several name/case analyses remain possible. |

Exact spans, source pages and qualifications: [cases.json](cases.json). Source hashes and access limits: [sources.json](sources.json).
