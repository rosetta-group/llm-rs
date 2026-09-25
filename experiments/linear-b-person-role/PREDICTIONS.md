# Held-out layout predictions

Each case is predicted after withholding every case from its physical tablet. Agreement is the
fraction of object-normalized training votes, not a calibrated probability. Source labels are
the qualified development annotations in [cases.json](cases.json).

| Case | Held-out object | Target | Gold | Prediction | Reason | Training agreement | Supporting training objects |
|---|---|---|---|---|---|---:|---|
| c001 | KN Ai(1) 63 | `pe-se-ro-jo` | PERSON | abstain | conflicting_objects | 66.7% | KN Ai(-) 824, PY An(6) 35, PY An(-) 292, PY An(4) 1, PY Jn(2) 658, PY An(3) 657 |
| c002 | KN Ai(-) 824 | `a-pi-qo-i-ta` | PERSON | abstain | conflicting_objects | 66.7% | KN Ai(1) 63, PY An(6) 35, PY An(-) 292, PY An(4) 1, PY Jn(2) 658, PY An(3) 657 |
| c003 | KN Ai(-) 824 | `do-e-ra` | DESIGNATION | DESIGNATION | predicted | 100.0% | PY An(-) 39, PY An(5) 427 |
| c004 | KN Am(2) 821 | `ko-pe-re-u` | PERSON | PERSON | predicted | 100.0% | KN As(2) 1516, KN As(2) 1517, PY Ae(1) 134, PY Ae(1) 108, PY Ae(1) 264, MY Au(-) 102, PY Jn(2) 658 |
| c005 | KN Am(2) 821 | `e-qe-ta` | DESIGNATION | abstain | conflicting_objects | 80.0% | PY Ae(1) 134, PY Ae(1) 108, PY Ae(1) 264, PY An(-) 18, PY An(-) 261 |
| c006 | KN As(2) 1516 | `a-ra-da-jo` | PERSON | PERSON | predicted | 100.0% | KN Am(2) 821, KN As(2) 1517, PY Ae(1) 134, PY Ae(1) 108, PY Ae(1) 264, MY Au(-) 102, PY Jn(2) 658 |
| c007 | KN As(2) 1516 | `pi-ja-si-ro` | PERSON | PERSON | predicted | 100.0% | KN Am(2) 821, KN As(2) 1517, PY Ae(1) 134, PY Ae(1) 108, PY Ae(1) 264, MY Au(-) 102, PY Jn(2) 658 |
| c008 | KN As(2) 1517 | `a-di-nwa-ta` | PERSON | PERSON | predicted | 100.0% | KN Am(2) 821, KN As(2) 1516, PY Ae(1) 134, PY Ae(1) 108, PY Ae(1) 264, MY Au(-) 102, PY Jn(2) 658 |
| c009 | KN As(2) 1517 | `ti-qa-jo` | PERSON | PERSON | predicted | 100.0% | KN Am(2) 821, KN As(2) 1516, PY Ae(1) 134, PY Ae(1) 108, PY Ae(1) 264, MY Au(-) 102, PY Jn(2) 658 |
| c010 | PY Ae(1) 134 | `ke-ro-wo` | PERSON | PERSON | predicted | 100.0% | KN Am(2) 821, KN As(2) 1516, KN As(2) 1517, PY Ae(1) 108, PY Ae(1) 264, MY Au(-) 102, PY Jn(2) 658 |
| c011 | PY Ae(1) 134 | `po-me` | DESIGNATION | abstain | conflicting_objects | 80.0% | KN Am(2) 821, PY Ae(1) 108, PY Ae(1) 264, PY An(-) 18, PY An(-) 261 |
| c012 | PY Ae(1) 108 | `qo-te-ro` | PERSON | PERSON | predicted | 100.0% | KN Am(2) 821, KN As(2) 1516, KN As(2) 1517, PY Ae(1) 134, PY Ae(1) 264, MY Au(-) 102, PY Jn(2) 658 |
| c013 | PY Ae(1) 108 | `a3-ki-pa-ta` | DESIGNATION | abstain | conflicting_objects | 80.0% | KN Am(2) 821, PY Ae(1) 134, PY Ae(1) 264, PY An(-) 18, PY An(-) 261 |
| c014 | PY Ae(1) 264 | `pi-ra-jo` | PERSON | PERSON | predicted | 100.0% | KN Am(2) 821, KN As(2) 1516, KN As(2) 1517, PY Ae(1) 134, PY Ae(1) 108, MY Au(-) 102, PY Jn(2) 658 |
| c015 | PY Ae(1) 264 | `a3-ki-pa-ta` | DESIGNATION | abstain | conflicting_objects | 80.0% | KN Am(2) 821, PY Ae(1) 134, PY Ae(1) 108, PY An(-) 18, PY An(-) 261 |
| c016 | PY An(6) 35 | `to-ko-do-mo` | DESIGNATION | abstain | tie | 50.0% | KN Ai(1) 63, KN Ai(-) 824, PY An(-) 292, PY An(4) 1, PY Jn(2) 658, PY An(3) 657 |
| c017 | PY An(-) 292 | `si-to-ko-wo` | DESIGNATION | abstain | tie | 50.0% | KN Ai(1) 63, KN Ai(-) 824, PY An(6) 35, PY An(4) 1, PY Jn(2) 658, PY An(3) 657 |
| c018 | PY An(4) 1 | `e-re-ta` | DESIGNATION | abstain | tie | 50.0% | KN Ai(1) 63, KN Ai(-) 824, PY An(6) 35, PY An(-) 292, PY Jn(2) 658, PY An(3) 657 |
| c019 | PY An(-) 18 | `to-ko-do-mo` | DESIGNATION | abstain | conflicting_objects | 80.0% | KN Am(2) 821, PY Ae(1) 134, PY Ae(1) 108, PY Ae(1) 264, PY An(-) 261 |
| c020 | PY An(-) 18 | `te-ko-to-na-pe` | DESIGNATION | abstain | too_few_objects | 100.0% | PY An(-) 261 |
| c021 | PY An(-) 261 | `a-pi-jo-to` | PERSON | abstain | too_few_objects | 100.0% | PY An(-) 18 |
| c022 | PY An(-) 261 | `ke-ro-si-ja` | DESIGNATION | abstain | unseen_signature | — | — |
| c023 | PY An(-) 261 | `o-wo-to` | PERSON | DESIGNATION | predicted | 100.0% | KN Am(2) 821, PY Ae(1) 134, PY Ae(1) 108, PY Ae(1) 264, PY An(-) 18 |
| c024 | PY An(-) 39 | `pu-ka-wo` | DESIGNATION | abstain | unseen_signature | — | — |
| c025 | PY An(-) 39 | `a-to-po-qo` | DESIGNATION | DESIGNATION | predicted | 100.0% | KN Ai(-) 824, PY An(5) 427 |
| c026 | PY An(-) 39 | `a-ko-so-ta` | PERSON | abstain | unseen_signature | — | — |
| c027 | PY An(-) 207 | `ke-ra-me-we` | DESIGNATION | abstain | too_few_objects | 100.0% | PY An(5) 427 |
| c028 | PY An(-) 207 | `ku-ru-so-wo-ko` | DESIGNATION | abstain | too_few_objects | 100.0% | PY An(5) 427 |
| c029 | PY An(5) 427 | `da-ko-ro` | DESIGNATION | abstain | too_few_objects | 100.0% | PY An(-) 207 |
| c030 | PY An(5) 427 | `a-to-po-qo` | DESIGNATION | DESIGNATION | predicted | 100.0% | KN Ai(-) 824, PY An(-) 39 |
| c031 | MY Au(-) 102 | `na-su-to` | PERSON | PERSON | predicted | 100.0% | KN Am(2) 821, KN As(2) 1516, KN As(2) 1517, PY Ae(1) 134, PY Ae(1) 108, PY Ae(1) 264, PY Jn(2) 658 |
| c032 | MY Au(-) 102 | `mo-i-da` | PERSON | PERSON | predicted | 100.0% | KN Am(2) 821, KN As(2) 1516, KN As(2) 1517, PY Ae(1) 134, PY Ae(1) 108, PY Ae(1) 264, PY Jn(2) 658 |
| c033 | MY Au(-) 102 | `a-to-po-qo` | DESIGNATION | abstain | unseen_signature | — | — |
| c034 | PY Jn(2) 658 | `ka-ke-we` | DESIGNATION | abstain | tie | 50.0% | KN Ai(1) 63, KN Ai(-) 824, PY An(6) 35, PY An(-) 292, PY An(4) 1, PY An(3) 657 |
| c035 | PY Jn(2) 658 | `ma-ka-wo` | PERSON | PERSON | predicted | 100.0% | KN Am(2) 821, KN As(2) 1516, KN As(2) 1517, PY Ae(1) 134, PY Ae(1) 108, PY Ae(1) 264, MY Au(-) 102 |
| c036 | PY Jn(2) 658 | `pi-ro-ne-ta` | PERSON | PERSON | predicted | 100.0% | KN Am(2) 821, KN As(2) 1516, KN As(2) 1517, PY Ae(1) 134, PY Ae(1) 108, PY Ae(1) 264, MY Au(-) 102 |
| c037 | PY An(3) 657 | `ma-re-wo` | PERSON | abstain | conflicting_objects | 66.7% | KN Ai(1) 63, KN Ai(-) 824, PY An(6) 35, PY An(-) 292, PY An(4) 1, PY Jn(2) 658 |
| c038 | PY An(3) 657 | `ne-da-wa-ta-o` | PERSON | abstain | conflicting_objects | 66.7% | KN Ai(1) 63, KN Ai(-) 824, PY An(6) 35, PY An(-) 292, PY An(4) 1, PY Jn(2) 658 |
