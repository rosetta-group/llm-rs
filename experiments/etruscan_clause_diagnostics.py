"""Post-run inspection only; never changes or selects the frozen predictor."""

import hashlib
import json
from collections import defaultdict

from etruscan import scaffolding as old,clause_coverage as model
from experiments.etruscan_clauses import OUT,load,verify,summaries
from experiments.etruscan_scaffolding import sha,write


def main():
    verify()
    results=json.loads((OUT/"results.json").read_text())
    prior,train,test,_=load()
    # Spellings are attached only to this post-run human audit, never predictions.
    spellings=defaultdict(set)
    for record in prior+test:
        for word in record["tokens"]:
            spellings[hashlib.sha256(word[:3].encode()).hexdigest()[:16]].add(word)
    learned=model.fit(train)
    diagnostics=[]
    for record in results["records"]["clause_coverage"]:
        p=record["prediction"]
        gold=set(map(old.frame_key,record["gold"]))
        accepted=set(map(old.frame_key,p["accepted"]))
        candidate={old.frame_key(c["frame"]) for c in p["candidates"]}
        diagnostics.append({"id":record["id"],"cohort":record["target_family"],
            "gold_edges":len(gold),"candidate_gold_edges":len(gold&candidate),
            "accepted_correct":len(gold&accepted),"accepted_wrong":len(accepted-gold),
            "best_graph_exact":set(map(old.frame_key,p["best_graph"]))==gold,
            "margin":p["margin"],"best_cost":p["best_cost"],
            "anchors":[dict(a,post_run_spellings=sorted(spellings[a["stem"]]),
                            relation_present_in_reference=a["relation"] in {f[0] for f in gold},
                            training_documents=learned["lexicon"][a["stem"]]["documents"],
                            template_sources=sorted({t["source"] for t in learned["templates"].get(a["stem"],[])}))
                       for a in p["anchors"]]})
    for name,records in results["records"].items():
        assert summaries(records)==results["summaries"][name]
    payload={"status":"post-run diagnostics; no predictor changes",
             "results_sha256":sha(OUT/"results.json"),"diagnostic_code_sha256":sha(__file__),
             "candidate_reference_edges":sum(r["candidate_gold_edges"] for r in diagnostics),
             "documents_without_anchors":sum(not r["anchors"] for r in diagnostics),
             "complex_documents_without_anchors":sum(not r["anchors"] for r in diagnostics if r["cohort"]=="complex"),
             "wrong_accepted_documents":[r["id"] for r in diagnostics if r["accepted_wrong"]],
             "records":diagnostics}
    write(OUT/"diagnostics.json",payload)
    print(json.dumps({k:v for k,v in payload.items() if k!="records"},indent=2))


if __name__=="__main__":
    main()
