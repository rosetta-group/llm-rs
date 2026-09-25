"""Post-run diagnosis only; never changes or selects the frozen predictor.

python -m experiments.etruscan_graph_diagnostics
"""

import itertools
import json

from etruscan import graph_alignment as model, scaffolding as old
from experiments.etruscan_graphs import OUT, load, verify
from experiments.etruscan_scaffolding import sha, write


def main():
    verify()
    results=json.loads((OUT/"results.json").read_text())
    _,train,_,views=load()
    views={v["id"]:v for v in views}
    lexicon=model.lexical_evidence(train)
    diagnostics=[]
    for record in results["records"]["complete_graphs"]:
        query=views[record["id"]]
        prediction=record["prediction"]
        gold=tuple(sorted(map(old.frame_key,record["gold"])))
        candidates={old.frame_key(c["frame"]):c for c in prediction["candidates"]}
        ranked=sorted(candidates,key=lambda e:(model.graph_cost((e,),candidates,query,lexicon),e))
        retained=set(ranked[:model.EDGE_LIMIT])
        for entity in query["entities"]:
            incident=next((e for e in ranked if entity in e[1:]),None)
            if incident is not None:
                retained.add(incident)
        # Exact enumeration over the frozen retained edges diagnoses beam loss.
        # It is not another prediction method or an alternative scored result.
        complete=[]
        for size in range(1,len(query["entities"])+1):
            for graph in itertools.combinations(sorted(retained),size):
                if model.valid(graph) and model.connected(graph,query["entities"]):
                    complete.append((model.graph_cost(graph,candidates,query,lexicon),graph))
        complete.sort()
        gold_rank=next((i+1 for i,(_,g) in enumerate(complete) if g==gold),None)
        diagnostics.append({"id":record["id"],"gold_edges":len(gold),
            "gold_edges_generated":sum(e in candidates for e in gold),
            "gold_edges_retained":sum(e in retained for e in gold),
            "gold_graph_rank_exhaustive_retained":gold_rank,
            "gold_graph_cost":model.graph_cost(gold,candidates,query,lexicon) if all(e in candidates for e in gold) else None,
            "beam_best_matches_exhaustive":bool(complete) and tuple(map(tuple,prediction["best_graph"]))==complete[0][1],
            "accepted":bool(prediction["accepted"]),
            "best_graph_exact":tuple(map(tuple,prediction["best_graph"]))==gold,
            "margin":prediction["margin"]})
    payload={"status":"post-run diagnostics; no model changes",
             "results_sha256":sha(OUT/"results.json"),
             "diagnostic_code_sha256":sha(__file__),"records":diagnostics}
    write(OUT/"diagnostics.json",payload)
    print(json.dumps(payload,indent=2))


if __name__=="__main__":
    main()
