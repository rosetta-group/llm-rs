"""Participant-complete role graphs on 16 new-to-evaluation monuments.

python -m experiments.etruscan_graphs build|freeze|run
"""

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import random

from etruscan import corpus,scaffolding as old,graph_alignment as model
from experiments.etruscan_scaffolding import child,spouse,gift,own,made,sha,write,shuffle_frames

OUT=Path("experiments/etruscan-graphs")
PRIOR=Path("experiments/etruscan-scaffolding")
NULLS=99

# The exact selection was made before any new-model test score. Gold is a graph
# within the five-relation ontology, not a complete translation of every clause.
SPECS=[
    ("AT 1.46",[[0,1],[2],[4,5]],[],[child(),child("e0","e2")],"daughter",
     "Supplied restored name readings; two parents. Age clause outside ontology."),
    ("ETP 181",[[0,1,2],[3],[4]],[],[child(),child("e0","e2")],"daughter",
     "Split papaslis a is retained but grouped inside the audited personal-name anchor."),
    ("ETP 287",[[0,1],[2]],[],[child()],"daughter",
     "Daughter relation editorially supplied; abbreviated names and no explicit daughter word."),
    ("Cr 3.14",[[1,2]],[],[gift()],"transfer","Donor given; recipient unspecified."),
    ("ETP 284",[[2,3]],[],[gift()],"transfer","Different mulu-family spelling, muluvunike."),
    ("ETP 186",[[0,3]],[],[gift()],"transfer","Discontinuous donor; mine is not on the frozen deictic list."),
    ("ETP 128",[[2],[3,4]],[],[gift("UNSPECIFIED","e0"),gift("UNSPECIFIED","e1")],"transfer",
     "Two named recipients, Venel and Velkhae Rasunies. Two transfer edges are required."),
    ("Cr 3.20",[[1],[2,3]],[],[gift("e0","e1")],"transfer","Supplied mi<ni> reading; donor and recipient specified."),
    ("Cr 3.12",[[1]],[],[gift("UNSPECIFIED","e0")],"transfer","Recipient, not donor, is named."),
    ("Cr 2.15",[[1,2]],[],[own()],"ownership","Object/person association; object supplied in English parentheses."),
    ("ETP 289",[[1]],[],[own()],"ownership","Plate of Flave; name written hvlaves."),
    ("ETP 344",[[1,2]],[],[own()],"ownership","Chalice of Laucie Mezenzies."),
    ("Cr 5.3",[[0,1],[2]],[],[child(),made()],"mixed","Parentage and tomb construction in one inscription."),
    ("Ta 1.191",[[0],[1],[3,5]],[],[child(),child("e0","e2")],"mixed",
     "Two parents; mother's name interrupted by clan. Death/age clause outside ontology."),
    ("Cl 1.324",[[0,1,4],[2],[3],[5,6]],[],[child(),child("e0","e2"),spouse("e0","e3")],"mixed",
     "Four people: subject has discontinuous cognomen; split rathum nasa grouped as one husband."),
    ("AV 6.1",[[0,3]],[],[made()],"making","First letter of Thucer is restored; discontinuous maker name."),
]


def build():
    rows=corpus.load_rows()
    records=[]
    for tid,spans,deities,frames,cohort,note in SPECS:
        row=rows[(rows.ID==tid)&rows.Translation.fillna("").str.strip().ne("")].iloc[0]
        records.append({"id":tid,"source_index":int(row.name),"raw":row.Etruscan,"translation":row.Translation,
                        "tokens":[w.replace("kh","ch") for w in corpus.tokens(row.Etruscan,numerals=True)],
                        "entities":[{"tokens":span,"kind":"DEITY" if i in deities else "PERSON"} for i,span in enumerate(spans)],
                        "gold":frames,"targets":frames,"target_family":cohort,"probe_tokens":[],"note":note})
    write(OUT/"manifest.json",records)
    morphology=old.name_morphology()
    write(OUT/"public.json",[old.public_view(r,morphology) for r in records])
    print(json.dumps(barrier(),indent=2))


def load():
    prior=json.loads((PRIOR/"manifest.json").read_text())
    old_views={r["id"]:r for r in json.loads((PRIOR/"public.json").read_text())}
    train=[{"public":old_views[r["id"]],"frames":r["gold"]} for r in prior]
    test=json.loads((OUT/"manifest.json").read_text())
    views=json.loads((OUT/"public.json").read_text())
    return prior,train,test,views


def barrier():
    prior,train,test,views=load()
    train_ids={r["id"] for r in prior}
    test_ids={r["id"] for r in test}
    assert len(train)==30 and len(test)==16 and len(test_ids)==16
    assert train_ids.isdisjoint(test_ids)
    assert {tuple(r["tokens"]) for r in prior}.isdisjoint({tuple(r["tokens"]) for r in test})
    morphology=old.name_morphology()
    assert views==[old.public_view(r,morphology) for r in test]
    for record in test:
        for frame in record["gold"]:
            old.frame_key(frame)
        assert model.valid(tuple(map(old.frame_key,record["gold"])))
        entities={f"e{i}" for i in range(len(record["entities"]))}
        assert model.connected(tuple(map(old.frame_key,record["gold"])),entities),record["id"]
    return {"training_monuments":len(train),"test_monuments":len(test),
            "test_edges":sum(len(r["gold"]) for r in test),
            "max_named_entities":max(len(v["entities"]) for v in views),
            "old_eight_now_training":True,"monuments_and_exact_texts_disjoint":True}


def freeze():
    if (OUT/"results.json").exists():
        raise RuntimeError("Cannot freeze after scoring")
    paths=["etruscan/graph_alignment.py","etruscan/scaffolding.py","etruscan/corpus.py","etruscan/classes.py",
           "experiments/etruscan_graphs.py","experiments/etruscan_scaffolding.py","tests/test_etruscan_graphs.py",
           *[str(OUT/n) for n in ("manifest.json","public.json","AUDIT.md","PROTOCOL.md")],
           *[str(PRIOR/n) for n in ("manifest.json","public.json","results.json","freeze.json")]]
    sources=[str(corpus.LARTH/n) for n in ("Etruscan.csv","ETP_POS.csv")]
    write(OUT/"freeze.json",{"created_utc":datetime.now(timezone.utc).isoformat(),"barrier":barrier(),
                            "files":{p:sha(p) for p in paths},"sources":{p:sha(p) for p in sources}})


def verify():
    frozen=json.loads((OUT/"freeze.json").read_text())
    for section in ("files","sources"):
        for path,expected in frozen[section].items():
            if sha(path)!=expected:
                raise RuntimeError(f"Frozen file changed: {path}")
    barrier()


def score_records(test,predictions):
    return [{"id":r["id"],"target_family":r["target_family"],"gold":r["gold"],"targets":r["gold"],
             "prediction":predictions[r["id"]]} for r in test]


def metrics(records):
    summary=old.score(records)
    p=summary["edge_precision"] or 0.0
    r=summary["edge_recall"]
    summary["edge_f1"]=2*p*r/(p+r) if p+r else 0.0
    summary["exact_graph_accuracy"]=summary["exact_graphs"]/summary["documents"]
    return summary


def run():
    if (OUT/"results.json").exists():
        raise FileExistsError("Results already exist")
    verify()
    prior,train,test,views=load()
    cached=model.geometry(train,views)
    single=model.geometry(train,views,multiple=False)
    variants={"complete_graphs":{},"single_mapping":{"cached":single},
              "partial_allowed":{"cover_all":False},"no_lexical":{"lexical":False}}
    records={}
    for name,options in variants.items():
        opts={"cached":cached,**options}
        records[name]=score_records(test,model.predict(train,views,**opts))
        print(f"Completed {name}",flush=True)
    records["old_local"]=score_records(test,{q["id"]:old.predict(train,q) for q in views})
    records["count_only"]=score_records(test,{q["id"]:old.count_baseline(train,q) for q in views})
    summaries={name:metrics(rs) for name,rs in records.items()}
    # Old eight remain development evidence; use their original 22 training cases.
    development=[r for r in prior if r["targets"]]
    dev_ids={r["id"] for r in development}
    dev_train=[ex for ex in train if ex["public"]["id"] not in dev_ids]
    dev_views=[ex["public"] for ex in train if ex["public"]["id"] in dev_ids]
    dev_predictions=model.predict(dev_train,dev_views)
    dev_candidates={r["id"]:{"target_edges_present":sum(old.frame_key(f) in
                       {old.frame_key(c["frame"]) for c in dev_predictions[r["id"]]["candidates"]} for f in r["targets"]),
                             "targets":len(r["targets"])} for r in development}
    rng=random.Random(240926)
    nulls=[]
    for i in range(NULLS):
        shuffled=shuffle_frames(train,rng)
        predictions=model.predict(shuffled,views,cached=cached)
        nulls.append(metrics(score_records(test,predictions)))
        if (i+1)%10==0:
            print(f"Translation permutation {i+1}/{NULLS}",flush=True)
    primary=summaries["complete_graphs"]
    p=(1+sum(s["edge_f1"]>=primary["edge_f1"] for s in nulls))/(NULLS+1)
    checks={"precision_ge_80pct":(primary["edge_precision"] or 0)>=0.8,
            "edge_recall_ge_60pct":primary["edge_recall"]>=0.6,
            "coverage_ge_50pct":primary["coverage"]>=0.5,
            "exact_graph_accuracy_ge_50pct":primary["exact_graph_accuracy"]>=0.5,
            "f1_beats_old_local_by_15pp":primary["edge_f1"]>=summaries["old_local"]["edge_f1"]+0.15,
            "f1_beats_count_by_15pp":primary["edge_f1"]>=summaries["count_only"]["edge_f1"]+0.15,
            "translation_null_p_le_05":p<=0.05}
    write(OUT/"results.json",{"freeze_sha256":sha(OUT/"freeze.json"),"records":records,"summaries":summaries,
                             "development_predictions":dev_predictions,"development_candidate_coverage":dev_candidates,
                             "nulls":nulls,"p_translation_shuffle":p,"gate_checks":checks,"passed":all(checks.values())})
    print(json.dumps({"summaries":summaries,"gate_checks":checks,"p":p,"passed":all(checks.values())},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("build","freeze","run"))
    args=parser.parse_args()
    {"build":build,"freeze":freeze,"run":run}[args.action]()


if __name__=="__main__":
    main()
