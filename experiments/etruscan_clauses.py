"""Frozen clause-coverage pilot. Run with build|develop|freeze|run."""

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import random

from etruscan import corpus,scaffolding as old,graph_alignment as base,clause_coverage as model
from experiments.etruscan_graphs import metrics,score_records
from experiments.etruscan_scaffolding import child,spouse,gift,own,made,sha,write,shuffle_frames

OUT=Path("experiments/etruscan-clauses")
PRIORS=[Path("experiments/etruscan-scaffolding"),Path("experiments/etruscan-graphs")]
NULLS=99

# Audited and fixed before evaluating the new predictor.
SPECS=[
    ("Ru 5.1",[[0,1],[2],[3,4]],[child(),child("e0","e2"),made()],"complex",
     "Split pes nalisa grouped as mother; two parents and tomb construction."),
    ("Cr 5.2",[[0],[1,2],[3,4]],[child("e0","e2"),child("e1","e2"),made(),["MADE","e1","OBJECT"]],"complex",
     "Split av le and laris al retained; two brothers commission construction. Four edges for three names."),
    ("Ta 1.182",[[0,1],[2],[3]],[child(),child("e0","e2"),made()],"complex",
     "Two parents and construction; supplied restored verb ending. Family/living modifiers outside ontology."),
    ("Cr 3.17",[[2,3],[4]],[child(),gift()],"complex",
     "Born from Velethnai represented as parentage, plus dedication by Larth."),
    ("Vs 1.28",[[1,2],[3]],[child(),own()],"complex",
     "Tomb supplied parenthetically; explicit son plus object association."),
    ("AT 1.34",[[2,3],[4]],[child(),own()],"complex",
     "Statlanes supplied restored letter; son supplied parenthetically."),
    ("ETP 335",[[2,3],[4]],[child(),own()],"complex",
     "Mirror association and parenthetically supplied son."),
    ("Vt 1.58",[[1,2],[3]],[child(),own()],"complex",
     "Tomb association plus son; klan spelling stays distinct from clan."),
    ("Ta 1.15",[[0,1],[2],[3,4]],[child(),child("e0","e2")],"kinship",
     "Two parents; death and age outside ontology."),
    ("Ta 1.96",[[0,1],[2],[4,5]],[child(),child("e0","e2")],"kinship",
     "Two parents; uninterpreted camthi eterau outside ontology."),
    ("Cl 1.2261",[[0,1,2],[3],[4,5]],[child(),child("e0","e2")],"kinship",
     "Split tiscusn al grouped as mother; two parents."),
    ("Ta 1.168",[[0,1],[2,3]],[spouse()],"kinship",
     "Restored names; death, age and three unnamed children outside named-participant ontology."),
    ("Cr 3.18",[[1,3]],[gift()],"transfer",
     "Discontinuous donor. Reference explicitly says given BY despite -si forms; do not reinterpret using morphology."),
    ("ETP 303",[[2],[3,4]],[gift("e0","e1")],"transfer",
     "Reference assigns Aranth donor and Thankhvil recipient."),
    ("Ve 3.2",[[0,1],[2,3]],[gift(),gift("e1")],"transfer",
     "Two donors; supplied restored second name and verb. Two edges share the same object."),
    ("ETP 120",[[2,3]],[made()],"making","Restored maker name; fashioned pooled with made."),
    ("Cm 2.32",[[1]],[own()],"ownership","Jug of Limurce."),
    ("Cm 2.65",[[1]],[own()],"ownership","Ceramic supplied in parentheses."),
    ("Vc 6.6",[[0]],[made()],"making","The work of Ruvfie encoded as MADE, not object possession."),
    ("ETP 331",[[1,2]],[own()],"ownership","Plate of Venu Platunalu; plate supplied parenthetically."),
]


def prior_data():
    records=[]
    views={}
    for directory in PRIORS:
        records.extend(json.loads((directory/"manifest.json").read_text()))
        views.update({v["id"]:v for v in json.loads((directory/"public.json").read_text())})
    return records,[{"public":views[r["id"]],"frames":r["gold"]} for r in records]


def build():
    rows=corpus.load_rows()
    records=[]
    for tid,spans,frames,cohort,note in SPECS:
        row=rows[(rows.ID==tid)&rows.Translation.fillna("").str.strip().ne("")].iloc[0]
        records.append({"id":tid,"source_index":int(row.name),"raw":row.Etruscan,"translation":row.Translation,
                        "tokens":[w.replace("kh","ch") for w in corpus.tokens(row.Etruscan,numerals=True)],
                        "entities":[{"tokens":span,"kind":"PERSON"} for span in spans],
                        "gold":frames,"targets":frames,"target_family":cohort,"note":note})
    write(OUT/"manifest.json",records)
    morphology=old.name_morphology()
    write(OUT/"public.json",[old.public_view(r,morphology) for r in records])
    print(json.dumps(barrier(),indent=2))


def load():
    prior,train=prior_data()
    return prior,train,json.loads((OUT/"manifest.json").read_text()),json.loads((OUT/"public.json").read_text())


def barrier():
    prior,train,test,views=load()
    assert len(train)==46 and len(test)==20
    assert len({r["id"] for r in prior+test})==66
    assert {tuple(r["tokens"]) for r in prior}.isdisjoint({tuple(r["tokens"]) for r in test})
    morphology=old.name_morphology()
    assert views==[old.public_view(r,morphology) for r in test]
    for record,view in zip(test,views):
        frames=tuple(map(old.frame_key,record["gold"]))
        assert base.valid(frames) and base.connected(frames,view["entities"])
        assert len(frames)<=model.MAX_EDGES
        used=[i for e in record["entities"] for i in e["tokens"]]
        assert len(used)==len(set(used)) and all(0<=i<len(record["tokens"]) for i in used)
    return {"training_monuments":46,"test_monuments":20,"test_edges":sum(len(r["gold"]) for r in test),
            "complex_monuments":sum(r["target_family"]=="complex" for r in test),
            "monuments_and_exact_texts_disjoint":True,"old_sixteen_are_development":True}


def develop():
    if (OUT/"freeze.json").exists():
        raise RuntimeError("Development action must precede freeze")
    prior,train=prior_data()
    development=prior[30:]
    views=[ex["public"] for ex in train[30:]]
    predictions=model.predict(train[:30],views)
    records=score_records(development,predictions)
    learned=model.fit(train)
    write(OUT/"development.json",{"status":"previously exposed cases, not new evaluation",
          "training_count":30,"records":records,"summary":metrics(records),
          "full_training_lexicon":learned["lexicon"]})
    print(json.dumps(metrics(records),indent=2))


def freeze():
    if (OUT/"results.json").exists():
        raise RuntimeError("Cannot freeze after scoring")
    paths=["etruscan/clause_coverage.py","etruscan/graph_alignment.py","etruscan/scaffolding.py",
           "etruscan/corpus.py","etruscan/classes.py","experiments/etruscan_clauses.py",
           "experiments/etruscan_graphs.py","experiments/etruscan_scaffolding.py","tests/test_etruscan_clauses.py",
           *[str(OUT/n) for n in ("manifest.json","public.json","development.json","AUDIT.md","PROTOCOL.md")],
           *[str(d/n) for d in PRIORS for n in ("manifest.json","public.json","results.json","freeze.json")]]
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


def summaries(records):
    result=metrics(records)
    result["complex"]=metrics([r for r in records if r["target_family"]=="complex"])
    return result


def run():
    if (OUT/"results.json").exists():
        raise FileExistsError("Results already exist")
    verify()
    _,train,test,views=load()
    cached=base.geometry(train,views)
    records={}
    for name,options in {"clause_coverage":{},"no_coverage":{"coverage":False},
                         "no_local_roles":{"local":False},"extended_graph":{"use_spans":False}}.items():
        records[name]=score_records(test,model.predict(train,views,cached=cached,**options))
        print(f"Completed {name}",flush=True)
    records["previous_graph"]=score_records(test,base.predict(train,views,cached=cached))
    records["count_only"]=score_records(test,{q["id"]:old.count_baseline(train,q) for q in views})
    scores={name:summaries(rs) for name,rs in records.items()}
    rng=random.Random(240927)
    nulls=[]
    for i in range(NULLS):
        shuffled=shuffle_frames(train,rng)
        nulls.append(summaries(score_records(test,model.predict(shuffled,views,cached=cached))))
        if (i+1)%10==0:
            print(f"Translation permutation {i+1}/{NULLS}",flush=True)
    primary=scores["clause_coverage"]
    previous=scores["previous_graph"]
    p=(1+sum(n["edge_f1"]>=primary["edge_f1"] for n in nulls))/(NULLS+1)
    checks={"precision_ge_80pct":(primary["edge_precision"] or 0)>=0.8,
            "precision_not_below_previous":(primary["edge_precision"] or 0)>=(previous["edge_precision"] or 0),
            "recall_ge_60pct":primary["edge_recall"]>=0.6,
            "coverage_ge_50pct":primary["coverage"]>=0.5,
            "complex_recall_ge_60pct":primary["complex"]["edge_recall"]>=0.6,
            "complex_exact_ge_50pct":primary["complex"]["exact_graph_accuracy"]>=0.5,
            "complex_exact_beats_previous_by_one":primary["complex"]["exact_graphs"]>=previous["complex"]["exact_graphs"]+1,
            "f1_beats_previous_by_10pp":primary["edge_f1"]>=previous["edge_f1"]+0.1,
            "f1_beats_extended_graph_by_05pp":primary["edge_f1"]>=scores["extended_graph"]["edge_f1"]+0.05,
            "translation_null_p_le_05":p<=0.05}
    write(OUT/"results.json",{"freeze_sha256":sha(OUT/"freeze.json"),"records":records,"summaries":scores,
          "nulls":nulls,"p_translation_shuffle":p,"gate_checks":checks,"passed":all(checks.values())})
    print(json.dumps({"summaries":scores,"gate_checks":checks,"p":p,"passed":all(checks.values())},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("build","develop","freeze","run"))
    {"build":build,"develop":develop,"freeze":freeze,"run":run}[parser.parse_args().action]()


if __name__=="__main__":
    main()
