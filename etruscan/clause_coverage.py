"""Weakly align anonymous token spans to predicates and require span coverage.

This is a bounded token-span pilot, not a full Etruscan grammar. Only training
graphs supervise predicate discovery; query translations never enter the model.
"""

import collections
import functools
import itertools

from etruscan import graph_alignment as base, scaffolding as old

MIN_ANCHOR_SUPPORT = 0.60
COVERAGE_WEIGHT = 0.70
LOCAL_WEIGHT = 0.25
MAX_EDGES = 4


def spans(view):
    return [(i,t["lexeme"]) for i,t in enumerate(view["sequence"]) if t["kind"]=="W"]


def role_offsets(view,frame,position):
    """Signed nearest-occurrence distances, clipped to four sequence positions."""
    result=[]
    for role in frame[1:]:
        if not role.startswith("e"):
            result.append(role)
            continue
        positions=[i for i,t in enumerate(view["sequence"]) if t.get("entity")==role]
        nearest=min(positions,key=lambda i:(abs(i-position),i))
        result.append(max(-4,min(4,nearest-position))/4)
    return tuple(result)


def fit(train):
    """Contrastive span/relation association with a background option.

    A stem votes once per monument. Lift against the training relation frequency
    discounts common labels. A unit background score competes with the relation
    scores, allowing weak or confounded stems to remain unassigned.
    """
    totals=collections.Counter()
    counts=collections.defaultdict(collections.Counter)
    for ex in train:
        relations={f[0] for f in ex["frames"]}
        totals.update(relations)
        for stem in {s for _,s in spans(ex["public"])}:
            counts[stem].update(relations)
            counts[stem]["DOCUMENTS"]+=1
    lexicon={}
    for stem,cs in counts.items():
        scores={r:((cs[r]+0.5)/(cs["DOCUMENTS"]+1))/((totals[r]+0.5)/(len(train)+1))
                for r in old.ARITY if cs[r]}
        denominator=1+sum(scores.values())
        support={r:value/denominator for r,value in scores.items()}
        winner=min(support,key=lambda r:(-support[r],r))
        lexicon[stem]={"relation":winner,"support":support[winner],
                       "relation_support":support,"background_support":1/denominator,
                       "documents":cs["DOCUMENTS"]}
    templates=collections.defaultdict(list)
    for ex in train:
        for pos,stem in spans(ex["public"]):
            entry=lexicon[stem]
            if entry["support"]<MIN_ANCHOR_SUPPORT:
                continue
            for frame in ex["frames"]:
                if frame[0]==entry["relation"]:
                    templates[stem].append({"offsets":role_offsets(ex["public"],frame,pos),
                                            "source":ex["public"]["id"]})
    return {"lexicon":lexicon,"templates":dict(templates),"bag":base.lexical_evidence(train)}


def anchors(query,learned):
    result=[]
    for pos,stem in spans(query):
        entry=learned["lexicon"].get(stem)
        if entry and entry["support"]>=MIN_ANCHOR_SUPPORT:
            result.append({"position":pos,"stem":stem,"relation":entry["relation"],"support":entry["support"]})
    return result


def offset_distance(a,b):
    if len(a)!=len(b):
        return 1.0
    return sum(min(1.0,abs(x-y)) if isinstance(x,(int,float)) and isinstance(y,(int,float))
               else float(x!=y) for x,y in zip(a,b))/max(1,len(a))


def attachment_cost(query,frame,anchor,learned,local=True):
    if frame[0]!=anchor["relation"]:
        return 1.0
    if not local:
        return 0.0
    offsets=role_offsets(query,frame,anchor["position"])
    possibilities=[offsets]
    if frame[0]=="SPOUSE_OF":
        possibilities.append(tuple(reversed(offsets)))
    return min((offset_distance(o,t["offsets"]) for o in possibilities
                for t in learned["templates"].get(anchor["stem"],[])),default=1.0)


def matching_cost(rows):
    """Each repeated occurrence needs a distinct edge; leaving one costs one.

    Different stem groups can explain the same edge. One occurrence can coexist
    with several coordinated edges; this is not a one-word/one-fact assumption.
    """
    @functools.lru_cache(None)
    def solve(i,used):
        if i==len(rows):
            return 0.0
        best=1.0+solve(i+1,used)
        for j,cost in enumerate(rows[i]):
            if not (used>>j)&1:
                best=min(best,cost+solve(i+1,used|(1<<j)))
        return best
    return solve(0,0)


def search(candidates,query,learned,coverage=True,local=True):
    evidence=anchors(query,learned)
    groups=collections.defaultdict(list)
    for i,a in enumerate(evidence):
        groups[a["stem"]].append(i)
    attachments={edge:tuple(attachment_cost(query,edge,a,learned,local) for a in evidence)
                 for edge in candidates}

    @functools.lru_cache(None)
    def costs(graph):
        lexical=base.graph_cost(graph,candidates,query,learned["bag"])
        local_cost=sum(min(attachments[e],default=0.0) for e in graph)/len(graph)
        uncovered=0.0
        if coverage and evidence:
            uncovered=sum(matching_cost(tuple(tuple(attachments[e][i] for e in graph) for i in indices))
                          for indices in groups.values())/len(evidence)
        return lexical+LOCAL_WEIGHT*local_cost+COVERAGE_WEIGHT*uncovered,uncovered,local_cost

    ranked_edges=sorted(candidates,key=lambda e:(costs((e,))[0],e))
    keep=set(ranked_edges[:base.EDGE_LIMIT])
    for entity in query["entities"]:
        incident=next((e for e in ranked_edges if entity in e[1:]),None)
        if incident is not None:
            keep.add(incident)
    # Preserve the best available attachment for every grounded token span.
    for i,_ in enumerate(evidence):
        if candidates:
            keep.add(min(candidates,key=lambda e:(attachments[e][i],costs((e,))[0],e)))
    ranked=[]
    for size in range(1,min(MAX_EDGES,len(keep))+1):
        for graph in itertools.combinations(sorted(keep),size):
            if base.valid(graph) and base.connected(graph,query["entities"]):
                ranked.append((costs(graph)[0],graph))
    ranked.sort()
    best_cost,best=ranked[0] if ranked else (None,())
    margin=ranked[1][0]-best_cost if len(ranked)>1 else None
    accept=bool(best and best_cost<=base.MAX_GRAPH_COST and (margin is None or margin>=base.MIN_MARGIN))
    return {"accepted":[list(e) for e in best] if accept else [],
            "best_graph":[list(e) for e in best],"best_cost":best_cost,"margin":margin,
            "anchors":evidence,"edges_retained":len(keep),"graphs_considered":len(ranked),
            "ranked_graphs":[{"graph":[list(e) for e in graph],"cost":cost,
                              "uncovered_span_cost":costs(graph)[1],"local_role_cost":costs(graph)[2]}
                             for cost,graph in ranked[:5]],
            "candidates":[dict(value,frame=list(edge)) for edge,value in sorted(candidates.items())]}


def predict(train,queries,cached=None,coverage=True,local=True,use_spans=True):
    cached=base.geometry(train,queries) if cached is None else cached
    learned=fit(train)
    if not use_spans:
        learned={**learned,"lexicon":{},"templates":{}}
    return {q["id"]:search(base.edge_candidates(train,q,cached[q["id"]]),q,learned,coverage,local)
            for q in queries}
