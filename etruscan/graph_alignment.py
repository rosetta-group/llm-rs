"""Exhaustive participant mappings and constrained whole-inscription role graphs.

This version composes evidenced edges into connected graphs that account for all
named entities. Missing cases remain unknown; no hidden translation is a feature.
"""

import collections
import itertools
import math

from etruscan import scaffolding as old

MAX_ENTITIES = 4
MAX_EDGE_COST = 1.0
EDGE_LIMIT = 18
BEAM = 128
COMPLEXITY = 0.18
LEXICAL_WEIGHT = 0.30
MIN_MARGIN = 0.10
MAX_GRAPH_COST = 0.65


def locations(view):
    result = collections.defaultdict(list)
    denominator = max(1,len(view["sequence"])-1)
    for i,t in enumerate(view["sequence"]):
        if "entity" in t:
            result[t["entity"]].append(i/denominator)
    return {e:sum(xs)/len(xs) for e,xs in result.items()}


def mappings(source,query):
    """Enumerate every injective assignment; no monotonic-order assumption."""
    left,right=sorted(source["entities"]),sorted(query["entities"])
    if max(len(left),len(right))>MAX_ENTITIES:
        raise ValueError("Outside the frozen four-entity pilot scope")
    if len(left)>len(right):
        return []
    sp,qp=locations(source),locations(query)
    src_deictic=sum(t["kind"]=="DEICTIC" for t in source["sequence"])
    qry_deictic=sum(t["kind"]=="DEICTIC" for t in query["sequence"])
    context_cost=0.1*min(abs(src_deictic-qry_deictic),1)
    result=[]
    for perm in itertools.permutations(right,len(left)):
        assignment=dict(zip(left,perm))
        costs=[]
        for a,b in assignment.items():
            x,y=source["entities"][a],query["entities"][b]
            cost=0.5*(x["kind"]!=y["kind"])
            for key,weight in (("case",0.45),("gender",0.15)):
                if x[key]!="?" and y[key]!="?" and x[key]!=y[key]:
                    cost+=weight
            cost+=0.2*abs(sp[a]-qp[b])
            costs.append(cost)
        result.append({"mapping":assignment,"cost":sum(costs)/max(1,len(costs))+context_cost})
    return sorted(result,key=lambda r:(r["cost"],sorted(r["mapping"].items())))


def geometry(train,queries,multiple=True):
    """Cache label-independent mappings for real and permuted training graphs."""
    output={}
    for query in queries:
        rows=[]
        for index,ex in enumerate(train):
            if ex["public"]["id"]==query["id"]:
                raise ValueError("Train/test monument overlap")
            if multiple:
                options=mappings(ex["public"],query)
            else:
                cost,assignment=old.align(ex["public"],query)
                options=[{"mapping":assignment,"cost":cost}]
            for option in options:
                if option["cost"]<=MAX_EDGE_COST:
                    rows.append(dict(option,source=ex["public"]["id"],training_index=index))
        output[query["id"]]=rows
    return output


def lexical_evidence(train):
    """Learn weak stem/relation associations from visible training graphs only.

    One count per monument, no supplied predicate-word alignment. English glosses
    and raw Etruscan spellings are absent: stems are opaque IDs in public inputs.
    """
    counts=collections.defaultdict(collections.Counter)
    for ex in train:
        stems={t["lexeme"] for t in ex["public"]["sequence"] if t["kind"]=="W"}
        relations={f[0] for f in ex["frames"]}
        for stem in stems:
            for relation in old.ARITY:
                counts[stem][relation]+=relation in relations
            counts[stem]["total"]+=1
    return {stem:{r:math.log((cs[r]+1)/(cs["total"]-cs[r]+1)) for r in old.ARITY}
            for stem,cs in counts.items()}


def edge_candidates(train,query,cached):
    candidates={}
    for row in cached:
        for frame in train[row["training_index"]]["frames"]:
            mapped=old.transfer(frame,row["mapping"])
            if mapped is None:
                continue
            previous=candidates.get(mapped)
            if previous is None or row["cost"]<previous["cost"]:
                candidates[mapped]={"cost":row["cost"],"source":row["source"],"mapping":row["mapping"]}
    return candidates


def mentioned(graph):
    return {arg for edge in graph for arg in edge[1:] if arg.startswith("e")}


def valid(graph):
    """Pilot assumptions: no kinship cycles, no child/spouse conflict, one object event type."""
    child=collections.defaultdict(set)
    spouses=set()
    object_types=set()
    for relation,*args in graph:
        if relation=="CHILD_OF":
            child[args[0]].add(args[1])
        elif relation=="SPOUSE_OF":
            spouses.add(frozenset(args))
        else:
            object_types.add(relation)
    if len(object_types)>1 or any(len(parents)>2 for parents in child.values()):
        return False
    if any(frozenset((a,b)) in spouses for a,parents in child.items() for b in parents):
        return False
    def cycle(node,path):
        if node in path:
            return True
        return any(cycle(parent,path|{node}) for parent in child.get(node,()))
    return not any(cycle(node,set()) for node in list(child))


def connected(graph,entities):
    if not graph or mentioned(graph)!=set(entities):
        return False
    adjacency=collections.defaultdict(set)
    for edge in graph:
        nodes=[x for x in edge[1:] if x!="UNSPECIFIED"]
        for a in nodes:
            adjacency[a].update(x for x in nodes if x!=a)
    reached=set()
    stack=[next(iter(entities))]
    while stack:
        node=stack.pop()
        if node not in reached:
            reached.add(node)
            stack.extend(adjacency[node]-reached)
    return set(entities)<=reached


def graph_cost(graph,candidates,query,lexicon):
    average=sum(candidates[e]["cost"] for e in graph)/len(graph)
    relations={e[0] for e in graph}
    stems={t["lexeme"] for t in query["sequence"] if t["kind"]=="W"}
    support=sum(max(0.0,*(lexicon.get(s,{}).get(r,0.0) for r in relations)) for s in stems)/max(1,len(stems))
    return average+COMPLEXITY*(len(graph)-1)-LEXICAL_WEIGHT*support


def search(candidates,query,lexicon,cover_all=True):
    """Bounded graph composition; generated-edge inventory is preserved separately."""
    ranked_edges=sorted(candidates,key=lambda e:(graph_cost((e,),candidates,query,lexicon),e))
    # Preserve each participant's cheapest incident edge as well as the global top.
    keep=set(ranked_edges[:EDGE_LIMIT])
    for entity in query["entities"]:
        first=next((e for e in ranked_edges if entity in e[1:]),None)
        if first is not None:
            keep.add(first)
    edges=sorted(keep)
    beam={()}
    complete={}
    for _ in range(max(1,len(query["entities"]))):
        expanded=set()
        for graph in beam:
            for edge in edges:
                if edge in graph:
                    continue
                new=tuple(sorted((*graph,edge)))
                if valid(new):
                    expanded.add(new)
        for graph in expanded:
            if not cover_all or connected(graph,query["entities"]):
                complete[graph]=graph_cost(graph,candidates,query,lexicon)
        # Missing-entity penalty applies to search priority, not the reported cost.
        beam=set(sorted(expanded,key=lambda g:(graph_cost(g,candidates,query,lexicon)
                    +0.4*len(set(query["entities"])-mentioned(g)),g))[:BEAM])
        if not beam:
            break
    ranked=sorted(complete,key=lambda g:(complete[g],g))
    best=ranked[0] if ranked else None
    margin=complete[ranked[1]]-complete[best] if len(ranked)>1 else None
    accepted=bool(best is not None and complete[best]<=MAX_GRAPH_COST
                  and (margin is None or margin>=MIN_MARGIN))
    return {"accepted":[list(e) for e in best] if accepted else [],
            "best_graph":[list(e) for e in best] if best else [],
            "best_cost":complete[best] if best else None,"margin":margin,
            "graphs_considered":len(complete),"edges_retained":len(edges),
            "ranked_graphs":[{"graph":[list(e) for e in g],"cost":complete[g]} for g in ranked[:5]]}


def predict(train,queries,cached=None,cover_all=True,lexical=True):
    if cached is None:
        cached=geometry(train,queries)
    lexicon=lexical_evidence(train) if lexical else {}
    out={}
    for query in queries:
        candidates=edge_candidates(train,query,cached[query["id"]])
        result=search(candidates,query,lexicon,cover_all)
        result["candidates"]=[dict(v,frame=list(e)) for e,v in sorted(candidates.items())]
        out[query["id"]]=result
    return out
