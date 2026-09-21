"""Domain-label classification with physical-folio units and independent controls."""
from collections import Counter,defaultdict
import hashlib
import re
import numpy as np
from voynich.data import cluster,normalise
from voynich.association_complex import radial,ridge_operator

DOMAIN={'herbal':'botanical','pharmaceutical':'botanical','balneological':'people/bathing',
        'astronomical':'celestial/diagrams','cosmological':'celestial/diagrams','zodiac':'celestial/diagrams'}


def forms(raw):
    text=normalise(raw)
    tokens=re.split(r'[.,\s\x1c\x1d!]+',text)
    return [t for t in tokens if t and '?' not in t and '*' not in t],sum(bool(t) and ('?' in t or '*' in t) for t in tokens)


def aggregate(corpus,assignments):
    groups=defaultdict(list);page_metadata={}
    for line in corpus['lines']:
        # Filter before reading/normalizing text: reserved folios never enter features.
        if assignments.get(line['folio'])!='train':continue
        page_metadata.setdefault(line['page'],{k:line[k] for k in ('section','folio','quire','hand')})
        if line['section'] in DOMAIN and line['locator']!='!':groups[cluster(line['folio'],'folio')].append(line)
    rows=[];excluded=[]
    for folio,lines in sorted(groups.items()):
        domains={DOMAIN[r['section']] for r in lines};quires={r['quire'] for r in lines}
        if len(domains)!=1 or len(quires)!=1:
            excluded.append(dict(folio=folio,reason='mixed domain or quire'));continue
        tokens=[];unknown=0
        for line in lines:
            words,n=forms(line['raw']);tokens.extend(words);unknown+=n
        if not tokens:
            excluded.append(dict(folio=folio,reason='no readable forms'));continue
        hands=Counter(r['hand'] for r in lines);kinds=Counter(r['kind'] for r in lines);pages=sorted({r['page'] for r in lines})
        rows.append(dict(folio=folio,domain=next(iter(domains)),quire=next(iter(quires)),
            pages=pages,sections=sorted({r['section'] for r in lines}),tokens=tokens,
            hand_signature='+'.join(sorted(hands)),hands=dict(hands),kinds=dict(kinds),loci=len(lines),unknown=unknown,
            position=85.5 if folio=='85-86' else float(folio)))
    coverage={}
    for section in sorted({r['section'] for r in page_metadata.values()}):
        subset=[r for r in page_metadata.values() if r['section']==section]
        coverage[section]=dict(pages=len(subset),folios=len({cluster(r['folio'],'folio') for r in subset}),
                              hands=dict(Counter(r['hand'] for r in subset)),quires=dict(Counter(r['quire'] for r in subset)))
    return rows,dict(original_categories=coverage,training_pages=len(page_metadata),excluded=excluded)


def hashed_counts(rows,view,bins=2048):
    x=np.zeros((len(rows),bins))
    for i,row in enumerate(rows):
        for word in row['tokens']:
            items=[word] if view=='words' else [word[j:j+n] for n in (1,2,3,4) for j in range(len(word)-n+1)]
            for item in items:
                bucket=int.from_bytes(hashlib.blake2b(item.encode(),digest_size=4).digest(),'big')%bins
                x[i,bucket]+=1
    return x


def controls(rows,strict=False):
    hands=sorted({h for r in rows for h in r['hands']});quires=sorted({r['quire'] for r in rows});result=[]
    for r in rows:
        lengths=np.array([len(w) for w in r['tokens']]);n=r['loci']
        values=[np.log1p(lengths.sum()),np.log1p(len(lengths)),np.log1p(n),np.log1p(len(r['pages'])),float(lengths.mean()),float(lengths.std()),r['unknown']/(len(lengths)+r['unknown'])]
        values += [r['kinds'].get(k,0)/n for k in ('P','L','C','R')]
        values += [r['hands'].get(h,0)/n for h in hands]
        if strict:values += [r['position']/100]+[float(r['quire']==q) for q in quires]
        result.append(values)
    return np.array(result)


def operators(rows,groups):
    groups=np.asarray(groups);n=len(rows)
    counts={view:hashed_counts(rows,view) for view in ('words','characters')}
    base=controls(rows);strong=controls(rows,True)
    maps={name:np.zeros((n,n)) for name in ('prior','controls','strict','words','characters','words_only','characters_only')}
    for group in sorted(set(groups)):
        train=np.flatnonzero(groups!=group);test=np.flatnonzero(groups==group)
        maps['prior'][np.ix_(test,train)]=1/len(train)
        kernels={}
        for name,x in [('controls',base),('strict',strong)]:
            mean=x[train].mean(0);scale=x[train].std(0);scale[scale<1e-12]=1
            kernels[name]=radial((x-mean)/scale,train)
        for name in ('controls','strict'):
            maps[name][np.ix_(test,train)]=ridge_operator(2*kernels[name],train,test)
        for view,x in counts.items():
            idf=np.log((1+len(train))/(1+(x[train]>0).sum(0)))+1
            features=np.log1p(x)*idf;features/=np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-12)
            kernel=np.clip(features@features.T,0,1)**2
            maps[view][np.ix_(test,train)]=ridge_operator(kernels['controls']+kernel,train,test)
            maps[view+'_only'][np.ix_(test,train)]=ridge_operator(2*kernel,train,test)
    return maps


def score(y,prediction,classes):
    matrix=np.zeros((classes,classes),dtype=int)
    for actual,guess in zip(y,prediction):matrix[int(actual),int(guess)]+=1
    count=matrix.sum(1)
    if (count==0).any():raise ValueError('Cannot score macro recall with a missing class')
    recall=np.diag(matrix)/count
    return dict(accuracy=float(np.trace(matrix)/matrix.sum()),macro_recall=float(recall.mean()),
                recall=recall.tolist(),confusion=matrix.tolist())


def predictions(operator,y,classes):
    values=operator@np.eye(classes)[y]
    # Resolve floating-point ties in fixed class order.
    return np.argmax(np.round(values,12),axis=1)


def shuffle(y,strata,rng):
    result=y.copy();strata=np.asarray(strata)
    for stratum in sorted(set(strata)):
        ix=np.flatnonzero(strata==stratum);result[ix]=rng.permutation(y[ix])
    return result


def movable(y,strata):
    strata=np.asarray(strata);total=0;groups=0
    for s in sorted(set(strata)):
        ix=np.flatnonzero(strata==s)
        if len(set(y[ix]))>1:total+=len(ix);groups+=1
    return dict(movable_folios=total,mixed_strata=groups,total_strata=len(set(strata)))
