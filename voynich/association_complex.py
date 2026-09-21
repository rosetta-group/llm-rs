"""Joint annotation profiles and grouped nonlinear association models."""
from collections import defaultdict
import hashlib
import re
import numpy as np

PARTS = {'root':r'roots?', 'leaf':r'leaf|leaves', 'flower':r'flowers?',
         'stem':r'stems?', 'twig':r'twigs?', 'bulb':r'bulbs?'}
SHAPES = ('round','triangular','conical','square','arrow')
BASE = list(PARTS) + [f'{part}_{color}' for part in PARTS for color in ('light','dark')]
BASE += list(SHAPES) + ['hairy','striped','spotted','edged','large','long','multiple_roots','split_roots']
BASE += ['count_one','count_two','count_three','count_four']
PAIRS = [(i,j) for i in range(len(BASE)) for j in range(i+1,len(BASE))]
NAMES = BASE + [BASE[i]+' + '+BASE[j] for i,j in PAIRS]


def describe(description):
    """Return explicit mentions only. A zero is not a biological absence label."""
    found=set()
    for clause in re.split(r'[,;.!]',description.lower()):
        if '?' in clause or re.search(r'\b(letters?|labels?|labeled|labelled|same as|f\d+[rv])\b',clause):
            continue
        for part,pattern in PARTS.items():
            if re.search(r'\b(?:'+pattern+r')\b',clause): found.add(part)
            if re.search(r'\b(?:light and dark|dark and light)\s+(?:'+pattern+r')\b',clause):
                found.update((part+'_light',part+'_dark'))
            for color in ('light','dark'):
                # Only modifiers between color and noun; do not cross "with light edge".
                modifiers=r'(?:(?:colou?red|large|long|hairy|fuzzy|round|triangular|conical|square)\s+)*'
                if re.search(r'\b'+color+r'\s+'+modifiers+r'(?:'+pattern+r')\b',clause):
                    found.add(part+'_'+color)
        for feature in SHAPES:
            if re.search(r'\b'+feature+r'\b',clause):found.add(feature)
        for feature,pattern in [('hairy',r'hairy|fuzzy'),('striped',r'striped|stripes?|twotone'),
                                ('spotted',r'spotted|spots?|speckled'),('edged',r'edges?|trim'),
                                ('large',r'large|enormous'),('long',r'long')]:
            if re.search(r'\b(?:'+pattern+r')\b',clause):found.add(feature)
        if re.search(r'\bmultiple\b.*\broots?\b',clause):found.add('multiple_roots')
        if re.search(r'\broots?\b.*\bsplit\b',clause):found.add('split_roots')
        for name,number in [('one','1'),('two','2'),('three','3'),('four','4')]:
            if re.search(r'\b(?:'+name+'|'+number+r')\b',clause):found.add('count_'+name)
    base=np.array([float(name in found) for name in BASE])
    return np.concatenate([base,np.array([base[i]*base[j] for i,j in PAIRS])])


def select_profiles(rows,metadata,assignments):
    maxima=defaultdict(int)
    for r in rows:maxima[r['page'],r['group']]=max(maxima[r['page'],r['group']],r['index'])
    selected=[]
    for r in rows:
        page=re.sub(r'^f101v[12]$','f101v',r['page']);meta=metadata.get(page)
        if (meta is None or assignments.get(meta['folio'])!='train' or meta['hand']!='1'
                or r['section']!='pharma' or r['object']!='plant'
                or not re.fullmatch(r'[a-z]+(?:\.[a-z]+)*',r['label'])):continue
        profile=describe(r['description'])
        if not profile[:len(BASE)].any():continue
        selected.append(dict(r,folio=meta['folio'],position=r['index']/max(1,maxima[r['page'],r['group']]),profile=profile))
    return selected


def features(rows):
    groups=sorted({r['group'] for r in rows});readers=sorted({r['transcriber'] for r in rows})
    control=np.array([[len(r['label'].replace('.','')),len(r['label'].split('.')),r['position']]
                     +[float(r['group']==g) for g in groups]+[float(r['transcriber']==t) for t in readers] for r in rows])
    text=np.zeros((len(rows),512))
    for i,r in enumerate(rows):
        label='^'+r['label']+'$'
        grams=[label[j:j+n] for n in (1,2,3,4) for j in range(len(label)-n+1)]
        grams += [label[j]+'~'+label[j+2] for j in range(len(label)-2)]
        for gram in grams:
            bucket=int.from_bytes(hashlib.blake2b(gram.encode(),digest_size=4).digest(),'big')%512
            text[i,bucket]+=1
    text/=np.maximum(np.linalg.norm(text,axis=1,keepdims=True),1e-12)
    return control,text


def radial(x,train):
    distances=np.maximum(0,(x*x).sum(1)[:,None]+(x*x).sum(1)[None,:]-2*x@x.T)
    block=distances[np.ix_(train,train)];positive=block[block>1e-10]
    bandwidth=float(np.median(positive)) if len(positive) else 1.
    return np.exp(-distances/bandwidth)


def ridge_operator(kernel,train,test,penalty=1.):
    """Centered kernel ridge, with intercept. Shape: held-out rows by training rows."""
    k=kernel[np.ix_(train,train)];cross=kernel[np.ix_(test,train)];n=len(train)
    centered=k-k.mean(0)[None,:]-k.mean(1)[:,None]+k.mean()
    cross=cross-cross.mean(1)[:,None]-k.mean(0)[None,:]+k.mean()
    h=np.eye(n)-np.ones((n,n))/n
    return cross@np.linalg.solve(centered+penalty*np.eye(n),h)+1/n


def operators(controls,text,folios):
    folios=np.asarray(folios);size=len(folios)
    result={name:np.zeros((size,size)) for name in ('control','additive','interactions','nonlinear')}
    cosine=np.clip(text@text.T,-1,1)
    for folio in sorted(set(folios)):
        train=np.flatnonzero(folios!=folio);test=np.flatnonzero(folios==folio)
        mean=controls[train].mean(0);scale=controls[train].std(0);scale[scale<1e-12]=1
        control=radial((controls-mean)/scale,train)
        kernels=dict(control=2*control,additive=control+cosine,
                     interactions=control+cosine**2,nonlinear=control+radial(text,train))
        for name,kernel in kernels.items():
            result[name][np.ix_(test,train)]=ridge_operator(kernel,train,test)
    return result


def cosine(a,b):
    return a@b.T/np.maximum(np.linalg.norm(a,axis=1)[:,None]*np.linalg.norm(b,axis=1)[None,:],1e-12)


def ranks(predicted,actual):
    similarities=cosine(predicted,actual);truth=np.diag(similarities)[:,None]
    above=(similarities>truth+1e-10).sum(1)
    tied=(np.abs(similarities-truth)<=1e-10).sum(1)-1
    return 1-(above+.5*tied)/(len(actual)-1)


def relational(predicted,actual):
    ix=np.triu_indices(len(actual),1)
    a=cosine(predicted,predicted)[ix];b=cosine(actual,actual)[ix]
    a-=a.mean();b-=b.mean();denom=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/denom) if denom>1e-12 else 0.


def target_folds(y,folios,pages):
    output=[]
    for folio in sorted(set(folios)):
        train=np.flatnonzero(folios!=folio);test=np.flatnonzero(folios==folio)
        count=y[train].sum(0);active=np.flatnonzero((count>=3)&(count<=len(train)-3))
        if not len(active):raise ValueError('No supported target descriptors in fold')
        mean=y[train][:,active].mean(0);scale=y[train][:,active].std(0)
        local_pages=[np.flatnonzero(pages[test]==p) for p in sorted(set(pages[test])) if (pages[test]==p).sum()>=3]
        output.append(dict(folio=folio,test=test,active=active,mean=mean,scale=scale,pages=local_pages))
    return output


def profile_scores(predicted,y,folds):
    mse=np.zeros(len(y));matching=np.full(len(y),np.nan);relations=[]
    for fold in folds:
        ix=fold['test'];active=fold['active'];scale=fold['scale'];mean=fold['mean']
        a=(y[ix][:,active]-mean)/scale;p=(predicted[ix][:,active]-mean)/scale
        mse[ix]=((a-p)**2).mean(1)
        for page in fold['pages']:
            matching[ix[page]]=ranks(p[page],a[page])
            relations.append((fold['folio'],relational(p[page],a[page])))
    return dict(mse=mse,matching=matching,relations=relations)


def holm(pvalues):
    names=sorted(pvalues,key=pvalues.get);result={};maximum=0
    for index,name in enumerate(names):
        maximum=max(maximum,min(1.,(len(names)-index)*pvalues[name]))
        result[name]=maximum
    return result
