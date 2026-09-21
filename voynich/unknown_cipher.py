"""Generic character/unit mappings, without a supplied cipher family or codebook."""
from collections import Counter
import math
import random
import time

import numpy as np

from voynich.decipher import ALPHABET, gram_ids, recover


def representations(ciphertext):
    candidates = [('characters', list(''.join(ciphertext.split())))]
    units = ciphertext.split()
    if len(units) > 1 and units != candidates[0][1]:
        candidates.append(('units', units))
    return candidates


def conditional_prior(log_quad):
    p=np.exp(log_quad).reshape((-1,len(ALPHABET)))
    return np.log(p/p.sum(axis=1,keepdims=True)).reshape(-1)


def language_score(values, conditional, letter_counts):
    if len(values)<4: return -1e6
    proportions=np.bincount(values,minlength=len(ALPHABET)).astype(float)/len(values)
    prior=(letter_counts+.1)/(letter_counts.sum()+.1*len(ALPHABET))
    seen=proportions>0
    kl=float(np.sum(proportions[seen]*np.log(proportions[seen]/prior[seen])))
    return float(conditional[gram_ids(values,len(ALPHABET))].mean())-kl


def variable_mapping(units, conditional, letter_counts, chunks, *, seed=42, restarts=6, steps=12000, cap=120):
    """An unconstrained homophonic unit-to-one/two-letter map, with bounded annealing."""
    started=time.monotonic(); rng=random.Random(seed); alphabet={c:i for i,c in enumerate(ALPHABET)}
    inventory=sorted(set(units));lookup={s:i for i,s in enumerate(inventory)}
    values=np.array([lookup[s] for s in units]);size=len(inventory)
    choices=[];weights=[]
    for length in (1,2):
        options=[(s,n) for s,n in chunks.items() if len(s)==length]
        choices.append([np.array([alphabet[c] for c in s]+([-1] if length==1 else []),dtype=np.int32) for s,n in options])
        weights.append([n for s,n in options])
    def sample():
        length=rng.randrange(2)
        return rng.choices(choices[length],weights=weights[length],k=1)[0].copy()
    def decode(key):
        result=key[values].reshape(-1)
        return result[result>=0]
    def score(key):
        return language_score(decode(key),conditional,letter_counts)
    initial=np.array([sample() for _ in inventory]);best_key=initial.copy();best=score(initial)
    initial_text=''.join(ALPHABET[c] for c in decode(initial));count=0;hit=False
    for restart in range(restarts):
        key=initial.copy() if restart==0 else best_key.copy()
        if restart:
            for _ in range(max(2,size//5)): key[rng.randrange(size)]=sample()
        current=score(key)
        for step in range(steps):
            if step%100==0 and time.monotonic()-started>=cap:
                hit=True;break
            count+=1;a=rng.randrange(size);saved=key[a].copy();b=None
            if rng.random()<.35 and size>1:
                b=rng.randrange(size);other=key[b].copy();key[a],key[b]=other,saved
            else:
                key[a]=sample()
            candidate=score(key);temperature=.035*(.015**(step/steps));gain=candidate-current
            if gain>=0 or rng.random()<math.exp(max(-700,gain/temperature)):
                current=candidate
                if current>best: best_key=key.copy();best=current
            else:
                key[a]=saved
                if b is not None: key[b]=other
        if hit: break
    return dict(recovered=''.join(ALPHABET[c] for c in decode(best_key)), initial=initial_text,
                language_score=best, proposals=count, cap_hit=hit, seconds=time.monotonic()-started,
                mapping={s:''.join(ALPHABET[c] for c in row if c>=0) for s,row in zip(inventory,best_key)})


def solve(ciphertext, prior, *, seed=42, restarts=6, steps=12000, cap=120):
    conditional=conditional_prior(prior['log_probabilities']);counts=prior['letter_counts'];candidates=[]
    for representation, units in representations(ciphertext):
        inventory=sorted(set(units))
        if len(inventory)<=len(ALPHABET):
            translated=dict(zip(inventory,ALPHABET));coded=''.join(translated[s] for s in units)
            started=time.monotonic()
            # Each restart is individually short; enforce the candidate cap between restarts.
            best=None;completed=0
            for restart in range(restarts):
                if time.monotonic()-started>=cap: break
                row=recover(coded,prior['log_probabilities'],counts,spaces=False,seed=seed+restart,
                            restarts=1,steps=steps)
                encoded=np.array([ALPHABET.index(c) for c in row['recovered']])
                row['language_score']=language_score(encoded,conditional,counts);completed+=1
                if best is None or row['language_score']>best['language_score']: best=row
            best.update(representation=representation,method='bijection',proposals=completed*steps,
                        cap_hit=completed<restarts,seconds=time.monotonic()-started)
            candidates.append(best)
        row=variable_mapping(units,conditional,counts,prior['chunks'],seed=seed,restarts=restarts,steps=steps,cap=cap)
        row.update(representation=representation,method='variable');candidates.append(row)
    winner=max(range(len(candidates)),key=lambda i:candidates[i]['language_score'])
    return dict(selected=winner,candidates=candidates,recovered=candidates[winner]['recovered'])


def chunk_counts(texts):
    counts=Counter()
    for text in texts:
        text=text.replace(' ','')
        counts.update(text);counts.update(text[i:i+2] for i in range(len(text)-1))
    return dict(counts)
