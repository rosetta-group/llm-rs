"""Ordered shape inventory. WORD is a graphic run, never a recognised person."""
import collections
import re
from linear_a import contexts
from linear_a.arithmetic import numeral_value

MARKERS = ('i-jo','u-jo','i-jo-qe','i-je-we','tu-ka-te','tu-ka-te-qe','tu-ka-te-re','ko-wo','ko-wa')


def counted_words(line, table):
    """Accept only WORD{1,3} NUMBER{1,2}; retain order and repeated words.

    No fractions, logograms, damage, internal quantities or cross-line merging.
    Roles use the pinned corpus's aggregate >=.8 logogram-share convention.
    """
    words, run, numbers = [], [], []
    def flush():
        if run:
            words.append('/'.join(run));run.clear()
    for ch in line:
        if contexts._is_fraction(ch):
            return None
        if contexts._is_numeral(ch):
            flush();numbers.append(numeral_value(ch));continue
        sign = contexts._sign_type(ch)
        if sign:
            if numbers or sign not in table or table[sign][1] >= .8:
                return None
            run.append(sign)
        elif ch.isspace() or ch in '\U00010100\U00010101|':
            flush()
        else:
            return None
    flush()
    n = sum(numbers)
    return {'words':words,'count':n} if 1 <= len(words) <= 3 and n in (1,2) else None


def inventory(data):
    table = contexts.sign_table(data)
    hits, rejected = [], collections.Counter()
    for doc,r in sorted(data.items()):
        if r.get('type')!='tablet':
            rejected['not_tablet']+=1;continue
        if not r.get('signs'):
            rejected['no_sign_layer']+=1;continue
        if r.get('conflicts') or any(s.get('certain') is not True for s in r['signs']):
            rejected['conflict_or_uncertain_sign']+=1;continue
        for n,line in enumerate((r.get('unicode_text') or '').splitlines(),1):
            found=counted_words(line,table)
            if found:
                hits.append({'document':doc,'object':r.get('parent_object') or doc,
                             'line_index':n,'raw':line,**found})
    return {'scope':'graphic shapes, no name or kinship labels', 'rows':hits,
            'objects':len({r['object'] for r in hits}),
            'shape_counts':dict(collections.Counter(f"{len(r['words'])}WORD+{r['count']}" for r in hits)),
            'rejected_records':dict(rejected), 'semantic_relations_inferred':0}


def marker_hits(records):
    out = {w:[] for w in MARKERS}
    for doc,item in records:
        for n,line in enumerate((item.get('content') or '').splitlines(),1):
            # Exact literal spellings only; damage and editorial signs are not stripped.
            counts=collections.Counter(t for t in re.split(r'[\s,/]+',line) if t)
            for w in MARKERS:
                if counts[w]:out[w].append({'id':doc,'heading':item.get('heading_short'),
                    'line_index':n,'occurrences':counts[w],'raw':line})
    return {w:{'occurrences':sum(r['occurrences'] for r in rows),
               'objects':len({r['id'] for r in rows}),'lines':rows} for w,rows in out.items()}
