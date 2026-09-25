"""Pinned passages for the known-answer key-recovery confirmation. See experiments/key-recovery-confirmation/PROTOCOL.md.

python -m experiments.key_recovery_confirmation_sources build | verify

build   extract every selected work from raw files already on disk (no network), filter 80-word chunks
        against the reference shingle set R and against every other selected work, cut one 5,200-letter
        passage per work (Occitan: two works give a second, distant passage), seal partitions.json and
        the committed sources.json manifest.
verify  recompute every raw and reference-input hash and rebuild the partitions in memory; the JSON
        must be byte-identical to the pinned partitions.json.

Works are fixed in WORKS and never chosen by score. Blocks and keys are drawn later, at prepare time.
"""
import argparse
from collections import Counter, OrderedDict
import hashlib
import json
import re
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile

from experiments import language_coverage_sources as lc
from experiments import language_expansion_sources as le
from experiments.language_coverage_sources import ROOT, chunks, exact_take, normalize_source
from experiments.rejection_transfer_v2 import exclusions, grams, read, seal
from voynich.corpora import conllu_sentences

OUT = ROOT / 'experiments/key-recovery-confirmation'
STATE = ROOT / 'artifacts/key-recovery-confirmation'
RAW = ROOT / 'artifacts/confirmation-sources/raw'
PASSAGE = 5200
GAP = 50_000                 # Occitan second passages start this many filtered letters after the first ends
MAX_LATIN_SHARE = 0.20       # German works must be German text, not Latin-dominant

SELECTION_RULE = (
    'Works fixed before any run by date, genre and length, never by model score; no substitution. '
    'Each work is normalized (language_coverage_sources.normalize_source: accents removed, j->i, k->c, w->uu, '
    'letters a-z only) and cut into 80-word chunks (chunks()). A chunk is removed if any 8-word shingle of it, '
    'with the 7-word tail of the previous kept chunk, is in R (every earlier train, calibration, challenge and '
    'released text) or occurs in any other selected work (whole text, all languages). The passage is the first '
    f'{PASSAGE} letters of the kept chunks (exact_take: spaces removed, last chunk trimmed); rows keep the whole '
    'chunks used. Occitan passages 5-6 are second passages of BmT-2884 and Français_13504 starting at least '
    f'{GAP} kept letters after the end of the last chunk of their first passage, filtered again against the first '
    'passage. A selected work that cannot yield the letters stops the build.')

# ---------------------------------------------------------------- catalogue
WS_LICENSE = ('Public-domain original and edition; Wikisource transcription CC BY-SA 4.0 '
              '(Wikimedia terms of use), see page history')
OPENMEDFR = 'https://raw.githubusercontent.com/OpenMedFr/texts/0d3112783556775fa30ab4bdac84a4c383cc0217/'
GESTE = 'https://raw.githubusercontent.com/Jean-Baptiste-Camps/Geste/71737a20632383cd5c88d9547ba15ea8feb1f641/txt/norm/'
GUM = 'https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/1fe635509c649e376dfb449d528424ab78f4eaee/'
LIC_OMF = 'CC BY-NC-SA 4.0 (OpenMedFr repository README; editions out of copyright or CC-licensed)'
LIC_GESTE = 'medieval text: Public Domain Mark 1.0; encoding/annotation: CC BY-SA 4.0 (Geste README)'
LIC_GUM = 'treebank CC BY-NC-SA 4.0 (GUM LICENSE.txt); underlying text: '
LIC_PG = 'Public domain in the USA (Project Gutenberg); PG header/footer licence text stripped'
LIC_REM = 'CC BY-SA 4.0 (ReM v2.1 TEI header)'
LIC_DIAKORP = 'CC BY-NC-SA 4.0 (Kučera and Stluka 2011, DIAKORP v5; Pettersson and Megyesi 2018, HistCorp)'
LIC_COMETA = 'CC BY 4.0 (Marinus Wiedner 2025, COMETA v1, Zenodo 15300719)'
REM_URL = 'https://zenodo.org/api/records/13982324/files/ReM-v2.1_tei.zip/content'
DIAKORP_URL = 'https://zenodo.org/records/10013189/files/czech-diakorp-txt.zip?download=1'
COMETA_URL = 'https://zenodo.org/api/records/15300719/files/{}/content'


def W(lang, id, title, author, date, genre, drop=None, sub=None, edition=None):
    return dict(language=lang, id=id, title=title, author=author, date=date, genre=genre, source='wikisource',
                wiki={'latin': 'la', 'italian': 'it', 'catalan': 'ca'}[lang], drop=drop, sub=sub,
                edition=edition, license=WS_LICENSE)


WORKS = [
    # ---------------- Latin (la.wikisource) ----------------
    W('latin', 'einhard-vita-karoli', 'Vita Karoli Magni', 'Einhard', 'c. 817-830', 'biography / history',
      drop=r'^Karolus gratia dei rex Francorum'),
    W('latin', 'navigatio-brendani', 'Navigatio Sancti Brendani Abbatis', 'anonymous', '9th-10th c.',
      'voyage narrative / hagiography'),
    W('latin', 'liutprand-antapodosis', 'Antapodosis', 'Liutprand of Cremona', 'c. 958-962', 'history',
      sub=[(r'\(\s*An\.\s*\d+\s*\)', ' ')], edition='Migne, Patrologia Latina 136'),
    W('latin', 'gesta-francorum', 'Gesta Francorum et aliorum Hierosolimitanorum', 'anonymous', 'c. 1100-1101',
      'crusade chronicle', sub=[(r'\[[ivxlc]+\]', ' ')]),
    W('latin', 'petrus-alfonsi-disciplina', 'Disciplina clericalis', 'Petrus Alfonsi', 'early 12th c.',
      'exempla / wisdom tales'),
    W('latin', 'isidore-etymologiae-4', 'Etymologiae, liber IV (De medicina)', 'Isidore of Seville', 'c. 625',
      'medical', sub=[(r'\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\.', ' ')]),
    # ---------------- Italian (it.wikisource) ----------------
    W('italian', 'fioretti', 'Fioretti di San Francesco', 'anonymous (Tuscan volgarizzamento)', 'late 14th c.',
      'hagiography / exempla', edition='A. Cesari 1860'),
    W('italian', 'passavanti-specchio', 'Specchio di vera penitenza', 'Jacopo Passavanti', 'c. 1354',
      'sermon treatise / exempla', sub=[(r'\([^()]*\d[^()]*\)', ' ')]),
    W('italian', 'giamboni-vizi', "Libro de' vizî e delle virtudi", 'Bono Giamboni', 'late 13th c.', 'moral allegory'),
    W('italian', 'fiore-di-virtu', 'Fiore di virtù', 'anonymous', 'early 14th c.', 'moral treatise', edition='A. Gelli'),
    W('italian', 'caterina-lettere', 'Lettere', 'Caterina da Siena', '1370-1380', 'letters',
      edition='Misciattelli 1922 (Tommaseo text)'),
    W('italian', 'bernardino-novellette', 'Novellette ed esempi morali', 'Bernardino da Siena', '1427 (sermons)',
      'sermon exempla'),
    # ---------------- Catalan (ca.wikisource) ----------------
    W('catalan', 'muntaner-cronica', 'Crònica', 'Ramon Muntaner', 'c. 1325-1328', 'chronicle',
      drop=r'^CAPITOL\b', edition='K. Lanz, Stuttgart 1844'),
    W('catalan', 'desclot-cronica', 'Crònica', 'Bernat Desclot', 'c. 1283-1288', 'chronicle', edition='J. Coroleu 1885'),
    W('catalan', 'llull-besties', 'Llibre de les bèsties', 'Ramon Llull', 'c. 1288-1289', 'animal fable',
      edition='1905 (normalized spelling)'),
    W('catalan', 'curial-guelfa', 'Curial e Güelfa (llibre I)', 'anonymous', 'c. 1440-1460', 'chivalric romance',
      edition='A. Rubió i Lluch 1901'),
    W('catalan', 'tirant', 'Tirant lo Blanc (vol. 1)', 'Joanot Martorell', '1460-1464 (pr. 1490)', 'chivalric romance',
      edition='M. Aguiló 1873-1905 (1490 princeps spelling, long s)'),
    W('catalan', 'pere-iv-cronica', 'Crònica de Pere el Cerimoniós', 'Pere III of Aragon (IV) and chancery',
      'c. 1375-1383', 'chronicle', sub=[(r'\bCap\.\s+[a-zA-ZàèéíòóúÀ-Ú]+\.', ' ')],
      edition='A. de Bofarull 1850 / 1885 reprint'),
    # ---------------- Old French ----------------
    dict(language='old_french', id='openmedfr:ErecF', source='openmedfr', title='Erec et Enide',
         author='Chrétien de Troyes', date='c. 1170 (ed. Foerster 1909)', genre='verse romance'),
    dict(language='old_french', id='openmedfr:ThebesC', source='openmedfr', title='Le Roman de Thèbes',
         author='anonymous', date='c. 1150 (ed. Constans 1890)', genre='verse romance'),
    dict(language='old_french', id='openmedfr:MeraugisF', source='openmedfr', title='Meraugis de Portlesguez',
         author='Raoul de Houdenc', date='early 13th c. (ed. Friedwagner 1897)', genre='verse romance'),
    dict(language='old_french', id='openmedfr:CleomHas', source='openmedfr', title='Li roumans de Cléomadès',
         author='Adenet le Roi', date='c. 1283 (ed. van Hasselt 1865-66)', genre='verse romance'),
    dict(language='old_french', id='geste:ed_HuonG', source='geste', title='Huon de Bordeaux', author='anonymous',
         date='early 13th c. (DEAF siglum HuonG: ed. Guessard/Grandmaison)', genre='chanson de geste'),
    dict(language='old_french', id='openmedfr:DancusM', source='openmedfr',
         title='Livre du roi Dancus (falconry treatise, medieval translation)', author='anonymous',
         date='13th c. (ed. Martin-Dairvault 1883)', genre='prose treatise'),
    # ---------------- English ----------------
    dict(language='english', id='gutenberg:2814', source='gutenberg', title='Dubliners', author='James Joyce',
         date='1914', genre='fiction (short stories)', license=LIC_PG,
         url='https://www.gutenberg.org/cache/epub/2814/pg2814.txt'),
    *[dict(language='english', id='gum:' + doc, source='gum', genre=genre, license=LIC_GUM + lic)
      for doc, genre, lic in [
          ('GUM_news_warhol', 'news', 'Wikinews CC BY 2.5'),
          ('GUM_essay_walking', 'essay', 'Open essay reader (openwa.pressbooks.pub); GUM lists essays under its '
                                         'CC BY-NC-SA 4.0 release'),
          ('GUM_academic_theropod', 'academic', 'PLOS ONE CC BY 4.0'),
          ('GUM_interview_herrick', 'interview', 'Wikinews CC BY 2.5'),
          ('GUM_textbook_history', 'textbook', 'OpenStax CC BY 4.0')]],
    # ---------------- German (ReM v2.1, norm forms) ----------------
    *[dict(language='german', id='rem:' + m, source='rem', rem=m, genre=genre, license=LIC_REM, url=REM_URL)
      for m, genre in [('M214', 'sermons'), ('M402', 'gospel lectionary'), ('M408', 'martyrology'),
                       ('M411', 'town law book'), ('M189', 'psalm fragments'), ('M353', 'charters')]],
    # ---------------- Czech (DIAKORP v5) ----------------
    *[dict(language='czech', id='czech:' + n, source='diakorp', diakorp=n, license=LIC_DIAKORP, url=DIAKORP_URL)
      for n in ['diakorp54-1380-1400', 'diakorp55-1495', 'diakorp58-1350-1400', 'diakorp59-1400-1450',
                'diakorp67-1350-1400', 'diakorp44-1500-1520']],
    # ---------------- Occitan (COMETA v1) ----------------
    *[dict(language='occitan', id='occitan:' + n, source='cometa', cometa=n, title=title,
           author='not stated in COMETA', date='medieval (COMETA; manuscript dates not homogenized)',
           genre=genre, license=LIC_COMETA, url=COMETA_URL.format(urllib.parse.quote(n + '.txt')))
      for n, title, genre in [
          ('BmT-2884', "Las leys d'amors (Toulouse, BmT 2884)", 'poetics / grammar treatise'),
          ('Français_13504', "Vie de saint Elzéar; vision de Marguerite d'Oingt; Vie de sainte Delphine", 'hagiography'),
          ('Arsenal_6355', 'La Vida de santa Enimia', 'verse hagiography'),
          ('NAF_11180', 'Planh de la Vierge', 'devotional prose')]],
]
SECOND_PASSAGES = ['occitan:BmT-2884', 'occitan:Français_13504']
LANGUAGES = ['latin', 'italian', 'catalan', 'old_french', 'english', 'german', 'czech', 'occitan']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_record(path, **extra):
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size, **extra)


# ---------------------------------------------------------------- Wikisource (Latin, Italian, Catalan)
WS_STRIP = ('style, script, table, sup, ol.references, .references, .reference, .mw-editsection, '
            '.mw-heading, h1, h2, h3, h4, h5, h6, .pagenum, .numeropagina, .ws-pagenum, .ws-noexport, '
            '.noprint, .metadata, .titolo, .sottotitolo, #headerContainer, .titulusHeaderBox, .centertext, .ct, .poem, '
            'div.poem, .ws-summary, .sisitem, .reflist, .error')
WS_SUBS = [
    (r'\[\s*\d+[a-z]?\s*\]', ' '),                 # bracketed note / folio numbers, e.g. [7b]
    (r'\(\s*\d+\s*\)', ' '),                       # footnote call numbers, e.g. (1)
    (r'\b\d+\.\d{4}[A-D]\|', ' '),                  # Migne PL column markers, e.g. 136.0789D|
]
MIN_WORDS = 8            # shorter paragraphs (titles, rubric numbers, signatures) are dropped
WS_RULE = ('MediaWiki parse-API JSON (raw bytes pinned, revid recorded), pages in index-link order. Removed: '
           + WS_STRIP + '. Drop capitals restored from image alt text. Only <p> paragraphs; bracketed and '
           'parenthesized numbers and Migne column markers removed; paragraphs >70% capitals or with fewer than '
           f'{MIN_WORDS} normalized words dropped; per-work drop/sub regexes as in WORKS.')


def ws_paragraphs(raw, work):
    """Prose <p> paragraphs of one parse-API response, apparatus removed."""
    from bs4 import BeautifulSoup
    parse = json.loads(raw)['parse']
    soup = BeautifulSoup(parse['text'], 'html.parser')
    root = soup.select_one('.mw-parser-output') or soup
    paragraphs = root.find_all('p')
    for img in root.select('img'):
        alt = img.get('alt') or ''
        if re.fullmatch(r'[A-ZÀ-ÝÇ]', alt):
            (img.find_parent('span', attrs={'typeof': 'mw:File'}) or img).replace_with(alt)
    for node in root.select(WS_STRIP):
        node.decompose()
    out = []
    for p in paragraphs:
        if getattr(p, 'decomposed', False):
            continue
        for br in p.select('br'):
            br.replace_with(' ')
        text = ' '.join(p.get_text().split())
        if work.get('drop') and re.search(work['drop'], text):   # drop rules see the unedited paragraph
            continue
        for pattern, repl in WS_SUBS + list(work.get('sub') or []):
            text = re.sub(pattern, repl, text)
        text = ' '.join(text.split())
        if caps_share(text) > 0.7:
            continue
        if len(normalize_source(text).split()) < MIN_WORDS:
            continue
        out.append(text)
    return parse['revid'], parse['title'], out


def caps_share(text):
    letters = [c for c in text if c.isalpha()]
    return sum(c.isupper() for c in letters) / len(letters) if letters else 0.0


def wikisource(work):
    files, paragraphs = [], []
    for path in sorted((RAW / work['language'] / work['id']).glob('*.json')):
        raw = path.read_bytes()
        if path.name == '000-index.json':
            data = json.loads(raw)['parse']
            files.append(file_record(path, page=data['title'], revid=data['revid'], role='index (link order only)',
                                     url=f"https://{work['wiki']}.wikisource.org/w/index.php?oldid={data['revid']}"))
            continue
        revid, title, ps = ws_paragraphs(raw, work)
        files.append(file_record(path, page=title, revid=revid,
                                 url=f"https://{work['wiki']}.wikisource.org/w/index.php?oldid={revid}"))
        paragraphs += ps
    work['url'] = files[0]['url']
    return '\n'.join(paragraphs), files, WS_RULE


# ---------------------------------------------------------------- Old French
ROMAN = re.compile(r'\.\s*[ivxlcdmjIVXLCDMJ]+[mc]?\s*\.')   # manuscript numerals like .xvj. / . VII .
OMF_RULE = ('#META# lines; *** START/END; page/folio marker lines (PageV01P001, Folio1r); column refs (361a.1); '
            '"# |" and "\\" separators; lines >=80% capitals (editorial titles, EXPLICIT rubrics); glued verse '
            'numbers; (fol. N x) refs; "[...]" lacunae; "*" and "§" marks; dot-delimited numerals (.xvj.); '
            'line-end hyphenation joined')
GESTE_RULE = 'glued line numbers at line start; dot-delimited manuscript numerals (. VII .)'


def join_hyphens(text):
    return re.sub(r'(?<=\w)-[ \t]*\n[ \t]*(?=\w)', '', text)


def caps_heading(line):
    letters = [c for c in line if c.isalpha()]
    return len(letters) >= 3 and sum(c.isupper() for c in letters) / len(letters) >= 0.8


def openmedfr_clean(text):
    text = text.lstrip('﻿').replace('\r\n', '\n').replace('\r', '\n')
    out = []
    for line in text.split('\n'):
        s = line.strip()
        if line.startswith('#META#') or not s or s.startswith('***') or s in ('\\', '# |') or s.startswith('-----'):
            continue
        if re.fullmatch(r'(?i)(page|folio)[\w.]*', s) or re.fullmatch(r'\(\d+[a-z]?\.\d+\)', s) or caps_heading(s):
            continue
        s = re.sub(r'^\d+(?=\D)', '', s)
        s = re.sub(r'\((?:fol|folio)\.?[^)]*\)', ' ', s)
        s = re.sub(r'(?i)\bfolio\s*\d+\s*[a-z]?\b', ' ', s)
        s = re.sub(r'^\d+\s*[.-]\s*—', '— ', s)
        out.append(s.replace('[...]', ' ').replace('*', '').replace('§', ' '))
    return ROMAN.sub(' ', join_hyphens('\n'.join(out)))


def geste_clean(text):
    lines = [re.sub(r'^\d+(?=\D)', '', l.strip()) for l in text.lstrip('﻿').split('\n') if l.strip()]
    return ROMAN.sub(' ', '\n'.join(lines))


def old_french(work):
    src, stem = work['id'].split(':')
    directory = RAW / 'old_french'
    path = directory / f'{src}_{stem}.txt'
    raw = path.read_text(encoding='utf-8')
    if src == 'openmedfr':
        text, url, rule, lic = openmedfr_clean(raw), OPENMEDFR + stem + '.txt', OMF_RULE, LIC_OMF
    else:
        text, url, rule, lic = geste_clean(raw), GESTE + stem + '.txt', GESTE_RULE, LIC_GESTE
    readme = directory / f'{src}_README.md'
    work.update(license=lic, url=url)
    return text, [file_record(path, url=url), file_record(readme, role='licence / provenance')], rule


# ---------------------------------------------------------------- English
DATELINE = (r'(?:(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day, )?(?:January|February|March|April|May|June|July|'
            r'August|September|October|November|December) \d{1,2}, \d{4}')
GUM_RULE = ('# text of the sentences of this newdoc only; dropped: headings (newpar_block head), figure captions '
            "and credits (newpar_block figure), datelines and 'By ...' bylines")
PG_RULE = 'PG header/footer and licence; cover, title, contents (text starts at THE SISTERS); all-caps story titles'


def gum_docs():
    docs = OrderedDict()
    for split in ('train', 'dev', 'test'):
        path = RAW / 'english' / f'gum_en_gum-ud-{split}.conllu'
        for line in path.read_text().splitlines():
            if line.startswith('# newdoc id = '):
                current = line.split(' = ', 1)[1]
                docs[current] = dict(split=split, sents=[], dropped=0, meta={})
            elif line.startswith('# meta::') and ' = ' in line:
                key, value = line[8:].split(' = ', 1)
                docs[current]['meta'][key] = value
        heading_left = 0
        for row in conllu_sentences(path):
            doc = row['id'].rsplit('-', 1)[0]
            block = row['metadata'].get('newpar_block', '')
            if block:
                kind, count = block.split()[0], int(re.search(r'\((\d+) s\)', block)[1])
                heading_left = count if kind in ('head', 'figure') else 0
            if heading_left:
                heading_left -= 1
                docs[doc]['dropped'] += 1
                continue
            text = row['text'].strip()
            if re.fullmatch(DATELINE, text) or re.fullmatch(r'By [A-Z][^.!?]{0,60}', text):
                docs[doc]['dropped'] += 1
                continue
            docs[doc]['sents'].append(row['text'])
    return docs


def pg_body(text):
    start = text.index('\n', text.index('*** START OF THE PROJECT GUTENBERG EBOOK'))
    end = text.index('*** END OF THE PROJECT GUTENBERG EBOOK')
    return text[start:end].replace('\r\n', '\n')


def dubliners(text):
    body = pg_body(text)
    body = body[body.index('\nTHE SISTERS\n'):]
    return '\n'.join(l for l in body.split('\n') if not caps_heading(l.strip()))


def english(work, cache={}):
    directory = RAW / 'english'
    if work['source'] == 'gutenberg':
        path = directory / 'pg2814.txt'
        return dubliners(path.read_text(encoding='utf-8')), [file_record(path, url=work['url'])], PG_RULE
    if 'gum' not in cache:
        cache['gum'] = gum_docs()
    doc = cache['gum'][work['id'].split(':')[1]]
    meta = doc['meta']
    work.update(title=meta.get('title', ''), author=meta.get('author', ''), date=meta.get('dateCreated', ''),
                source_url=meta.get('sourceURL', ''), url=GUM + f"en_gum-ud-{doc['split']}.conllu")
    files = [file_record(directory / f'gum_en_gum-ud-{s}.conllu', url=GUM + f'en_gum-ud-{s}.conllu')
             for s in ('train', 'dev', 'test')]
    files += [file_record(directory / f'gum_{n}', url=GUM + n, role='licence / provenance')
              for n in ('LICENSE.txt', 'README.md')]
    return ' '.join(doc['sents']), files, GUM_RULE + f" ({doc['dropped']} sentences dropped)"


# ---------------------------------------------------------------- German (ReM), Czech, Occitan
REM_RULE = 'ReM norm attribute of every <w> in <body> (publisher normalized surface form, never lemma)'
CZECH_RULE = '# header lines removed; "[...]" lacunae removed (as language_expansion_sources.czech_document)'
OCCITAN_RULE = 'line-end hyphenation joined; "[...]" lacunae removed (as language_expansion_sources.occitan_document)'


def rem_root(ident):
    with zipfile.ZipFile(lc.RAW / 'ReM-v2.1_tei.zip') as archive:
        return ET.fromstring(archive.read(f'ReM-v2.1_tei/tei/{ident}.xml'))


def rem_words(root, ident):
    words = []
    for word in root.findall('.//t:body//t:w', lc.NS):
        if word.get('norm') is None:
            raise ValueError('Missing normalized surface form: ' + ident)
        words.append(word.get('norm'))
    return words


def german(work):
    root = rem_root(work['rem'])
    usage = [e.text for e in root.findall('.//t:langUsage/t:language', lc.NS) if e.text and e.text != '-']
    ms = root.find('.//t:msIdentifier', lc.NS)
    work.update(title=root.find('.//t:title', lc.NS).text, author='not stated in ReM header',
                date='Middle High German, ReM 1050-1350 (no date in TEI header)', variety=', '.join(usage),
                manuscript=' '.join(' '.join(e.text.split()) for e in ms if e.text)[:200])
    path = lc.RAW / 'ReM-v2.1_tei.zip'
    return ' '.join(rem_words(root, work['rem'])), [file_record(path, url=REM_URL, member=f"ReM-v2.1_tei/tei/{work['rem']}.xml")], REM_RULE


def czech_document(name, text):
    """language_expansion_sources.czech_document with the confirmation date window 1300-1520."""
    meta = dict(re.findall(r'^# ([^:]+):[ \t]*(.*)$', text, re.M))
    dates = [int(x) for x in re.findall(r'\d{4}', meta['originDate'])]
    if not dates or min(dates) < 1300 or max(dates) > 1520:
        raise ValueError('Not a selected medieval Czech source: ' + name)
    body = '\n'.join(line for line in text.splitlines() if not line.startswith('#'))
    return meta, re.sub(r'\[\s*\.\.\.\s*\]', ' ', body)


def czech(work):
    path = le.RAW / 'diakorp.zip'
    with zipfile.ZipFile(path) as archive:
        member = f"czech/diakorp/txt/{work['diakorp']}.txt"
        meta, text = czech_document(work['diakorp'], archive.read(member).decode('utf-8'))
    work.update(title=meta['title'].strip(), author=meta.get('author', '').strip() or 'anonymous',
                date=meta['originDate'].strip(), genre=meta.get('genre', '').strip())
    return text, [file_record(path, url=DIAKORP_URL, member=member),
                  file_record(le.RAW / 'diakorp-readme.txt', role='licence / provenance',
                              url='https://sprakbanken-clarin.lingfil.uu.se/histcorp/readme/czech-readme-diakorp')], CZECH_RULE


def occitan_clean(text):
    text = re.sub(r'(?<=\w)-[ \t]*\n[ \t]*(?=\w)', '', text)
    return re.sub(r'\[\s*\.\.\.\s*\]', ' ', text)


def occitan(work):
    path = le.RAW / (work['cometa'] + '.txt')
    meta = read(le.RAW / 'cometa-metadata.json')
    md5 = [f['checksum'] for f in meta['files'] if f['key'] == path.name]
    if md5 != ['md5:' + hashlib.md5(path.read_bytes()).hexdigest()]:
        raise ValueError('COMETA file does not match the Zenodo record checksum: ' + path.name)
    return occitan_clean(path.read_text()), [
        file_record(path, url=work['url'], zenodo_md5=md5[0]),
        file_record(le.RAW / 'cometa-metadata.json', url='https://zenodo.org/api/records/15300719',
                    role='record metadata (titles, md5 checksums)')], OCCITAN_RULE


def extract(work):
    work = dict(work)
    handler = dict(wikisource=wikisource, openmedfr=old_french, geste=old_french, gutenberg=english, gum=english,
                   rem=german, diakorp=czech, cometa=occitan)[work['source']]
    text, files, rule = handler(work)
    return dict(work, text=text, files=files, extraction=rule)


# ---------------------------------------------------------------- reference R
def reference(paths=None):
    """8-word shingles of every earlier train, calibration, challenge and released text.

    Conservative superset of scratch reference.py: exclusions(), released v2 answers, both earlier partitions,
    every LLCT and HisCat document, ReM prose outside the ungraded challenge pool, all language-expansion
    documents, historical prose and verse, and every string of >= 8 words in artifacts/*/evaluator-only/*.json.
    `paths` pins the evaluator-only file list at verify time.
    """
    _, shingles, inputs = exclusions()

    def use(path):
        inputs[str(path.relative_to(ROOT))] = sha(path)
        return read(path)

    for answer in use(ROOT / 'artifacts/rejection-transfer-v2/released-answers.json').values():
        for p in answer['passages']:
            shingles.update(grams(normalize_source(p['plaintext'])))
    for rel in ('artifacts/language-coverage/partitions.json', 'artifacts/language-expansion/partitions.json'):
        parts = use(ROOT / rel)
        for prior in parts['priors'].values():
            for kind in ('train', 'calibration'):
                shingles.update(grams(' '.join(r['text'] for r in prior[kind])))
        for pair in parts['passages'].values():
            for p in pair:
                shingles.update(grams(' '.join(r['text'] for r in p['rows'])))

    def add(doc):
        shingles.update(grams(' '.join(r['text'] for r in chunks(doc))))

    released = {e['language']: set(e['source_groups'])
                for e in use(ROOT / 'experiments/language-coverage/released-source-ids.json')['evaluated']}
    for rel in ('la_llct-ud-train.conllu', 'la_llct-ud-dev.conllu', 'la_llct-ud-test.conllu',
                'LlibredelsFets_60000corrected.txt', 'ReM-v2.1_tei.zip'):
        inputs[str((lc.RAW / rel).relative_to(ROOT))] = sha(lc.RAW / rel)
    for doc in lc.latin_documents():
        add(doc)
    for doc in lc.catalan_documents(lc.RAW / 'LlibredelsFets_60000corrected.txt'):
        add(doc)
    with zipfile.ZipFile(lc.RAW / 'ReM-v2.1_tei.zip') as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith('.xml'):
                continue
            root = ET.fromstring(archive.read(name))
            genre = root.find('.//t:classCode', lc.NS)
            ident = name.rsplit('/', 1)[-1][:-4]
            group = re.match(r'M\d+', ident)[0]
            bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10
            if genre is not None and genre.text == 'P' and (bucket != 0 or group in released['german']):
                add(dict(id=ident, group=ident, text=' '.join(w.get('norm') or '' for w in
                                                             root.findall('.//t:body//t:w', lc.NS))))
    for row in read(le.RAW / 'downloads.json'):
        inputs[row['path']] = sha(ROOT / row['path'])
    for docs in le.documents().values():
        for doc in docs.values():
            add(doc)
    for row in use(ROOT / 'artifacts/historical-prose/texts.json'):
        shingles.update(grams(normalize_source(' '.join(row['paragraphs']))))
    for row in use(ROOT / 'artifacts/historical-verse/texts.json'):
        shingles.update(grams(normalize_source(' '.join(row['lines']))))

    def strings(x):
        if isinstance(x, str):
            if x.count(' ') >= 7:
                yield x
        elif isinstance(x, dict):
            for v in x.values():
                yield from strings(v)
        elif isinstance(x, list):
            for v in x:
                yield from strings(v)

    if paths is None:
        paths = [str(p.relative_to(ROOT)) for p in sorted(ROOT.glob('artifacts/*/evaluator-only/*.json'))]
    for rel in paths:
        for s in strings(use(ROOT / rel)):
            shingles.update(grams(normalize_source(s)))
    return shingles, dict(sorted(inputs.items())), paths


# ---------------------------------------------------------------- filter and cut
def letters(rows):
    return sum(len(r['text'].replace(' ', '')) for r in rows)


def filter_rows(rows, blocked):
    """Drop a chunk if blocked(shingles of chunk plus 7-word tail of the previous kept chunk) names a reason."""
    kept, removed, tail = [], [], []
    for row in rows:
        candidate = ' '.join(tail + row['text'].split())
        reason = blocked(grams(candidate))
        if reason:
            removed.append(dict(id=row['id'], letters=len(row['text'].replace(' ', '')), reason=reason))
            continue
        kept.append(row)
        tail = candidate.split()[-7:]
    return kept, removed


def cut(rows, budget=PASSAGE):
    """The first `budget` letters of `rows`: record shape of the earlier partitions."""
    selected = exact_take(rows, budget)
    ids = {r['id'] for r in selected}
    return dict(document=rows[0]['document'], plaintext=''.join(r['text'] for r in selected),
                rows=[r for r in rows if r['id'] in ids])


def after_gap(rows, last_id, gap=GAP):
    """Rows starting once at least `gap` letters of `rows` have passed after the row `last_id`."""
    index = [r['id'] for r in rows].index(last_id) + 1
    skipped = 0
    while index < len(rows) and skipped < gap:
        skipped += len(rows[index]['text'].replace(' ', ''))
        index += 1
    if skipped < gap:
        raise ValueError(f'No room for a second passage: {skipped} letters after {last_id}')
    return rows[index:], skipped


# Latin function words that are not Middle High German word forms (no 'in', 'ab', 'per', 'die').
LATIN_MARKERS = frozenset(
    'et est qui quod non ad cum ut sed autem enim quia eius sunt deus dei domini dominus domine super ex sicut '
    'vel atque ergo esse fuit nos vos ego mihi michi tibi omnes omnia erat eum eis quam quae hec haec'.split())


def marker_rate(text):
    words = normalize_source(text).split()
    return sum(w in LATIN_MARKERS for w in words) / len(words)


def latin_share(text, latin_rate):
    """Estimated share of Latin words: Latin marker rate of the text over the rate in Latin reference text."""
    return min(1.0, marker_rate(text) / latin_rate)


def german_latin_check(docs):
    reference = ' '.join([d['text'] for d in lc.latin_documents()] +
                         [d['text'] for d in docs if d['language'] == 'latin'])
    rate = marker_rate(reference)
    return dict(latin_marker_rate=round(rate, 4),
                shares={d['id']: round(latin_share(d['text'], rate), 4) for d in docs if d['language'] == 'german'})


def partitions(docs, R):
    rows = {d['id']: list(chunks(dict(id=d['id'], group=d['id'], text=d['text']))) for d in docs}
    own = {i: grams(' '.join(r['text'] for r in rs)) for i, rs in rows.items()}
    counts = Counter(g for s in own.values() for g in s)
    holders = {}
    for i, s in own.items():
        for g in s:
            if counts[g] > 1:
                holders.setdefault(g, []).append(i)
    passages, records = {l: [] for l in LANGUAGES}, {l: [] for l in LANGUAGES}
    firsts = {}
    for d in docs:
        mine = own[d['id']]

        def blocked(candidate):
            if candidate & R:
                return 'reference'
            if any(counts[g] - (g in mine) > 0 for g in candidate):
                return 'other selected work'
            return None

        kept, removed = filter_rows(rows[d['id']], blocked)
        total = letters(rows[d['id']])
        available = letters(kept)
        if available < PASSAGE:
            raise ValueError(f"{d['id']}: only {available} clean letters; stop, do not substitute")
        passage = cut(kept)
        firsts[d['id']] = (kept, passage)
        last = passage['rows'][-1]['id']
        position = [r['id'] for r in rows[d['id']]].index(last)
        before = {r['id'] for r in rows[d['id']][:position + 1]}
        meta = {k: v for k, v in d.items() if k not in ('text', 'files', 'drop', 'sub', 'wiki', 'rem',
                                                        'diakorp', 'cometa')}
        record = dict(meta, work=d['id'], passage=1, raw_sha256=sorted({f['sha256'] for f in d['files']}),
                      raw_letters=total, clean_letters_available=available, letters_removed=total - available,
                      fraction_removed=round((total - available) / total, 4),
                      letters_removed_by_reference=sum(r['letters'] for r in removed if r['reason'] == 'reference'),
                      letters_removed_by_other_works=sum(r['letters'] for r in removed if r['reason'] != 'reference'),
                      letters_removed_before_passage_end=sum(r['letters'] for r in removed if r['id'] in before),
                      chunks=len(rows[d['id']]), chunks_removed=[r['id'] for r in removed],
                      shared_shingles_with_other_works=dict(sorted(Counter(
                          o for g in mine if g in holders for o in holders[g] if o != d['id']).items())),
                      first_row=passage['rows'][0]['id'], last_row=last,
                      text_sha256=hashlib.sha256(d['text'].encode()).hexdigest(),
                      plaintext_sha256=hashlib.sha256(passage['plaintext'].encode()).hexdigest())
        passages[d['language']].append(passage)
        records[d['language']].append(record)
    for ident in SECOND_PASSAGES:
        kept, first = firsts[ident]
        later, skipped = after_gap(kept, first['rows'][-1]['id'])
        first_grams = grams(' '.join(r['text'] for r in first['rows']))
        later, removed = filter_rows(later, lambda c: 'first passage' if c & first_grams else None)
        if letters(later) < PASSAGE:
            raise ValueError(f'{ident}: second passage lacks letters')
        passage = cut(later)
        doc_rows = [r['id'] for r in rows[ident]]
        raw_gap = letters(rows[ident][doc_rows.index(first['rows'][-1]['id']) + 1:doc_rows.index(passage['rows'][0]['id'])])
        base = next(r for r in records['occitan'] if r['work'] == ident)
        passages['occitan'].append(passage)
        records['occitan'].append(dict(base, passage=2, first_row=passage['rows'][0]['id'],
                                       last_row=passage['rows'][-1]['id'], gap_clean_letters=skipped,
                                       gap_raw_letters=raw_gap,
                                       letters_removed_against_first_passage=sum(r['letters'] for r in removed),
                                       plaintext_sha256=hashlib.sha256(passage['plaintext'].encode()).hexdigest()))
    audit(passages, R, own, counts)
    return dict(passages=passages), records


def audit(passages, R, own, counts):
    """Final check on each passage's joined rows, including joins across removed chunks."""
    for language, items in passages.items():
        assert len(items) == 6, (language, len(items))
        texts = [p['plaintext'] for p in items]
        assert len(set(texts)) == 6 and all(len(t) == PASSAGE and t.isalpha() for t in texts)
        for p in items:
            g = grams(' '.join(r['text'] for r in p['rows']))
            mine = own[p['document']]
            if g & R or any(counts[x] - (x in mine) > 0 for x in g):
                raise ValueError('Overlap survived filtering: ' + p['document'])
    firsts = {p['document']: p for p in passages['occitan'][:4]}
    for p in passages['occitan'][4:]:
        first = firsts[p['document']]
        assert not {r['id'] for r in p['rows']} & {r['id'] for r in first['rows']}
        assert not grams(' '.join(r['text'] for r in p['rows'])) & grams(' '.join(r['text'] for r in first['rows']))


def dumps(value):
    """Exactly the bytes seal() writes."""
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'


def assemble(paths=None):
    docs = [extract(w) for w in WORKS]
    for language in LANGUAGES:
        assert sum(d['language'] == language for d in docs) == (4 if language == 'occitan' else 6), language
    shares = german_latin_check(docs)
    too_latin = {k: v for k, v in shares['shares'].items() if v > MAX_LATIN_SHARE}
    if too_latin:
        raise ValueError(f'Latin-dominant German works (report, do not substitute): {too_latin}')
    R, inputs, paths = reference(paths)
    parts, records = partitions(docs, R)
    for record, passage in zip(records['german'], parts['passages']['german']):
        record['passage_latin_share'] = round(latin_share(
            ' '.join(r['text'] for r in passage['rows']), shares['latin_marker_rate']), 4)
    return docs, parts, records, shares, R, inputs, paths


def build():
    if (STATE / 'partitions.json').exists() or (OUT / 'sources.json').exists():
        raise FileExistsError('Confirmation sources already sealed')
    docs, parts, records, shares, R, inputs, paths = assemble()
    seal(STATE / 'partitions.json', parts)
    raw = {}
    for d in docs:
        for f in d['files']:
            raw.setdefault(f['path'], dict(f, works=[]))['works'].append(d['id'])
    seal(OUT / 'sources.json', dict(
        purpose='Fresh known-answer passages for the key-recovery confirmation (PROTOCOL.md). Passages only; '
                'blocks and keys are drawn at prepare time.',
        selection_rule=SELECTION_RULE, passage_letters=PASSAGE, occitan_second_passage_gap=GAP,
        works=[{k: v for k, v in d.items() if k not in ('text', 'files', 'drop', 'sub', 'wiki')} |
               dict(files=[f['path'] for f in d['files']], drop=d.get('drop'),
                    sub=[list(s) for s in d.get('sub') or []] or None) for d in docs],
        raw_files=sorted(raw.values(), key=lambda f: f['path']),
        passages=records, german_latin_share=dict(
            shares, maximum=MAX_LATIN_SHARE, markers=sorted(LATIN_MARKERS),
            method='rate of Latin function words (markers) in the work divided by their rate in LLCT plus the six '
                   'selected Latin works; capped at 1'),
        reference=dict(shingles=len(R), inputs=inputs, evaluator_only_paths=paths,
                       rule='exclusions(); rejection-transfer-v2 released answers; language-coverage and '
                            'language-expansion partitions (priors and passages); all LLCT and HisCat documents; '
                            'ReM prose outside the ungraded challenge pool; every language-expansion document; '
                            'historical prose and verse; every >=8-word string of the evaluator-only files'),
        partitions_path=str((STATE / 'partitions.json').relative_to(ROOT)),
        partitions_sha256=sha(STATE / 'partitions.json')))
    summary()


def verify():
    manifest = read(OUT / 'sources.json')
    for f in manifest['raw_files']:
        if sha(ROOT / f['path']) != f['sha256']:
            raise ValueError('Raw source drift: ' + f['path'])
    for path, digest in manifest['reference']['inputs'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('Reference input drift: ' + path)
    if sha(STATE / 'partitions.json') != manifest['partitions_sha256']:
        raise ValueError('Pinned partitions.json changed')
    docs, parts, records, shares, R, inputs, _ = assemble(manifest['reference']['evaluator_only_paths'])
    if inputs != manifest['reference']['inputs'] or len(R) != manifest['reference']['shingles']:
        raise ValueError('Reference set differs')
    if json.loads(dumps(records)) != manifest['passages']:
        raise ValueError('Passage records differ')
    rebuilt = hashlib.sha256(dumps(parts).encode()).hexdigest()
    if rebuilt != manifest['partitions_sha256']:
        raise ValueError(f'Rebuilt partitions differ: {rebuilt}')
    print('verified', rebuilt, len(manifest['raw_files']), 'raw files')


def summary():
    manifest = read(OUT / 'sources.json')
    for language, items in manifest['passages'].items():
        for r in items:
            print(f"{language:10} {r['work']:32} p{r['passage']} clean {r['clean_letters_available']:7} "
                  f"removed {100 * r['fraction_removed']:5.2f}%")
    print('raw files', len(manifest['raw_files']), 'sources.json', sha(OUT / 'sources.json'),
          'partitions.json', manifest['partitions_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['build', 'verify', 'summary'])
    globals()[parser.parse_args().command]()
