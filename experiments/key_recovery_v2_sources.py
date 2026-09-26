"""Pinned sources for the second known-answer confirmation: three broadened priors and 48 fresh passages.

python -m experiments.key_recovery_v2_sources build | verify | summary

build   extract every catalogued work from raw files already on disk (no network); filter 80-word chunks against
        the reference shingle set R; assign TEST / CALIBRATION / TRAIN roles by the fixed ROLE_RULES (never by any
        model score); fill the latin_broad2, german_broad2 and catalan_broad2 priors (400,000 train letters by
        water-filling, 2 x 10,000 calibration letters); cut six 5,200-letter TEST passages per language; seal
        artifacts/key-recovery-confirmation-v2/partitions.json and the committed sources.json manifest.
verify  recompute every raw and reference-input hash and rebuild in memory; the partitions JSON must be
        byte-identical to the pinned file.

Extraction logic is moved in from the scratch builders of this round (sources_v2.py and web.py for Occitan and
Latin, extract_v2.py for Italian, Old French, English and Czech, catalan_v2.py and works_catalan.py for Catalan,
and the unused round-one candidates of confirmation_sources.py). Round-one handlers are reused from
key_recovery_confirmation_sources. Blocks later pair passages (1,2), (3,4), (5,6) in the order written here.
"""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import re
import urllib.parse
import zipfile
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup

from experiments import key_recovery_confirmation_sources as kr
from experiments import language_coverage_sources as lc
from experiments import language_expansion_sources as le
from experiments.language_coverage_sources import ROOT, chunks, exact_take, normalize_source
from experiments.rejection_transfer_v2 import grams, read, seal
from voynich.corpora import conllu_sentences

OUT = ROOT / 'experiments/key-recovery-confirmation-v2'
STATE = ROOT / 'artifacts/key-recovery-confirmation-v2'
RAW = kr.RAW
PASSAGE = 5200
GAP = 50_000                 # Italian second passages start this many filtered letters after the first ends
BUDGET = 400_000             # train letters per new prior
CALIBRATION = 10_000         # calibration letters per calibration group
GERMAN_MIN = 10_000          # German groups need this many clean letters to be TEST or CALIBRATION
N_TEST, N_CAL = 6, 2
LANGUAGES = ['latin', 'italian', 'catalan', 'old_french', 'english', 'german', 'czech', 'occitan']
NEW_PRIORS = dict(latin='latin_broad2', german='german_broad2', catalan='catalan_broad2')
LEGACY = dict(latin='latin_broad', german='german_broad', catalan='catalan')
UNCHANGED = ['italian', 'old_french', 'english', 'czech', 'occitan']
ITALIAN_SECOND = ['gutenberg:44549', 'gutenberg:67931']     # the two largest eligible Italian works
LATIN_HOSTS = {'thelatinlibrary.com': [b'thelatinlibrary.com'], 'Corpus Corporum': [b'Corpus Corporum', b'mlat.uzh.ch']}
# Found by the raw scan below; a drift in the raw files or the catalogue stops the build.
EXPECTED_LATIN_HOSTS = {
    'nithard-historiae': ['Corpus Corporum'], 'widukind-res-gestae': ['Corpus Corporum'],
    'richer-historiae': ['Corpus Corporum'], 'thietmar-chronicon': ['Corpus Corporum'],
    'glaber-historiae': ['Corpus Corporum'], 'otto-gesta-friderici': ['thelatinlibrary.com'],
    'william-tyre-historia': ['thelatinlibrary.com'], 'carpini-historia-mongalorum': ['thelatinlibrary.com']}

ROLE_RULES = [
    'A. Author grouping: all works by one author (and anonymous works of one cycle/text family, and parallel '
    'versions/copies of one text) form one group and share one role.',
    'B. Exclusions from ALL roles: Latin works whose text came from thelatinlibrary.com or Corpus Corporum (raw '
    'Wikisource header "fons" scanned); Italian 20th-century critical editions (alberti-famiglia, '
    'tristano-riccardiano, anonimo-romano-cronica, sanseverino-meraviglie), prose-genovesi (Genoese), Boccaccio; '
    "Old French Cligès (Chrétien is a training author) and the Fille du comte de Ponthieu pair; English Lippmann and "
    'all GUM documents; Czech diakorp24 (~23% Latin), diakorp5 (Nicodemus); Czech works by an author of the Czech '
    "prior's training texts (Štítný); Catalan rejected downloads (_rejected).",
    'C. Order: within each language, eligible groups are ordered by sha256(group id) ascending.',
    'D. Unchanged priors (italian, old_french, english, czech, occitan): the first 6 eligible groups are TEST (one '
    'work each: the one with most clean letters). Italian has only 4 eligible works: all 4, plus second passages '
    '(>= 50,000 filtered letters after the first ends) of the two largest (Sidrach, Leonardo) as passages 5-6. '
    'Czech: pre-1500 groups first (hash order), then 1500-1595 DIAKORP groups (hash order); period flagged. Hus '
    'Dcerka (modernised spelling) eligible but flagged.',
    'E. New priors (latin, german, catalan): hash order, first 6 eligible groups TEST, next 2 CALIBRATION, rest '
    'TRAIN. Uncorrected-OCR works only TRAIN. German eligible groups: ReM v2.1 documents not used in round one and '
    'not in legacy german_broad train/calibration, grouped by ReM group id (M###) and merged by text family '
    '(REM_FAMILIES); only groups with >= 10,000 clean letters may be TEST/CALIBRATION.',
    'F. TRAIN pool of each new prior = its TRAIN groups + the legacy prior rows (latin_broad: ITTB, LLCT; '
    'german_broad: GSD, ReM prose; catalan: Llibre dels Fets; language-coverage partitions) as separate groups; '
    'unused on-disk round-one candidates join. Exactly 400,000 letters by water-filling (each group min(letters, '
    'cap)), first letters of each group, exact_take of 80-word chunks. CALIBRATION = first 10,000 letters of each '
    'of the 2 calibration groups. Train/calibration rows are filtered against R only (legacy rows are taken as '
    'partitioned: they are the source of R). TEST works are filtered against R, all v2 train/calibration text and '
    'every other TEST work (8-word shingles), then cut to their first 5,200 letters. Blocks pair (1,2),(3,4),(5,6).',
    'G. Unchanged priors: every train/calibration row of every earlier prior maps back to a source text whose '
    '8-word shingles are all in R (asserted).']
FILTER_RULE = (
    'normalize_source (accents removed, j->i, k->c, w->uu, letters a-z), 80-word chunks (chunks()); a chunk is '
    'removed if any 8-word shingle of it, with the 7-word tail of the previous kept chunk, is blocked. Train and '
    'calibration: blocked = R. TEST: blocked = R, whole texts of every v2 train/calibration work, whole texts of every '
    'other TEST work (all languages); Italian second passages also against the first passage.')

# ---------------------------------------------------------------- licences
WS_LICENSE = kr.WS_LICENSE
WS_LIC_V2 = ('Public-domain original and edition; Wikisource transcription CC BY-SA 4.0 (page header states '
             'CC BY-SA 3.0 / GFDL), see page history')
WS_LIC_CA = ('Public-domain medieval text and pre-1926 (or public-domain) edition; Wikisource transcription '
             'CC BY-SA 4.0 (Wikimedia terms of use), see page history')
IA_PD = 'Public domain (edition published {year}; editor(s) died more than 70 years ago{extra}); IA item {rights}'
LIC_PG = kr.LIC_PG
LIC_PG_IT = ('Public domain in the USA (Project Gutenberg); edition 19th/early-20th c., editor died >70 years ago '
             'where known; PG header/footer stripped')
PG_URL = 'https://www.gutenberg.org/cache/epub/{0}/pg{0}.txt'
UDANTE = ('https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-UDante/'
          'e02420457780c6fbb503ba39a7d8798ab6a8645c/')
LIC_UDANTE = ('UD_Latin-UDante: LICENSE.txt says CC BY-SA 4.0; README says CC BY-NC-SA 3.0 (conflict upstream; '
              'research use satisfies both)')


# ---------------------------------------------------------------- catalogue
def WSH(lang, id, title, author, date, genre, form, host, drop=None, sub=None, edition=None, notes='',
        verse=False, start=None, stop=None, **kw):
    """Wikisource rendered page HTML (sources_v2: the API answered 429, so the plain page was pinned)."""
    return dict(language=lang, id=id, title=title, author=author, date=date, genre=genre, form=form,
                source='ws_html', host=host, drop=drop, sub=sub, edition=edition, notes=notes, verse=verse,
                start=start, stop=stop, license=WS_LICENSE, **kw)


def IA(lang, id, title, author, date, genre, form, ident, file, edition, license, kind, region, header=None,
       notes='', **kw):
    """Internet Archive OCR text layer plus item metadata (sources_v2)."""
    return dict(language=lang, id=id, title=title, author=author, date=date, genre=genre, form=form, source='ia',
                ident=ident, file=file, edition=edition, license=license, kind=kind, region=region, header=header,
                notes=notes, **kw)


def WSJ(lang, id, dir, title, author, date, genre, form='prose', wiki='it', drop=None, sub=None, notes='',
        edition=None, license=WS_LIC_V2, **kw):
    """Wikisource parse-API JSON read with the pinned round-one ws_paragraphs (extract_v2)."""
    return dict(language=lang, id=id, source='ws_json', dir=dir, wiki=wiki, title=title, author=author, date=date,
                genre=genre, form=form, drop=drop, sub=sub, edition=edition, license=license, notes=notes, **kw)


def PGW(lang, n, title, author, date, genre, start, end=None, form='prose', notes='', lic=LIC_PG, **kw):
    return dict(language=lang, id=f'gutenberg:{n}', source='pg', pg=n, path=f'{lang}-v2/pg{n}.txt', title=title,
                author=author, date=date, genre=genre, form=form, start=start, end=end, license=lic, notes=notes,
                url=PG_URL.format(n), **kw)


def OMF(stem, sub, title, author, date, genre, form='verse', notes='', **kw):
    return dict(language='old_french', id='openmedfr:' + stem, source='omf', stem=stem,
                path=f'{sub}/openmedfr_{stem}.txt', title=title, author=author, date=date, genre=genre, form=form,
                license=kr.LIC_OMF, notes=notes, url=kr.OPENMEDFR + stem + '.txt', **kw)


def GE(stem, title, author, date, notes='', **kw):
    return dict(language='old_french', id='geste:' + stem, source='geste', stem=stem, title=title, author=author,
                date=date, genre='chanson de geste', form='verse', license=kr.LIC_GESTE, notes=notes,
                url=kr.GESTE + stem + '.txt', **kw)


def CZ(name, notes='', **kw):
    return dict(language='czech', id='czech:' + name, source='diakorp', diakorp=name, form='prose',
                license=kr.LIC_DIAKORP, url=kr.DIAKORP_URL, notes=notes, **kw)


def CAT(id, title, author, date, genre, dir=None, form='prose', drop=None, sub=None, edition=None, notes='',
        poem_lines=False, drop_latin=False, **kw):
    """Catalan parse-API JSON read with catalan_v2.paragraphs (sidenotes and editorial brackets removed)."""
    return dict(language='catalan', id=id, title=title, author=author, date=date, genre=genre, form=form,
                source='ws_catalan', dir=dir or 'catalan-v2/' + id, drop=drop, sub=sub, edition=edition,
                notes=notes, poem_lines=poem_lines, drop_latin=drop_latin, license=WS_LIC_CA, **kw)


def UD(id, title, prefix, date, genre):
    return dict(language='latin', id=id, title=title, author='Dante Alighieri', date=date, genre=genre,
                source='udante', prefix=prefix, license=LIC_UDANTE, url=UDANTE, group='author:dante-alighieri',
                notes='UDante sentences with sent_id prefix ' + prefix + ' (round-one on-disk candidate, unused)')


def X(lang, id, title, author, reason, **kw):
    """A catalogued work excluded from every role (rule B); not extracted."""
    return dict(language=lang, id=id, title=title, author=author, exclude=reason, **kw)


OCC, LAT = 'occitan', 'latin'
WSO, WSL, WSF = 'wikisource.org', 'la.wikisource.org', 'fr.wikisource.org'
OCR = 'rule E: uncorrected OCR (TRAIN only)'
POST1500 = 'PERIOD FLAG: DIAKORP 1500-1600, later than the 1350-1460 prior.'

WORKS = [
    # ================= Old Occitan (sources_v2) =================
    WSH(OCC, 'mascaro-libre-memorias', 'Libre de Memorias', 'Jacme Mascaro', 'c. 1336-1390 (finished 1390)',
        'municipal chronicle (Béziers)', 'prose', WSO,
        notes='Edition not named on the page (text matches the 19th-c. printed Libre de memorias, e.g. Barbier, RLR '
              '1895; public domain). Consul / office-holder name lists are <li> items and are not extracted; '
              'roman-numeral dates removed.'),
    WSH(OCC, 'esquerrier-comtes-foix', 'Canonica dels comtes de Foix', 'Arnaud Esquerrier', 'c. 1456-1461',
        'dynastic chronicle', 'prose', WSO,
        notes='Edition not named on the page (editorial (brackets) as in Pasquier and Courteault 1895, public '
              'domain). 12 of 13 linked chapters exist (Lo detze comte is a red link). Roman-numeral dates removed.'),
    WSH(OCC, 'costumas-seix', 'Usatges, libertats et costumas de Seix', 'anonymous (consuls of Seix)',
        '1280 (confirmation by Philip III; copy as printed 1893)', 'customary law', 'prose', WSO,
        drop=r'^Acta fuerunt', edition='Bulletin périodique de la Société ariégeoise 10 (1893)',
        notes='Short: below 20,000 letters.'),
    WSH(OCC, 'girart-rossilho-oxford', 'Girart de Rossilho (Oxford, Bodl. Canonici misc. 63)', 'anonymous',
        'c. 1150-1180 (ms. 13th c.)', 'chanson de geste', 'verse', WSF, verse=True,
        edition='W. Foerster, diplomatic print, Romanische Studien 5 (1880)',
        notes='Diplomatic transcription: unexpanded abbreviations, manuscript word division (many run-together '
              'words), long s. Oxford copy has French (Poitevin) admixture. Line and laisse numbers removed.'),
    IA(OCC, 'breviari-amor-1', "Le Breviari d'amor, tome I (vv. 1-15797)", 'Matfre Ermengaud', '1288-c. 1292',
       'encyclopedic didactic poem', 'verse', 'lebreviaridamor01ermeuoft', 'lebreviaridamor01ermeuoft_djvu.txt',
       'G. Azaïs, Béziers/Paris 1862', IA_PD.format(year=1862, extra='; Azaïs d. 1888', rights='NOT_IN_COPYRIGHT'),
       'verse', (r'^AYS\S*\s+COMENSA', r'^Ermengaud,\s+Matfre'), header=r'BREVIARI|^\s*-\s*[\dIl]+\s*-\s*$',
       notes='Uncorrected OCR (U. Toronto scan, good quality). Apparatus lines (sigla A-D, em dashes) removed.'),
    IA(OCC, 'daurel-beton', 'Daurel et Beton', 'anonymous', 'late 12th / early 13th c.', 'chanson de geste',
       'verse', 'daureletbetoncha00meyeuoft', 'daureletbetoncha00meyeuoft_djvu.txt', 'P. Meyer, SATF 1880',
       IA_PD.format(year=1880, extra='; Meyer d. 1917', rights='NOT_IN_COPYRIGHT'), 'verse',
       (r'^So\s+es\s+lo\s+romans\s+de\s+Daurel', r'^VOCABULAIRE'), header=r'DAUREL\s+E\s+DE\s+BETO|^RuBB',
       notes='Uncorrected OCR (U. Toronto scan). Editorial brackets kept as letters. Introduction excluded by region.'),
    IA(OCC, 'guillaume-barre', 'Guillaume de la Barre', 'Arnaut Vidal de Castelnaudary', '1318',
       'verse romance', 'verse', 'guillaumedelabar00arnauoft', 'guillaumedelabar00arnauoft_djvu.txt',
       'P. Meyer, SATF 1895', IA_PD.format(year=1895, extra='; Meyer d. 1917', rights='NOT_IN_COPYRIGHT'), 'verse',
       (r'^Aquest\s+libre\s+fe\s+Ar', r'^VOCABULAIRE'), header=r'GUILLAUME\s+DE\s+LA\s+BARRE',
       notes='Uncorrected OCR (U. Toronto scan). Folio refs (f. 40 a) removed.'),
    IA(OCC, 'jaufre', 'Jaufre', 'anonymous', 'c. 1180-1230', 'Arthurian verse romance', 'verse',
       'jaufreeinaltprovenbreu', 'jaufreeinaltprovenbreu_djvu.txt', 'H. Breuer after W. Foerster, Göttingen 1925',
       IA_PD.format(year=1925, extra='; Breuer d. 1936, Foerster d. 1915', rights='Public Domain Mark 1.0'), 'verse',
       (r"^'un conte de bona maniera", r'^Anmerkungen,'), header=r'Digitized by|^\s*\d+[a-d]\s*$',
       notes='Tesseract 5 OCR (good). German introduction and apparatus (sigla A/B) removed.'),
    # ================= Latin (sources_v2) =================
    WSH(LAT, 'nithard-historiae', 'Historiarum libri IV', 'Nithard', '841-843', 'history', 'prose', WSL),
    WSH(LAT, 'annales-bertiniani-hincmar', 'Annales Bertiniani, pars tertia', 'Hincmar of Reims', '861-882',
        'annals', 'prose', WSL),
    WSH(LAT, 'widukind-res-gestae', 'Res gestae Saxonicae', 'Widukind of Corvey', 'c. 967-973', 'history', 'prose',
        WSL, start=r'^Flore virginali', edition='Migne, Patrologia Latina 137 (after Waitz, MGH)'),
    WSH(LAT, 'richer-historiae', 'Historiae', 'Richer of Saint-Rémi', 'c. 991-998', 'history', 'prose', WSL,
        stop=r'Post Historiae finem', edition='Migne, Patrologia Latina 138'),
    WSH(LAT, 'thietmar-chronicon', 'Chronicon', 'Thietmar of Merseburg', '1012-1018', 'chronicle', 'prose', WSL),
    WSH(LAT, 'glaber-historiae', 'Historiae sui temporis', 'Rodulfus Glaber', 'c. 1030-1046', 'history', 'prose',
        WSL, edition='Migne, Patrologia Latina 142'),
    WSH(LAT, 'anselm-proslogion', 'Proslogion (with Gaunilo and responsio)', 'Anselm of Canterbury', '1077-1078',
        'theology', 'prose', WSL, notes='Wikisource fons: incognitus.'),
    WSH(LAT, 'otto-gesta-friderici', 'Gesta Friderici imperatoris', 'Otto of Freising and Rahewin', '1157-1160',
        'history', 'prose', WSL),
    WSH(LAT, 'william-tyre-historia', 'Historia rerum in partibus transmarinis gestarum', 'William of Tyre',
        'c. 1170-1184', 'crusade history', 'prose', WSL),
    WSH(LAT, 'geoffrey-historia-regum', 'Historia regum Britanniae', 'Geoffrey of Monmouth', 'c. 1136',
        'legendary history', 'prose', WSL, notes='Wikisource fons: incognitus.'),
    WSH(LAT, 'hugh-soliloquium', 'Soliloquium de arrha animae', 'Hugh of Saint Victor', 'c. 1130s',
        'devotional dialogue', 'prose', WSL, notes='Wikisource fons: incognitus.'),
    WSH(LAT, 'carpini-historia-mongalorum', 'Libellus historicus (Historia Mongalorum)', 'John of Plano Carpini',
        '1247', 'travel account', 'prose', WSL),
    WSH(LAT, 'salimbene-cronica', 'Cronica, pars I', 'Salimbene de Adam', 'c. 1283-1288', 'chronicle', 'prose', WSL,
        edition='F. Bernini, Laterza 1942 (BEIC scan)',
        notes='Italian critical-edition right (20 years) expired; running chapter index lines are dotted leaders.'),
    WSH(LAT, 'petrarch-secretum', 'Secretum (De secreto conflictu curarum mearum)', 'Francesco Petrarca',
        'c. 1347-1353', 'dialogue', 'prose', WSL, notes='Wikisource fons: incognitus.'),
    WSH(LAT, 'piccolomini-duobus-amantibus', 'Historia de duobus amantibus', 'Enea Silvio Piccolomini', '1444',
        'novella (epistolary)', 'prose', WSL),
    WSH(LAT, 'poggio-facetiae', 'Liber facetiarum', 'Poggio Bracciolini', '1438-1452', 'jests / anecdotes', 'prose',
        WSL, notes='Wikisource fons: incognitus.'),
    WSH(LAT, 'cusanus-docta-ignorantia', 'De docta ignorantia', 'Nicholas of Cusa', '1440', 'philosophy', 'prose',
        WSL, notes='Wikisource fons: incognitus.'),
    WSH(LAT, 'kempis-imitatio', 'De imitatione Christi', 'Thomas a Kempis', 'c. 1418-1427', 'devotional', 'prose',
        WSL, notes='Wikisource fons: incognitus.'),
    IA(LAT, 'hildegard-causae-curae', 'Causae et curae', 'Hildegard of Bingen', 'c. 1150-1160', 'medical', 'prose',
       'hildegardiscaus01hildgoog', 'hildegardiscaus01hildgoog_djvu.txt', 'P. Kaiser, Teubner, Leipzig 1903',
       "Public domain: 12th-c. text; 1903 edition (US public domain; editor's death date not found, German 25-year "
       'right for scientific editions long expired); IA item NOT_IN_COPYRIGHT', 'prose',
       (r'^BEATAE HILDEGARDIS CAUSAE ET CURAE', r'^INDEX RERUM ET NOMINUM'),
       header=r'HILDEGARDIS|CAUSAE ET CURAE|ed\. Kaiser', train_only=OCR,
       notes='Tesseract 5 OCR of a Google scan (good). Physica parallel passages kept (same author).'),
    IA(LAT, 'mondeville-chirurgia', 'Chirurgia', 'Henri de Mondeville', 'c. 1306-1320', 'medical (surgery)', 'prose',
       'b24856307', 'b24856307_djvu.txt', 'J. L. Pagel, Berlin 1892 (Hirschwald)',
       IA_PD.format(year=1892, extra='; Pagel d. 1912', rights='Wellcome Library, Public Domain Mark 1.0'), 'prose',
       (1420, 33500), header=r'^\s*Tract\.\s|^\s*Prooemium\.\s*$|^\s*\d+\*?\s*$', train_only=OCR,
       notes='ABBYY 11 OCR (Wellcome). German preface, commentary and registers excluded by region and a '
             'German-function-word line filter; variant footnotes removed.'),
    # ---------------- Latin: unused round-one on-disk candidates (confirmation_sources.py) ----------------
    WSJ(LAT, 'abelard-historia-calamitatum', 'latin/abelard-historia-calamitatum', 'Historia calamitatum',
        'Peter Abelard', 'c. 1132', 'autobiographical letter', wiki='la', license=WS_LICENSE,
        notes='Round-one on-disk candidate, unused.'),
    WSJ(LAT, 'albertus-de-animalibus-22', 'latin/albertus-de-animalibus-22', 'De animalibus, liber XXII',
        'Albertus Magnus', 'c. 1260', 'natural history (animals, with medical uses)', wiki='la', license=WS_LICENSE,
        sub=[(r'\bı0\b|\b\d+\b', ' '), (r'([^\W\d_])[-­]\s+([^\W\d_])', r'\1\2'), (r'\|\|', ' ')],
        edition='Stadler 1916-1920 (OCR)', train_only=OCR,
        notes='Round-one on-disk candidate, unused. Uncorrected OCR (dotless i, line numbers, hyphenation).'),
    WSJ(LAT, 'legenda-aurea', 'latin/legenda-aurea', 'Legenda aurea (selected legends)', 'Jacobus de Voragine',
        'c. 1260', 'hagiography', wiki='la', license=WS_LICENSE, edition='Graesse 1846',
        notes='Round-one on-disk candidate, unused. Wikisource fons: GoogleBooks.'),
    WSJ(LAT, 'gesta-romanorum', 'latin/gesta-romanorum', 'Gesta Romanorum (Oesterley ed.)', 'anonymous',
        'early 14th c.', 'exempla / moralized tales', wiki='la', license=WS_LICENSE, edition='Oesterley 1872',
        sub=[(r'\[Moralisatio\.?\]', ' ')], notes='Round-one on-disk candidate, unused.'),
    UD('udante-monarchia', 'Monarchia', 'Mon', 'c. 1312-1318', 'political treatise'),
    UD('udante-dve', 'De vulgari eloquentia', 'DVE', 'c. 1303-1305', 'linguistic / rhetorical treatise'),
    UD('udante-epistole', 'Epistolae', 'Epi', '1304-1320', 'letters'),
    UD('udante-questio', 'Questio de aqua et terra', 'Que', '1320', 'natural-philosophy disputation'),
    # ================= Italian (extract_v2) =================
    X('italian', 'alberti-famiglia', 'I libri della famiglia', 'Leon Battista Alberti',
      'rule B: 20th-c. critical edition (Grayson 1960)'),
    WSJ('italian', 'masuccio-novellino', 'italian-v2/masuccio-novellino', 'Il Novellino (prologo, novelle I-II)',
        'Masuccio Salernitano', 'c. 1450-1475 (pr. 1476)', 'novelle', edition='L. Settembrini, Napoli 1874',
        notes='Same title as, but unrelated to, the anonymous 13th-c. Novellino prior. SAL 75%.'),
    X('italian', 'anonimo-romano-cronica', 'Cronica, chapters I-VIII', 'Anonimo romano',
      'rule B: 20th-c. critical edition (Porta 1979)'),
    X('italian', 'tristano-riccardiano', 'La leggenda di Tristano (Tristano Riccardiano)', 'anonymous',
      'rule B: 20th-c. critical edition (Di Benedetto 1942)'),
    X('italian', 'prose-genovesi', 'Prose religiose genovesi del sec. XIV', 'anonymous',
      'rule B: Genoese (Ligurian) dialect'),
    X('italian', 'sanseverino-meraviglie', 'Libro piccolo di meraviglie', 'Jacopo da Sanseverino',
      'rule B: 20th-c. critical edition (Guglielminetti 1985)'),
    PGW('italian', 44549, 'Il libro di Sidrach (Tuscan volgarizzamento)', 'anonymous (tr. of the French Sidrac)',
        '14th c. (ed. A. Bartoli 1868)', 'encyclopedic dialogue / natural lore, medicine',
        start=r'^\s*LIBRO DI SIDRAC\s*$', end=r'^\s*XXIIII\. XXVIIII\.', lic=LIC_PG_IT, drop=r'^Cap\.',
        sub=[(r'\bCap\.\s+[IVXLCDM]+\.?', ' ')],
        notes='Footnote paragraphs and call numbers removed; Bartoli preface cut; astrological tables cut.'),
    PGW('italian', 45126, 'La seconda e terza guerra punica (volgarizzamento)', 'Leonardo Bruni (tr. anon.)',
        '15th c. volgare of Bruni 1421 (ed. A. Ceruti 1875)', 'history',
        start=r'^\s*DELLA SECONDA E TERZA GUERRA PUNICA\s*$', end=r'^\s*INDICE\s*$', lic=LIC_PG_IT,
        drop=r'^(Questo libro scrisse|Nota che questo libro)',
        notes='Ceruti preface, NOTE blocks and scribal/ownership colophons removed.'),
    PGW('italian', 67931, 'Frammenti letterari e filosofici', 'Leonardo da Vinci', 'c. 1480-1519 (ed. E. Solmi 1900)',
        'fables, notes, treatise fragments', start=r'^LE FAVOLE\.\s*$', end=r'^NOTE\.\s*$', lic=LIC_PG_IT,
        notes='Solmi preface and sigla table cut; [Sidenote] variants removed; spelling partly modernized.'),
    # ================= Old French (extract_v2) =================
    GE('ed_AimeriD', 'Aymeri de Narbonne', 'Bertrand de Bar-sur-Aube (attr.)', 'early 13th c. (ed. Demaison 1887)',
       group='cycle:aymeri-de-narbonne'),
    GE('ed_MortAymC', 'La Mort Aymeri de Narbonne', 'anonymous', 'c. 1180-1200 (ed. Couraye du Parc 1884)',
       group='cycle:aymeri-de-narbonne'),
    GE('ed_GuiBourgG_pos', 'Gui de Bourgogne', 'anonymous', 'c. 1211 (ed. Guessard-Michelant 1859)'),
    GE('ed_FloovG_pos', 'Floovant', 'anonymous', 'late 12th c. (ed. Guessard-Michelant 1859)'),
    GE('ed_OtinG_pos', 'Otinel', 'anonymous', '13th c. (ed. Guessard-Michelant 1859)'),
    WSJ('old_french', 'villehardouin-conqueste', 'old_french-v2/villehardouin-conqueste',
        'De la conqueste de Constantinople', 'Geoffroi de Villehardouin', 'c. 1207-1213', 'crusade chronicle',
        wiki='fr', sub=[(r'\[An \d+\.?\]', ' ')],
        edition='Du Cange text, ed. C.-B. Petitot, Collection des mémoires 1re série t. 1, 1824',
        notes='Original Old French only. Same events as Clari (PROFITEROLE) but a different text.'),
    OMF('BenTroieC_1partial', 'old_french-v2', 'Le Roman de Troie (vol. 1 of the edition)', 'Benoît de Sainte-Maure',
        'c. 1165 (ed. Constans 1904)', 'verse romance', notes='OpenMedFr file covers volume 1 only.'),
    OMF('OmbreB2', 'old_french-v2', "Le Lai de l'Ombre", 'Jean Renart', 'early 13th c. (ed. Bédier 1913)',
        'lai / short verse narrative'),
    X('old_french', 'openmedfr:JAvesnesFilleB', 'La Fille du comte de Pontieu (15th-c. version)', 'anonymous',
      'rule B: Fille du comte de Ponthieu pair'),
    X('old_french', 'openmedfr:FillePonth1B1', 'La Fille du comte de Ponthieu (13th-c. version 1)', 'anonymous',
      'rule B: Fille du comte de Ponthieu pair'),
    X('old_french', 'openmedfr:Cliges', 'Cligès', 'Chrétien de Troyes', 'rule B: Chrétien is a training author'),
    # ================= English (extract_v2) =================
    PGW('english', 5827, 'The Problems of Philosophy', 'Bertrand Russell', '1912', 'philosophy (non-fiction)',
        start=r'^CHAPTER I\. APPEARANCE AND REALITY', end=r'^BIBLIOGRAPHICAL NOTE'),
    PGW('english', 7213, 'My Life and Work', 'Henry Ford (with Samuel Crowther)', '1922',
        'memoir / business (non-fiction)', start=r'^INTRODUCTION\s*$(?![\s\S]{0,200}^I\. THE BEGINNING)',
        end=r'^INDEX\s*$(?![\s\S]{0,50}^INTRODUCTION)'),
    PGW('english', 6435, 'The Principles of Scientific Management', 'Frederick Winslow Taylor', '1911',
        'management (non-fiction)', start=r'^INTRODUCTION\s*$'),
    PGW('english', 852, 'Democracy and Education', 'John Dewey', '1916', 'philosophy of education (non-fiction)',
        start=r'^Chapter One: Education as a Necessity of Life\s*$(?![\s\S]{0,300}^Chapter Two)',
        drop=r'^Chapter [A-Z][a-z-]+:'),
    PGW('english', 15776, 'The Economic Consequences of the Peace', 'John Maynard Keynes', '1919',
        'economics (non-fiction)', start=r'^CHAPTER I\s*$(?![\s\S]{0,200}^CHAPTER II)',
        notes='Footnote sections and statistical tables removed.'),
    PGW('english', 408, 'The Souls of Black Folk', 'W. E. B. Du Bois', '1903', 'essays (non-fiction)',
        start=r'^I\.\s*\nOf Our Spiritual Strivings', drop_indented=1,
        notes='Chapter verse epigraphs mostly removed (indented lines).'),
    PGW('english', 24, 'O Pioneers!', 'Willa Cather', '1913', 'fiction (novel)', start=r'^PART I\.\s*$',
        drop_indented=4, notes='Indented verse dropped.'),
    PGW('english', 416, 'Winesburg, Ohio', 'Sherwood Anderson', '1919', 'fiction (short stories)',
        start=r'^THE BOOK OF THE GROTESQUE\s*$(?![\s\S]{0,100}^PART ONE)', notes='Irving Howe introduction cut.'),
    PGW('english', 14314, 'Etiquette in Society, in Business, in Politics and at Home', 'Emily Post', '1922',
        'advice / how-to (non-fiction)', start=r'^CHAPTER I\s*$(?![\s\S]{0,300}^CHAPTER II\s*$)',
        end=r'^=INDEX=\s*$', notes='Richard Duffy introduction cut.'),
    X('english', 'gutenberg:6456', 'Public Opinion', 'Walter Lippmann', 'rule B: English Lippmann'),
    *[X('english', 'gum:' + d, d, 'GUM', 'rule B: GUM document') for d in
      ['GUM_court_property', 'GUM_essay_ghost', 'GUM_essay_merit', 'GUM_speech_albania', 'GUM_textbook_spacetime']],
    # ================= Czech (extract_v2) =================
    WSJ('czech', 'chelcicky-trojiem-lidu', 'czech-v2/chelcicky-trojiem-lidu', 'O trojiem lidu', 'Petr Chelčický',
        'c. 1424-1425', 'religious treatise', wiki='cs', group='author:petr-chelcicky', period='pre-1500',
        notes='cs.wikisource (source ceskacitanka.cz, edition not named; Old Czech spelling kept). PD-old-70.'),
    WSJ('czech', 'chelcicky-cierkvi', 'czech-v2/chelcicky-cierkvi', 'O cierkvi svaté', 'Petr Chelčický', 'c. 1440',
        'religious treatise', wiki='cs', group='author:petr-chelcicky', period='pre-1500',
        notes='Source ceskacitanka.cz. PD-old-70.'),
    WSJ('czech', 'hus-dcerka', 'czech-v2/hus-dcerka', 'Dcerka (O poznání cesty pravé k spasení)', 'Jan Hus',
        'c. 1412-1413', 'devotional treatise', wiki='cs', period='pre-1500',
        flags=['SPELLING FLAG: MKP e-book 2011, modernised Czech spelling (ou, ě, j), unlike DIAKORP transcriptions.'],
        notes='Licence PD-old-70.'),
    X('czech', 'stitny-sasiech', 'Kniežky o šašiech', 'Tomáš Štítný ze Štítného',
      "rule B: Štítný is an author of the Czech prior's training texts (diakorp50)"),
    CZ('diakorp29-1350-1400', 'Prayer collection; shares prayers with round-one diakorp54 (filtered).'),
    *[CZ(n, POST1500) for n in ['diakorp108-1543', 'diakorp18-1558', 'diakorp21-1552', 'diakorp31-1580',
                                'diakorp32-1585', 'diakorp37-1532', 'diakorp51-1565', 'diakorp53-1581']],
    X('czech', 'czech:diakorp24-1595', 'Kvalt na pohany', 'Bartoloměj Paprocký z Hlahol',
      'rule B: about 23% Latin'),
    X('czech', 'czech:diakorp5-1577', 'Čtení Nikodémovo', 'anonymous', 'rule B: Gospel of Nicodemus'),
    # ================= Catalan (catalan_v2 / works_catalan) =================
    CAT('llull-primera-segona-intencio', 'Libre de la primera e segona intenció', 'Ramon Llull', 'c. 1282-1287',
        'moral-theological treatise', edition='Obres de Ramón Llull, ed. J. Rosselló, Palma 1886',
        group='author:ramon-llull'),
    CAT('llull-mil-proverbis', 'Libre de mil proverbis', 'Ramon Llull', '1302', 'proverbs / moral aphorisms',
        poem_lines=True, edition='Obres de Ramón Llull, ed. J. Rosselló, Palma 1886', group='author:ramon-llull',
        notes='Numbered proverbs laid out as <div class=poem> lines: poem_paragraphs (numbers removed).'),
    CAT('llull-gentil', 'Libre del gentil e los tres savis', 'Ramon Llull', 'c. 1274-1276', 'religious dialogue',
        dir='catalan/llull-gentil', edition='Obres de Ramón Llull, Rosselló 1886', group='author:ramon-llull',
        notes='Round-one on-disk candidate, unused.'),
    CAT('metge-somni', 'Lo somni', 'Bernat Metge', '1399', 'philosophical dialogue',
        edition='1891 edition (Lo sompni, 1891 djvu)',
        notes='Same author as Valter e Griselda (in R as an exclusion-only text); distinct original work.'),
    CAT('perellos-purgatori', 'Viatge al Purgatori de Sant Patrici', 'Ramon de Perellós', '1398',
        'travel / visionary account', edition="Llegendes de l'altra vida, ed. R. Miquel i Planas 1914"),
    CAT('visio-tundal', 'Visió de Tundal (Historia de Tuglat)', 'anonymous (Catalan version)', '14th-15th c.',
        'visionary legend', edition="Llegendes de l'altra vida, ed. R. Miquel i Planas 1914",
        notes='The parallel Historia de Gaudal is not used.'),
    CAT('filla-rey-hungria', "Historia de la filla del rey d'Hongria", 'anonymous', '14th c.',
        'prose novella / legend', edition="Histories d'altre temps, ed. R. Miquel i Planas 1910"),
    CAT('paris-viana', 'Historia de Paris e Viana', 'anonymous', '15th c. (pr. 1495)', 'prose romance',
        edition="Histories d'altre temps, ed. R. Miquel i Planas 1910"),
    CAT('carcer-amor', "Lo càrcer d'amor", 'Bernardí Vallmanya (transl. of Diego de San Pedro)', '1493',
        'sentimental romance (translation)', sub=[(r'\(\s*L\.\s*\d+\s*\)', ' ')],
        edition='ed. R. Miquel i Planas 1906 (1493 text)'),
    CAT('seneca-tragedies', 'Les tragèdies de Sèneca', 'Antoni de Vilaragut (translator)', 'late 14th c.',
        'prose translation (drama rendered in prose)', edition='ed. R. Miquel i Planas 1914',
        notes='Interlinear variants: only the underlined base reading is kept.'),
    CAT('art-be-morir', 'Art de bé morir', 'anonymous (translation of Ars moriendi)', '1493 (Valencia print)',
        'devotional', edition='facsimile ed. 1905 (A. Aguiló)',
        drop=r"^D'aquesta obreta|^La pre[sſ]ent reproduccio",
        sub=[('ꝑ', 'per'), ('ꝓ', 'pro'), ('q̃', 'que'), ('p̃', 'pre'), ('ã', 'an'), ('ẽ', 'en'),
             ('õ', 'on'), ('ũ', 'un'), ('ĩ', 'in'), ('ṽ', 'vn'), ('̇', ''), ('⹒', ' ')],
        notes='Diplomatic transcription of the 1493 print; abbreviations expanded.'),
    CAT('homilies-organya', "Homilies d'Organyà", 'anonymous', 'late 12th - early 13th c.', 'sermons',
        edition='Antics documents de llengua catalana, 1915', sub=[(r'^[^«»]{0,600}?\s\.?S\.\s', ' ')],
        notes='Latin gospel pericope opening each homily removed; short Latin quotations remain.'),
    CAT('eiximenis-regles', 'Regles de bona criança (Terç del Crestià excerpt)', 'Francesc Eiximenis', 'c. 1384',
        'conduct / table manners', edition='J. Balari i Jovany 1889'),
    CAT('ordinacions-cort', 'Ordinacions de la Casa i Cort (Pere III/IV)', 'Pere el Cerimoniós (chancery)', '1344',
        'legal / court ordinances', edition='P. de Bofarull, CODOIN ACA V, 1850',
        group='family:crown-of-aragon-chancery',
        train_only='rule A: chancery family shares authors (Jaume I) with the legacy Llibre dels Fets TRAIN group'),
    CAT('alcanyis-regiment', 'Regiment preservatiu e curatiu de la pestilència', 'Lluís Alcanyís', 'c. 1490',
        'medical', edition='A. Chinchilla, Anales históricos de la medicina, 1845-1848',
        drop=r'^(Ningun historiador|Mucho tiempo|Estos datos|Está escrita|Recipe\.|Omnipotens)',
        notes='Castilian introduction, Latin prayer and Latin Recipe formulas dropped.'),
    CAT('llibre-coch', 'Llibre del Coch', 'Robert de Nola', 'pr. 1520 (compiled late 15th c.)', 'cookery',
        edition='1520 Barcelona print (facsimile)'),
    CAT('corella-prosa', 'Prose works (Leander i Hero; Tragèdia de Caldesa; Triümfo de les dones)',
        'Joan Roís de Corella', 'c. 1458-1462', 'mythological / sentimental prose',
        notes='Three short prose works by one author, one entry.'),
    CAT('cartes-cancelleria', 'Royal and parliamentary letters and edicts (collection)',
        'chancery of Jaume I, Jaume II, Pere III/IV, Jaume III of Majorca; Parlament; Joana Enríquez; Joan II',
        '1250-1475', 'letters / chancery', drop=r'^(En nom de Deu tot piados|\d+\.\s*(\.\s*)+|Nota:)',
        drop_latin=True, group='family:crown-of-aragon-chancery',
        train_only='rule A: chancery family shares authors (Jaume I) with the legacy Llibre dels Fets TRAIN group'),
    CAT('documents-dret', 'Treaties, ordinances and parliamentary acts (collection)', 'various (incl. Jaume I)',
        '1270-1436', 'legal / administrative', drop=r'^(Certificat de traducció|Nota:)', drop_latin=True,
        group='family:crown-of-aragon-chancery',
        train_only='rule A: chancery family shares authors (Jaume I) with the legacy Llibre dels Fets TRAIN group'),
    CAT('sermonari-marsella', 'Sermonari de Marsella', 'anonymous', '14th c.', 'sermons',
        dir='catalan/sermonari-marsella', edition='1911 (journal transcription)', drop=r'[àèéíòóú].*[àèéíòóú]',
        notes='Round-one on-disk candidate, unused. Paragraphs with two or more accented vowels (1911 editorial '
              'Catalan) dropped.'),
]
# Fetched only as an exclusion text (already used elsewhere): joins R, never a role.
VALTER = CAT('x-metge-valter', 'Historia de Valter e Griselda', 'Bernat Metge', '1388',
             'novella (translation of Petrarch)', dir='catalan-v2/_exclusion-only/x-metge-valter')
REJECTED = RAW / 'catalan-v2/_rejected'

# German text families (rule A/E): ReM group ids (M###) whose titles show one text, one author or parallel versions.
REM_FAMILIES = {
    'rem-family:genesis': (['M028', 'M087', 'M088', 'M116', 'M138', 'M149'],
                           'Wiener Genesis and the Vorauer Bücher Mosis 1-5 (Genesis, Joseph = reworked Wiener '
                           'Genesis Joseph, Moses, Marienlob, Balaam)'),
    'rem-family:kaiserchronik': (['M055', 'M121', 'M213'],
                                 'Kaiserchronik and its separately transmitted episodes (Crescentia, Trierer Silvester)'),
    'rem-family:physiologus': (['M154', 'M155', 'M156', 'M157'], 'Physiologus versions (prose and Millstätter verse)'),
    'rem-family:gospels': (['M067', 'M318', 'M323', 'M402', 'M520'],
                           'German gospel translations / evangelistaries (parallel versions of one text)'),
    'rem-family:psalter': (['M040', 'M185', 'M186', 'M187', 'M188', 'M189', 'M191', 'M192', 'M193', 'M194', 'M195',
                            'M242'], 'German psalter versions (incl. Wiener Notker psalter)'),
    'rem-family:benedictine-rule': (['M324', 'M358', 'M506', 'M508'], 'Benediktinerregel translations'),
    'rem-family:gottfried-tristan': (['M341', 'M342'], 'Gottfried von Straßburg, Tristan (two manuscripts)'),
    'rem-family:prosa-lancelot': (['M505', 'M548'], 'Prosa-Lancelot (two manuscripts)'),
    'rem-family:lilie': (['M327', 'M354'], 'Die Lilie (prose part, verse part)'),
    'rem-family:predigt-christi-geburt': (['M159', 'M252'], 'Predigt von Christi Geburt (M/G T 38), two copies'),
    'rem-family:millstaetter-predigten': (['M161', 'M329'], 'Millstätter Predigtsammlung'),
    'rem-family:leipziger-predigten': (['M167', 'M509', 'M536'], 'Leipziger Predigten'),
    'rem-family:mitteldeutsche-predigten': (['M328', 'M330', 'M504'], 'Mitteldeutsche Predigten'),
    'rem-family:kraeuterbuch': (['M125', 'M126'], 'Kräuterbuch, Innsbrucker and Prüller Fassung'),
    'rem-family:psalter-prayer-instructions': (['M084', 'M085', 'M086'], 'Gebetsanweisungen zum Psalter'),
    'rem-family:aegidius': (['M005', 'M006'], 'Aegidius (Trierer, Höxterer)'),
    'rem-family:mariensequenz': (['M140', 'M141'], 'Mariensequenz (Muri, St. Lambrecht/Seckau)'),
    'rem-family:frau-ava': (['M022', 'M023', 'M024'], 'Frau Ava (one author)'),
    'rem-family:lambrecht-alexander': (['M008', 'M009', 'M226'],
                                       'Pfaffe Lambrecht (Alexanderlied, Tobias) and the Straßburger Alexander'),
    'rem-family:rudolf-von-ems': (['M336', 'M359'], 'Rudolf von Ems (one author)'),
    'rem-family:koeln-hagen': (['M349', 'M350', 'M529'], 'Kölner Urkunden and Gottfried Hagen (charters, chronicle)'),
    'rem-family:nuernberger-stadtbuch': (['M338', 'M540'], 'Nürnberger Stadtbuch'),
    'rem-family:augsburger-urkunden': (['M344', 'M345'], 'Augsburger Urkunden'),
    'rem-family:freiburger-urkunden': (['M346', 'M347'], 'Freiburger Urkunden'),
    'rem-family:mittelfraenkische-urkunden': (['M544', 'M545'], 'Mittelfränkische Urkunden'),
    'rem-family:schwabenspiegel': (['M339', 'M410'], 'Schwabenspiegel and the Freisinger Rechtsbuch based on it'),
    'rem-family:flore': (['M307', 'M546'], 'Flore und Blanscheflur versions'),
}
REM_GENRES = {'P': 'prose', 'V': 'verse', 'PV': 'prose and verse', 'U': 'charters', '-': 'not classified'}


def rem_m(ident):
    return re.match(r'M\d+', ident)[0]


def rem_group(ident, families=REM_FAMILIES):
    m = rem_m(ident)
    for family, (members, _) in families.items():
        if m in members:
            return family
    return 'rem:' + m


# ---------------------------------------------------------------- file records
@lru_cache(maxsize=None)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path):
    return str(path.relative_to(ROOT))


def file_record(path, **extra):
    return dict(path=rel(path), sha256=sha(path), bytes=path.stat().st_size, **extra)


# ---------------------------------------------------------------- Wikisource HTML (sources_v2)
V2_STRIP = ('style, script, table, sup, ol.references, .references, .reference, .mw-editsection, .mw-heading, '
            'h1, h2, h3, h4, h5, h6, .pagenum, .numeropagina, .ws-pagenum, .ws-noexport, .noprint, .metadata, '
            '.titolo, .sottotitolo, #headerContainer, .titulusHeaderBox, .headertemplate, .centertext, .ct, '
            '.ws-summary, .sisitem, .reflist, .error, .OptionText, span[style*="visibility:hidden"], .navigate')
V2_SUBS = [
    (r'\[\s*\d+[a-z]?\s*\]', ' '),                     # bracketed note / folio numbers
    (r'\(\s*\d+\s*\)', ' '),                           # footnote call numbers
    (r'\b\d+\.\d{4}[A-D]\|', ' '),                      # Migne column markers
    (r'\bf\.\s*\d+\s*[rv]\b', ' '),                     # folio refs (f. 1r)
    (r'\.{4,}', ' '),                                   # dotted leaders
    (r'\[\d{4}[A-D]\]|\b\d+\.\d{4}[A-D]?\|', ' '),      # Migne column markers [0234B], 138.0169|
]
GLOBAL_DROP = r'cc_id|J\. P\. Migne'                    # Corpus Corporum / Migne metadata line
V2_ROMAN = [(r'\b[IVXLCDM][IVXLCDMc]*[IVXLCDM](?:e|en|a)?\b', ' '),   # roman numerals: MCCXLVII, MIIcLXXXVI
            (r'\.\s*[ivxlcdmj]+\s*\.', ' ')]                            # dotted manuscript numerals .xiiii.
MIN_WORDS = kr.MIN_WORDS
WSH_RULE = ('Rendered Wikisource page HTML (revid = wgRevisionId), pages in file order. Removed: ' + V2_STRIP +
            f'. Prose: <p> only, >= {MIN_WORDS} normalized words, >70% capitals dropped; verse: .poem lines, laisse/'
            'line numbers removed; bracketed numbers, folio refs, Migne markers, dotted leaders, roman numerals '
            'removed; Corpus Corporum / Migne metadata lines dropped; per-work drop/sub/start/stop.')


def ws_blocks(raw, w):
    """Prose: <p> paragraphs outside apparatus. Verse (w['verse']): lines of .poem blocks."""
    soup = BeautifulSoup(raw, 'html.parser')
    root = soup.select_one('#mw-content-text .mw-parser-output') or soup.select_one('.mw-parser-output')
    for img in root.select('img'):
        alt = img.get('alt') or ''
        if re.fullmatch(r'[A-ZÀ-ÝÇ]', alt):
            (img.find_parent('span', attrs={'typeof': 'mw:File'}) or img).replace_with(alt)
    strip = V2_STRIP + ('' if w.get('verse') else ', .poem, div.poem')
    for node in root.select(strip):
        node.decompose()
    out = []
    if w.get('verse'):
        for p in root.select('.poem p'):
            for br in p.select('br'):
                br.replace_with('\n')
            for line in p.get_text().split('\n'):
                line = ' '.join(line.split())
                for pattern, repl in V2_SUBS + V2_ROMAN + list(w.get('sub') or []):
                    line = re.sub(pattern, repl, line)
                line = re.sub(r'^\d+\.?\s*', '', line.strip())      # laisse / line numbers glued to the verse
                if len(normalize_source(line).split()) >= 2:
                    out.append(line)
        return out
    for p in root.find_all('p'):
        for br in p.select('br'):
            br.replace_with(' ')
        text = ' '.join(p.get_text().split())
        if (w.get('drop') and re.search(w['drop'], text)) or re.search(GLOBAL_DROP, text):
            continue
        for pattern, repl in V2_SUBS + V2_ROMAN + list(w.get('sub') or []):
            text = re.sub(pattern, repl, text)
        text = ' '.join(text.split())
        if kr.caps_share(text) > 0.7:
            continue
        if len(normalize_source(text).split()) < MIN_WORDS:
            continue
        out.append(text)
    return out


def page_title(raw):
    m = re.search(rb'"wgPageName":"([^"]*)"', raw)
    return json.loads(b'"' + m[1] + b'"').replace('_', ' ') if m else None


def ws_html(w):
    files, paras = [], []
    for p in sorted((RAW / (w['language'] + '-v2') / w['id']).glob('*.html')):
        raw = p.read_bytes()
        rev = int(re.search(rb'"wgRevisionId":(\d+)', raw)[1])
        rec = file_record(p, page=page_title(raw), revid=rev, url=f"https://{w['host']}/w/index.php?oldid={rev}")
        if p.name.startswith('000-index'):
            rec['role'] = 'index (link order only)'
        else:
            paras.extend(ws_blocks(raw, w))
        files.append(rec)
    if w.get('start'):
        paras = paras[next(i for i, x in enumerate(paras) if re.search(w['start'], x)):]
    if w.get('stop'):
        paras = paras[:next(i for i, x in enumerate(paras) if re.search(w['stop'], x))]
    return '\n'.join(paras), files, WSH_RULE


# ---------------------------------------------------------------- Internet Archive OCR (sources_v2)
FRENCH = set('les des du dans est une sont pour cette nous aux avec ces été être était leçon corr cf voy ms '
             'où elle sur comme lisez faut peut texte vers note voir ici'.split())
GERMAN = set('der die das und ist nicht mit von zu den dem sich bei wird sind auch als wie nach eine einer ein '
             'im auf aus dass daß werden hat wurde fehlt vgl oder zum zur'.split())
STRONG = {'dans', 'pour', 'cette', 'avec', 'sont', 'été', 'était', 'leçon', 'corr', 'voy', 'cf', 'ms', 'lisez',
          'und', 'der', 'die', 'das', 'nicht', 'fehlt', 'vgl', 'ist'}
IA_RULE = ('djvu.txt OCR layer; region = edition text only (marker lines or explicit line span); dropped: running '
           'heads, page numbers, lines with em-dash apparatus / ms. / sigla, numbered footnotes, lines with French '
           'or German function words, all-caps headings; leading/trailing line numbers and roman numerals removed; '
           'prose de-hyphenated and re-joined into paragraphs. OCR errors are not corrected.')


def foreign(line, lang):
    words = [x.strip('.,;:!?()[]«»"\'') for x in line.lower().split()]
    if lang == 'latin':            # French/German markers only; 'et', 'in', 'est' are Latin
        hits = [x for x in words if x in GERMAN or x in (FRENCH - {'est'})]
    else:
        hits = [x for x in words if x in FRENCH or x in GERMAN]
    return len(hits) >= 2 or any(x in STRONG for x in hits)


def ia_lines(text, w):
    lines = text.replace('\r', '').split('\n')
    reg = w['region']
    if isinstance(reg[0], int):
        a, b = reg
    else:
        a = next(i for i, l in enumerate(lines) if re.search(reg[0], l.strip()))
        b = next(i for i, l in enumerate(lines) if i > a and re.search(reg[1], l.strip()))
    return lines[a:b], (a, b)


def ia_clean(text, w):
    lines, span = ia_lines(text, w)
    lang, verse = w['language'], w['kind'] == 'verse'
    kept = []
    for raw in lines:
        s = ' '.join(raw.split())
        if not s:
            kept.append('')
            continue
        if w.get('header') and re.search(w['header'], s):
            continue
        if re.search(r'\s[—–]\s|[—–]\s*\d|^\s*[—–-]|\bms\.|\bMs\.|\bMS\.|\bQ\.\s*\d|Digitized|Google', s):
            continue                                              # apparatus / page furniture
        if re.fullmatch(r'[\W\d_]*', s) or re.fullmatch(r'[\divxlcIVXLC.\-* ]+', s):
            continue                                              # page numbers, signatures
        if verse and len(re.findall(r'\b[A-E]\b', s)) >= 2:
            continue                                              # sigla-heavy apparatus
        if re.match(r'^\s*[\d*]+\)\s', s) and (len(s) < 70 or re.search(r':|\b[A-Z]\.', s)):
            continue                                              # numbered footnotes
        if foreign(s, lang):
            continue
        letters_ = [c for c in s if c.isalpha()]
        if len(letters_) >= 4 and sum(c.isupper() for c in letters_) / len(letters_) > 0.7:
            continue                                              # headings / running titles
        s = re.sub(r'\(\s*/?\.?\s*\d+\s*[a-d]?\s*\)|\{\s*/\.\s*\d+\s*[a-d]?\s*\)', ' ', s)   # (f. 40 a)
        s = re.sub(r'^\s*\d{1,5}[.,]?\s+', '', s)                 # leading line numbers
        s = re.sub(r'\s+\d{1,5}[a-d]?\s*$', '', s)                # trailing line numbers / folio refs
        for pattern, repl in V2_ROMAN + list(w.get('sub') or []):
            s = re.sub(pattern, repl, s)
        kept.append(s)
    if verse:
        out = [l for l in kept if len(normalize_source(l).split()) >= 2]
    else:
        body = '\n'.join(kept)
        body = re.sub(r'(?<=[^\W\d_])[-¬]\s*\n\s*(?=[^\W\d_])', '', body)        # de-hyphenate line breaks
        out = [' '.join(p.split()) for p in re.split(r'\n\s*\n', body)]
        out = [p for p in out if len(normalize_source(p).split()) >= MIN_WORDS]
    return out, span


def ia(w):
    d = RAW / (w['language'] + '-v2') / w['id']
    p, m = d / w['file'], d / f"ia-metadata-{w['ident']}.json"
    meta = json.loads(m.read_bytes())
    md = meta.get('metadata', {})
    sha1 = next((f.get('sha1') for f in meta.get('files', []) if f['name'] == w['file']), None)
    paras, span = ia_clean(p.read_text(errors='replace'), w)
    w['url'] = f"https://archive.org/download/{w['ident']}/{w['file']}"
    files = [file_record(p, url=w['url'], ia_sha1=sha1, sha1_matches=hashlib.sha1(p.read_bytes()).hexdigest() == sha1,
                         line_span=list(span), ocr=md.get('ocr'),
                         ia_rights=md.get('possible-copyright-status') or md.get('licenseurl') or md.get('rights')),
             file_record(m, url=f"https://archive.org/metadata/{w['ident']}",
                         role='item metadata (fetched later; includes volatile fields)')]
    return '\n'.join(paras), files, IA_RULE


# ---------------------------------------------------------------- Wikisource JSON, PG, OpenMedFr, Geste, DIAKORP
WSJ_RULE = 'MediaWiki parse-API JSON, key_recovery_confirmation_sources.ws_paragraphs: ' + kr.WS_RULE
PG_RULE = ('PG header/footer; front matter before the start marker and back matter after the end marker; '
           '[Illustration/Footnote/Sidenote] blocks and FOOTNOTES sections; short-line (verse/table) blocks; all-caps '
           'headings; note paragraphs; call numbers; digit-heavy tables; per-work drop/sub/drop_indented.')


def ws_json(w):
    files, paras = [], []
    for p in sorted((RAW / w['dir']).glob('*.json')):
        data = json.loads(p.read_bytes())['parse']
        url = f"https://{w['wiki']}.wikisource.org/w/index.php?oldid={data['revid']}"
        if p.name.startswith('000-index'):
            files.append(file_record(p, page=data['title'], revid=data['revid'], url=url, role='index (link order only)'))
            continue
        revid, title, ps = kr.ws_paragraphs(p.read_bytes(), w)
        files.append(file_record(p, page=title, revid=revid, url=url))
        paras += ps
    w['url'] = next(f['url'] for f in files if 'role' not in f)
    return '\n'.join(paras), files, WSJ_RULE


def pg(w):
    path = RAW / w['path']
    body = kr.pg_body(path.read_text(encoding='utf-8'))
    body = body[re.search(w['start'], body, re.M).start():]
    if w.get('end'):
        body = body[:re.search(w['end'], body, re.M).start()]
    body = re.sub(r'\[(?:Illustration|Footnote|Sidenote)[^\]]*\]', ' ', body, flags=re.S)
    body = re.sub(r'^FOOTNOTES?:\s*$[\s\S]*?(?=^CHAPTER |\Z)', ' ', body, flags=re.M)   # end-of-chapter note blocks
    out = []
    for para in re.split(r'\n\s*\n', body):
        lines = [l for l in para.split('\n') if l.strip()]
        if w.get('drop_indented'):
            lines = [l for l in lines if not re.match(r' {%d,}' % w['drop_indented'], l)]
        if len(lines) >= 2 and max(len(l.strip()) for l in lines) < 55:
            continue                                         # verse / epigraph / table blocks (PG prose wraps at ~70)
        text = ' '.join(' '.join(lines).split())
        if not text or kr.caps_share(text) > 0.7:
            continue
        if re.match(r'^(\[\d+\]|\(\d+\)|NOTE:|NOTES:|FOOTNOTES?:)', text):
            continue
        if w.get('drop') and re.search(w['drop'], text):
            continue
        text = re.sub(r'\[\d+\]|\(\d+\)', ' ', text)
        for a, b in w.get('sub') or []:
            text = re.sub(a, b, text)
        if sum(c.isdigit() for c in text) > 0.2 * max(1, sum(c.isalnum() for c in text)):
            continue                                         # statistical tables
        if len(normalize_source(text).split()) < MIN_WORDS:
            continue
        out.append(text)
    return '\n'.join(out), [file_record(path, url=w['url'])], PG_RULE


def omf(w):
    path = RAW / w['path']
    lines = []
    for l in path.read_text(encoding='utf-8').lstrip('﻿').replace('\r\n', '\n').replace('\r', '\n').split('\n'):
        s = l.strip()
        if re.fullmatch(r'Version\w*', s) or s.startswith('# |'):
            continue
        lines.append(l)
    readme = RAW / 'old_french/openmedfr_README.md'
    return kr.openmedfr_clean('\n'.join(lines)), [file_record(path, url=w['url']),
                                                  file_record(readme, role='licence / provenance')], kr.OMF_RULE


def geste(w):
    path = RAW / 'old_french' / f"geste_{w['stem']}.txt"
    readme = RAW / 'old_french/geste_README.md'
    return kr.geste_clean(path.read_text(encoding='utf-8')), [file_record(path, url=w['url']),
                                                             file_record(readme, role='licence / provenance')], kr.GESTE_RULE


def diakorp_meta(raw):
    return dict(re.findall(r'^# ([^:]+):[ \t]*(.*)$', raw, re.M))


def diakorp(w):
    path = le.RAW / 'diakorp.zip'
    member = f"czech/diakorp/txt/{w['diakorp']}.txt"
    with zipfile.ZipFile(path) as archive:
        raw = archive.read(member).decode('utf-8')
    meta = diakorp_meta(raw)
    dates = [int(x) for x in re.findall(r'\d{4}', meta['originDate'])]
    if not dates or min(dates) < 1300 or max(dates) > 1600:
        raise ValueError('Outside the 1300-1600 window: ' + w['diakorp'])
    body = '\n'.join(l for l in raw.splitlines() if not l.startswith('#'))
    w.update(title=meta['title'].strip(), author=meta.get('author', '').strip() or 'anonymous',
             date=meta['originDate'].strip(), genre=meta.get('genre', '').strip(),
             period='pre-1500' if max(dates) < 1500 else '1500-1595',
             member_sha256=hashlib.sha256(raw.encode()).hexdigest())
    return re.sub(r'\[\s*\.\.\.\s*\]', ' ', body), [
        file_record(path, url=kr.DIAKORP_URL, member=member),
        file_record(le.RAW / 'diakorp-readme.txt', role='licence / provenance',
                    url='https://sprakbanken-clarin.lingfil.uu.se/histcorp/readme/czech-readme-diakorp')], kr.CZECH_RULE


def udante(w):
    u = RAW / 'latin/udante'
    out = []
    for split in ('train', 'dev', 'test'):
        for row in conllu_sentences(u / f'la_udante-ud-{split}.conllu'):
            if row['id'].startswith(w['prefix']):
                out.append(' '.join(row['words']))
    names = ('LICENSE.txt', 'README.md', 'la_udante-ud-train.conllu', 'la_udante-ud-dev.conllu',
             'la_udante-ud-test.conllu')
    return '\n'.join(out), [file_record(u / n, url=UDANTE + n) for n in names], (
        '# text of every sentence whose sent_id starts with the work prefix (written words)')


# ---------------------------------------------------------------- Catalan (catalan_v2)
CA_LATIN = set('''et est quod cum ad dei gratia domini dominus regis rex anno sunt vel ejus eius super hujus huius ego
quam atque enim sed ut nobis vestris vestri dictus dicti dicto dictis dictam presentibus presente quondam universi
idcirco mandantes itaque testes signum data datum infans illustrissimi inclitus noverint habens juravit
fecit apposui subscripsi vidi legi meum nostri nostris michi mandavit etiam predicti premissa'''.split())
CA_LATIN_MAX = 0.12
SIDENOTES = '.sidenote-right, .sidenote-left, .sidenote, .marginnote, .ws-sidenote'
CA_SUBS = [
    (r'\[[^\]]*\?[^\]]*\]', ' '),                         # [flayrosos ?]  editorial conjecture
    (r'\[\s*(f|fol|full|foli|pag|p)\.?\s[^\]]*\]', ' '),  # [f. 102 v. del Ms. ...]  folio reference
    (r'\[\s*(sic|sic\.)\s*\]|\(\s*sic\s*\)', ' '),
    (r'\[[^\]]*\d[^\]]*\]', ' '),                         # any bracket with a number
    (r'\[\s*(\.\s*){2,}\]|\[\s*…\s*\]', ' '),             # [...] lacuna mark
    (r'\(\s*f\.\s[^)]*\)', ' '),                           # (f. monilia)  editorial reading
    (r'\*', ' '),
]
CA_RULE = ('MediaWiki parse-API JSON; editorial sidenotes (' + SIDENOTES + ') removed; interlinear collation keeps '
           'the underlined base reading; editorial conjectures, folio refs, sic, numbered brackets and lacuna marks '
           'removed; then key_recovery_confirmation_sources.ws_paragraphs (poem_lines works: .poem lines of each page '
           f'joined, margin numbers removed); drop_latin works drop paragraphs with Latin function-word rate > '
           f'{CA_LATIN_MAX}.')


def ca_latin_rate(text):
    words = normalize_source(text).split()
    return sum(x in CA_LATIN for x in words) / len(words) if words else 0.0


def strip_sidenotes(raw):
    d = json.loads(raw)
    soup = BeautifulSoup(d['parse']['text'], 'html.parser')
    nodes = soup.select(SIDENOTES)
    inter = [s for s in soup.select('span[style*="inline-block"]') if s.find('u') and s.find('br')]
    if not nodes and not inter:
        return raw
    for n in nodes:
        n.decompose()
    for s in inter:
        s.replace_with(' ' + s.find('u').get_text() + ' ')
    d['parse']['text'] = str(soup)
    return json.dumps(d).encode()


def poem_paragraphs(raw, w):
    parse = json.loads(raw)['parse']
    soup = BeautifulSoup(parse['text'], 'html.parser')
    root = soup.select_one('.mw-parser-output') or soup
    for s in root.select('span[style*="left:-4em"], span[style*="position:absolute"], sup, .pagenum, '
                         'ol.references, .reference, .mw-editsection, #headerContainer, .ws-noexport, .noprint'):
        s.decompose()
    out = []
    for div in root.select('div.poem'):
        for br in div.select('br'):
            br.replace_with('\n')
        kept = []
        for text in (' '.join(l.split()) for l in div.get_text().split('\n')):
            if not text or (w.get('drop') and re.search(w['drop'], text)):
                continue
            for pattern, repl in kr.WS_SUBS + list(w.get('sub') or []):
                text = re.sub(pattern, repl, text)
            text = ' '.join(text.split())
            if text and kr.caps_share(text) <= 0.7:
                kept.append(text)
        para = ' '.join(kept)
        if len(normalize_source(para).split()) >= MIN_WORDS:
            out.append(para)
    return parse['revid'], parse['title'], out


def ca_paragraphs(raw, w):
    raw = strip_sidenotes(raw)
    w = dict(w, sub=CA_SUBS + list(w.get('sub') or []))
    revid, title, out = (poem_paragraphs if w.get('poem_lines') else kr.ws_paragraphs)(raw, w)
    if w.get('drop_latin'):
        out = [p for p in out if ca_latin_rate(p) <= CA_LATIN_MAX]
    return revid, title, out


def ws_catalan(w):
    files, paras = [], []
    for p in sorted((RAW / w['dir']).glob('*.json')):
        data = json.loads(p.read_bytes())['parse']
        url = f"https://ca.wikisource.org/w/index.php?oldid={data['revid']}"
        if p.name == '000-index.json':
            files.append(file_record(p, page=data['title'], revid=data['revid'], url=url, role='index (link order only)'))
            continue
        revid, title, ps = ca_paragraphs(p.read_bytes(), w)
        files.append(file_record(p, page=title, revid=revid, url=url))
        paras.extend(ps)
    w['url'] = next(f['url'] for f in files if 'role' not in f)
    return '\n'.join(paras), files, CA_RULE


# ---------------------------------------------------------------- German (ReM v2.1)
def german_catalogue():
    """Every ReM document as a work, grouped by M### and REM_FAMILIES; round-one and legacy status from pinned data."""
    round_one = [w['rem'] for w in kr.WORKS if w['language'] == 'german']
    legacy = read(lc.STATE / 'partitions.json')['priors']['german_broad']
    legacy_groups = {r['group'] for kind in ('train', 'calibration') for r in legacy[kind]} - {'german'}
    round_one_groups = {rem_group(i) for i in round_one}
    legacy_families = {rem_group(m) for m in legacy_groups}
    out = []
    with zipfile.ZipFile(lc.RAW / 'ReM-v2.1_tei.zip') as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith('.xml'):
                continue
            root = ET.fromstring(archive.read(name))
            ident = name.rsplit('/', 1)[-1][:-4]
            genre = root.find('.//t:classCode', lc.NS)
            usage = [e.text for e in root.findall('.//t:langUsage/t:language', lc.NS) if e.text and e.text != '-']
            ms = root.find('.//t:msIdentifier', lc.NS)
            code = genre.text if genre is not None else '-'
            group = rem_group(ident)
            work = dict(language='german', id='rem:' + ident, source='rem', rem=ident, group=group,
                        title=root.find('.//t:title', lc.NS).text, author='not stated in ReM header',
                        date='Middle High German, ReM 1050-1350 (no date in TEI header)', genre=REM_GENRES[code],
                        form=REM_GENRES[code], variety=', '.join(usage),
                        manuscript=' '.join(' '.join(e.text.split()) for e in ms if e.text)[:200] if ms is not None else '',
                        license=kr.LIC_REM, url=kr.REM_URL, member=name)
            if group in round_one_groups:
                work['exclude'] = 'rule A/E: round-one work, or same ReM group / text family as one'
            elif rem_m(ident) in legacy_groups:
                work['exclude'] = ('rule E/F: ReM group in legacy german_broad train/calibration (enters TRAIN only '
                                   'through the legacy rows)')
            elif group in legacy_families:
                work['train_only'] = 'rule A: same text family as legacy german_broad training text'
            if not work.get('exclude'):
                work['_text'] = ' '.join(kr.rem_words(root, ident))
            out.append(work)
    return out


def rem(w):
    return w.pop('_text'), [file_record(lc.RAW / 'ReM-v2.1_tei.zip', url=kr.REM_URL, member=w['member'])], kr.REM_RULE


HANDLERS = dict(ws_html=ws_html, ia=ia, ws_json=ws_json, pg=pg, omf=omf, geste=geste, diakorp=diakorp,
                udante=udante, ws_catalan=ws_catalan, rem=rem)
INTERNAL = {'_text', 'drop', 'sub', 'start', 'stop', 'end', 'verse', 'kind', 'region', 'header', 'ident', 'file',
            'dir', 'wiki', 'path', 'stem', 'pg', 'drop_indented', 'diakorp', 'prefix', 'poem_lines', 'drop_latin',
            'rem', 'host', 'member'}


def extraction_params(w):
    out = {}
    for key in sorted(INTERNAL - {'_text'}):
        value = w.get(key)
        if value in (None, False, '', []):
            continue
        out[key] = [list(x) for x in value] if key == 'sub' else list(value) if isinstance(value, tuple) else value
    return out


def extract(w):
    w = dict(w, group=w.get('group') or w['id'])
    text, files, rule = HANDLERS[w['source']](w)
    return dict(w, text=text, files=files, extraction=rule)


def latin_hosts(works):
    """Transcription sources named in the raw Wikisource pages (header 'fons' and its link)."""
    found = {}
    for w in works:
        if w['language'] != 'latin' or w['source'] not in ('ws_html', 'ws_json'):
            continue
        d = RAW / (w['dir'] if w['source'] == 'ws_json' else 'latin-v2/' + w['id'])
        blob = b''.join(p.read_bytes() for p in sorted(d.iterdir()) if p.is_file())
        hits = sorted({name for name, markers in LATIN_HOSTS.items() if any(m in blob for m in markers)})
        if hits:
            found[w['id']] = hits
    return found


def catalogue():
    works = [dict(w) for w in WORKS] + german_catalogue()
    hosts = latin_hosts(works)
    if hosts != EXPECTED_LATIN_HOSTS:
        raise ValueError(f'Latin transcription sources changed: {hosts}')
    for w in works:
        if w['id'] in hosts:
            w['exclude'] = 'rule B: Wikisource transcription source ' + ' / '.join(hosts[w['id']])
            w['transcription_source'] = hosts[w['id']]
    rejected = {p.parent.name for p in REJECTED.rglob('*.json')}
    assert not rejected & {w['id'] for w in works}, rejected
    ids = [w['id'] for w in works]
    assert len(ids) == len(set(ids)), Counter(ids).most_common(3)
    round_one = {w['id'] for w in kr.WORKS}
    assert all(w.get('exclude') for w in works if w['id'] in round_one), 'round-one work in the v2 catalogue'
    return works


# ---------------------------------------------------------------- reference R
def reference(paths=None):
    """R = key_recovery_confirmation_sources.reference() plus the whole text of every round-one confirmation work,
    every COMETA file on disk and the exclusion-only Valter e Griselda."""
    R, inputs, paths = kr.reference(paths)
    round_one = []
    for w in kr.WORKS:
        d = kr.extract(w)
        R.update(grams(' '.join(r['text'] for r in chunks(dict(id=d['id'], group=d['id'], text=d['text'])))))
        for f in d['files']:
            inputs[f['path']] = f['sha256']
        round_one.append(d['id'])
    manifest = kr.OUT / 'sources.json'
    if sorted(w['id'] for w in read(manifest)['works']) != sorted(round_one):
        raise ValueError('Round-one works differ from the sealed round-one manifest')
    inputs[rel(manifest)] = sha(manifest)
    cometa = []
    for p in sorted(le.RAW.glob('*.txt')):
        if p.name.startswith('diakorp'):
            continue
        R.update(grams(' '.join(r['text'] for r in chunks(dict(id=p.stem, group=p.stem,
                                                               text=kr.occitan_clean(p.read_text()))))))
        inputs[rel(p)] = sha(p)
        cometa.append(p.stem)
    valter = extract(VALTER)
    R.update(grams(' '.join(r['text'] for r in chunks(dict(id=valter['id'], group=valter['id'], text=valter['text'])))))
    for f in valter['files']:
        inputs[f['path']] = f['sha256']
    return R, dict(sorted(inputs.items())), paths, dict(round_one_works=round_one, cometa_files=cometa,
                                                        exclusion_only=[valter['id']])


def check_prior_rows(R):
    """Rule G: every train/calibration row of every earlier prior maps back to a source whose shingles are in R."""
    from experiments import language_id
    legacy = language_id.texts()
    sources = {}
    for d in [*lc.latin_documents(), *lc.rem_documents(lc.RAW / 'ReM-v2.1_tei.zip'),
              *lc.catalan_documents(lc.RAW / 'LlibredelsFets_60000corrected.txt'),
              *[d for docs in le.documents().values() for d in docs.values()]]:
        for row in chunks(d):
            sources[row['id']] = row['text']
    report = {}
    for path in (lc.STATE / 'partitions.json', le.STATE / 'partitions.json'):
        for name, prior in read(path)['priors'].items():
            for kind in ('train', 'calibration'):
                rows, missing = prior[kind], 0
                for row in rows:
                    m = re.fullmatch(r'([a-z_]+):([01]):(\d+)', row['id'])
                    text = legacy[m[1]][int(m[2])][int(m[3])] if m else sources[row['id']]
                    if not text.replace(' ', '').startswith(row['text']):
                        raise ValueError(f'Prior row does not match its source: {name} {row["id"]}')
                    missing += len(grams(text) - R)
                if missing:
                    raise ValueError(f'Rule G: {name} {kind} has {missing} shingles outside R')
                report[f'{rel(path)}:{name}:{kind}'] = len(rows)
    return report


# ---------------------------------------------------------------- roles, water-filling, filters
def group_hash(group_id):
    return hashlib.sha256(group_id.encode()).hexdigest()


def assign_roles(groups, n_test, n_cal=0, min_letters=0):
    """Fixed role rule. groups: dicts with id, clean (letters after the R filter), optional train_only (reason) and
    tier (int, lower first). Order = (tier, sha256(id)). The first n_test eligible groups are TEST, the next n_cal
    CALIBRATION; the rest TRAIN when n_cal (new priors) else unused. Eligible = not train_only and clean >= min_letters.
    Too few eligible groups stop the build."""
    ordered = sorted(groups, key=lambda g: (g.get('tier', 0), group_hash(g['id'])))
    counts, out = Counter(), {}
    for order, g in enumerate(ordered, 1):
        eligible = not g.get('train_only') and g['clean'] >= min_letters
        if eligible and counts['test'] < n_test:
            role = 'test'
        elif eligible and counts['calibration'] < n_cal:
            role = 'calibration'
        else:
            role = 'train' if n_cal else 'unused'
        counts[role] += 1
        why = None if eligible else (g.get('train_only') or f'fewer than {min_letters} clean letters')
        out[g['id']] = dict(role=role, order=order, sha256=group_hash(g['id']), not_eligible=why)
    if counts['test'] < n_test or counts['calibration'] < n_cal:
        raise ValueError(f'Too few eligible groups: {dict(counts)}; stop, do not improvise')
    return out


def water_fill(sizes, budget):
    """{group: letters taken}: each group min(size, cap) with the integer cap that fills `budget` exactly; the
    remainder (< number of capped groups) adds one letter to the first capped groups in the given order."""
    total = sum(sizes.values())
    if total < budget:
        raise ValueError(f'Train pool has only {total} letters for a budget of {budget}')
    lo, hi = 0, max(sizes.values())
    while lo < hi:                      # largest cap with sum(min(size, cap)) <= budget
        mid = (lo + hi + 1) // 2
        if sum(min(s, mid) for s in sizes.values()) <= budget:
            lo = mid
        else:
            hi = mid - 1
    take = {g: min(s, lo) for g, s in sizes.items()}
    remainder = budget - sum(take.values())
    for g, s in sizes.items():
        if remainder and s > lo:
            take[g] += 1
            remainder -= 1
    assert sum(take.values()) == budget and not remainder
    return take, lo


def letters(rows):
    return sum(len(r['text'].replace(' ', '')) for r in rows)


def test_blocker(R, role_grams, own, counts, mine):
    """Blocked reason for a TEST chunk: R, any v2 train/calibration text, or any other TEST work."""
    def blocked(candidate):
        if candidate & R:
            return 'reference'
        if candidate & role_grams:
            return 'v2 train/calibration'
        if any(counts[g] - (g in mine) > 0 for g in candidate):
            return 'other test work'
        return None
    return blocked


# ---------------------------------------------------------------- assembly
def build_groups(docs):
    groups = {}
    for d in docs:
        groups.setdefault(d['language'], {}).setdefault(d['group'], []).append(d)
    for language in groups:
        for g in groups[language].values():
            g.sort(key=lambda d: d['id'])
    return groups


def group_entry(gid, works, tier=None):
    reasons = {w.get('train_only') for w in works if w.get('train_only')}
    entry = dict(id=gid, clean=sum(w['clean'] for w in works), train_only=' / '.join(sorted(reasons)) or None)
    if tier is not None:
        entry['tier'] = tier
    return entry


def pick(works):
    """The TEST work of a group: most clean letters (ties: smaller id)."""
    return sorted(works, key=lambda w: (-w['clean'], w['id']))[0]


def legacy_groups(language):
    prior = read(lc.STATE / 'partitions.json')['priors'][LEGACY[language]]
    rows = prior['train'] + prior['calibration']
    if language == 'latin':
        split = [('legacy:ittb', [r for r in rows if r['group'] == 'latin']),
                 ('legacy:llct', [r for r in rows if r['group'] != 'latin'])]
    elif language == 'german':
        split = [('legacy:gsd', [r for r in rows if r['group'] == 'german']),
                 ('legacy:rem-prose', [r for r in rows if r['group'] != 'german'])]
    else:
        split = [('legacy:llibre-dels-fets', rows)]
    return split


def assemble(paths=None):
    works = catalogue()
    docs = [extract(w) for w in works if not w.get('exclude')]
    R, inputs, paths, extras = reference(paths)
    prior_check = check_prior_rows(R)
    for d in docs:
        rows = list(chunks(dict(id=d['id'], group=d['group'], text=d['text'])))
        kept, removed = kr.filter_rows(rows, lambda g: 'reference' if g & R else None)
        d.update(rows=rows, kept=kept, raw_letters=letters(rows), clean=letters(kept),
                 removed_by_reference=letters(rows) - letters(kept))
    groups = build_groups(docs)
    roles = {}
    for language in LANGUAGES:
        gs = groups[language]
        if language in NEW_PRIORS:
            entries = [group_entry(g, ws) for g, ws in gs.items()]
            roles[language] = assign_roles(entries, N_TEST, N_CAL, GERMAN_MIN if language == 'german' else 0)
        elif language == 'italian':
            assert len(gs) == 4, sorted(gs)
            roles[language] = assign_roles([group_entry(g, ws) for g, ws in gs.items()], 4)
        elif language == 'czech':
            entries = [group_entry(g, ws, tier=0 if all(w['period'] == 'pre-1500' for w in ws) else 1)
                       for g, ws in gs.items()]
            roles[language] = assign_roles(entries, N_TEST)
        else:
            roles[language] = assign_roles([group_entry(g, ws) for g, ws in gs.items()], N_TEST)
    by_role = lambda language, role: sorted((g for g, r in roles[language].items() if r['role'] == role),
                                            key=lambda g: roles[language][g]['order'])
    # ---- new priors
    priors, prior_letters = {}, {}
    role_works = []
    for language, name in NEW_PRIORS.items():
        pools = [(g, [r for w in groups[language][g] for r in w['kept']]) for g in by_role(language, 'train')]
        pools += legacy_groups(language)
        take, cap = water_fill({g: letters(rows) for g, rows in pools}, BUDGET)
        train = [row for g, rows in pools if take[g] for row in exact_take(rows, take[g])]
        calibration = [row for g in by_role(language, 'calibration')
                       for row in exact_take([r for w in groups[language][g] for r in w['kept']], CALIBRATION)]
        priors[name] = dict(train=train, calibration=calibration)
        prior_letters[name] = dict(
            cap=cap, train_letters=letters(train), calibration_letters=letters(calibration),
            train=[dict(group=g, source='legacy' if g.startswith('legacy:') else 'v2', available=letters(rows),
                        taken=take[g]) for g, rows in pools],
            calibration=[dict(group=g, available=sum(w['clean'] for w in groups[language][g]), taken=CALIBRATION)
                         for g in by_role(language, 'calibration')])
        for role in ('train', 'calibration'):
            role_works += [w for g in by_role(language, role) for w in groups[language][g]]
    role_grams = set()
    for w in role_works:
        role_grams |= grams(' '.join(r['text'] for r in w['rows']))
    # ---- TEST works
    tests = {language: [pick(groups[language][g]) for g in by_role(language, 'test')] for language in LANGUAGES}
    own = {w['id']: grams(' '.join(r['text'] for r in w['rows'])) for ws in tests.values() for w in ws}
    counts = Counter(g for s in own.values() for g in s)
    passages, records, firsts = {l: [] for l in LANGUAGES}, {l: [] for l in LANGUAGES}, {}
    latin_rate = kr.marker_rate(' '.join(d['text'] for d in lc.latin_documents()))
    for language in LANGUAGES:
        for w in tests[language]:
            kept, removed = kr.filter_rows(w['rows'], test_blocker(R, role_grams, own, counts, own[w['id']]))
            available = letters(kept)
            if available < PASSAGE:
                raise ValueError(f"{w['id']}: only {available} clean letters after the TEST filter; stop")
            passage = kr.cut(kept)
            firsts[w['id']] = (kept, passage)
            passages[language].append(passage)
            records[language].append(passage_record(w, 1, passage, kept, removed, roles[language][w['group']],
                                                    latin_rate))
    for ident in sorted(ITALIAN_SECOND, key=lambda i: group_hash(i)):
        w = next(x for x in tests['italian'] if x['id'] == ident)
        largest = sorted(tests['italian'], key=lambda x: -x['clean'])[:2]
        assert {x['id'] for x in largest} == set(ITALIAN_SECOND), [x['id'] for x in largest]
        kept, first = firsts[ident]
        later, skipped = kr.after_gap(kept, first['rows'][-1]['id'], GAP)
        first_grams = grams(' '.join(r['text'] for r in first['rows']))
        later, removed = kr.filter_rows(later, lambda c: 'first passage' if c & first_grams else None)
        if letters(later) < PASSAGE:
            raise ValueError(f'{ident}: second passage lacks letters')
        passage = kr.cut(later)
        passages['italian'].append(passage)
        base = next(r for r in records['italian'] if r['work'] == ident)
        records['italian'].append(dict(base, passage=2, first_row=passage['rows'][0]['id'],
                                       last_row=passage['rows'][-1]['id'], gap_clean_letters=skipped,
                                       letters_removed_against_first_passage=sum(r['letters'] for r in removed),
                                       plaintext_sha256=hashlib.sha256(passage['plaintext'].encode()).hexdigest()))
    audit(passages, priors, R, role_grams, own, counts, roles, groups)
    parts = dict(priors=priors, passages=passages)
    return dict(works=works, docs=docs, groups=groups, roles=roles, parts=parts, records=records,
                prior_letters=prior_letters, R=R, inputs=inputs, paths=paths, extras=extras, prior_check=prior_check,
                latin_rate=latin_rate)


def passage_record(w, number, passage, kept, removed, role, latin_rate):
    last = passage['rows'][-1]['id']
    position = [r['id'] for r in w['rows']].index(last)
    before = {r['id'] for r in w['rows'][:position + 1]}
    record = dict(language=w['language'], work=w['id'], group=w['group'], group_order=role['order'],
                  group_sha256=role['sha256'], passage=number, title=w.get('title'), author=w.get('author'),
                  date=w.get('date'), genre=w.get('genre'), raw_sha256=sorted({f['sha256'] for f in w['files']}),
                  raw_letters=w['raw_letters'], clean_letters_after_reference=w['clean'],
                  clean_letters_available=letters(kept),
                  letters_removed_by_reference=sum(r['letters'] for r in removed if r['reason'] == 'reference'),
                  letters_removed_by_train_calibration=sum(r['letters'] for r in removed
                                                           if r['reason'] == 'v2 train/calibration'),
                  letters_removed_by_other_tests=sum(r['letters'] for r in removed if r['reason'] == 'other test work'),
                  letters_removed_before_passage_end=sum(r['letters'] for r in removed if r['id'] in before),
                  chunks=len(w['rows']), first_row=passage['rows'][0]['id'], last_row=last,
                  text_sha256=hashlib.sha256(w['text'].encode()).hexdigest(),
                  plaintext_sha256=hashlib.sha256(passage['plaintext'].encode()).hexdigest())
    for key in ('period', 'flags'):
        if w.get(key):
            record[key] = w[key]
    if w['language'] in ('german', 'czech'):
        record['passage_latin_share'] = round(kr.latin_share(' '.join(r['text'] for r in passage['rows']), latin_rate), 4)
    return record


def audit(passages, priors, R, role_grams, own, counts, roles, groups):
    """Final checks on the sealed data."""
    for language, items in passages.items():
        assert len(items) == 6, (language, len(items))
        texts = [p['plaintext'] for p in items]
        assert len(set(texts)) == 6 and all(len(t) == PASSAGE and t.isalpha() for t in texts), language
        for p in items:
            g = grams(' '.join(r['text'] for r in p['rows']))
            mine = own[p['document']]
            if g & R or g & role_grams or any(counts[x] - (x in mine) > 0 for x in g):
                raise ValueError('Overlap survived filtering: ' + p['document'])
    firsts = {p['document']: p for p in passages['italian'][:4]}
    for p in passages['italian'][4:]:
        first = firsts[p['document']]
        assert not {r['id'] for r in p['rows']} & {r['id'] for r in first['rows']}
        assert not grams(' '.join(r['text'] for r in p['rows'])) & grams(' '.join(r['text'] for r in first['rows']))
    round_one = {w['id'] for w in kr.WORKS}
    test_docs = {p['document'] for items in passages.values() for p in items}
    for language, name in NEW_PRIORS.items():
        prior = priors[name]
        assert letters(prior['train']) == BUDGET and letters(prior['calibration']) == N_CAL * CALIBRATION, name
        train_groups = {r['group'] for r in prior['train'] if not r['id'].startswith(('latin:', 'german:'))}
        cal_groups = {r['group'] for r in prior['calibration']}
        assert not train_groups & cal_groups, (name, train_groups & cal_groups)
        docs = {r['document'] for r in prior['train'] + prior['calibration']}
        assert not docs & test_docs and not docs & round_one, name
        tests = {g for g, r in roles[language].items() if r['role'] == 'test'}
        assert not tests & (train_groups | cal_groups), name
    assert not test_docs & round_one


# ---------------------------------------------------------------- build / verify
def dumps(value):
    return kr.dumps(value)


def manifest(a):
    raw = {}
    for d in a['docs']:
        for f in d['files']:
            entry = raw.setdefault(f['path'], {k: v for k, v in f.items() if k != 'member'} | dict(works=[], licences=[]))
            entry['works'].append(d['id'])
            if d.get('license') and d['license'] not in entry['licences']:
                entry['licences'].append(d['license'])
    docs = {d['id']: d for d in a['docs']}
    works = []
    for w in a['works']:
        d = docs.get(w['id'])
        rec = {k: v for k, v in w.items() if k not in INTERNAL}
        rec['group'] = w.get('group') or w['id']
        rec['extraction_params'] = extraction_params(w)
        if d is None:
            rec['role'] = 'excluded'
        else:
            role = a['roles'][w['language']][d['group']]
            rec.update({k: v for k, v in d.items() if k in ('title', 'author', 'date', 'genre', 'period', 'url',
                                                             'member_sha256')})
            rec.update(role=role['role'], group_order=role['order'], group_sha256=role['sha256'],
                       files=[f['path'] for f in d['files']], member=w.get('member'), extraction=d['extraction'],
                       raw_letters=d['raw_letters'], clean_letters_after_reference=d['clean'],
                       text_sha256=hashlib.sha256(d['text'].encode()).hexdigest())
        works.append(rec)
    group_table = {}
    for language, rs in a['roles'].items():
        group_table[language] = [dict(group=g, **r, works=[w['id'] for w in a['groups'][language][g]],
                                      clean_letters_after_reference=sum(w['clean'] for w in a['groups'][language][g]))
                                 for g, r in sorted(rs.items(), key=lambda x: x[1]['order'])]
    return dict(
        purpose='Second known-answer key-recovery confirmation: three broadened character priors (latin_broad2, '
                'german_broad2, catalan_broad2) and six fresh TEST passages per language. Blocks and keys are '
                'drawn later; blocks pair passages (1,2), (3,4), (5,6).',
        role_rules=ROLE_RULES, filter_rule=FILTER_RULE, passage_letters=PASSAGE, italian_second_passage_gap=GAP,
        prior_train_letters=BUDGET, calibration_letters_per_group=CALIBRATION, german_min_clean_letters=GERMAN_MIN,
        group_id_rule='Singleton groups use the work id; merged groups use a named id (author:, cycle:, family:, '
                      'rem-family:); ReM groups rem:M###. Works inside a group are in id order.',
        german_families={k: dict(members=v[0], reason=v[1]) for k, v in REM_FAMILIES.items()},
        merges={'old_french': {'cycle:aymeri-de-narbonne': ['geste:ed_AimeriD', 'geste:ed_MortAymC']},
                'czech': {'author:petr-chelcicky': ['chelcicky-trojiem-lidu', 'chelcicky-cierkvi']},
                'latin': {'author:dante-alighieri': ['udante-monarchia', 'udante-dve', 'udante-epistole',
                                                     'udante-questio']},
                'catalan': {'author:ramon-llull': ['llull-gentil', 'llull-mil-proverbis',
                                                   'llull-primera-segona-intencio'],
                            'family:crown-of-aragon-chancery': ['cartes-cancelleria', 'documents-dret',
                                                                'ordinacions-cort']}},
        groups=group_table, works=works, raw_files=sorted(raw.values(), key=lambda f: f['path']),
        priors=a['prior_letters'], passages=a['records'], latin_marker_rate=round(a['latin_rate'], 4),
        reference=dict(shingles=len(a['R']), inputs=a['inputs'], evaluator_only_paths=a['paths'], **a['extras'],
                       prior_rows_checked=a['prior_check'],
                       rule='key_recovery_confirmation_sources.reference() (exclusions(); released answers; '
                            'language-coverage and language-expansion partitions; all LLCT and HisCat documents; ReM '
                            'prose outside the ungraded challenge pool; every language-expansion document; historical '
                            'prose and verse; every >=8-word string of the evaluator-only files) plus the whole text '
                            'of every round-one confirmation work, every COMETA file on disk and the exclusion-only '
                            'Valter e Griselda'),
        partitions_path=rel(STATE / 'partitions.json'))


def build():
    if (STATE / 'partitions.json').exists() or (OUT / 'sources.json').exists():
        raise FileExistsError('Confirmation v2 sources already sealed')
    a = assemble()
    m = manifest(a)
    m['partitions_sha256'] = hashlib.sha256(dumps(a['parts']).encode()).hexdigest()
    dumps(m)                                    # serializable before anything is written
    seal(STATE / 'partitions.json', a['parts'])
    if hashlib.sha256((STATE / 'partitions.json').read_bytes()).hexdigest() != m['partitions_sha256']:
        raise ValueError('Sealed partitions differ from their in-memory bytes')
    seal(OUT / 'sources.json', m)
    summary()


def verify():
    m = read(OUT / 'sources.json')
    for f in m['raw_files']:
        if sha(ROOT / f['path']) != f['sha256']:
            raise ValueError('Raw source drift: ' + f['path'])
    for path, digest in m['reference']['inputs'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('Reference input drift: ' + path)
    if sha(STATE / 'partitions.json') != m['partitions_sha256']:
        raise ValueError('Pinned partitions.json changed')
    a = assemble(m['reference']['evaluator_only_paths'])
    if a['inputs'] != m['reference']['inputs'] or len(a['R']) != m['reference']['shingles']:
        raise ValueError('Reference set differs')
    rebuilt = manifest(a)
    rebuilt['partitions_sha256'] = m['partitions_sha256']
    if json.loads(dumps(rebuilt)) != m:
        raise ValueError('Manifest differs')
    digest = hashlib.sha256(dumps(a['parts']).encode()).hexdigest()
    if digest != m['partitions_sha256'] or dumps(a['parts']).encode() != (STATE / 'partitions.json').read_bytes():
        raise ValueError(f'Rebuilt partitions differ: {digest}')
    print('verified', digest, len(m['raw_files']), 'raw files')


def summary():
    m = read(OUT / 'sources.json')
    for language, items in m['passages'].items():
        for r in items:
            print(f"{language:10} {r['work']:34} p{r['passage']} group #{r['group_order']:<3} "
                  f"clean {r['clean_letters_available']:7}")
    for name, p in m['priors'].items():
        print(name, 'cap', p['cap'], 'train', p['train_letters'], 'calibration', p['calibration_letters'])
        for g in p['calibration']:
            print('   calibration', g['group'], g['available'])
    print('raw files', len(m['raw_files']), 'sources.json', sha(OUT / 'sources.json'),
          'partitions.json', m['partitions_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['build', 'verify', 'summary'])
    globals()[parser.parse_args().command]()
