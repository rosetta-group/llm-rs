"""Etruscan phase 0: pin sources and count what the corpora hold.

    python -m experiments.etruscan_phase0

See experiments/etruscan/SCOPE.md. Writes sources.json and phase0.json; description only, no test.
"""

import collections
import hashlib
import json
import statistics
from pathlib import Path

import pandas as pd

from etruscan import corpus

OUT = Path("experiments/etruscan")
LARTH_REV = "daf4972175f45b48188fe36671db3a0e081e5130"
LARTH_URL = f"https://raw.githubusercontent.com/GianlucaVico/Larth-Etruscan-NLP/{LARTH_REV}/"
LARTH_FILES = {"LICENSE": "LICENSE", "README.md": "README.md", "DATA_README.md": "Data/README.md"}
LIRE = Path("artifacts/etruscan-sources/lire")
LIRE_URL = "https://zenodo.org/api/records/8431452/files/"
# Letters Etruscan does not write; a text using them is probably Latin, Umbrian or OCR noise.
NON_ETRUSCAN = set("obdg")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    out = []
    for path in sorted(corpus.LARTH.iterdir()):
        remote = LARTH_FILES.get(path.name, ("Data/" + path.name))
        out.append({"name": "GianlucaVico/Larth-Etruscan-NLP", "revision": LARTH_REV,
                    "path": str(path), "url": LARTH_URL + remote, "sha256": sha256(path),
                    "bytes": path.stat().st_size, "license": "CC BY 4.0"})
    for path in sorted(LIRE.iterdir()):
        out.append({"name": "LIRE (Latin Inscriptions of the Roman Empire)", "revision": "v3.0, Zenodo record 8431452",
                    "path": str(path), "url": LIRE_URL + path.name + "/content", "sha256": sha256(path),
                    "bytes": path.stat().st_size, "license": "CC BY 4.0"})
    return {"retrieved": "2026-09-23", "sources": out}


def describe(texts, glossed):
    toks = [w for _, _, ws in texts for w in ws]
    clean = [w for w in toks if not corpus.damaged(w)]
    lengths = [len(w) for w in clean]
    counts = collections.Counter(clean)
    covered = sum(n for w, n in counts.items() if w in glossed)
    return {
        "texts": len(texts),
        "texts_with_2plus_tokens": sum(len(ws) >= 2 for _, _, ws in texts),
        "tokens": len(toks),
        "damaged_tokens": len(toks) - len(clean),
        "types": len(counts),
        "hapax_types": sum(n == 1 for n in counts.values()),
        "tokens_per_text_mean": round(len(toks) / len(texts), 2),
        "token_length_median": statistics.median(lengths),
        "share_tokens_12plus_letters": round(sum(n >= 12 for n in lengths) / len(lengths), 3),
        "share_texts_with_o_b_d_g": round(sum(any(set(w) & NON_ETRUSCAN for w in ws) for _, _, ws in texts) / len(texts), 3),
        "share_tokens_glossed": round(covered / len(clean), 3),
        "glossed_types_attested": sum(w in glossed for w in counts),
        "top_20": counts.most_common(20),
    }


def latin_pool():
    table = pd.read_parquet(LIRE / "LIRE_v3-0.parquet",
                            columns=["type_of_inscription_clean", "inscr_type", "clean_text_interpretive_word"])
    epitaph = table.type_of_inscription_clean.eq("epitaph") | table.inscr_type.fillna("").eq("tituli sepulcrales")
    words = table.loc[epitaph, "clean_text_interpretive_word"].fillna("").str.split()
    words = words[words.str.len() > 0]
    return {"records": len(table), "epitaphs_nonempty": int(len(words)),
            "tokens": int(words.str.len().sum()), "tokens_per_text_median": float(words.str.len().median()),
            "selection": "type_of_inscription_clean == 'epitaph' or inscr_type == 'tituli sepulcrales' (Christian epitaphs excluded)"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sources.json").write_text(json.dumps(sources(), indent=1) + "\n")
    rows = corpus.load_rows()
    duplicates = int(rows.duplicated(subset=["ID", "Etruscan", "key"]).sum())
    rows = rows.drop_duplicates(subset=["ID", "Etruscan", "key"])
    texts = corpus.texts(rows)
    glossed = corpus.glossed_words()
    result = {
        "larth_rows": int(len(rows) + duplicates),
        "exact_duplicate_rows_dropped": duplicates,
        "glossed_word_list_types": len(glossed),
        "ETP": describe([t for t in texts if t[0] == "ETP"], glossed),
        "CIEP": describe([t for t in texts if t[0] == "CIEP"], glossed),
        "all": describe(texts, glossed),
        "latin_epitaph_pool": latin_pool(),
    }
    (OUT / "phase0.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items()}, indent=1, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
