"""Corrected secondary metric for the Linear B control (not frozen; does not affect the gate).

The frozen ``control`` stage compared Greek-script headwords with the Latin-script cognate cited
by Wiktionary, so ``best_form_is_cited_cognate`` was always 0. Here the cited cognate is spelled
with the same Linear B rules and compared with the matched Greek form.

    python -m experiments.linear_a_round_one_report
"""

import json
from pathlib import Path

from linear_a import lexicons
from linear_a.matching import Matcher
from linear_a.spelling import spell

OUT = Path("experiments/linear-a")


def main():
    theta = json.loads((OUT / "freeze.json").read_text())["theta"]
    greek = Matcher({"Greek": lexicons.build_lexicon("AncientGreek", min_syllables=2)})
    myc = lexicons.mycenaean_words(min_syllables=2)
    cited = {w: spell(r["cognate"].split(",")[0]) for w, r in myc.items() if r["cognate"]}
    matched = [w for w in cited if greek.best_form(w, "Greek")[0] <= theta]
    spelled_equal = [w for w in cited if cited[w] == w]
    result = {
        "with_cited_cognate": len(cited),
        "cited_cognate_spells_identically": len(spelled_equal),
        "exact_greek_match": len(matched),
        "exact_match_and_cognate_spells_identically": len(set(matched) & set(spelled_equal)),
    }
    (OUT / "control-cognates.json").write_text(json.dumps(result, indent=1) + "\n")
    print(result)


if __name__ == "__main__":
    main()
