"""Fetch the pinned lexicon and annotation inputs; verify existing copies."""
import json
from experiments.discovery_sources import ROOT, fetch


def main():
    lexicon=json.loads((ROOT/'experiments/segmentation-sources.json').read_text())['lexicon']
    annotation=json.loads((ROOT/'experiments/association/freeze.json').read_text())['source']
    for source in (lexicon,annotation):
        fetch(source)
    print('Verified Morph-it! lexicon and Grove/Stolfi annotations')


if __name__=='__main__': main()
