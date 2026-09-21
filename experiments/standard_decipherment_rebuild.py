"""Rebuild ignored fitted models from the committed freeze, without retuning."""
import numpy as np

from experiments.historical_sources import restore
from experiments.segmentation import corpus
from experiments.standard_decipherment import ROOT, OUT, STATE, read, write, hashes, verify
from voynich.decipher import normalize, language_model
from voynich.description_length import CharacterPrior
from voynich.segmentation import fit
from voynich.unknown_cipher import chunk_counts


def rebuild():
    frozen = read(OUT/'freeze.json')
    if hashes() != frozen['code_sha256']:
        raise ValueError('Frozen source changed')
    stories = restore()
    modern, _ = corpus('UD_Italian-ISDT', 'train')
    modern = [normalize(' '.join(r['words'])) for r in modern]
    prose = [normalize(' '.join(r['paragraphs'])) for r in stories if r['split']=='train']
    STATE.mkdir(parents=True, exist_ok=True)
    if not (STATE/'prior.npz').exists():
        CharacterPrior.fit(modern+prose).save(STATE/'prior.npz')
    if not (STATE/'legacy-prior.npz').exists():
        probabilities, counts = language_model(modern+prose, spaces=False)
        np.savez(STATE/'legacy-prior.npz', log_probabilities=probabilities, letter_counts=counts)
    if not (STATE/'chunks.json').exists():
        write(STATE/'chunks.json',chunk_counts(modern+prose))
    if not (STATE/'segmenter.json').exists():
        lexicon = read(ROOT/'artifacts/segmentation/model.json')['lexicon']
        weight = frozen['development']['selected_segmenter']['weight']
        write(STATE/'segmenter.json',fit(modern+prose*weight,lexicon))
    verify()
    print('Rebuilt models match every committed input hash; no evaluation references read.')


if __name__ == '__main__':
    rebuild()
