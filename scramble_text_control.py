"""Build a deterministic within-line control without loading a model."""

from voynich.data import build


if __name__ == "__main__":
    import json
    print(json.dumps(build("voynich_transliterations/GC2a-n.txt", "artifacts/data/gc-shuffle",
                          "experiments/splits/folio-42.json", control="shuffle"), indent=2))
