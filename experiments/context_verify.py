"""Cross-check the context scorer with the original stride-one evaluator."""

import fcntl
import os

import torch

from experiments.cloud import write
from experiments.context import OUT, ROOT, PLAN, check_targets, prepare, read
from voynich.character import CharacterModel, score
from voynich.evaluate import summarize


def main():
    os.chdir(ROOT)
    torch.set_num_threads(2)
    checks = []
    with open("artifacts/replication.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for dataset, contexts in [("gc", [32]), ("naibbe", [8, 16, 32, 64, 128])]:
            source, config, docs, _ = prepare(dict(dataset=dataset, seed=42), read(PLAN))
            model = CharacterModel("gru", config["width"], config["layers"], config["context"])
            model.load_state_dict(torch.load(source["path"] + "/best.pt", map_location="cpu", weights_only=True))
            model.to("mps")
            for context in contexts:
                reference = score(model, docs, context=context, stride=1, batch_size=16)
                saved = read(OUT / f"{dataset}-s42-c{context}.json")
                check_targets(reference, saved["pages"])
                bpc = summarize(reference)["overall"]["bits_per_character"]
                error = abs(bpc - saved["summary"]["overall"]["bits_per_character"])
                if error >= 1e-5:
                    raise ValueError(f"Evaluator mismatch: {dataset}, context {context}, error {error}")
                checks.append(dict(dataset=dataset, context=context, reference_bpc=bpc,
                                   absolute_bpc_error=error, targets_match=True))
                print(checks[-1], flush=True)
            del model
            torch.mps.empty_cache()
    write(OUT / "reference-verification.json", dict(evaluator="voynich.character.score",
          stride=1, device="mps", checks=checks, test_scored=False))


if __name__ == "__main__":
    main()
