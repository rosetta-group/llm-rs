"""Reproduce the validation baseline suite: python -m experiments.baselines."""

from pathlib import Path

from voynich.baselines import benchmark
from voynich.data import build, digest, load_documents, write_json
from voynich.report import baseline_markdown
from voynich.sources import fetch_sources, import_control


def main():
    fetch_sources()
    source = "voynich_transliterations/GC2a-n.txt"
    manifest = "experiments/splits/folio-42.json"
    for name, settings in [("gc", {}), ("gc-shuffle", dict(control="shuffle")),
                           ("gc-merged", dict(boundary="merged")),
                           ("gc-separate", dict(boundary="separate"))]:
        build(source, f"artifacts/data/{name}", manifest, **settings)
    build(source, "artifacts/data/gc-quire", "experiments/splits/quire-42.json", group="quire")
    build("artifacts/sources/ZL3b-n.txt", "artifacts/data/zl", manifest, allow_other_source=True,
          pages=sorted(d["page"] for d in load_documents("artifacts/data/gc")))
    for name, filename in [("timm", "timm-generated.txt"), ("naibbe", "naibbe-reference.txt")]:
        import_control(f"artifacts/sources/{filename}", f"artifacts/data/{name}")
    for name in ("gc", "gc-shuffle", "gc-merged", "gc-separate", "gc-quire", "zl", "timm", "naibbe"):
        dataset = Path("artifacts/data")/name
        result = dict(dataset=str(dataset), dataset_sha256=digest(dataset/"documents.json"),
                      split="validation", models=benchmark(load_documents(dataset)))
        output = Path("artifacts/results")/f"{name}.json"
        write_json(output, result)
        output.with_suffix(".md").write_text(baseline_markdown(result))
        print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
