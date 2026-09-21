"""Dataset, baseline, and experiment commands: python -m voynich --help."""

import argparse
import json
from pathlib import Path

from .data import build, digest, load_documents, write_json


def require_test_release(args):
    if args.split == "test" and not args.release_test:
        raise ValueError("Final test is sealed. Freeze choices first, then pass --release-test.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("build", help="Parse IVTFF and freeze grouped splits")
    data.add_argument("source")
    data.add_argument("--output", default="artifacts/data/gc")
    data.add_argument("--manifest", default="experiments/splits/folio-42.json")
    data.add_argument("--seed", type=int, default=42)
    data.add_argument("--group", choices=["folio", "quire"], default="folio")
    data.add_argument("--boundary", choices=["marked", "merged", "separate"], default="marked")
    data.add_argument("--reading", choices=["mask", "first", "last"], default="mask")
    data.add_argument("--control", choices=["intact", "shuffle"], default="intact")
    data.add_argument("--match-source", action="store_true", help="Reuse split groups for a different transcription")
    data.add_argument("--match-pages", help="Another dataset; restrict to its paragraph pages")

    base = sub.add_parser("baseline", help="Fit on train; score validation by default")
    base.add_argument("dataset")
    base.add_argument("--output", required=True)
    base.add_argument("--models", nargs="+", choices=["frequency", "ngram3", "ngram5", "layout", "copy"])
    base.add_argument("--split", choices=["validation", "test"], default="validation")
    base.add_argument("--release-test", action="store_true")

    compare = sub.add_parser("compare", help="Paired cluster bootstrap for two score files")
    compare.add_argument("reference")
    compare.add_argument("candidate")
    compare.add_argument("--reference-model", help="Baseline name, if reference contains several models")
    compare.add_argument("--candidate-model", help="Baseline name, if candidate contains several models")
    compare.add_argument("--group", choices=["folio", "quire"], default="folio")
    compare.add_argument("--output", required=True)

    matrix = sub.add_parser("matrix", help="Write configs without starting training")
    matrix.add_argument("model")
    matrix.add_argument("--dataset", default="artifacts/data/gc")
    matrix.add_argument("--output", default="experiments/generated")
    matrix.add_argument("--steps", type=int, default=100)
    matrix.add_argument("--layers", type=int, default=4)
    matrix.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    matrix.add_argument("--contexts", nargs="+", type=int, default=[16, 64, 256])

    evaluate = sub.add_parser("evaluate", help="Evaluate a frozen model or saved adapter")
    evaluate.add_argument("config")
    evaluate.add_argument("--adapter")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--split", choices=["validation", "test"], default="validation")
    evaluate.add_argument("--release-test", action="store_true")
    evaluate.add_argument("--context", type=int, help="Change evaluation context for the same checkpoint")
    evaluate.add_argument("--stride", type=int)

    tokenizer = sub.add_parser("tokenizer", help="Learn symbol groups on training pages only")
    tokenizer.add_argument("dataset")
    tokenizer.add_argument("--output", required=True)
    tokenizer.add_argument("--merges", type=int, default=64)

    fetch = sub.add_parser("fetch", help="Download pinned sources and verify checksums")
    fetch.add_argument("--manifest", default="experiments/sources.json")
    fetch.add_argument("--output", default="artifacts/sources")
    control = sub.add_parser("control", help="Import published synthetic text with chronological splits")
    control.add_argument("source")
    control.add_argument("--output", required=True)
    control.add_argument("--lines-per-page", type=int, default=29)

    args = parser.parse_args()
    if args.command == "fetch":
        from .sources import fetch_sources
        result = fetch_sources(args.manifest, args.output)
    elif args.command == "control":
        from .sources import import_control
        result = import_control(args.source, args.output, args.lines_per_page)
    elif args.command == "build":
        pages = sorted(d["page"] for d in load_documents(args.match_pages)) if args.match_pages else None
        result = build(args.source, args.output, args.manifest, args.seed, args.group,
                       boundary=args.boundary, reading=args.reading, control=args.control,
                       allow_other_source=args.match_source, pages=pages)
    elif args.command == "baseline":
        from .baselines import benchmark
        from .report import baseline_markdown
        require_test_release(args)
        result = dict(dataset=args.dataset, dataset_sha256=digest(Path(args.dataset)/"documents.json"),
                      split=args.split, models=benchmark(load_documents(args.dataset), args.split, args.models))
        write_json(args.output, result)
        Path(args.output).with_suffix(".md").write_text(baseline_markdown(result))
        result = {name: row["summary"]["overall"] for name, row in result["models"].items()}
    elif args.command == "compare":
        from .evaluate import paired_interval
        ref, cand = [json.loads(Path(p).read_text()) for p in (args.reference, args.candidate)]
        if args.reference_model:
            ref = ref["models"][args.reference_model]
        if args.candidate_model:
            cand = cand["models"][args.candidate_model]
        result = paired_interval(ref["pages"], cand["pages"], args.group)
        write_json(args.output, result)
    elif args.command == "matrix":
        from configs.config_model import FineTuningConfig
        written = []
        for seed in args.seeds:
            for context in args.contexts:
                for selection in ("outer", "middle", "random"):
                    name = f"{Path(args.dataset).name}-{selection}-c{context}-s{seed}"
                    config = FineTuningConfig(model_path=args.model, dataset=args.dataset,
                        output_dir=f"training_run_outputs/{name}", run_name=name, seed=seed,
                        layer_selection=selection, layer_count=args.layers, context=context,
                        stride=max(1, context//2), max_steps=args.steps,
                        eval_steps=args.steps, save_steps=args.steps, dtype="bfloat16")
                    path = Path(args.output)/f"{name}.json"
                    write_json(path, config.model_dump())
                    written.append(str(path))
        result = dict(configs=written, note="Equal steps and adapter sizes within each context; no runs started.")
    elif args.command == "tokenizer":
        from .tokenization import train_tokenizer
        result = train_tokenizer(args.dataset, args.output, args.merges)
    else:
        from configs.config_model import FineTuningConfig
        from fine_tuning import evaluation_rows, load_model
        from .evaluate import summarize
        require_test_release(args)
        config = FineTuningConfig.model_validate_json(Path(args.config).read_text())
        if args.context is not None or args.stride is not None:
            values = config.model_dump()
            values["context"] = args.context or config.context
            values["stride"] = args.stride or max(1, values["context"]//2)
            config = FineTuningConfig.model_validate(values)
        docs = [d for d in load_documents(config.dataset) if d["split"] == args.split]
        model, tokenizer, _ = load_model(config)
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
        rows = evaluation_rows(model, tokenizer, docs, config.context, config.stride, config.eval_batch_size)
        result = dict(config=config.model_dump(), adapter=args.adapter, split=args.split,
                      dataset_sha256=digest(Path(config.dataset)/"documents.json"), summary=summarize(rows), pages=rows)
        write_json(args.output, result)
        result = result["summary"]["overall"]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
