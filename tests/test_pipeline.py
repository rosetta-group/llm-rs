import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from voynich.baselines import Ngram, copy_distribution, layout_keys
from voynich.data import build, documents, make_manifest, normalise, read_ivtff, shuffle_line
from voynich.evaluate import paired_interval, summarize, target_signature
from voynich.windows import CausalCollator, encode_document, token_windows


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "sample.txt"
        rows = ["#=IVTFF v101 2.0 M 6"]
        for i in range(1, 13):
            for side in "rv":
                rows += [f"<f{i}{side}> <! $Q={chr(65+(i-1)//2)} $I=H $L=A $H=1>",
                         f"<f{i}{side}.1,@P0> <%>ab.cd,ef.@155;.?<$>",
                         f"<f{i}{side}.2,@L0> label"]
        self.source.write_text("\n".join(rows))

    def tearDown(self):
        self.temp.cleanup()

    def test_single_line_paragraph_and_metadata(self):
        corpus = read_ivtff(self.source)
        line = corpus["lines"][0]
        self.assertTrue(line["paragraph_start"] and line["paragraph_end"])
        self.assertEqual(line["quire"], "A")
        self.assertEqual(line["raw"], "<%>ab.cd,ef.@155;.?<$>")
        docs = documents(corpus, make_manifest(corpus))
        self.assertEqual(len(docs), 24)
        self.assertEqual(docs[0]["text"], "ab.cd,ef.\x9b.?")

    def test_paragraph_end_alone_preserves_boundary(self):
        with self.source.open("a") as handle:
            handle.write("\n<f12v.3,+P0> next.paragraph<$>\n")
        corpus = read_ivtff(self.source)
        docs = documents(corpus, make_manifest(corpus))
        self.assertTrue(docs[-1]["text"].endswith("?\n\nnext.paragraph"))

    def test_grouped_splits_are_reproducible(self):
        corpus = read_ivtff(self.source)
        for group in ("folio", "quire"):
            manifest = make_manifest(corpus, group=group)
            self.assertEqual(manifest, make_manifest(corpus, group=group))
            docs = documents(corpus, manifest)
            for value in {d[group] for d in docs}:
                self.assertEqual(len({d["split"] for d in docs if d[group] == value}), 1)

    def test_rosettes_foldout_cannot_cross_splits(self):
        from voynich.data import validate_manifest
        corpus = read_ivtff(self.source)
        corpus["lines"] += [{**corpus["lines"][0], "folio": name} for name in ("85", "86", "Ros")]
        for seed in range(10):
            manifest = make_manifest(corpus, seed=seed)
            self.assertEqual(len({manifest["assignments"][g] for g in ("85", "86", "Ros")}), 1)
        manifest["assignments"].update({"85": "train", "86": "test"})
        with self.assertRaisesRegex(ValueError, "Rosettes"):
            validate_manifest(corpus, manifest)

    def test_source_drift_is_rejected(self):
        build(self.source, self.root/"out", self.root/"manifest.json")
        self.source.write_text(self.source.read_text()+"\n# changed\n")
        with self.assertRaisesRegex(ValueError, "Source changed"):
            build(self.source, self.root/"out", self.root/"manifest.json")

    def test_unknown_pages_cannot_leak_into_reused_split(self):
        corpus = read_ivtff(self.source)
        manifest = make_manifest(corpus)
        del manifest["assignments"]["1"]
        with self.assertRaisesRegex(ValueError, "missing"):
            documents(corpus, manifest)

    def test_boundary_variants_preserve_other_symbols(self):
        raw = "<%>a,b.c!%[d:e]{fg}@155;?<->h<$>"
        self.assertEqual(normalise(raw), "a,b.c!%?fg\x9b?\x1ch")
        self.assertEqual(normalise(raw, "merged", "first"), "ab.c!%dfg\x9b?\x1ch")
        self.assertEqual(normalise(raw, "separate", "last"), "a.b.c!%efg\x9b?\x1ch")

    def test_shuffle_changes_training_text_and_keeps_forms(self):
        corpus = read_ivtff(self.source)
        manifest = make_manifest(corpus)
        before = documents(corpus, manifest)
        after = documents(corpus, manifest, control="shuffle")
        self.assertNotEqual([d["text"] for d in before], [d["text"] for d in after])
        self.assertEqual([d["split"] for d in before], [d["split"] for d in after])
        import re
        for a, b in zip(before, after):
            self.assertEqual(Counter(re.split("[.,]", a["text"])), Counter(re.split("[.,]", b["text"])))
        self.assertEqual(shuffle_line("one.two,three.four", 42, "f1r.1"),
                         shuffle_line("one.two,three.four", 42, "f1r.1"))

    def test_wrapped_locus(self):
        self.source.write_text("#=IVTFF Eva- 2.0 M 6\n<f1r> <! $Q=A>\n<f1r.1,@P0> <%>ab./\n/cd<$>\n")
        line = read_ivtff(self.source)["lines"][0]
        self.assertEqual(line["raw"], "<%>ab.cd<$>")

    def test_tokenizer_never_learns_heldout_merges(self):
        from voynich.tokenization import train_tokenizer
        from voynich.data import write_json
        from transformers import AutoTokenizer
        dataset = self.root/"data"
        write_json(dataset/"documents.json", [dict(page="train", text="abab."*20, split="train"),
                                               dict(page="heldout", text="zzzz."*20, split="test")])
        audit = train_tokenizer(dataset, self.root/"tokenizer", merges=20)
        tokenizer = AutoTokenizer.from_pretrained(self.root/"tokenizer", local_files_only=True)
        self.assertEqual(audit["training_pages"], ["train"])
        self.assertNotIn("zz", tokenizer.get_vocab())
        self.assertEqual(tokenizer.decode(tokenizer.encode("zzzz.", add_special_tokens=False)), "zzzz.")

    def test_published_control_has_chronological_gaps(self):
        from voynich.sources import import_control
        from voynich.data import load_documents
        source = self.root/"control.txt"
        source.write_text("# generator settings\n" + "aa bb\n"*100)
        import_control(source, self.root/"control", lines_per_page=2)
        docs = load_documents(self.root/"control")
        indices = {s: [int(d["folio"]) for d in docs if d["split"] == s]
                   for s in ("train", "validation", "test")}
        self.assertEqual(min(indices["validation"])-max(indices["train"]), 2)
        self.assertEqual(min(indices["test"])-max(indices["validation"]), 2)
        self.assertTrue(all(d["text"] == "aa.bb\naa.bb" for d in docs))


class WindowTests(unittest.TestCase):
    def test_targets_scored_once_at_every_stride(self):
        for length in (2, 17, 33, 100):
            ids, valid, units = list(range(length)), [False]+[True]*(length-1), [0]+[1]*(length-1)
            for context, stride in ((1, 1), (16, 1), (16, 8), (16, 16)):
                seen = []
                windows = list(token_windows(ids, valid, units, context, stride))
                for w in windows:
                    self.assertLessEqual(len(w["input_ids"]), context+1)
                    self.assertEqual(w["labels"][0], -100)
                    seen += [w["start"]+i for i, x in enumerate(w["labels"]) if x != -100]
                self.assertEqual(seen, list(range(1, length)))
                self.assertEqual(sum(w["units"] for w in windows), length-1)

    def test_padding_does_not_relabel_context_or_real_eos(self):
        batch = CausalCollator(0)([
            dict(input_ids=[1, 0, 3], attention_mask=[1, 1, 1], labels=[-100, 0, 3]),
            dict(input_ids=[1, 4], attention_mask=[1, 1], labels=[-100, 4])])
        self.assertEqual(batch["labels"].tolist(), [[-100, 0, 3], [-100, 4, -100]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1], [1, 1, 0]])

    def test_invalid_stride(self):
        with self.assertRaises(ValueError):
            list(token_windows([0, 1], [False, True], [0, 1], 2, 3))

    def test_offsets_and_uncertainty(self):
        from fine_tuning import tiny_model
        _, tokenizer = tiny_model()
        ids, valid, units = encode_document(tokenizer, "ab?\x9b")
        self.assertEqual(tokenizer.decode(ids[1:]), "ab?\x9b")
        self.assertEqual(sum(valid), 3)
        self.assertEqual(sum(units), 3)


class StatisticsTests(unittest.TestCase):
    def test_ngram_beats_frequency_on_alternation(self):
        docs = [dict(text="ab"*100)]
        unigram, bigram = Ngram(0).fit(docs), Ngram(1).fit(docs)
        self.assertGreater(bigram.distribution("a")[ord("b")], unigram.distribution("")[ord("b")])
        self.assertAlmostEqual(float(bigram.distribution("a").sum()), 1)
        self.assertGreater(bigram.distribution("a")[ord("z")], 0)

    def test_layout_uses_no_future_characters(self):
        self.assertEqual(list(layout_keys("ab\nc")), list(layout_keys("ab\ncXYZ"))[:4])

    def test_copy_model_uses_only_history(self):
        p = copy_distribution("abcd.abcd.abc")
        self.assertIsNotNone(p)
        self.assertEqual(int(np.argmax(p)), ord("d"))

    def test_metrics_weight_tokens_and_bootstrap_groups(self):
        ref = [dict(page=f"f{i}r", folio=str(i), quire="A", currier="A", section="herbal", hand="1",
                    nll=20, correct=5, tokens=10, units=10,
                    target_sha256=target_signature("a"*10)) for i in range(3)]
        candidate = [{**r, "nll": 10} for r in ref]
        interval = paired_interval(ref, candidate, draws=100)
        self.assertGreater(interval["interval_95"][0], 0)
        self.assertEqual(interval["groups"], 3)
        self.assertEqual(summarize(ref)["overall"]["loss_nats"], 2)
        candidate[0]["units"] = 9
        with self.assertRaises(ValueError):
            paired_interval(ref, candidate)

    def test_comparison_rejects_different_equal_length_targets(self):
        ref = [dict(page=str(i), folio=str(i), nll=20, units=2,
                    target_sha256=target_signature("ab")) for i in range(3)]
        other = [{**r, "target_sha256": target_signature("ba")} for r in ref]
        with self.assertRaisesRegex(ValueError, "identical targets"):
            paired_interval(ref, other)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            paired_interval(ref+[ref[0]], ref)

    def test_rosettes_are_one_bootstrap_cluster(self):
        ref = [dict(page=str(i), folio=folio, nll=20, units=2, target_sha256=target_signature("ab"))
               for i, folio in enumerate(("85", "86", "Ros", "1"))]
        result = paired_interval(ref, [{**r, "nll": 10} for r in ref], draws=100)
        self.assertEqual(result["groups"], 2)

    def test_test_partition_requires_explicit_release(self):
        from argparse import Namespace
        from voynich.__main__ import require_test_release
        with self.assertRaisesRegex(ValueError, "sealed"):
            require_test_release(Namespace(split="test", release_test=False))
        require_test_release(Namespace(split="validation", release_test=False))


class NeuralTests(unittest.TestCase):
    def test_replication_requires_a_gain_and_keeps_the_chosen_layers(self):
        import math
        from unittest.mock import patch
        from configs.config_model import FineTuningConfig
        from experiments.replications import prepare

        def score(bpc):
            return dict(summary={"overall": {"bits_per_character": bpc}}, pages=[
                dict(page=str(i), folio=str(i), nll=bpc*10*math.log(2), units=10,
                     target_sha256=target_signature("a"*10)) for i in range(3)])

        def source(path):
            path = str(path)
            if path.endswith("gc.json"):
                return {"models": {"copy": score(2.7)}}
            if path.endswith("run.json"):
                return dict(status="complete", config=FineTuningConfig().model_dump(),
                            layers=[0, 3, 20, 23] if "random" in path else [0, 1, 26, 27])
            return score(2.5 if "random" in path else 2.6)

        with patch("experiments.replications.read", side_effect=source), patch("experiments.replications.write_json") as write:
            plan = prepare()
            self.assertEqual(plan["seeds"], [42, 43, 44, 45, 46])
            self.assertEqual(len(plan["jobs"]), 7)
            self.assertTrue(all(call.args[1]["trainable_layers"] == [0, 3, 20, 23] for call in write.call_args_list))
        with patch("experiments.replications.read", side_effect=source), patch("experiments.replications.write_json") as write:
            with patch("experiments.replications.paired_interval", return_value={"interval_95": [-.1, .1]}):
                self.assertEqual(prepare()["jobs"], [])
                write.assert_not_called()

    def test_earlier_best_checkpoint_is_exported_with_matching_scores(self):
        import contextlib
        import io
        import math
        from unittest.mock import patch
        from configs.config_model import FineTuningConfig
        from fine_tuning import WeightedTrainer, evaluation_rows, load_model, run_fine_tuning
        from peft import PeftModel
        from voynich.data import write_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [dict(page=split, folio=split, quire=split, currier="A", section="test",
                         hand="1", split=split, text="ab.cd."*8) for split in ("train", "validation")]
            write_json(root/"data/documents.json", docs)
            write_json(root/"data/manifest.json", {"fixture": True})
            config = FineTuningConfig(tiny=True, device="cpu", dataset=str(root/"data"),
                output_dir=str(root/"run"), layer_count=2, context=16, stride=8, max_steps=3,
                gradient_accumulation_steps=1, eval_steps=1, save_steps=1, load_best_model_at_end=True)
            evaluate = WeightedTrainer.evaluate

            def prefer_first(trainer, *args, **kwargs):
                metrics = evaluate(trainer, *args, **kwargs)
                # Force selection of step 1 while keeping the genuine page scores.
                metrics["eval_bits_per_character"] += 10*trainer.state.global_step
                return metrics

            with patch.object(WeightedTrainer, "evaluate", prefer_first), contextlib.redirect_stdout(io.StringIO()):
                run_fine_tuning(config)
            run = json.loads((root/"run/run.json").read_text())
            saved = json.loads((root/"run/adapted.json").read_text())
            first = json.loads((root/"run/validation/step-000001.json").read_text())
            self.assertEqual(run["optimizer_steps"], 3)
            self.assertEqual(run["selected_step"], 1)
            self.assertEqual(saved["pages"], first["pages"])
            model, tokenizer, _ = load_model(config)
            model = PeftModel.from_pretrained(model, root/"run", is_trainable=False)
            actual = summarize(evaluation_rows(model, tokenizer, docs[1:], 16, 8))["overall"]
            self.assertTrue(math.isclose(actual["bits_per_character"],
                                        saved["summary"]["overall"]["bits_per_character"], rel_tol=1e-6))

    def test_only_requested_lora_layers_are_trainable(self):
        from fine_tuning import get_lora_target_modules, select_layers, tiny_model
        from peft import get_peft_model, LoraConfig, TaskType
        model, _ = tiny_model()
        layers = select_layers(4, 2)
        self.assertEqual(layers, [0, 3])
        model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=2,
                              target_modules=get_lora_target_modules(layers)))
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all("lora_" in name and ("layers.0." in name or "layers.3." in name) for name in trainable))
        self.assertEqual(len(select_layers(28, 4, "random", 42)), 4)


if __name__ == "__main__":
    unittest.main()
