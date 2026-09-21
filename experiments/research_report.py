"""Build the interim PDF, Markdown, and figures from a frozen result snapshot.

Requires matplotlib and reportlab in a separate reporting environment.
Run with --capture after refreshing experiments.learning_curves to take a new snapshot.
Run without arguments to reproduce the saved report without reading live runs.
"""

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle, Preformatted,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/report"
FIG = OUT / "figures"
PDF = ROOT / "output/pdf/voynich-research-report.pdf"
BLUE, TEAL, ORANGE, GRAY = "#315EA8", "#087E83", "#C76B29", "#728096"


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def capture():
    paths = ["experiments/learning-curve-results.json", "experiments/results.json",
             "experiments/sources.json", "experiments/splits/folio-42.json"]
    curves = read(paths[0])
    assert curves["test_scored"] is False
    baselines = {}
    for name in ("gc", "gc-shuffle", "gc-merged", "gc-separate", "gc-quire", "zl", "timm", "naibbe"):
        path = f"artifacts/results/{name}.json"
        baselines[name] = read(path)
        assert baselines[name]["split"] == "validation"
        paths.append(path)
    runs = {}
    for name, summary in curves["runs"].items():
        run_path = f"training_run_outputs/{name}/run.json"
        run = read(run_path)
        paths.append(run_path)
        verified = None
        if run["status"] == "complete":
            base = f"training_run_outputs/{name}"
            adapter = f"{base}/adapter_model.safetensors"
            selected = f"{base}/checkpoint-{run['selected_step']}/adapter_model.safetensors"
            verified = sha(adapter) == sha(selected)
            assert verified
        for row in summary["checkpoints"]:
            path = f"training_run_outputs/{name}/validation/step-{row['step']:06d}.json"
            score = read(path)
            expected = {p["page"]: (p["units"], p["target_sha256"]) for p in baselines["gc"]["models"]["copy"]["pages"]}
            assert expected == {p["page"]: (p["units"], p["target_sha256"]) for p in score["pages"]}
            paths.append(path)
        # The portable config supplies the pinned model ID, without a machine-specific cache path.
        run["config"]["model_path"] = "mlx-community/Qwen3-1.7B-bf16"
        runs[name] = dict(metadata=run, adapter_matches_selected=verified)
    return dict(captured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                curves=curves, baselines=baselines, pilots=read("experiments/results.json")["qwen"],
                runs=runs, input_sha256={p: sha(p) for p in paths})


def plots(data):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#273548", "text.color": "#273548",
                         "axes.edgecolor": "#B9C3CF", "savefig.facecolor": "white"})
    base = data["baselines"]["gc"]["models"]
    runs = data["curves"]["runs"]
    random = runs["qwen-random-c64-s42-n3000"]
    copy = base["copy"]["summary"]["overall"]["bits_per_character"]
    best = random["best_so_far"]["bits_per_character"]

    def save(fig, name):
        fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
        fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.8), layout="constrained")
    labels = ["Frozen Qwen", "Frequency", "5-character context", "Layout + spelling",
              "3-character context", "Spelling + copy", "Qwen random: 3,000 updates"]
    values = [data["pilots"]["random"]["frozen"]["overall"]["bits_per_character"]]
    values += [base[n]["summary"]["overall"]["bits_per_character"] for n in ("frequency", "ngram5", "layout", "ngram3", "copy")]
    values.append(best)
    ax.barh(labels, values, color=[GRAY]*5+[ORANGE, TEAL], height=.62)
    ax.invert_yaxis()
    ax.set(xlabel="Validation bits per character (lower is better)", xlim=(0, 4.9))
    for i, value in enumerate(values):
        ax.text(value+.06, i, f"{value:.3f}", va="center", fontsize=10)
    save(fig, "comparison")

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.1), layout="constrained", gridspec_kw={"height_ratios": [2, 1]})
    for name, color, marker in (("random", TEAL, "o"), ("outer", BLUE, "s")):
        run = runs[f"qwen-{name}-c64-s42-n3000"]
        rows = run["checkpoints"]
        axes[0].plot([r["step"] for r in rows], [r["bits_per_character"] for r in rows],
                     color=color, marker=marker, label=f"{name.capitalize()} layers ({run['status']})", linewidth=2)
    axes[0].axhline(copy, color=ORANGE, linestyle="--", label=f"Spelling + copy: {copy:.3f}")
    axes[0].set(ylabel="Validation bits/character", xlim=(400, 3100), ylim=(2.46, 2.76))
    axes[0].legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=3)
    rows = random["checkpoints"]
    gains = [a["bits_per_character"]-b["bits_per_character"] for a, b in zip(rows, rows[1:])]
    axes[1].bar([r["step"] for r in rows[1:]], gains, width=280, color=TEAL)
    axes[1].set(xlabel="Optimizer updates", ylabel="Gain since previous\n500-update checkpoint", xlim=(400,3100), ylim=(0,.14))
    for row, gain in zip(rows[1:], gains):
        axes[1].text(row["step"], gain+.005, f"{gain:.3f}", ha="center", fontsize=9)
    save(fig, "learning-curves")

    selected = random["selected_scores"]
    grouped = defaultdict(lambda: [0.,0.,0])
    ref = {p["page"]: p for p in base["copy"]["pages"]}
    for page in selected["pages"]:
        row = grouped[page["folio"]]
        row[0] += ref[page["page"]]["nll"]
        row[1] += page["nll"]
        row[2] += page["units"]
    folios = sorted(grouped, key=lambda k: int(k))
    values = [(grouped[k][0]-grouped[k][1])/math.log(2)/grouped[k][2] for k in folios]
    fig, axes = plt.subplots(1,2,figsize=(9,4.5),layout="constrained",gridspec_kw={"width_ratios":[1.5,1]})
    axes[0].barh([f"f{k}" for k in folios], values, color=[TEAL if x>0 else ORANGE for x in values])
    axes[0].invert_yaxis()
    axes[0].axvline(0,color=GRAY,linewidth=1)
    axes[0].axvline(copy-best,color=BLUE,linestyle="--",label="Pooled gain")
    axes[0].set(xlabel="Copy loss - Qwen loss (bits/character)",title="Gain by held-out folio")
    axes[0].legend(frameon=False,fontsize=9)
    xpos=np.arange(2)
    for shift, name, summary, color in ((-.18,"Copy",base["copy"]["summary"],ORANGE),(.18,"Qwen",selected["summary"],TEAL)):
        ys=[summary["currier"][g]["bits_per_character"] for g in ("A","B")]
        axes[1].bar(xpos+shift,ys,width=.35,label=name,color=color)
        for x,y in zip(xpos+shift,ys): axes[1].text(x,y+.04,f"{y:.3f}",ha="center",fontsize=9)
    axes[1].set(xticks=xpos,xticklabels=["Currier A\n9,821 characters","Currier B\n9,931 characters"],ylabel="Validation bits/character",ylim=(0,3.5),title="Gain in both text varieties")
    axes[1].legend(frameon=False,fontsize=9)
    save(fig,"folio-gains")

    fig,axes=plt.subplots(1,3,figsize=(9,3.1),layout="constrained",sharey=True)
    for ax,key,title in zip(axes,("gc-shuffle","timm","naibbe"),("Shuffled Voynich","Timm-Schinner sample","Naibbe ciphertext")):
        models=data["baselines"][key]["models"]
        strongest=min(models,key=lambda n:models[n]["summary"]["overall"]["bits_per_character"])
        vals=[models[n]["summary"]["overall"]["bits_per_character"] for n in ("frequency",strongest)]
        ax.bar(["Frequency",strongest],vals,color=[GRAY,ORANGE],width=.6)
        for x,y in enumerate(vals): ax.text(x,y+.08,f"{y:.3f}",ha="center")
        ax.set(title=title,ylim=(0,4.8))
    axes[0].set_ylabel("Validation bits/character")
    save(fig,"controls")
    return sum(v>0 for v in values), len(values)


class Report:
    def __init__(self, stamp):
        self.story=[]
        self.md=[]
        self.stamp=stamp
        self.styles=getSampleStyleSheet()
        for name,size,leading,color in (("BodyText",10.3,14.6,"#273548"),("Heading1",25,29,"#163153"),
                                       ("Heading2",13,17,TEAL)):
            self.styles[name].fontSize=size
            self.styles[name].leading=leading
            self.styles[name].textColor=colors.HexColor(color)
            self.styles[name].spaceAfter=9
        self.styles.add(ParagraphStyle("Caption",fontSize=8.4,leading=11,textColor=colors.HexColor(GRAY),spaceAfter=12))
        self.styles.add(ParagraphStyle("CodeBlock",fontName="Courier",fontSize=8.7,leading=12,
                                       backColor=colors.HexColor("#F1F5F8"),borderPadding=10,spaceAfter=14))

    @staticmethod
    def html(text):
        text=escape(text)
        text=re.sub(r"\*\*(.*?)\*\*",r"<b>\1</b>",text)
        text=re.sub(r"`(.*?)`",r'<font name="Courier" size="8.5">\1</font>',text)
        return re.sub(r"\[(.*?)\]\((.*?)\)",r'<a href="\2" color="#315EA8">\1</a>',text)

    def p(self,text,style="BodyText"):
        self.story.append(Paragraph(self.html(text),self.styles[style]))
        self.md.extend([text,""])

    def heading(self,title,first=False):
        if not first: self.story.append(PageBreak())
        self.p(title,"Heading1")
        self.md[-2] = ("# " if first else "## ")+title

    def sub(self,title):
        self.p(title,"Heading2")
        self.md[-2]="### "+title

    def figure(self,name,caption,width=490):
        from PIL import Image as PILImage
        with PILImage.open(FIG/f"{name}.png") as im: w,h=im.size
        self.story.append(Image(str(FIG/f"{name}.png"),width=width,height=width*h/w))
        self.md.extend([f"![{caption}](figures/{name}.png)",""])
        self.p(caption,"Caption")

    def table(self,rows,widths):
        cells=[[Paragraph(self.html(str(v)),self.styles["Caption"] if i else self.styles["BodyText"]) for v in row] for i,row in enumerate(rows)]
        table=Table(cells,colWidths=widths,hAlign="LEFT",repeatRows=1)
        table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#E9F0F5")),
                                   ("VALIGN",(0,0),(-1,-1),"TOP"),("BOTTOMPADDING",(0,0),(-1,-1),6),
                                   ("TOPPADDING",(0,0),(-1,-1),6),
                                   ("LINEBELOW",(0,0),(-1,-1),.3,colors.HexColor("#DDE4EA"))]))
        self.story.extend([table,Spacer(1,10)])
        self.md.extend(["| "+" | ".join(map(str,rows[0]))+" |","| "+" | ".join(["---"]*len(rows[0]))+" |"])
        self.md.extend("| "+" | ".join(map(str,row))+" |" for row in rows[1:])
        self.md.append("")

    def code(self,text):
        self.story.append(Preformatted(text,self.styles["CodeBlock"]))
        self.md.extend(["```text",text,"```",""])

    def finish(self):
        def footer(canvas,doc):
            canvas.setStrokeColor(colors.HexColor("#DDE4EA"))
            canvas.line(48,42,547,42)
            canvas.setFont("Helvetica",8)
            canvas.setFillColor(colors.HexColor(GRAY))
            canvas.drawString(48,29,f"VOYNICH / INTERIM RESEARCH REPORT / {self.stamp[:10]}")
            canvas.drawRightString(547,29,str(doc.page))
        SimpleDocTemplate(str(PDF),pagesize=(595,842),leftMargin=48,rightMargin=48,topMargin=44,bottomMargin=58,
                          title="Voynich: prediction before interpretation",author="llm-rs research project").build(self.story,onFirstPage=footer,onLaterPages=footer)
        (OUT/"REPORT.md").write_text("\n".join(self.md))


def build(data):
    npositive,nfolios=plots(data)
    r=Report(data["captured_at"])
    base=data["baselines"]["gc"]["models"]
    runs=data["curves"]["runs"]
    random=runs["qwen-random-c64-s42-n3000"]
    outer=runs["qwen-outer-c64-s42-n3000"]
    selected=random["selected_scores"]
    copy=base["copy"]["summary"]["overall"]["bits_per_character"]
    best=random["best_so_far"]
    gain=copy-best["bits_per_character"]
    lo,hi=best["vs_copy"]["interval_95"]
    stamp=data["captured_at"].replace("T"," ").replace("+00:00"," UTC")

    r.heading("Voynich: prediction before interpretation",True)
    r.p(f"Interim experiment report | Snapshot: {stamp}","Caption")
    r.p("This project tests whether adapting a small part of Qwen improves prediction of Voynich transcription text.")
    r.p(f"**Main finding:** the completed 3,000-update run reaches **{best['bits_per_character']:.3f} bits per character**, versus **{copy:.3f}** for the strongest implemented baseline: **{gain/copy:.1%} lower loss**.")
    r.p("**Bits per character (BPC):** the model's prediction loss divided by the number of scored transcription characters; lower is better.")
    r.p("**Validation:** pages excluded from parameter fitting but used to choose settings and checkpoints.")
    r.figure("comparison","Figure 1. Same GC validation target text: 29 pages, 19,752 scored characters. Frozen Qwen is the saved pre-adaptation reference. These bars compare prediction loss, not translation accuracy.")
    r.p("1. **What was done.** Repaired data preparation and scoring; measured five simple predictors; ran three 400-update pilots and a completed 3,000-update random-layer experiment.")
    r.p(f"2. **What is still running.** Outer-layer training is {outer['status']}; this snapshot includes checkpoints through {outer['checkpoints'][-1]['step']:,} updates. The five-seed study and neural text controls are not yet results.")
    r.p("3. **Why this matters.** Qwen now captures predictive regularities missed by our current baseline. The result does not identify what any Voynich word means.")

    r.heading("01 / What the experiment measures")
    r.p("The experiment predicts the next piece of a transcription from preceding text.")
    r.p("**Thesis:** compare models on identical held-out characters before making claims about language or meaning.")
    r.p("**Token:** one input unit used by a model; Qwen subwords and baseline characters are different units.")
    r.p("**LoRA adapter:** a small set of trainable weight updates; here, 1,245,184 parameters across four of Qwen's 28 layers.")
    r.p("**Folio:** a manuscript leaf; its front and back stay in the same split to reduce leakage.")
    r.code("Parse GC2a-n.txt and preserve page metadata\nGroup related pages into fixed splits\nFit predictors on training pages\nScore each validation target once\nSelect checkpoints using validation loss\nKeep final-test scores sealed")
    r.table([["Split","Pages","Folio groups","Characters used"],["Training",148,69,"125,614"],
             ["Validation",29,15,"19,752"],["Final test",30,14,"Not scored"]],[130,80,120,169])
    r.p("1. **Representation.** The main source, `GC2a-n.txt`, uses v101 transcription, not EVA. Paragraph text is modeled. Line breaks, paragraph breaks and uncertain boundaries are preserved; unreadable `?` targets are excluded. A normalized character is not necessarily one original manuscript glyph.")
    r.p("2. **Loss.** BPC = summed negative log probability / (scored characters x ln 2). Assigning probability 1/2 to a character costs 1 bit; assigning 1/4 costs 2. An 8% BPC reduction does not mean 8 percentage points more accuracy.")
    r.p("3. **The copying baseline.** It mixes 80% smoothed prediction from the previous three characters with 20% copy continuation. It searches the previous 256 characters for strings of length 2, 3 or 4, allowing one mismatch. Its 2.699 BPC score is not attributable to copying alone.")
    r.p("4. **Fair comparisons.** Hash checks confirm all primary checkpoints and the copy baseline score the same targets. Qwen's subword accuracy is not directly comparable with baseline character accuracy. The frequency predictor is a learned non-contextual reference, not uniform random guessing.")

    r.heading("02 / Longer training changed the result")
    r.p("The learning curve records held-out loss after each 500 optimizer updates.")
    r.p("**Thesis:** 3,000 updates established a useful gain, but the last 500 added little compared with the first extensions.")
    r.p("**Optimizer update:** one weight adjustment; here it accumulates gradients from two one-window batches.")
    r.figure("learning-curves","Figure 2. Fresh 3,000-update schedules, seed 42, context 64, stride 32. The lower panel shows the random-layer run's incremental gain. Missing outer checkpoints are unfinished work, not extrapolated values.")
    r.p("1. **The earlier result was budget-specific.** At 400 updates, random layers scored 2.878, outer layers 2.937 and middle layers 3.155; all lost to the 2.699 baseline. These pilots used a different cosine learning-rate schedule, so their endpoints are not points on the curves above.")
    r.p("2. **Returns are diminishing.** Random layers improved by 0.111 BPC from 500 to 1,000 updates, but only 0.0056 from 2,500 to 3,000. This schedule lowers the learning rate toward zero; the curve does not prove that all further training would fail.")
    r.p("3. **Layer location remains unresolved.** At 2,000 updates, random layers scored 2.5040 and outer layers 2.5052. This one-seed comparison gives no persuasive reason to call the outer layers uniquely language-dependent. We have not tested preservation of English or Italian abilities.")

    r.heading("03 / How convincing is the gain?")
    r.p("The completed random-layer model improves prediction across held-out manuscript material.")
    r.p(f"**Thesis:** the gain is consistent across these folios, but validation reuse and a single training seed limit the claim.")
    r.p("**Paired bootstrap:** repeatedly sample the same folio groups for both models and recalculate their loss difference.")
    r.p("**Training seed:** controls random initialization and training order; variation across seeds is a separate uncertainty.")
    r.figure("folio-gains",f"Figure 3. Random layers, selected step 3,000. Positive values favor Qwen; {npositive}/{nfolios} folios improve. The dashed line is the character-weighted pooled gain, not the unweighted average of folio bars.")
    r.p(f"1. **Measured effect.** Copy loss minus Qwen loss is **{gain:.4f} BPC**. A paired bootstrap over 15 folio groups gives a 95% interval of **[{lo:.4f}, {hi:.4f}]**, using 2,000 draws. All values favor Qwen in this comparison.")
    r.p("2. **Selection limits certainty.** The same 29 pages chose the checkpoint and support the interval. The interval is exploratory; it does not account for model selection, training-seed variability or pretraining exposure to public Voynich text. The final test has not been scored.")
    r.p("3. **Two varieties improve.** Currier A falls from 2.905 to 2.631 BPC; B falls from 2.495 to 2.336. A/B are statistical text varieties, not established source languages. The 22 A pages and 7 B pages have similar character totals; their page counts alone would misstate their weights.")

    r.heading("04 / Prediction is not decipherment")
    r.p("Controls test whether a result could arise without recovering meaning.")
    r.p("**Thesis:** lower loss on Voynich is only informative about meaning when competing explanations are tested.")
    r.p("**Control text:** altered or generated text designed to preserve some properties while changing others.")
    r.figure("controls","Figure 4. Existing simple-model results on three control datasets. Compare bars within each panel; different target strings and alphabets prevent interpreting absolute BPC differences between panels as a language ranking. Neural control runs are pending.")
    r.p("1. **Local structure survives shuffling.** Shuffling words within each line leaves the copy baseline at 2.864 BPC, still well below its 4.245 frequency reference. This destroys the original order while retaining words and line membership. Neural gains must be measured against each control's own baseline.")
    r.p("2. **Generators provide an alternative.** The [Timm-Schinner self-citation materials](https://github.com/TorstenTimm/SelfCitationTextgenerator) provide algorithmically generated text. Our sampled control's copy score is 2.091 BPC. Predictable generated text shows why predictability alone is not a meaning test.")
    r.p("3. **Known plaintext provides a method check.** [Greshko's Naibbe cipher](https://github.com/greshko/naibbe-cipher) encrypts Latin and Italian reversibly. Its strongest current baseline here is the layout model at 1.868 BPC. Predicting this ciphertext and recovering its held-out plaintext are distinct tasks; neither has established a Voynich translation.")
    r.p("4. **Current gaps.** Each synthetic control uses one published sample, split into chronological blocks. These are not independent generator realizations. ZL/EVA, boundary variants and held-out quires have baseline results only; the main neural gain has not yet passed those robustness checks.")

    r.heading("05 / What to do next")
    r.p("Use a **replicate-then-explain** research sequence.")
    r.p("**Thesis:** spend the next local compute budget on explaining the 0.216 BPC gain before extending the same run again.")
    r.p("**Context ablation:** shorten the preceding text available to the same trained checkpoint, while keeping evaluation targets fixed.")
    r.code("Finish the two 3,000-update primary runs\nSelect the promising configuration\nRepeat with seeds 43, 44, 45 and 46\nRun shuffled, Timm-Schinner and Naibbe controls\nIf gains survive replication:\n    Compare 16-token and 64-token context on the same targets\n    Test stronger baselines and alternative transcriptions\nIf independent evidence supports interpretation:\n    Validate known-plaintext recovery\n    Test constrained Voynich mappings on unseen material")
    r.p("1. **Finish the queued evidence.** The existing local monitor is set to complete the primary comparison, then run four extra seeds and three neural controls. Keep the winning four layer positions fixed across seeds. This measures training stability for one subset, not robustness across random layer subsets.")
    r.p("2. **Find the source of the gain.** Evaluate the selected checkpoint at contexts 16 and 64 with identical target masks; repeat on controls. If the gain remains with short context, local patterns are sufficient for that gain. If longer context helps Voynich more than controls, test what distant information matters. A 256-token extension should be separate: the present adapter trained at context 64.")
    r.p("3. **Challenge the baseline and representation.** Add a stronger variable-length character predictor or small character model, tune on validation, and repeat matched comparisons on ZL/EVA and boundary variants. Record character exposure and local runtime; equal update counts alone do not equalize data exposure across tokenizations.")
    r.p("4. **Set a translation gate.** First test recovery of held-out known plaintext from newly generated Naibbe examples with held-out keys and texts. For Voynich, require independently annotated text-image associations within section and hand, then consistent predictions on unseen pages. Fluent English or Italian is not a correctness test. Defer SAEs and broad layer-localization claims until a specific causal question exists.")
    r.p("**Longer runs become worthwhile** if repeat seeds show a stable advantage and a revised schedule or context budget tests a stated hypothesis. Do not open final-test scores until the comparison and selection rule are fixed.")

    r.heading("06 / Provenance and reproducibility")
    r.p("This is a frozen snapshot of local experiments, not a completed decipherment study.")
    r.p(f"**Snapshot:** {stamp}. **Scope:** validation only; no final-test scoring. Training continued independently while this report was written.")
    r.table([["Setting","Value"],
             ["Model","mlx-community/Qwen3-1.7B-bf16; 28 layers"],
             ["Pinned model revision","9cd6692855d3e06772228e9a962b2606359b2d24"],
             ["Layer positions (zero-based)","Random: 0, 3, 20, 23. Outer: 0, 1, 26, 27."],
             ["Adapter","Rank 8; alpha 16; dropout 0; seven attention/MLP projections per layer"],
             ["Training","3,000 optimizer updates; batch 1; accumulation 2; seed 42"],
             ["Context / schedule","64 tokens; stride 32; learning rate 0.0001; cosine decay"],
             ["Checkpoint selection","Validation every 500 updates; lowest validation BPC"],
             ["Hardware","Local Apple Silicon MPS; BF16; 36 GiB unified memory"],
             ["Training data","2,935 overlapping windows; targets scored once per evaluation"]],[135,364])
    r.sub("Recorded evidence")
    r.p("`experiments/report/snapshot.json` contains the numerical snapshot and SHA-256 hashes of its inputs. The completed random run's exported adapter was checked against its selected checkpoint. All primary checkpoint target hashes match the copy baseline.")
    r.p("The report builder reads saved scores only. `python -m experiments.research_report` reproduces the saved report using matplotlib and reportlab. Use `--capture` only after refreshing `experiments.learning_curves` to create a newer snapshot; review the narrative when experiment status changes.")
    r.p("The frozen Qwen reference used evaluation batch size 2; the long adapted runs use 1. BF16 batch-shape rounding can cause small differences. The comparison with the copy baseline uses the same scored target characters.")
    r.sub("Primary sources and research boundary")
    r.p("[Yale's manuscript description](https://beinecke.library.yale.edu/beinecke/collections/beinecke-cipher-voynich-manuscript) describes the text as undeciphered. This study has no accepted Voynich plaintext labels and makes no translation claim.")
    r.p("[Timm & Schinner (2019), A possible generating algorithm of the Voynich manuscript](https://doi.org/10.1080/01611194.2019.1596999), with the authors' code and sample materials linked on page 5.")
    r.p("[Greshko (2025), The Naibbe cipher](https://doi.org/10.1080/01611194.2025.2566408), with the author's reversible cipher implementation and datasets linked on page 5.")
    r.p("Sources checked on 2026-09-20. Their broader claims have not been independently reproduced here. See `RESEARCH_PLAN.md` for the wider reading list.","Caption")
    r.finish()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture",action="store_true")
    args=parser.parse_args()
    FIG.mkdir(parents=True,exist_ok=True)
    PDF.parent.mkdir(parents=True,exist_ok=True)
    snapshot=OUT/"snapshot.json"
    if args.capture:
        snapshot.write_text(json.dumps(capture(),indent=2)+"\n")
    data=json.loads(snapshot.read_text())
    build(data)
    print(PDF)
    print(OUT/"REPORT.md")


if __name__=="__main__":
    main()
