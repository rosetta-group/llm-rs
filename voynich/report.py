def baseline_markdown(result):
    lines = [f"# Baselines: {result['split']}", "", f"Dataset: `{result['dataset']}`.", "",
             "Lower bits per character is better. Accuracy predicts the next normalized transcription character.", "",
             "| Model | Bits/character | Accuracy | Scored characters |", "|---|---:|---:|---:|"]
    for name, model in result["models"].items():
        row = model["summary"]["overall"]
        lines.append(f"| {name} | {row['bits_per_character']:.4f} | {row['accuracy']:.2%} | {row['units']:,} |")
    for field in ("currier", "section"):
        lines += ["", f"## By {field}", "", "| Group | Model | Bits/character | Characters |", "|---|---|---:|---:|"]
        for name, model in result["models"].items():
            for group, row in model["summary"][field].items():
                lines.append(f"| {group} | {name} | {row['bits_per_character']:.4f} | {row['units']:,} |")
    lines += ["", "These are prediction measurements, not evidence of a translation.", ""]
    return "\n".join(lines)
