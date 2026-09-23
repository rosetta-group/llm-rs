import json, time, urllib.error, urllib.request, pathlib
# Run from the repository root: python experiments/linear_a_damos_crawl.py
root = pathlib.Path("artifacts/linear-a-sources/damos")
tables = json.load(open(root / "filter.json"))["tables"]
# Knossos (csort 10) last: mainland documents are the development split.
ids = sorted({x["id"] for x in tables}, key=lambda i: (min(t["csort"] for t in tables if t["id"] == i) == 10, i))
out = root / "items"; out.mkdir(exist_ok=True)
failed = []
for n, i in enumerate(ids):
    path = out / f"{i}.json"
    if path.exists():
        continue
    started = time.time()
    ok = False
    for attempt in range(3):
        try:
            req = urllib.request.Request(f"https://damos.hf.uio.no/ajaxitem/{i}/",
                                         headers={"User-Agent": "llm-rs research crawl (1 req/s)"})
            data = urllib.request.urlopen(req, timeout=30).read()
            json.loads(data)
            path.write_bytes(data)
            ok = True
            break
        except urllib.error.HTTPError as e:
            if e.code == 500:
                break  # server error for this document; recorded, not retried
            time.sleep(5)
        except Exception:
            time.sleep(5)
    if not ok:
        failed.append(i)
    time.sleep(max(0.0, 1.0 - (time.time() - started)))
    if n % 250 == 0:
        print(f"{n}/{len(ids)} failed={len(failed)}", flush=True)
(root / "failed.json").write_text(json.dumps(failed))
print("done", len(ids), "failed", len(failed), flush=True)
