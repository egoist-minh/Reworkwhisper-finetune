"""What does the model write instead of a loanword it drops, and does the train
label say the same thing? Reads the cross-domain audit CSV for segments whose
reference carries one of the dropped loanwords, aligns ref/hyp word by word, and
reports the substitution the model made. Then counts how often that substitution
appears in the v6-corpus TRAIN labels."""
import csv, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

AUDIT = Path(sys.argv[1])
CORPUS = Path(sys.argv[2])
WORDS = sys.argv[3].split(",")

def toks(s):
    return re.findall(r"[0-9a-zA-Zàáâãèéêìíòóôõùúăđĩũơưăạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹý]+", s.lower())

subs = defaultdict(Counter)
with AUDIT.open(encoding="utf-8") as f:
    for row in csv.DictReader(f):
        r, h = toks(row["ref"]), toks(row["hyp"])
        for w in WORDS:
            if w not in r:
                continue
            for i, t in enumerate(r):
                if t != w:
                    continue
                lo, hi = max(0, i - 3), min(len(h), i + 4)
                window = h[lo:hi]
                if w in window:
                    subs[w]["<KEPT>"] += 1
                else:
                    subs[w][" ".join(window[max(0, i - lo - 1):i - lo + 2]) or "<EMPTY>"] += 1

print("=== what the model writes where it drops the loanword ===")
for w in WORDS:
    if subs[w]:
        print(f"{w:12s} {subs[w].most_common(6)}")

train_text = []
for m in sorted(CORPUS.glob("manifest.*.jsonl")):
    for line in m.open(encoding="utf-8"):
        rec = json.loads(line)
        if rec.get("split") != "test":
            train_text.append(rec["text"].lower())
blob = "\n".join(train_text)
print(f"\n=== train labels: {len(train_text)} records ===")
print("word          foreign-form hits")
for w in WORDS:
    n = len(re.findall(rf"\b{re.escape(w)}\b", blob))
    print(f"{w:12s}  {n}")
