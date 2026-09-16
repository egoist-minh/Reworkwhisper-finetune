# v6 pha 2 — tái phơi nhiễm trên corpus đậm (`mixed-noisy-v1` + hai lô TTS)

Chạy trên máy GPU thuê theo `docs/server-finetune.md`. File này chỉ ghi phần khác recipe
chung. Các quyết định dưới đây được chốt qua một lượt hỏi đáp ngày 16/09/2026; mỗi mục ghi
lại cả lựa chọn đã chốt lẫn lựa chọn bị loại, để lượt sau không phải bàn lại.

Kế hoạch này **thay thế** `docs/v6-density-ladder-plan.md` cho hướng đi tiếp. Thang mật độ
đã chạy hết và trần của nó là ~62% retention (`docs/v6-ondomain-plan.md` §8).

---

## 0. Tiền đề — đọc trước, đừng bỏ qua

### 0.1 Pha 2 không mang thông tin mới. Nó đổi thứ tự tiếp xúc.

Hai phép đo, cả hai đều chống lại cách mô tả "pha 2 = thêm dữ liệu sạch":

- `mixed-noisy-v1` train có 5.082 segment; **4.846 (95,4%) đã nằm trong `v6-corpus` train**,
  trong đó 4.727 trùng nhãn từng ký tự. Phần chỉ có ở `mixed-noisy-v1` là 236 segment /
  0,31 h, sáu cuộc `paid_meeting_legacy_*`, toàn synthetic.
- Hai lô TTS `paid-meeting-vi-0247-0296` và `-0297-0339` (9,06 h) **cũng đã nằm trong pha
  1**: run `v6-100h-steps` train trên `v6-corpus` cộng add-on, tổng 42.794 segment /
  106,69 h / mật độ 3,55% (`docs/v6-100h-steps-chay-tay.md:10-14`).

Nên cơ chế duy nhất kế hoạch này tác động là **mật độ ở lượt tiếp xúc cuối**: pha 1 thấy
vật liệu này hoà trong 3,55%, pha 2 cho thấy lại ở mật độ cao hơn khoảng ba lần và không
kèm gì khác. Mọi câu trong báo cáo dạng "bổ sung dữ liệu chất lượng cao" là sai sự thật.

Điểm làm nhẹ bớt: `step-800` × batch 16 = 12.800 mẫu trên 42.794, tức khi dừng nó mới đi
được ~30% một epoch. Phần lớn vật liệu nó gặp đúng một lần hoặc chưa lần nào.

### 0.2 Phép so với v5 không còn sạch

Đã cân nhắc và **cố tình bỏ**: chạy pha 2 trên đúng `mixed-noisy-v1` nguyên bản sẽ cho một
phép thử một biến so với v5 (v5 = LoRA tươi trên chính corpus đó). Trộn thêm hai lô TTS làm
mất tính đối chứng — thua v5 thì không tách được "100 giờ pha 1 có hại" khỏi "hai lô TTS có
hại". Đổi lại được cơ hội thắng cao hơn trong một lượt duy nhất. Ghi rõ giới hạn này khi
báo cáo.

### 0.3 ValCER không đọc được kết quả này

`v6-ondomain-15h` ValCER 3,54% / `v6-ondomain-29h` 3,52% — chênh 0,02 điểm — trong khi
cross-domain chênh 2,26 điểm (8,75% so với 11,01%). Val hiện tại là 365 segment cắt từ
corpus v6, mật độ từ ngoại lai ~2,5%, nên nó chấm phần tiếng Việt chứ không chấm hành vi
giữ từ ngoại lai. Hệ quả cụ thể: `checkpoints/best/` chọn bằng val CER **trỏ sai** — ở
`v6-100h-steps`, retention đỉnh ở step 800 còn val CER giảm tiếp tới step 5350.

Do đó kế hoạch này lưu checkpoint theo lịch step cố định rồi chọn tay (§5). Val chỉ tồn tại
để `src.train.train()` không raise `RuntimeError`, không tham gia quyết định nào.

---

## 1. Điểm xuất phát

| | |
|---|---|
| adapter pha 1 | `rework-whisper-v6-org/v6-100h-steps-checkpoints`, thư mục `step-800/` |
| rank / alpha | **16 / 32** — cùng hình dạng v5 |
| retention cross-domain của nó | **62,82% (thô), 61,18% (λ=0,75)** — `Outputs/v6-100h-steps-metrics/cross_domain.step-800*.json` |
| base model | `vinai/PhoWhisper-large` |

Đã cân nhắc và bỏ: `v6-corpus-r32/checkpoints/best` (rank 32) — ba nấc
`Outputs/ladder-phaseA/` đã khởi từ đó và đều dưới 56% retention. Cũng bỏ phương án train
lại pha 1 ở rank khác: cả v5 (thắng) lẫn `v6-ondomain-15h` (trần 61,9%) đều rank 16, nên
rank không phân biệt được thắng/thua trong bằng chứng hiện có và không đáng tiêu giờ GPU ở
lượt này.

⚠ **Điểm xuất phát đã trượt sẵn ràng buộc thứ ba của §6.** Đọc thẳng từ
`Outputs/v6-100h-steps-metrics/`, `step-800` cho no-loanword CER **7,87% thô** và
**6,40% ở λ=0,75**, trong khi ngưỡng là ≤ 6,13%. Chỉ λ=0,5 mới qua (5,70%), nhưng ở
đó retention rơi còn 57,29%. Nên pha 2 không chỉ phải nâng retention: nó phải nâng
retention **và** hạ no-loanword CER cùng lúc, từ một điểm xuất phát đang hỏng cả
hai. Nếu bảng §5 cho thấy retention lên mà no-loanword CER không xuống, đó là dấu
hiệu LR 2e-4 đè quá tay — hạ về 1e-4 ở lượt sau, đúng như §3 đã nói.

Toàn bộ đường cong của pha 1, để khỏi phải đề xuất lại "train 100 giờ lâu hơn":
step-400 48,95% · **step-800 62,82%** · step-1200 47,70% · step-1600 48,25% ·
step-5350 (đúng 2 epoch trọn) 46,30%. Đỉnh nhọn ở 0,3 epoch rồi sập, và phẳng trong
dải 46–48% suốt 4.550 step sau đó.

⚠ Nhánh `training.init_adapter` **bỏ qua `cfg.lora` hoàn toàn** (`src/train.py:210-216`).
Rank đến theo adapter. Đừng thêm `--override lora.rank=...` — nó không có tác dụng và chỉ
làm `config.json` đông cứng ghi một con số sai.

⚠ Phải nạp `step-800/`, **không** nạp thư mục nào do `save_with_lambda` viết ra — những thư
mục đó đã nướng λ vào `lora_alpha`. `src/config.py:233-239` chỉ kiểm sự tồn tại của
`adapter_config.json`, không phát hiện được adapter đã bake.

---

## 2. Corpus pha 2 — số đo thật, dựng trên máy thuê

Toàn bộ bảng dưới đây **đã đo trên máy Windows ngày 16/09/2026** từ
`dataset/v6-corpus/` và `dataset/v6-corpus-addon/` (cả hai có sẵn ở đây, trái với
điều `CLAUDE.md` còn ghi). Đây không còn là ước lượng, và bước dựng trên máy thuê
chỉ việc tái lập đúng những con số này.

| phần | segment | giờ | mật độ từ ngoại lai |
|---|---:|---:|---:|
| 53 cuộc `mixed-noisy-v1` (phần train), manifest lấy từ `v6-corpus` | 4.481 | 6,66 | **7,22%** |
| 93 cuộc TTS của `v6-corpus-addon` | 8.636 | 9,07 | **17,87%** |
| **train pha 2** | **13.117** | **15,73** | **13,14%** |
| val (4 cuộc ở `configs/experiment.yaml:43`) | 365 | 0,79 | — |
| test | 654 | 1,58 | — |
| *(đối chiếu)* train của `v6-corpus`, tức pha 1 | 34.972 | 101,23 | **2,48%** |

Mật độ ở lượt tiếp xúc cuối do đó là **13,14% so với 3,55% của pha 1** — gấp 3,7
lần, đúng bậc độ lớn §0.1 dự đoán, nay có số đo thay cho ước lượng.

**Rủi ro chưa được nêu ở bản trước — tỷ lệ audio thật tụt ba lần.** Trong 13.117
segment train chỉ có **447 segment YouTube (3,4%)**; v5 học trên 4.481 segment
trong đó cũng đúng 447 segment YouTube, tức **10,0%**. Pha 2 nâng mật độ từ ngoại
lai nhưng đồng thời pha loãng phần audio thật xuống còn một phần ba, mà retention
đang hỏng chính trên audio thật ngoài corpus. Ghi điều này vào báo cáo cạnh giới hạn
ở §0.2; nếu pha 2 trượt, đây là nghi phạm thứ hai bên cạnh prior của pha 1.

### 2.1 Cách dựng

⚠ **Hai lô TTS không nằm trong `v6-corpus`.** Chúng nằm ở thư mục riêng
`v6-corpus-addon`, và manifest của chúng tên là `manifest.paid_meeting_0247.jsonl`
… `manifest.paid_meeting_0339.jsonl` — **không** có tiền tố `paid-meeting-vi-`.
Glob theo tiền tố đó khớp **0 file**, và bản dựng sẽ lặng lẽ ra 4.481 segment thay
vì 13.117. Kho `audio/` cũng tách làm hai, nên phải link theo từng cuộc chứ không
link được một thư mục audio duy nhất.

```bash
cd ~/speech/Fine_tune_wf
PYTHONPATH=. python scripts/_build_phase2.py
```

`scripts/_build_phase2.py` — viết một lần, dùng một lần, không cần commit:

```python
import glob, os, shutil
from pathlib import Path

SRC = Path("dataset/v6-corpus")
ADD = Path("dataset/v6-corpus-addon")
MIX = Path("dataset/mixed-noisy-v1")
OUT = Path("dataset/v6-phase2")
(OUT / "audio").mkdir(parents=True, exist_ok=True)

def ids(root):
    return sorted(Path(f).name[len("manifest."):-len(".jsonl")]
                  for f in glob.glob(str(root / "manifest.*.jsonl")))

wanted = [(m, SRC) for m in ids(MIX)] + [(m, ADD) for m in ids(ADD)]

copied, missing = 0, []
for m, root in wanted:
    man = root / f"manifest.{m}.jsonl"
    if not man.exists():
        missing.append(m)
        continue
    shutil.copy(man, OUT / man.name)
    link = OUT / "audio" / m
    if not link.exists():
        os.symlink((root / "audio" / m).resolve(), link, target_is_directory=True)
    copied += 1
print("copied", copied, "| missing", len(missing), missing)

from src.config import load
from src.data import load_manifests, resolve_splits, split_stats
cfg = load("configs/experiment.yaml",
           overrides=["data.dataset_path=dataset/v6-phase2"])
print(split_stats(resolve_splits(load_manifests(cfg.data.dataset_path),
                                 cfg.data.val_meetings)))
```

⚠ `resolve_splits` nhận **`cfg.data.val_meetings`**, không nhận `cfg`
(`src/data.py:66`). Bản trước truyền `cfg` và sẽ chết ngay ở dòng kiểm.

**Cổng kiểm — lệch thì dừng, đừng chạy tiếp:**

- `copied` = **140** (53 cuộc của `mixed-noisy-v1` trừ 6, cộng 93 cuộc add-on), `missing` = **6**, đúng sáu cuộc `paid_meeting_legacy_*`. Đã
  đối chiếu tại chỗ: đó là toàn bộ phần lệch giữa `mixed-noisy-v1` và `v6-corpus`,
  và 93 cuộc add-on không trùng cuộc nào của `v6-corpus`.
- `split_stats` in **`{'train': 13117, 'val': 365, 'test': 654}`**. Con số khác
  nghĩa là danh sách cuộc sai — kiểm `ls dataset/v6-corpus-addon/manifest.* | head`.
- Sáu cuộc `legacy` (236 segment / 0,31 h synthetic) bỏ được. Muốn giữ thì chép
  manifest **và** link audio của chúng từ `dataset/mixed-noisy-v1`; lúc đó train là
  13.353. Chọn một và ghi lại đã chọn cái nào — mọi con số step ở §3 tính theo
  13.117.

Đã cân nhắc và bỏ: `scripts/build_mixed_dataset.py` (bắt mọi record `verified: true`,
mà hai lô TTS chưa qua soát tay — nới điều kiện đó là gỡ một cổng chặn dùng chung để
giải quyết một trường hợp) và `scripts/merge_ondomain15h_tts_batch.py` (chép audio
thật, thừa khi audio đã nằm sẵn trên cùng máy).

---

## 3. Tham số train

```
training.init_adapter     = outputs/v6-100h-steps/checkpoints/step-800
training.epochs           = 2
training.batch_size       = 16
training.grad_accum_steps = 1
training.gradient_checkpointing = false
training.checkpoint_steps = 200
training.eval_steps       = 400
training.val_limit        = 64
eval.batch_size           = 64
```

**Không override `learning_rate`, `epochs=3` hay `eval.batch_size=8`.**
`configs/experiment.yaml` đã đặt sẵn `learning_rate: 2.0e-4`, `epochs: 3`,
`eval.batch_size: 8`; ba override đó trong bản trước chỉ chép lại giá trị mặc định
và làm người đọc tưởng chúng là quyết định của lượt này. Lập luận giữ 2e-4 vẫn
đứng: ba nấc `ladder-phaseA` chạy 1e-4 và đều kẹt dưới 56% retention, còn thứ đang
hỏng **chính là** prior của pha 1, nên LR thấp giữ lại đúng thứ cần bỏ. Lưới an
toàn là ràng buộc no-loanword CER ở §6; trượt nó thì lượt sau hạ về 1e-4 với bằng
chứng trong tay.

**2 epoch, không phải 3.** 13.117 / 16 = 820 step mỗi epoch, nên 2 epoch =
**1.640 step**. Đỉnh retention của pha 1 nằm ở step 800 và của v5 ở step 801, cả
hai cùng LR 2e-4 — epoch thứ ba rơi trọn vào vùng mà chính bản kế hoạch cũ dự đoán
là overfit. Với `checkpoint_steps=200`, 1.640 step cho **8 ứng viên (step-200 …
step-1600)** nằm hết trong vùng sống, thay vì 8 ứng viên trải trên một dải mà 5 đã
đoán trước là chết. Cùng số ứng viên, độ phân giải quanh đỉnh gấp rưỡi, rẻ hơn 10
phút GPU.

**`val_limit=64`.** §0.3 đã kết luận val không tham gia quyết định nào; nó chỉ tồn
tại để `train()` không raise `RuntimeError`. Để nguyên, mỗi lượt eval decode 365
segment val cộng toàn bộ VIVOS — khoảng 15 phút cho cả run, đổi lấy đúng những con
số mà kế hoạch này cấm mình dùng. `training.val_limit` cắt cả val lẫn ood
(`src/pipeline.py:333-341`), đưa chi phí đó về ~2 phút.

**`eval.batch_size=64`, không phải 8.** Ba ngưỡng ở §6 là số của chính v5, đo tại
`eval.batch_size=64` (chú thích khối `gates` trong `configs/experiment.yaml`). Chấm
ứng viên ở batch 8 rồi so với ngưỡng đo ở batch 64 là so hai đường decode khác nhau
— đúng lỗi mà chú thích ấy được viết ra để chặn. Batch lớn hơn đồng thời rút ngắn
mọi lượt decode ở §5 và §6.

⚠ `checkpoint_steps` lưu độc lập với lượt eval (`src/train.py:358-360`), và
`save_total_limit=2` chỉ xoay vòng `checkpoint-N` của Trainer, không đụng `step-N`.
`checkpoints/best/` vẫn được ghi theo val CER — **kệ nó, đừng dùng**.

---

## 4. Lệnh chạy, theo thứ tự

```bash
export PYTHONPATH=. TRANSFORMERS_AUTO_CONVERSION=0
export HF_TOKEN=<token org_download, đuôi HTgk>

# adapter pha 1
huggingface-cli download rework-whisper-v6-org/v6-100h-steps-checkpoints \
    --include 'step-800/*' --local-dir outputs/v6-100h-steps/checkpoints
ls outputs/v6-100h-steps/checkpoints/step-800/adapter_config.json   # phải tồn tại

R=v6-phase2-mixed
OV="--override data.dataset_path=dataset/v6-phase2
    --override data.real_bench_path=null
    --override data.cross_domain_path=dataset/cross-domain-bench
    --override training.epochs=2
    --override training.batch_size=16 --override training.grad_accum_steps=1
    --override training.gradient_checkpointing=false
    --override training.checkpoint_steps=200
    --override training.eval_steps=400
    --override training.val_limit=64
    --override eval.batch_size=64"

A="--override run_id=$R
   --override training.init_adapter=outputs/v6-100h-steps/checkpoints/step-800"

python -m src.pipeline --stage prepare $OV $A
python -m src.pipeline --stage train   $OV $A
```

**`--stage prepare`, không phải `--stage baseline`.** `prepare` làm đúng hai việc
mà các bước sau cần: đóng băng `config.json` (nguồn `gates.*` và `normalization` mà
§5/§6 đọc lại) và viết `validated_manifest.jsonl`. `baseline` làm thêm một việc:
decode base model trên test (654 segment) cộng toàn bộ VIVOS cộng real-bench, để
sinh `metrics/baseline.json` — mốc của tier 1 và tier 2. Quyết định của kế hoạch
này không dùng tier nào trong hai tier đó; nó nằm trọn ở ba ràng buộc cross-domain
ở §6, vốn là ngưỡng **tuyệt đối** so với v5 chứ không so base model. Docstring của
`stage_prepare` (`src/pipeline.py`) nói thẳng lý do nó được tách ra: số của base
model không đổi và đã có sẵn ở `Outputs/benchmark-2026-09-10/`.

`--stage train` vẫn đọc `data.ood_eval_path` cho lượt eval trong lúc train, nên vẫn
dựng `dataset/vivos` bằng `scripts/fetch_vivos.py`. Vẫn cần scp
`dataset/cross-domain-bench.zip` (119 MB) từ máy Windows cho §6.

**Không chạy `--stage sweep-gate`.** `select_lambda` chọn λ bằng val CER và OOD
CER, mà val mù (§0.3) — sweep của v6 đã chọn λ=0,5 mà không hề thấy retention rơi,
và v5 phải override tay đúng vì lý do này.

Chạy trong `tmux`. Đứt giữa chừng thì `--stage train --resume`, **đừng xoá
`checkpoints/`**.

### 4.1 Đã cân nhắc và bỏ — nhánh LoRA tươi

Một nhánh thứ hai với `training.init_adapter=null` (LoRA tươi trên
`vinai/PhoWhisper-large`, cùng corpus pha 2) là lượt chạy có xác suất qua gate cao
nhất trong bằng chứng hiện có: mọi cấu hình từng vượt 66% retention đều là LoRA
tươi, mọi cấu hình khởi từ adapter train trên corpus loãng đều dưới 63%. Giá chỉ
~20 phút GPU.

**Bỏ vì nó không thoả ràng buộc phải dùng corpus 100 giờ** — LoRA tươi không chạm
vào `v6-corpus` ở bất kỳ đâu, nên kết quả không được duyệt dù có thắng. Ghi lại ở
đây để lượt sau không đề xuất lại.

Nhánh A thoả ràng buộc đó **qua `init_adapter`**, không phải qua corpus nó train:
run pha 2 chỉ thấy 15,73 giờ, còn 100 giờ đi vào model qua adapter `step-800` của
`v6-100h-steps`. Cách hiểu này **đã được xác nhận ngày 16/09/2026**: ràng buộc là
"đã từng train trên 100 giờ", không phải "lượt train cuối phải chạy trên 100 giờ".
Nhánh A hợp lệ; nhánh LoRA tươi vẫn không, vì nó không chạm 100 giờ ở bất kỳ đâu.

---

## 5. Chọn checkpoint và λ — một lệnh, trên YouTube test

Bộ chọn là **hai cuộc YouTube của split test — `3nuCdzuyqng` và `7B24A9GfHAo`,
114 + 114 = 228 segment / 60,0 phút** (đếm lại tại chỗ trên manifest, cả
`mixed-noisy-v1` lẫn `v6-corpus` đều ra 228, và không segment nào trong hai cuộc này
bị QC đánh `bo`). `scripts/benchmark_run.py` cũng báo 228 trên cùng hai cuộc
(`Outputs/benchmark-2026-09-10 (3)/benchmark-table.N3.md`), nên hai đường decode
đồng ý về số segment. Con số **214** chỉ có ở bảng tay `Outputs/bang-tong-hop-*.csv`,
là số cũ, và bảng đó đã được biết là trôi khỏi manifest — đừng trích từ nó. Lý do dùng nó thay vì `cross-domain-bench`: bộ nào vừa dùng để chọn vừa dùng
để báo cáo thì số cuối không còn là ước lượng không thiên lệch, mà khoảng cách đang
xét giữa v6 và v5 chỉ 0,63 điểm CER.

⚠ Hạn chế giữ nguyên: YouTube test **không** độc lập hoàn toàn — 790 segment
YouTube trong corpus pha 2 đến từ cùng kho video, dù khác cuộc. Nó là bộ chọn tốt
hơn `cross-domain-bench`, không phải bộ chọn hoàn hảo.

Đã cân nhắc và bỏ: cắt đôi `cross-domain-bench` (mỗi nửa ~150 segment, khoảng tin
cậy rộng gấp rưỡi, mà CI của v5 đã là [7,03 – 10,17]); và cắt val mới từ chính
corpus pha 2 (val đó đo "giữ từ ngoại lai trong phân phối train" — đúng cái bẫy đã
giết v6, khi in-domain retention leo 71,6% còn cross-domain kẹt 48,6%).

Dựng bench một lần:

```bash
mkdir -p dataset/youtube-test/audio
PYTHONPATH=. python - <<'PY'
import json
from pathlib import Path
for m in ("3nuCdzuyqng", "7B24A9GfHAo"):
    rows = [json.loads(l) for l in
            open(f"dataset/v6-phase2/manifest.{m}.jsonl", encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["split"] == "test"]
    Path(f"dataset/youtube-test/manifest.{m}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    Path(f"dataset/youtube-test/audio/{m}").symlink_to(
        Path(f"dataset/v6-phase2/audio/{m}").resolve(), target_is_directory=True)
    print(m, len(rows))       # phải in 114 và 114
PY
```

Rồi một lệnh cho mỗi bước:

```bash
# bước 1 — 8 checkpoint ở lambda=1,0 (adapter nguyên, chưa bị save_with_lambda thu nhỏ)
python -m scripts.select_checkpoint --run-dir outputs/$R \
    --bench dataset/youtube-test --batch-size 64 \
    --out outputs/$R/metrics/selection-steps.json

# bước 2 — 5 lambda, chỉ trên checkpoint thắng
python -m scripts.select_checkpoint --run-dir outputs/$R \
    --bench dataset/youtube-test --batch-size 64 \
    --steps <N thắng> --lambdas 0 0.25 0.5 0.75 1.0 \
    --out outputs/$R/metrics/selection-lambda.json
```

Tiêu chí — **retention cao nhất; hoà thì CER thấp hơn; vẫn hoà thì step nhỏ hơn** —
nằm trong `scripts/select_checkpoint.py` chứ không đọc bằng mắt từ bảng, để quy tắc
không thể chạy theo số sau khi đã nhìn thấy số. Tiền lệ cho bước 2: v5 ship ở
λ=0,75, và λ cao kéo retention **lên** chứ không xuống — nhưng đừng ghim 0,75 sẵn,
đó là tham số đã tối ưu cho corpus khác.

Hai điểm khiến bản này rẻ hơn và sạch hơn bản trước:

- **Một tiến trình, không phải 13.** `scripts/score_cross_domain_model.py` nạp lại
  PhoWhisper-large mỗi lần gọi, khoảng một phút GPU mỗi lần, mười ba lần, chỉ để
  xếp hạng những adapter cùng ngồi trên một base đóng băng. Ở đây base nạp một lần
  và mỗi ứng viên là một `load_adapter` cộng một `set_lambda`.
- **Cùng đường decode với §6.** Bản trước chọn bằng `scripts/benchmark_run.py` (cắt
  30 giây rồi ghép lại) và gate bằng `src.gate._eval_split`. Hai đường cho ra hai
  con số khác nhau — chính chỗ này từng làm tier4a lệch từ 48,01% xuống 35,07%
  (`CLAUDE.md`). Chọn ở đường này rồi gate ở đường kia là để kẻ thắng phụ thuộc vào
  đường decode ngang với phụ thuộc vào adapter. `select_checkpoint.py` dùng đúng
  `_eval_split` + `score_cross_domain` mà §6 dùng.

Chi phí: 13 lượt decode 228 segment ở batch 64, một lần nạp model.

---

## 6. Gate — ba ràng buộc, ngưỡng lấy thẳng từ config

```bash
python -m scripts.score_cross_domain_model \
    --model vinai/PhoWhisper-large \
    --adapter outputs/$R/checkpoints/step-<N thắng> \
    --cross-domain-path dataset/cross-domain-bench \
    --config outputs/$R/config.json --batch-size 64 \
    --out outputs/$R/metrics/cross-domain-final.json
```

λ khác 1,0 thì bake trước bằng `src.lora.save_with_lambda` rồi trỏ `--adapter` vào
thư mục đã bake.

| ràng buộc | ngưỡng | ai giữ |
|---|---:|---|
| `cross-domain-bench` CER (299 segment) | ≤ **7,40%** | v5 |
| `cross-domain-bench` retention (1.283 lần xuất hiện) | ≥ **66,48%** | v5 |
| CER trên 48 segment **không** có từ ngoại lai | ≤ 6,13% | v5 |

⚠ **Đừng override ba ngưỡng này.** `configs/experiment.yaml` đã mang đúng ba con số
trên, nên `--config outputs/$R/config.json` là đủ. Bản trước override thành
`cer_max=0.0819` và `retention_min=0.6664` — đó là bộ số **cũ, đã bị loại**, đo
trước khi `src/asr.py` chuyển sang đường decode chống lặp (temperature fallback +
`compression_ratio_threshold`). Chú thích ngay trong config nói rõ: một segment bị
lặp đang gánh 0,79 điểm CER của con số 0,0819, nên gate theo nó vừa **so hai đường
decode khác nhau** vừa **nới ngưỡng 0,79 điểm** — đủ để cho qua một model thực chất
tệ hơn v5, tức đúng thất bại mà cả kế hoạch này sinh ra để tránh.

⚠ **Đừng dùng `scripts/gate_at_lambda.py` cho kế hoạch này.** Ba lý do, mỗi lý do
đủ để hỏng lượt thuê:

1. Nó bắt buộc `metrics/lambda_sweep.csv` và **từ chối** một λ không phải hàng
   trong đó (`scripts/gate_at_lambda.py:78-88`) — mà §4 cố ý không chạy
   `sweep-gate`, nên file ấy không tồn tại.
2. Nó nạp `checkpoints/best` (`scripts/gate_at_lambda.py:95-97`), tức checkpoint
   chọn theo val CER — đúng cái §0.3 và §3 bảo đừng dùng. Thư mục ấy **có tồn
   tại**, nên nó sẽ gate nhầm model mà không báo lỗi gì.
3. Nó không có cờ `--override`; `--override gates.cross_domain_cer_max=...` bị
   argparse từ chối ngay.

`score_cross_domain_model.py` in cả ba con số cộng khoảng tin cậy bootstrap, trên
đúng đường decode của gate. So tay với bảng trên, và ghi cả ba vào báo cáo.

Ràng buộc thứ ba không phải mục tiêu mà là lưới an toàn, và nó không thừa: 15,7% ký
tự của bộ cross-domain nằm ở 48 segment đó, và `v6-ondomain-29h` đã cho thấy ép
retention làm model bịa tiếng Anh ở chỗ không có tiếng Anh — no-loanword CER vọt
lên 10,83%. **Cả ba phải qua.** Không qua thì không đẩy lên HF, v5 giữ production.
Không hạ ngưỡng để cho qua.

---

## 7. Ngân sách

| bước | bản trước | bản này |
|---|---:|---:|
| tải corpus 14,7 GB + add-on 1,57 GB, giải nén | ~15 phút | ~15 phút |
| dựng `dataset/v6-phase2` + `dataset/youtube-test` | ~1 phút | ~2 phút |
| `--stage baseline` → `--stage prepare` | ~5 phút | **~0** |
| train nhánh A (3 epoch → 2 epoch) | ~30 phút | **~20 phút** |
| eval trong lúc train (`val_limit=64`) | ~15 phút | **~2 phút** |
| chọn checkpoint + λ, một lần nạp model, batch 64 | ~52 phút | **~25 phút** |
| gate cuối | ~10 phút | ~8 phút |
| benchmark 6 bộ, 3.985 segment (§8) | ~15 phút (chỉ ViMedCSS) | **~25 phút** |
| **tổng** | **~2 giờ 10** | **~1 giờ 50** |

Thuê ba giờ vẫn đúng: phần dư giờ đây đủ
cho một lượt train lại **có thông tin** — ví dụ hạ về 1e-4 nếu ràng buộc
no-loanword CER trượt — chứ không chỉ đủ chạy lại y hệt.

Số 0,728 giây mỗi step đo trên H200 (`docs/h200-timing-probe-2026-09-11.md`). Các
con số decode là ngoại suy từ nó; đo lượt decode đầu tiên rồi hiệu chỉnh phần còn
lại thay vì tin bảng này.

---

## 8. Benchmark — sáu bộ, cùng lượt thuê

Gate ở §6 chỉ trả lời "có ship được không". Bảng benchmark trả lời "so với ai thì
đứng đâu", và đó là thứ đi ra ngoài phòng kỹ thuật. Chạy trong cùng lượt thuê: trả
máy rồi mới thiếu một hàng là phải thuê lại.

**Ba model nền đã decode xong và đang nằm ở `Outputs/benchmark-2026-09-10 (3)/`** —
`vinai_phowhisper_large`, `winhsss_reworkwhisper_large_v5`, `scribe_v2`. Không
decode lại chúng: chép nguyên thư mục đó lên máy thuê, decode **một** model mới vào
cùng thư mục, rồi `benchmark_report.py` dựng lại bảng cho cả bốn. Decode lại ba model
nền là trả tiền GPU cho những con số đã có, và với `scribe_v2` là trả tiền API thật.

| bộ | n | giờ | nguồn |
|---|---:|---:|---|
| `vimedcss-hard` | 658 | 1,38 | HF `tensorxt/ViMedCSS`, split `hard` |
| `vimedcss-test` | 1.614 | 3,39 | HF `tensorxt/ViMedCSS`, split `test` |
| `youtube-test` | 228 | 1,00 | manifest, `split=test` + `source=youtube` |
| `synthetic-test` | 426 | 0,58 | manifest, `split=test` + `source=synthetic` |
| `vivos` | 760 | — | `dataset/vivos` |
| `cross-domain` | 299 | 1,30 | `dataset/cross-domain-bench` |
| **tổng** | **3.985** | **~7,7** | |

⚠ `youtube-test` ở đây là **228**, giống hệt §5 — `benchmark_run.py` và
`_eval_split` đồng ý với nhau về số segment. Con số **214** chỉ xuất hiện ở bảng tay
`Outputs/bang-tong-hop-*.csv`, và bảng đó đã được biết là trôi khỏi manifest. Dùng
`benchmark-table.N3.md` làm nguồn, đừng dùng file CSV.

### 8.1 Lệnh

```bash
# adapter thắng ở §5, đã bake lambda nếu lambda != 1,0, đặt ở một đường dẫn NGẮN và
# ổn định: nó đi thẳng vào tên file output và là khoá resume.
ADP=outputs/v6-phase2-adapter

python -m scripts.benchmark_run --backend hf \
    --model vinai/PhoWhisper-large --adapter $ADP \
    --suites vimedcss-hard,vimedcss-test,youtube-test,synthetic-test,vivos,cross-domain \
    --path youtube-test=dataset/v6-phase2 \
    --path synthetic-test=dataset/v6-phase2 \
    --path vivos=dataset/vivos \
    --path cross-domain=dataset/cross-domain-bench \
    --batch-size 64 \
    --out "Outputs/benchmark-2026-09-10 (3)"

python -m scripts.benchmark_report --dir "Outputs/benchmark-2026-09-10 (3)" \
    --variant N3 --baseline vinai/PhoWhisper-large
```

Ba điều dễ sai:

- **`--adapter` phải có mặt.** Tên file output là `slug(model+adapter)`, và
  `done_ids()` resume theo đúng tên đó. Thiếu `--adapter`, output trùng tên với lượt
  decode base model cũ, `benchmark_run` coi là "đã xong" và **báo cáo hypothesis của
  base model như thể của adapter** — im lặng, không lỗi.
- **`--path` trỏ `dataset/v6-phase2`** cho hai bộ manifest. Ba model nền decode trên
  `mixed-noisy-v1`; đã đối chiếu: cả hai corpus cho đúng 228 segment YouTube từ
  `3nuCdzuyqng` (114) và `7B24A9GfHAo` (114), và 426 segment synthetic từ
  `paid_meeting_test_0001-0003`. Cùng audio, khác đường dẫn.
- **Bộ `cross-domain` bị decode hai lần** — một lần ở §6 bằng `_eval_split` cho gate,
  một lần ở đây bằng đường cắt-30-giây của `benchmark_run`. Cố ý: hai con số thuộc
  hai đường decode và **không được trộn vào nhau**. Số gate lấy từ §6, số bảng lấy từ
  `benchmark-table.N3.md`. Giá là ~2 phút decode 299 segment.

### 8.2 Đọc kết quả

`benchmark_report.py` in hai bảng: ma trận CER gọn và bảng chi tiết có khoảng tin cậy
95% cộng loanword retention mỗi ô. So **dọc** một cột (cùng audio, khác model), không
so ngang — sáu bộ khác độ khó.

Mốc phải vượt, lấy từ bảng N3 hiện có (v5 là model đang chạy production):

| bộ | v5 | PhoWhisper-large | scribe v2 |
|---|---:|---:|---:|
| `cross-domain` CER | **8,19** | 13,32 | 6,25 |
| `cross-domain` retention | **66,64%** | 27,83% | 83,63% |
| `youtube-test` CER | **5,77** | 15,92 | 6,44 |
| `synthetic-test` CER | **1,61** | 4,26 | — |
| `vimedcss-hard` CER | **16,42** | 19,43 | — |
| `vimedcss-test` CER | **14,21** | 15,93 | — |
| `vivos` CER | 3,34 | **2,28** | — |

⚠ Hàng `vivos` của N0 từng cho 81% CER và đó là **hiện tượng giả do chữ hoa** —
reference VIVOS viết ALL-CAPS còn hypothesis viết thường. N3 đã chuẩn hoá nên số ở
trên dùng được; đừng trích hàng VIVOS từ `benchmark-table.N0.md`.

⚠ Bốn cột ViMedCSS trong `bang-tong-hop-*.csv` dùng chuẩn hoá khác bench này, ghi rõ
ngay trong file đó là **không so trực tiếp được**. Chỉ trích từ `benchmark-table.N3.md`.

Chi phí: 3.985 segment ở batch 64. Neo duy nhất đang có là ~15 phút cho 2.272 segment
ViMedCSS ở batch 8, nên ~25 phút là ước lượng thô — **đo bộ đầu tiên rồi suy ra phần
còn lại** thay vì tin con số này. `benchmark_run` resume theo `segment_id`, nên đứt
giữa chừng chỉ mất phần chưa xong.

---

## 9. Trước khi trả máy

- Chép **`outputs/v6-phase2-mixed/`** và **`Outputs/benchmark-2026-09-10 (3)/`** đầy
  đủ về máy Windows. Đã mất một lượt đo vì chép muộn.
- Đẩy 8 checkpoint lên HF bằng `scripts/push_checkpoints_live.py --once`. Không đẩy
  thì lượt sau muốn chấm lại một bản khác phải train lại từ đầu.
- Ghi `experiments/task_ledger.md` và `provenance.md`.
- Cập nhật `Outputs/bang-tong-hop-*.csv` **từ** `benchmark-table.N3.md`, không gõ tay
  từ trí nhớ — đó là cách file CSV đó trôi khỏi manifest ngay từ đầu.
