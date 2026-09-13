# Kế hoạch fine-tune trên `v6-corpus`

Chạy trên server GPU thuê, theo quy trình chung ở `docs/server-finetune.md`. File này chỉ
ghi phần khác với recipe chung: nguồn dữ liệu, override, thứ tự stage, ngân sách thời gian
đo trên chính corpus thật (không phải ước lượng).

## 0. Việc đã xong

`rework-whisper-v6-org/v6-corpus` (private, `v6-corpus.tar` 14,7 GB) đã build và push —
xem `v6-corpus-merge-blockers.md` trong memory. Bản trên HF là 35.544 record.

**Bản trên máy thuê đã thêm 4 cuộc YouTube train** (12/09/2026) có ở
`dataset/youtube-meetings` trên máy Windows nhưng không có trong drop
`youtube-meeting-1`: `dGT3YW0AdD8`, `iyeFAuuEBl4`, `rIFrrmm8ILY`, `xKDHjUoUN54` — 447 seg,
1,96 h, `verified: true`, không trùng test/val. `CHECKSUMS.txt` đã sinh lại (36.291 file,
verify OK). **Bản trên HF chưa có 4 cuộc này** — dựng lại corpus trên máy mới thì phải
thêm tay, hoặc push lại tar.

| split | segment | giờ | cuộc |
|---|---:|---:|---:|
| train | 34.972 | 101,24 | 290 |
| val | 365 | 0,79 | 4 |
| test | 654 | 1,58 | 5 |

Train chia 49,98 h synthetic (243 cuộc) / 51,25 h YouTube (47 cuộc).

`sha256` trong manifest YouTube **không** khớp hash của wav trên đĩa — đúng với cả cuộc cũ
lẫn cuộc mới thêm, là dữ liệu kế thừa từ pipeline pilot, không phải dấu hiệu hỏng file.
Dùng `CHECKSUMS.txt` để kiểm toàn vẹn, không dùng field đó.

## 1. Trước khi gõ lệnh

Theo `docs/server-finetune.md` §0: hỏi IP máy mới, sửa `~/.ssh/config`, khảo sát driver/
vCPU/RAM, kiểm máy có đang chạy job khác không.

## 2. Mã nguồn + môi trường

Theo §1–§2 của `docs/server-finetune.md`, không đổi gì.

## 3. Dữ liệu — khác với recipe chung

`docs/server-finetune.md` §3 trỏ tới `winhsss/mixed-noisy-v1` (public). Lần này lấy
`rework-whisper-v6-org/v6-corpus` (**private**, cần `HF_TOKEN` scope org — dùng token
`org_download`, không dùng `write_dataset`, token đó scope user sẽ ra 404):

```bash
mkdir -p dataset
.venv/bin/python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('rework-whisper-v6-org/v6-corpus','v6-corpus.tar',
      repo_type='dataset', local_dir='.', token='<HF_TOKEN org_download>'))"
tar -xf v6-corpus.tar -C dataset
PYTHONPATH=. .venv/bin/python scripts/checksum_dataset.py --root dataset/v6-corpus --mode verify
```

Phải thấy `OK -- dataset matches CHECKSUMS.txt`. **Giữ token trên máy tới khi `sweep-gate`
push xong** (§5 dùng lại chính token này để đẩy model) — chỉ xoá ở bước cuối (§7), không
xoá ngay sau tải như lần tải corpus trước.

`data.ood_eval_path` (tier 2) vẫn là `dataset/vivos`, không nằm trong corpus này, dựng
riêng như mọi lần:

```bash
PYTHONPATH=. .venv/bin/python scripts/fetch_vivos.py --out dataset/vivos
```

`data.real_bench_path` đặt `null` — quyết định đã chốt (§4 dưới), không dựng
`dataset/real-meetings-bench` cho lần này.

## 4. Override — khác `configs/experiment.yaml` mặc định

Config hiện ghi `run_id: v4-mixed-r16`, `dataset_path: dataset/mixed-noisy-v1` — đó là
run đã publish, **không sửa file**, override ở dòng lệnh như mọi lần:

```bash
export PYTHONPATH=. TRANSFORMERS_AUTO_CONVERSION=0
export HF_TOKEN=<org_download token>   # src/hub.py:push_adapter reads it from env, not config
R=v6-corpus-r32
OV="--override run_id=$R
    --override data.dataset_path=dataset/v6-corpus
    --override data.real_bench_path=null
    --override lora.rank=32 --override lora.alpha=64
    --override training.batch_size=16 --override training.grad_accum_steps=1
    --override training.gradient_checkpointing=false
    --override training.epochs=1
    --override training.eval_steps=364
    --override eval.batch_size=64
    --override hub.repo_id=rework-whisper-v6-org/Reworkwhisper-large-v6"
```

`hub.repo_id` chưa tồn tại trên HF (kiểm bằng `api/models/...` trước khi viết plan này —
`Repository not found`), tên sạch. `hub.private` mặc định `true` trong config, giữ nguyên
— các model đã publish trước đó (`Reworkwhisper-large-v4`, `-v5`) đều để private.

Mọi lựa chọn giống `docs/server-finetune.md` §4 (đã đo trên H200, không lặp lại lý do ở
đây), trừ một chỗ khác: **`training.eval_steps=364`**, tính thẳng từ corpus thật thay vì
đoán. `train.batch_size=16`, `grad_accum_steps=1` → 34.972 seg / 16 ≈ **2.186 step một
epoch**; `total_steps // 6 ≈ 364` cho 6 lượt eval giữa chừng, đúng khoảng 5–8 lượt
khuyến nghị.

⚠ **`lora.rank=32` chưa có bằng chứng so với rank 16** — vẫn là đề xuất, không phải kết
luận đã kiểm chứng, giữ nguyên cảnh báo từ `docs/server-finetune.md`. Nếu muốn có so sánh
rank16 vs rank32 cho lần publish này thì cần chạy thêm một lượt `--override lora.rank=16
--override lora.alpha=32`, tốn thêm ~30 phút GPU — quyết định trước khi thuê máy, không
phải sau.

## 5. Thứ tự stage

Khác với `docs/h200-timing-probe-2026-09-11.md` (chỉ đo `baseline`+`train`, không chạy
`sweep-gate` vì đó là lượt đo giá, không phải run thật): lần này chạy đủ bốn bước, đúng
`docs/runbook.md` phần B:

```bash
python -m src.pipeline --stage baseline  $OV
python -m src.pipeline --stage train     $OV
python -m src.pipeline --stage sweep-gate $OV --override hub.push=True
```

`sweep-gate` chọn λ, chấm tier 1/2 (tier 4a bỏ qua vì `real_bench_path=null` — luật gate
coi `pass: None` là không chặn `overall_pass`), và chỉ push lên HF khi `overall_pass` **và**
`hub.push: true`. Không λ nào giữ được ngân sách OOD CER thì dừng cứng, không chọn đại —
khi đó `hub.repo_id` không được tạo, không có gì để dọn.

Chạy trong `tmux`, đúng mẫu ở `docs/server-finetune.md` §4 — nối cả ba lệnh trong một
session để không phải canh SSH ba lần.

## 6. Ngân sách thời gian — tính lại trên số thật

`docs/h200-timing-probe-2026-09-11.md` ước 2.730 step cho corpus 100 giờ; corpus thật là
**2.186 step/epoch**, ít hơn ước lượng, nên cột "thuận lợi" của báo cáo đó là cận trên hợp
lý cho `train`, không phải cận dưới:

| Khoản | Ước tính |
|---|---:|
| Tải 14,7 GB (đo được 58–120 MB/s) | 2–5 phút |
| Giải nén tar | 1–2 phút |
| `--stage baseline` | 8–12 phút |
| `--stage train` (2.186 step × 0,728–1,0 s) | 26–36 phút tính toán, cộng 6 lượt eval giữa chừng |
| `--stage sweep-gate` (5 λ, tier 1+2, không tier 4a) | 30–70 phút |
| **Tổng** | **~1,5–2,5 giờ thuận lợi, ~4–5 giờ bi quan** |

Vẫn khuyến nghị **thuê 6 giờ** như `docs/server-finetune.md` — lượt đầu hỏng phải chạy lại
là chuyện thường, và mọi số trên là lần chạy trót lọt.

## 7. Sau khi gate pass

Theo `docs/server-finetune.md` §5: chép `outputs/<run_id>.zip` và `run.log` về **ngay khi
xong**, đừng đợi hết phiên. Rồi ghi `experiments/task_ledger.md` và `provenance.md` —
không có DVC/W&B, đây là toàn bộ hệ thống lưu vết của project.

Thu hồi (xoá khỏi máy, không revoke trên HF vì token này còn dùng cho việc khác của org)
`HF_TOKEN` khỏi máy thuê **sau khi `sweep-gate` push xong**, trước khi trả máy.
