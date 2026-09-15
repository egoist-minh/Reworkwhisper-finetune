# v6-100h-steps — lượt 100 h công bằng, đo luôn đường cong retention theo số bước

Chốt 15/09. Một lượt train, hai câu trả lời: lượt 101 h ở cấu hình công bằng mà
professor yêu cầu, **và** phép thử cho chẩn đoán "train càng lâu càng mất từ ngoại lai".

---

## 0. Chẩn đoán đang kiểm

Fine-tune ghi đè một **luật** của base model bằng một **bảng tra cứu**.

Bằng chứng (đo 15/09, `scratchpad/retention_diff.py` trên 299 segment / 1.283 occurrence
của `dataset/cross-domain-bench`):

- v5 giữ 237 occurrence mà `v6-ondomain-15h` đánh mất; chiều ngược lại chỉ 18. Bất đối
  xứng 13:1.
- **Mọi type v6 đánh mất đều xuất hiện nhiều hơn trong corpus train của v6.** `team`:
  v5 thấy 81 lần và giữ; `-15h` thấy 143 lần và mất. `digital`: v5 thấy **0** lần mà vẫn
  giữ; `-15h` thấy 9 lần và mất. `loyalty`: cả hai đều 0 lần — v5 giữ, v6 mất.

Nên đây không phải lỗ hổng từ vựng. v5 giữ được từ nó chưa từng thấy; v6 mất từ nó thấy
hàng trăm lần. v5 giữ **theo luật**, v6 giữ **theo trí nhớ**.

Biến sắp đúng thứ tự cả bốn lượt LoRA tươi là **số bước tối ưu**, không phải mật độ:

| run | bước | mật độ | retention |
|---|---:|---:|---:|
| v5 (`v4-mixed-r16`) | **801** | 7,13% | **0,666** |
| `v6-ondomain-15h` | 1.317 | 7,46% | 0,619 |
| `v6-ondomain-29h` | 2.157 | 7,42% | 0,595 |
| `v6-corpus-r32` | 2.186 | 2,48% | 0,486 |

Mật độ không sắp được thứ tự — 7,46% và 7,42% gần bằng nhau mà retention lệch 2,4 pp.
Số bước sắp đúng cả bốn. **v5 thắng vì nó dừng sớm nhất, không phải vì corpus tốt nhất.**

Cơ chế giả định: mỗi bước tối ưu CER tiếng Việt, và cách rẻ nhất để giảm CER tiếng Việt
là phiên âm từ ngoại lai thành âm tiết Việt. Retention là thiệt hại phụ, và không có gì
trong hàm loss biết nó đang xảy ra.

## 0.1 Vì sao lượt 100 h là phép thử tốt nhất

5.350 bước = **6,7 lần** ngân sách bước của v5. Lưu checkpoint dọc đường thì một lượt train
quét được cả trục số-bước trên **cùng một corpus** — không lẫn biến corpus như bốn điểm
dữ liệu trên.

Và có một điểm chưa ai đo: **checkpoint ở bước ~800** = đúng ngân sách bước của v5, trên
corpus rộng gấp 12 lần. Đây là ứng viên mạnh nhất của lượt này.

---

## 1. Sửa code trước khi thuê máy (0 phút GPU)

**Trạng thái 15/09: cả bốn mục dưới đã sửa xong và `pytest tests/` xanh (244 passed,
2 skipped).** Phần chưa làm được trên máy này là lượt train tí hon ở §1.1 — máy không
cài torch (`ModuleNotFoundError: No module named 'torch'`), nên đó là việc đầu tiên
trên máy thuê, trước khi bắt đầu lượt 100 h.

### 1.1 Lưu adapter ở mọi vòng eval, không chỉ khi ValCER tốt lên

`RobustEvalTrackingCallback.on_evaluate` ([src/train.py:349](../src/train.py#L349)) trước
đây chỉ gọi `model.save_pretrained(best_dir)` khi `stopping.rounds_without_improvement == 0`.
Không có checkpoint nào khác sống sót, nên không có gì để chấm.

**Thay đổi:** thêm một dòng lưu vô điều kiện vào `checkpoints/step-<global_step>/`, song
song với `best`:

```python
model.save_pretrained(str(out / "checkpoints" / f"step-{state.global_step}"))
```

Adapter rank 16 là 115.487.384 byte (đo trên `Outputs/v3-r16/adapter/`), nên 14 vòng cộng
`best/` là **~1,7 GB**. `archive_run` loại thư mục tên `checkpoint-<số>` (optimizer state
của Trainer) — `step-<số>` **không** khớp bộ lọc đó nên tự động đi vào zip.

Máy Windows hết chỗ, nên **không chép zip về**: ba adapter đáng giữ đi thẳng lên Hub bằng
`scripts/push_run_adapters.py` (`best/`, bản retention cao nhất ở §3a, và `step-5350`),
phần còn lại bỏ cùng máy thuê. Chỉ scp `run.log`, `outputs/<run>/metrics/` và thư mục
bench về.

Nhưng cả hai đường trên đều chỉ chạy **sau** khi train xong, mà máy thuê có thể tắt giữa
chừng — hết hạn mức, bị thu hồi. Lượt dừng ở bước 3.000 thì không còn gì.
`scripts/push_checkpoints_live.py` chạy song song trong tmux, cứ 120 giây quét
`checkpoints/` và đẩy `step-<N>` mới lên một repo private, mỗi cái một thư mục con. Nó
không đụng vào tiến trình train và không bao giờ raise ra ngoài, nên lỗi mạng ở đây không
giết được lượt chạy. `best/` không cần đẩy: `on_evaluate` ghi `best/` và `step-<N>` từ
cùng một model trong cùng một lệnh, nên mọi `best/` đều trùng byte với một `step-<N>`.

**Verify:** `pytest tests/` — xanh, cộng một test mới chặn đúng rủi ro đắt nhất:
`archive_run` phải giữ `checkpoints/step-*` (lọc của nó chỉ bỏ `checkpoint-<số>`). Mất
bước đó là cả lượt thuê không còn gì để chấm.

**Còn nợ:** lượt train tí hon
(`training.limit=20 training.val_limit=20 training.eval_steps=5`) — phải thấy nhiều hơn
một thư mục `checkpoints/step-*`. Cần torch, nên chạy trên máy thuê trước lượt 100 h.

### 1.2 Phụ thuộc

`docs/plan-cat-pipeline-va-so-thi-nghiem.md` §1 phải xong trước: `stage_prepare` tồn tại,
`ood` đã bỏ khỏi eval trong train, `validate` chấp nhận `ood_eval_path: null`. Không có
§1.3 thì vẫn phải chạy `scripts/fetch_vivos.py` trên máy thuê.

**Đã xong** (commit `ef36306`, 15/09).

### 1.3 Tắt được early stopping — nếu không, lượt này có thể tự dừng trước bước 5.350

`_EarlyStoppingState(patience=3)` trước đây ghim cứng trong [src/train.py](../src/train.py):
ba vòng eval liên tiếp ValCER không cải thiện là `control.should_training_stop = True`.

Đây không phải rủi ro lý thuyết. ValCER hai lượt gần nhất (val 365 segment):

| | vòng 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| `v6-ondomain-15h` | 0,04355 | 0,03865 | 0,03625 | 0,03554 | 0,03540 |
| `v6-ondomain-29h` | 0,05758 | 0,04428 | 0,04064 | 0,03877 | 0,03524 |

Đơn điệu giảm, nhưng khoảng cách vòng 4→5 của `-15h` chỉ **0,014 pp**. Val 1.179 segment
của lượt này đo yên hơn val 365 segment, nên plateau đọc ra sạch hơn — và đó chính là vấn
đề: 13 vòng trên một đường cong đã phẳng từ giữa lượt thì ba vòng liên tiếp không cải
thiện gần như chắc chắn xảy ra. Dừng sớm ở bước ~2.800 là mất **cả hai** deliverable: số
cuối cho professor, và hai điểm cuối của đường cong retention. Thêm nữa, thứ chọn
checkpoint là retention cross-domain, một đại lượng ValCER không nhìn thấy — để ValCER kết
thúc lượt train là để sai người cầm lái.

**Thay đổi:** `training.early_stopping_patience`, mặc định `3` (đúng hành vi cũ, không
lượt nào đổi), `null` = không bao giờ dừng sớm. Lượt này chạy `null` — checkpoint được
chọn sau bằng retention cross-domain, một đại lượng ValCER không nhìn thấy, nên để ValCER
kết thúc lượt train là để sai người cầm lái.

### 1.4 `benchmark_run.py` phải nạp được adapter

§3b cũ viết `--model outputs/v6-100h-steps/adapter`. Hai chỗ sai, cả hai đều chỉ lộ ra sau
khi đã trả tiền máy:

1. `outputs/<run>/adapter/` do `save_with_lambda` viết trong `sweep-gate` — mà lượt này
   **không chạy `sweep-gate`**, nên thư mục đó không tồn tại.
2. `HFBackend` gọi `load_for_eval(model)` không truyền adapter: `--model` là một model
   đầy đủ, không phải thư mục PEFT. Nó sẽ không nạp adapter dù đường dẫn có thật.

**Thay đổi:** thêm `--adapter` cho backend `hf` (`load_for_eval` vốn đã nhận tham số này —
`scripts/score_cross_domain_model.py` dùng từ trước). Tên file kết quả lấy cả
`--model` lẫn `--adapter`, vì đó là thứ duy nhất phân biệt hai lượt và `done_ids()` resume
theo nó: decode base+adapter dưới tên base trần sẽ trùng tên với lượt base có sẵn trong
`Outputs/benchmark-2026-09-10 (3)/`, bị coi là "đã decode xong", và báo cáo sẽ trình
hypothesis của base model như thể của model mới.

---

## 2. Cấu hình lượt chạy

Cấu hình công bằng — giống v5 ở mọi thứ trừ corpus, để so được trực tiếp.

| | giá trị | lý do |
|---|---|---|
| `lora.rank` / `alpha` | 16 / 32 | hình dạng đạt retention 66,6% ở v5. `v6-corpus-r32` chạy rank 32 nên chưa bao giờ so được với v5 |
| `training.epochs` | **2** | v5 chạy 3 — đây là chỗ duy nhất lượt này lệch khỏi cấu hình v5, và là cái giá của trần 3 giờ (§4) |
| `learning_rate` | 2.0e-4 | như v5 |
| `batch` × `grad_accum` | 16 × 1 | batch hiệu dụng 16, bằng v5 |
| `gradient_checkpointing` | false | chỉ T4 cần |
| `init_adapter` | null | LoRA tươi. Curriculum hai pha đã đo và thua (`v6-dense-{1,3,4}`) |
| `eval_steps` | **400** | 13 vòng trên 5.350 bước; mỗi vòng lưu một adapter |
| `training.val_limit` | **null** | `ManifestDataset.limit()` lấy `records[:n]` không xáo, nên cap 100 chỉ chấm được `paid_meeting_0001` — ba cuộc họp thêm ở dưới không vòng nào chạm tới. Chấm đủ 1.179 segment, ~70 giây/vòng ở batch 64 |
| `eval.batch_size` | 64 | CER lệch theo batch — mọi phép đo trong lượt phải cùng số này |
| `data.ood_eval_path` | null | không stage nào đọc nữa sau §1.3 của plan kia |
| `early_stopping_patience` | **null** | §1.3. Không có dòng này thì ValCER có quyền kết thúc lượt train |
| `data.val_meetings` | **7 cuộc** | mặc định 4 cuộc + `zlKBfNzfh50`, `coteccons-agm-2025`, `jB1P4bqLwDY` — xem dưới |
| corpus | `v6-corpus` + add-on TTS | **42.794** segment train → **2.675** bước/epoch |

### 2.1 Corpus: `v6-corpus` cộng add-on 9,07 h TTS đậm

`v6-corpus.tar` trên Hub là bản 12/09 và **không chứa** hai lô TTS đã giao:

| lô | cuộc | giờ | density |
|---|---:|---:|---:|
| `paid-meeting-vi-0247-0296` | 50 | 4,91 | **21,47%** |
| `paid-meeting-vi-0297-0339` | 43 | 4,15 | **14,02%** |
| `v6-corpus` train (trước khi thêm) | — | 97,62 | 2,34% |

Đây là đòn bẩy density duy nhất còn lại chưa dùng, và nó đúng vào biến §0 đang truy: v5
đạt retention 0,666 ở density 7,13%, `v6-corpus-r32` đạt 0,486 ở 2,48%. Thêm hai lô đẩy
train lên **3,55%**.

`scripts/build_v6_corpus_addon.py` đóng riêng phần mới (1,57 GB) ở đúng layout của corpus
để máy thuê giải nén đè lên `dataset/v6-corpus/`; dựng lại rồi đẩy lại cả 15 GB thì mất
nguyên buổi uplink. `load_manifests` quét `manifest.*.jsonl` nên không cần bước trộn nào.
Manifest của hai lô ghi `"split": "train"` mà `resolve_splits` chỉ nhận `demo`/`test`, nên
script remap sang `demo` — cùng phép remap `scripts/ingest_paid_dataset_v2.py` dùng.

`voice_id` của cả hai lô **trùng test 0 cuộc**, nên tính disjoint của test giữ nguyên. Lô
0247-0296 trùng 5 voice của val, nhưng val vốn đã leak sang train từ trước nên không thêm
hại mới.

Đo lại bằng `load_manifests` + `resolve_splits`, không lấy từ card HF:

| | segment | giờ | density |
|---|---:|---:|---:|
| train | **42.794** | 106,69 | **3,55%** |
| val | 1.179 | 4,40 | 7,12% |
| test | 654 | 1,58 | 7,06% |

**Đừng kỳ vọng 3,55% lật được v5.** Vẫn chưa bằng một nửa 7,13%, và lần lọc density trước
đẩy retention 48,6% lên 61,9% mà vẫn thiếu 4,6 pp. Đây là cải thiện xác suất, không phải
lời giải.

Corpus đổi nên **không so được bước/epoch với `v6-corpus-r32`** (2.186 so với 2.675). Khi
trình số thì nói đúng tên: khác corpus, khác số epoch, khác cả rank.

Mười ba vòng eval rơi vào **400, 800, 1.200, … , 5.200** (bội số của 400). Bước cuối 5.350
không có vòng eval, nhưng `on_train_end` ở §1.1 vẫn lưu nó, nên trên đĩa có **14 thư mục
`step-*`** cộng `best/`.

`eval_steps=400` thay vì 800 mua hai thứ, giá 6 phút eval và ~0,9 GB đĩa:

- **`step-400` và `step-800` kẹp hai bên ngân sách 801 bước của v5.** Với 800 thì chỉ có
  một điểm trùng v5; với 400 thì trả lời được cả câu "đỉnh có nằm SỚM hơn v5 không" — mà
  theo chính chẩn đoán ở §0 (retention giảm đơn điệu theo bước) thì `step-400` phải là
  điểm cao nhất của cả lượt.
- Độ phân giải gấp đôi ở đoạn dốc nhất của đường cong, tức đoạn 400–1.600, nơi ba lượt cũ
  (801 / 1.317 / 2.157 bước) nằm chen nhau.

`data.val_meetings` thêm **3 cuộc họp YouTube** đang nằm trong train, không chạy
`scripts/select_val_meetings.py`:

| meeting_id | chủ đề | seg | giờ | density |
|---|---|---|---|---|
| `coteccons-agm-2025` | đại hội cổ đông Coteccons — xây dựng, chủ tịch nói nguyên câu tiếng Anh có phiên dịch | 265 | 1,19 | 9,16% |
| `jB1P4bqLwDY` | tọa đàm đầu tư khởi nghiệp (NIC, ThinkZone) — `startup`, `investor`, `angel` | 301 | 1,33 | 7,85% |
| `zlKBfNzfh50` | "chuyện người làm nghề" — digital marketing, SEO | 248 | 1,09 | 2,41% |

Val cũ là 4 cuộc, 365 segment, 0,79 h, chủ đề trùng test gần như hoàn toàn và 253 segment
có từ ngoại lai. Val mới: **1.179 segment, 4,40 h, density 7,12%, 791 segment có từ ngoại
lai** — density test là **7,06%**, tức val giờ đo trên cùng nồng độ code-switch với bộ sẽ
chấm ở §3b, và đủ segment để ValCER thôi trồi sụt vài điểm giữa các vòng.

Hai cuộc đầu là code-switching mức câu **ngoài domain công nghệ** — v6-corpus không có
cuộc nào như vậy trong val trước đây, mà cả test lẫn cross-domain-bench đều có. Ba cuộc
marketing khác (`W71X_KzfJg8`, `Kltz8ZJ9HOw`, `lSlkfTiWKBk`) **ở lại train**: density
1,3–1,8%, thêm vào thì 3,0 h của chúng chiếm 87% khối token của val và kéo density gộp
xuống 2,80%.

Chọn checkpoint vẫn bằng retention cross-domain ở §3a, không bằng ValCER —
`checkpoints/best/` là "tốt nhất trên ValCER", mà ValCER không nhìn thấy retention.

Không chạy `sweep-gate`. Không cổng tự động. λ chốt cứng, benchmark decode cả 1,0 và 0,75.

```bash
R=v6-100h-steps
OV="--override run_id=$R
    --override data.dataset_path=dataset/v6-corpus
    --override training.epochs=2
    --override data.real_bench_path=null
    --override data.ood_eval_path=null
    --override data.val_meetings=[paid_meeting_0001,paid_meeting_0002,paid_meeting_0011,rCd8DSMk3-c,zlKBfNzfh50,coteccons-agm-2025,jB1P4bqLwDY]
    --override data.cross_domain_path=dataset/cross-domain-bench
    --override training.learning_rate=2.0e-4
    --override training.batch_size=16 --override training.grad_accum_steps=1
    --override training.gradient_checkpointing=false
    --override training.eval_steps=400
    --override training.val_limit=null
    --override training.early_stopping_patience=null
    --override eval.batch_size=64"

python -m src.pipeline --stage prepare $OV || exit 1
python -m src.pipeline --stage train   $OV || exit 1
```

Run đứt: `--stage train --resume`. **Đừng xoá `checkpoints/`.**

---

## 3. Chấm điểm — hai giai đoạn

Chấm đủ 6 suite cho 4 checkpoint là không vừa ngân sách. Nên: **lọc bằng cross-domain
trước, benchmark đầy đủ chỉ cho bản thắng.**

### 3a. Sàng — cross-domain, 4 checkpoint (~10 phút)

`scripts/score_cross_domain_model.py`, cùng đường decode chống lặp, `eval.batch_size=64`:

```bash
for S in 400 800 1600 5350; do
  PYTHONPATH=. .venv/bin/python -m scripts.score_cross_domain_model \
      --model vinai/PhoWhisper-large \
      --adapter outputs/v6-100h-steps/checkpoints/step-$S \
      --cross-domain-path dataset/cross-domain-bench \
      --batch-size 64 \
      --out outputs/v6-100h-steps/metrics/cross_domain.step-$S.json || exit 1
done
```

- `step-400` ← **sớm hơn** v5. Nếu §0 đúng thì đây là retention cao nhất cả lượt
- `step-800` ← đúng ngân sách bước của v5 (801), trên corpus gấp 12 lần. Đặt cược ở đây
- `step-1600` ← điểm giữa, để biết đường cong dốc hay phẳng
- `step-5350` ← bước cuối, số professor cần

13 checkpoint còn lại vẫn nằm trên đĩa. Nếu bốn điểm này lộ ra một đỉnh thì chấm thêm
hàng xóm của nó — 2,5 phút một điểm, không cần train lại gì.

Mỗi lần ~2,5 phút, ghi `cer` / `retention` / `no_loanword_cer` trên 299 segment. Đây là
4 điểm của đường cong retention-theo-bước — **deliverable chính của lượt này**, có ra
adapter tốt hay không cũng phải lấy được.

Mốc so, v5 dưới đúng đường decode này: **0,0740 CER / 0,6648 retention / 0,0613
no-loanword CER**.

### 3b. Benchmark đầy đủ — chỉ bản thắng (~30 phút)

Bản có retention cao nhất ở 3a đi qua `scripts/benchmark_run.py` trên **5 suite**, để có
bảng tổng cho 17/09:

```bash
WIN=step-800                     # thay bằng bản thắng ở 3a
python -m scripts.benchmark_run --backend hf \
    --model vinai/PhoWhisper-large \
    --adapter outputs/v6-100h-steps/checkpoints/$WIN \
    --suites cross-domain,youtube-test,synthetic-test,vimedcss-test,vimedcss-hard \
    --path cross-domain=dataset/cross-domain-bench \
    --path youtube-test=dataset/v6-corpus \
    --path synthetic-test=dataset/v6-corpus \
    --batch-size 8 \
    --out Outputs/bench-v6-100h-steps
```

`--adapter` là λ=1,0 (adapter chưa scale). Muốn cột λ=0,75 thì `src.lora.save_with_lambda`
ghi ra một bản đã bake λ rồi trỏ `--adapter` vào bản đó — đúng cách `v6-ondomain-15h` đã
làm (`cross_domain_lambda075.json` ghi `"adapter": "/tmp/lam075"`).

Quy mô: 299 + 228 + 426 + 1.612 + 658 = **3.223 segment**.

**Bỏ suite `vivos`** — ref của nó viết HOA toàn bộ còn hyp viết thường, nên 81% CER là
hiện vật của quy ước chữ hoa, không phải phép đo. Không đưa dòng đó lên bảng.

**Không phải decode lại các hệ để so.** `Outputs/benchmark-2026-09-10 (3)/` đã có sẵn
per-segment của `winhsss_reworkwhisper_large_v5`, `scribe_v2` và
`vinai_phowhisper_large` trên **đủ cả 6 suite**. `scripts/benchmark_report.py` nối theo
`segment_id`, chỉ model mới cần decode.

Chấm: `python -m scripts.benchmark_report` trên thư mục gộp → bảng CER/WER, khoảng tin
cậy bootstrap, retention, và paired bootstrap dCER so với base.

Hai mục tiêu tiên quyết mà bảng phải trả lời: **ViMedCSS** (`-test` + `-hard`) và
**youtube-test so với ElevenLabs Scribe**.

**`--batch-size 8`** ở đây, không phải 64 — để cùng cấu hình với cột v5 đã đo. CER lệch
theo batch, đổi batch là làm hỏng khả năng so cột.

---

## 4. Ngân sách 3 giờ

| hạng mục | phút |
|---|---:|
| dựng máy + tải `v6-corpus.tar` (14,7 GB) + add-on (1,57 GB), có `hf_transfer` | 30–45 |
| lượt train tí hon §1.1 | 3 |
| train 5.350 bước × 0,728 s | 65 |
| eval val 13 vòng × 1.179 segment | 15 |
| 3a: sàng cross-domain, 4 checkpoint | 10 |
| 3b: benchmark đầy đủ 3.223 segment, bản thắng | 30 |
| đẩy adapter cuối lên Hub + scp `metrics/`, `run.log`, bench về | 3 |
| **tổng** | **156–171** |

Vừa trần 180 phút, dư 9–24 phút. **2 epoch là bắt buộc, không phải lựa chọn:** 3 epoch là
8.025 bước, cộng 20 vòng eval thành 164 phút tính từ lúc train chạy, tức phải tải xong
16 GB trong 16 phút. Không xảy ra.

Hai khoản đã cắt và không mất gì: `hf_transfer` cộng việc tải chồng lấn với `pip install`
(§2 của guide), và watcher đẩy checkpoint trong lúc train nên §8 chỉ còn đẩy bước cuối.

**Biến số lớn nhất vẫn là tốc độ tải corpus**
— 15 phút chênh lệch ở dòng đầu bảng lớn hơn mọi thứ khác cộng lại. Thứ tự cắt nếu đồng hồ
chạy chậm hơn dự kiến:

1. Sàng 3 checkpoint thay vì 4 (bỏ `step-1600` — điểm giữa, mất ít thông tin nhất vì
   `step-400`/`step-800`/`step-5350` đã kẹp cả hai đầu) — **−2,5 phút**.
2. Bỏ `vimedcss-test` (1.612 segment, suite lớn nhất) khỏi 3b, giữ `vimedcss-hard` —
   **−15 phút**. Mất một cột của bảng summit, nên là lựa chọn cuối.
3. **Không bao giờ cắt 3a.** Đường cong retention là lý do lượt này tồn tại; nếu chỉ kịp
   một thứ thì đó là thứ phải kịp.

**Chép kết quả về trước khi trả máy** — đã mất một lượt đo vì chép muộn.

---

## 5. Dự đoán, ghi trước để sau còn đối chiếu

- Checkpoint 5.350 bước: retention **thấp hơn** 0,486 của `v6-corpus-r32`.
- Checkpoint 800 bước: **cao nhất trong bốn**, có cơ hội vượt 0,62.
- Đường cong retention theo bước: **giảm đơn điệu**.

Nếu đường cong **không** giảm thì chẩn đoán sai — và ta biết ngay trong lượt thuê này
thay vì lượt sau.

## 6. Đọc kết quả

| kết quả | đọc là gì |
|---|---|
| đường cong giảm đơn điệu | chẩn đoán đúng. Đòn bẩy là **điểm dừng**, không phải corpus. Lượt sau: dừng sớm theo retention, và sửa `checkpoints/best` để chọn theo retention thay vì ValCER |
| step-800 ≥ 0,6648 | có model thay được v5. Chấm nốt các checkpoint lân cận để tìm đỉnh |
| đường cong phẳng | số bước không phải biến. Quay lại H3 (nhãn máy) và H4 (audio TTS) trong `docs/plan-cat-pipeline-va-so-thi-nghiem.md` |
| đường cong tăng | chẩn đoán ngược dấu. Dừng mọi suy luận dựa trên nó, viết lại từ đầu |

Cổng: không có cổng tự động. Người đọc số rồi quyết push. Nếu không checkpoint nào vượt
v5 thì bảng 17/09 trình v5 + phân tích này.
