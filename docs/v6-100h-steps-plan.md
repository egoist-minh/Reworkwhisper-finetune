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

6.558 bước = **8 lần** ngân sách bước của v5. Lưu checkpoint dọc đường thì một lượt train
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

Adapter rank 16 là 115.487.384 byte (đo trên `Outputs/v3-r16/adapter/`), nên 16 vòng là
**~1,85 GB**. `archive_run` loại thư mục tên `checkpoint-<số>` (optimizer state của
Trainer) — `step-<số>` **không** khớp bộ lọc đó nên tự động đi vào zip. Zip to thêm ~1,85 GB
(safetensors không nén, `archive_run` chạy compresslevel=1); đó là giá của việc có gì để
chấm, và scp vẫn nằm trong 10 phút đã tính ở §4.

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

### 1.3 Tắt được early stopping — nếu không, lượt này có thể tự dừng trước bước 6.400

`_EarlyStoppingState(patience=3)` trước đây ghim cứng trong [src/train.py](../src/train.py),
và nó đọc **đúng cái ValCER mà §2 vừa hạ xuống 100 segment**. Ba vòng eval liên tiếp không
cải thiện là `control.should_training_stop = True`.

Đây không phải rủi ro lý thuyết. ValCER hai lượt gần nhất (val đầy đủ 365 segment):

| | vòng 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| `v6-ondomain-15h` | 0,04355 | 0,03865 | 0,03625 | 0,03554 | 0,03540 |
| `v6-ondomain-29h` | 0,05758 | 0,04428 | 0,04064 | 0,03877 | 0,03524 |

Đơn điệu giảm, nhưng khoảng cách vòng 4→5 của `-15h` chỉ **0,014 pp**. Cắt val còn 100
segment (27% số segment) thì nhiễu lớn hơn ~1,9 lần, thừa sức nuốt một khoảng cách cỡ đó —
và lượt này có 16 vòng, tức plateau cuối dài hơn hẳn. Dừng sớm ở bước ~3.200 là mất **cả
hai** deliverable: số cuối cho professor, và hai điểm cuối của đường cong retention.

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
| `training.epochs` | 3 | như v5 |
| `learning_rate` | 2.0e-4 | như v5 |
| `batch` × `grad_accum` | 16 × 1 | batch hiệu dụng 16, bằng v5 |
| `gradient_checkpointing` | false | chỉ T4 cần |
| `init_adapter` | null | LoRA tươi. Curriculum hai pha đã đo và thua (`v6-dense-{1,3,4}`) |
| `eval_steps` | **400** | 16 vòng trên 6.558 bước; mỗi vòng lưu một adapter |
| `training.val_limit` | **100** | vòng eval chỉ còn tồn tại để **kích hoạt việc lưu checkpoint**. Chọn checkpoint giờ theo retention cross-domain, không theo ValCER — nên không cần decode đủ 365 segment val mỗi vòng. Cắt mỗi vòng từ ~2,5 phút xuống ~0,7 phút |
| `eval.batch_size` | 64 | CER lệch theo batch — mọi phép đo trong lượt phải cùng số này |
| `data.ood_eval_path` | null | không stage nào đọc nữa sau §1.3 của plan kia |
| `early_stopping_patience` | **null** | §1.3. Không có dòng này thì ValCER 100 segment có quyền kết thúc lượt train |
| corpus | `dataset/v6-corpus` | **34.972** segment train → **2.186** bước/epoch |

Số corpus đo lại 15/09 bằng `load_manifests` + `resolve_splits` với `data.val_meetings`
hiện tại của `configs/experiment.yaml`, không lấy từ card HF: **train 34.972 / val 365 /
test 654**. Card ghi 34.525 — chênh 447 segment, đúng kiểu sai lệch đã ghi nhận giữa card
và manifest. Hệ quả: **6.558 bước** (không phải 6.474), tức +84 bước ≈ +1 phút. 2.186
bước/epoch khớp đúng con số của `v6-corpus-r32` trong bảng §0, nên hai lượt so được.

Mười sáu vòng eval rơi vào **400, 800, 1.200, … , 6.400** (bội số của 400). Bước cuối
6.558 không có vòng eval, nên **checkpoint muộn nhất là `step-6400`**, không phải bước cuối
cùng; chênh 158 bước, nói đúng tên khi trình số.

`eval_steps=400` thay vì 800 mua hai thứ, giá 6 phút eval và ~0,9 GB đĩa:

- **`step-400` và `step-800` kẹp hai bên ngân sách 801 bước của v5.** Với 800 thì chỉ có
  một điểm trùng v5; với 400 thì trả lời được cả câu "đỉnh có nằm SỚM hơn v5 không" — mà
  theo chính chẩn đoán ở §0 (retention giảm đơn điệu theo bước) thì `step-400` phải là
  điểm cao nhất của cả lượt.
- Độ phân giải gấp đôi ở đoạn dốc nhất của đường cong, tức đoạn 400–1.600, nơi ba lượt cũ
  (801 / 1.317 / 2.157 bước) nằm chen nhau.

`data.val_meetings` **giữ nguyên 4 cuộc mặc định**, không chạy `scripts/select_val_meetings.py`.
Lý do: val lớn hơn chỉ để đọc xu hướng ValCER, mà lượt này không chọn checkpoint bằng
ValCER và (sau §1.3) cũng không dừng bằng nó. Tiết kiệm cả bước dựng lẫn thời gian decode.

Không chạy `sweep-gate`. Không cổng tự động. λ chốt cứng, benchmark decode cả 1,0 và 0,75.

```bash
R=v6-100h-steps
OV="--override run_id=$R
    --override data.dataset_path=dataset/v6-corpus
    --override data.real_bench_path=null
    --override data.ood_eval_path=null
    --override data.cross_domain_path=dataset/cross-domain-bench
    --override training.epochs=3
    --override training.learning_rate=2.0e-4
    --override training.batch_size=16 --override training.grad_accum_steps=1
    --override training.gradient_checkpointing=false
    --override training.eval_steps=400
    --override training.val_limit=100
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
for S in 400 800 1600 6400; do
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
- `step-6400` ← checkpoint muộn nhất có eval, số professor cần (lượt kết thúc ở 6.558)

12 checkpoint còn lại vẫn nằm trong zip. Nếu bốn điểm này lộ ra một đỉnh thì chấm thêm
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
| dựng máy + tải `v6-corpus.tar` (14,7 GB, private HF) | 30–45 |
| train 6.558 bước × 0,728 s | 80 |
| eval val 16 vòng × 100 segment | 12 |
| 3a: sàng cross-domain, 4 checkpoint | 10 |
| 3b: benchmark đầy đủ 3.223 segment, bản thắng | 30 |
| scp `outputs/v6-100h-steps/` + `Outputs/bench-v6-100h-steps/` về | 10 |
| **tổng** | **171–186** |

Vượt trần 180 phút ở đầu bi quan (186), và **biến số lớn nhất vẫn là tốc độ tải corpus**
— 15 phút chênh lệch ở dòng đầu bảng lớn hơn mọi thứ khác cộng lại. Thứ tự cắt nếu đồng hồ
chạy chậm hơn dự kiến:

1. Sàng 3 checkpoint thay vì 4 (bỏ `step-1600` — điểm giữa, mất ít thông tin nhất vì
   `step-400`/`step-800`/`step-6400` đã kẹp cả hai đầu) — **−2,5 phút**.
2. Bỏ `vimedcss-test` (1.612 segment, suite lớn nhất) khỏi 3b, giữ `vimedcss-hard` —
   **−15 phút**. Mất một cột của bảng summit, nên là lựa chọn cuối.
3. **Không bao giờ cắt 3a.** Đường cong retention là lý do lượt này tồn tại; nếu chỉ kịp
   một thứ thì đó là thứ phải kịp.

**Chép kết quả về trước khi trả máy** — đã mất một lượt đo vì chép muộn.

---

## 5. Dự đoán, ghi trước để sau còn đối chiếu

- Checkpoint 6.400 bước: retention **thấp hơn** 0,486 của `v6-corpus-r32`.
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
