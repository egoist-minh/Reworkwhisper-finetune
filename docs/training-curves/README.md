# Training curves + λ tradeoff — bốn lần fine-tune

Model gọi bằng **đúng tên trên HuggingFace**, không dùng nhãn "v1/v2/v3" — nhãn đó đá nhau với số
của HF (`v1 → v3 → v4`), với run ID (`dot1-v1-lora` / `v1c-r16-valfix` / `v3-r16`) và với tên
dataset (`paid-dataset-v2`).

Lần 4 **chưa publish** nên gọi bằng run ID `v4-mixed-r16`. Đừng gọi tắt là "v4": tên
`Reworkwhisper-large-v4` đã thuộc về lần 3.

Trình bày theo **hai bước, đúng thứ tự này**:

1. **Training curve** — model có học không: train loss giảm, CER giảm.
2. **Tradeoff curve** — chọn λ bao nhiêu và trả giá gì.

Training curve của lần 3 **có** đường OOD. Nó ở λ=1.0 (adapter chưa scale) nên kết ở 4.20% so với
base 2.28% — nhìn trần ra là quên nặng, trong khi slide kết quả báo 2.49% và đánh "✓ không quên".
Đừng để người xem tự đọc con số đó: nói ngay rằng nó là λ=1.0, rồi chuyển sang tradeoff curve để
giải thích. Đường OOD chính là lý do phải có bước 2, không phải lỗi cần che.

| Lần | Training curve | Tradeoff curve | Model xuất bản |
|---|---|---|---|
| 1 · 23/07 | `reworkwhisper-large-v1_training-curve.png` | — (không sweep λ) | `reworkwhisper-large-v1` |
| 2 · 30/07 | `reworkwhisper-large-v3_training-curve.png` | `reworkwhisper-large-v3_lambda-tradeoff.png` | `reworkwhisper-large-v3-0.5lamda` |
| 3 · 04/08 | `Reworkwhisper-large-v4_training-curve.png` | `Reworkwhisper-large-v4_lambda-tradeoff.png` | `Reworkwhisper-large-v4` |
| 4 · 17/08 | `v4-mixed-r16_training-curve.png` | `v4-mixed-r16_lambda-tradeoff.png` + `v4-mixed-r16_lambda-by-lambda.png` | chưa publish |

Ảnh của lần 1 và 2 copy từ `D:\phowhisper-finetune-exp\outputs\<run>\`. Ảnh của lần 3 và 4 sinh bằng
`scripts/plot_training_curve.py` và `scripts/plot_lambda_tradeoff.py`.

**5 ảnh của lần 1–3 đã nhúng vào `docs/finetune-slides.html`** (deck 18 slide), mỗi ảnh kèm chú
thích rút từ file này. Đường dẫn ảnh trong slide là tương đối (`training-curves/*.png`) nên phải
giữ nguyên vị trí thư mục này cạnh file HTML. **3 ảnh của lần 4 chưa vào deck** — hiện chỉ nhúng
trong `docs/finetune-results-report-v4-mixed-r16.md`.

Tên file ảnh gắn với **model xuất bản** của lần đó cho dễ tìm, nhưng nội dung mọi curve là **λ=1.0**
— xem mục dưới.

---

## λ nằm ở đâu trong quy trình

λ **không phải hyperparameter train**. Thứ tự thực tế:

1. `run_train` train adapter ở biên độ gốc. Scaling nội tại của LoRA là `alpha/r`
   (rslora: `alpha/sqrt(r)`), [`src/lora.py:52-57`](../../src/lora.py#L52-L57). Không có λ ở đây.
2. Train xong mới tới stage `sweep-gate`: `set_lambda(model, lam)` nhân `layer.scaling` với λ
   rồi decode lại, [`src/pipeline.py:239-246`](../../src/pipeline.py#L239-L246).

Hệ quả cho hai loại đường trên training curve:

- **Train loss**: λ-independent. Không phải "đo ở λ=1", mà là λ chưa tồn tại. Về số thì nó là
  loss của adapter chưa scale, tức trùng model λ=1.0.
- **Các điểm CER**: đúng là λ=1.0 — eval callback decode model đang train, chưa scale.

Nếu bị hỏi "sao không có training loss ở λ=0.5": không có khái niệm đó. λ=0.5 là phép scale
hậu-train, không sinh ra quỹ đạo loss nào. Muốn 1 con số loss ở λ=0.5 thì phải forward lại
toàn bộ train set với trọng số đã scale — ra một điểm, không phải curve, và chưa ai làm.

---

## Bước 1 — Training curve

### Lần 1 — `reworkwhisper-large-v1_training-curve.png`

- Đường đỏ ghi "eval CER%" nhưng **thực chất là test CER**: `dataset-dot1` dùng chung một file
  cho val và test (500 segment). Đường cam là WER cùng tập đó. Trục phải: train loss, linear,
  log mỗi 25 step → 5 điểm.
- λ=1.0 — trùng đúng model xuất bản, nên **điểm cuối 18.53% = số "sau" trên slide kết quả**.
  Nguồn: `dot1-v1-lora/benchmark_results.csv` (n=500).
- **Không có base trên ảnh.** Số "trước" của slide (29.65%) là base; curve bắt đầu ở epoch 1
  đã là sau train (23.93%).
- VIVOS 1.78% → 4.42% trên slide không nằm trên curve — lần này không eval VIVOS trong lúc train.
  Hai số đó từ kernel Kaggle `winhkento/benchmark-asr`, **không có file raw trong repo**.

### Lần 2 — `reworkwhisper-large-v3_training-curve.png`

- "val CER%" / "val WER%" — tập **val 250 segment**, tách rời test (đây là "valfix": lần 1 không
  tách). Trục phải: train loss, linear, log mỗi 25 step.
- λ=1.0, tức đúng model `reworkwhisper-large-v3`. Model xuất bản là
  `reworkwhisper-large-v3-0.5lamda` (λ=0.5) → **không có số nào trên ảnh trùng slide kết quả**.
- `best val CER 1.4%` trên ảnh **không phải** `1.65` trên slide: khác tập (val 250 câu vs test
  236 câu), khác λ (đồ thị 1.0, slide 0.5). Gần nhau chỉ là trùng hợp.
- Dừng ở step 175/246 (epoch 2.11) do early stopping — cosine scheduler chưa anneal xong.
- Run này **mất raw log**, chỉ còn PNG. Các số val CER theo step gõ tay trong
  `phowhisper-finetune-exp/docs/v1c-paid-lora-report.md` §5.3.

### Lần 3 — `Reworkwhisper-large-v4_training-curve.png`

- Cùng bố cục twin-axis như hai lần trước: trục trái error %, trục phải train loss.
- Trục phải: train loss **log scale** (loss chạy 31 → 0.06; linear thì bước 1 nuốt hết phần còn
  lại). Mean theo block 25 step → 33 điểm, cùng độ thô với hai lần trước (`--every` đổi được).
- Đường đỏ: val CER 1.19% → 1.05% → 1.01% (val 250 segment, eval mỗi epoch).
- Đường cam: **OOD (VIVOS) CER — không phải WER.** 5.59% → 4.94% → 4.20%. Khác nghĩa với đường
  cam của hai lần trước (kia là WER). Xem cảnh báo ở đầu file trước khi trình bày. Bỏ được bằng
  `--no-ood` nếu cần một bản không có nó.
- Không có val WER: entry eval val trong `trainer_state.json` là dict rỗng, `training.csv`
  không có cột WER.
- Kiểm tra nhất quán: `lambda_sweep.csv` tại λ=1.0 cho val 1.023% / OOD 4.143%; curve kết ở
  1.010% / 4.197%. Lệch nhỏ vì gate decode lại, không phải sai số liệu.

### Lần 4 — `v4-mixed-r16_training-curve.png`

- Cùng bố cục và cùng script như lần 3, `--every 25` → 36 điểm train loss trên 885 step.
- Đường đỏ: val CER 3.98% → 3.24% → 3.23% (val 365 segment = 250 giả lập + 115 YouTube).
- Đường cam: OOD (VIVOS) CER 5.08% → 4.80% → 4.67%, **ở λ=1.0** — cảnh báo đầu file áp nguyên
  vào lần này. Bản λ=0.25 xuất xưởng cho 2.26%, tức gần bằng base 2.28%.
- **Đường tím (val loss) là điểm mới đáng nhìn**: 0.1303 → 0.1192 → **0.1246**, quay đầu ở
  epoch 3 trong khi train loss vẫn xuống tới 0.0035. Overfit nhẹ. Không có early stopping —
  `checkpoints/best/` trùng byte với `checkpoint-885/`.
- Kiểm tra nhất quán: curve kết ở OOD 4.674% còn `lambda_sweep.csv` tại λ=1.0 cho 4.257%.
  Lệch 0.42pp vì hai đường decode khác nhau (Trainer `predict_with_generate` vs `_eval_split`).
  Đừng trộn hai cột.

---

## Bước 2 — Tradeoff curve

### Lần 2 — `reworkwhisper-large-v3_lambda-tradeoff.png`

Grid λ = `[0, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]` cộng một điểm `attn-λ1.00` (chỉ scale
attention, bỏ fc1/fc2). Trục x = CER val, trục y = CER VIVOS, góc dưới-trái là lý tưởng.
Ngân sách OOD **+0.2pp** so với base 2.16% → trần 2.37%.

λ=0.5 cho val 2.54% / VIVOS 2.21% — nằm ngay dưới trần. λ=0.6 đã vượt trần (2.49%). λ=1.0 bật
lên 4.32%, tức ×2 base. **Khác lần 3 ở chỗ này**: ở lần 2 ngân sách thực sự chặn, không chỉ luật
khuỷu chặn.

### Lần 3 — `Reworkwhisper-large-v4_lambda-tradeoff.png`

Grid thưa hơn: `[0, 0.25, 0.5, 0.75, 1.0]` (`configs/experiment.yaml` → `sweep.lambdas`).
Ngân sách OOD **+2.0pp** so với base 2.28% → trần 4.28%.

| λ | val CER | OOD CER | Δval so bước trước | ΔOOD so bước trước | Giá / 1pp lợi |
|---|---|---|---|---|---|
| 0.0 (base) | 5.83% | 2.28% | — | — | — |
| 0.25 | 2.56% | 2.34% | −3.27pp | +0.06pp | 0.017 |
| **0.5** | **1.60%** | **2.49%** | −0.96pp | +0.15pp | 0.155 |
| 0.75 | 1.20% | 3.26% | −0.40pp | +0.77pp | 1.95 |
| 1.0 | 1.02% | 4.14% | −0.18pp | +0.88pp | 4.98 |

Điểm cần nói rõ với giáo sư: **cả 5 mức λ đều nằm trong ngân sách OOD** (4.14% < 4.28%). Nên
luật cũ "lấy λ lớn nhất còn trong ngân sách" chọn λ=1.0 — đúng luật nhưng sai tinh thần, vì
bước 0.5 → 1.0 chỉ mua thêm 0.58pp lợi trong miền mà trả 1.65pp OOD. Luật hiện tại là
cost/benefit có ngưỡng khuỷu (`elbow_ratio_threshold: 10.0`,
[`src/pipeline.py:32-80`](../../src/pipeline.py#L32-L80)): tỉ số ở bước 0.5→0.75 là 1.95, vượt
quá 10× tỉ số bước trước (0.155) → chặn, **λ=0.5 thắng**.

`--chosen` của script không tự tính lại luật này — `src.pipeline.select_lambda` là nơi duy nhất
sở hữu nó.

### Lần 4 — hai ảnh, `v4-mixed-r16_lambda-tradeoff.png` và `v4-mixed-r16_lambda-by-lambda.png`

Cùng grid `[0, 0.25, 0.5, 0.75, 1.0]` và cùng ngân sách **+2.0pp** như lần 3 (base OOD 2.283%
→ trần 4.283%). Hai khác biệt về hình:

- `lambda_sweep.csv` của run này **có thêm hai cột** `val_cer_synthetic` / `val_cer_youtube`, nên
  cả hai ảnh vẽ ba đường val thay vì một. Script tự phát hiện; sweep nào thiếu cột thì vẫn ra một
  đường như cũ.
- Ảnh thứ hai dùng `--style by-lambda` (λ trên trục x). Ảnh scatter không cho thấy trực tiếp "lát
  nào kéo λ\*"; ảnh này thì có.

| λ | val CER | OOD CER | Δval so bước trước | ΔOOD so bước trước | Giá / 1pp lợi |
|---|---|---|---|---|---|
| 0.0 (base) | 10.34% | 2.283% | — | — | — |
| **0.25** | **5.48%** | **2.261%** | −4.85pp | **−0.02pp** | **−0.0046** |
| 0.5 | 3.64% | 2.619% | −1.85pp | +0.36pp | 0.193 |
| 0.75 | 3.32% | 3.343% | −0.32pp | +0.72pp | 2.26 |
| 1.0 | 3.26% | 4.257% | −0.06pp | +0.91pp | 15.4 |

**Cột cuối là chỗ luật elbow chạm trường hợp biên.** Bước đầu làm OOD CER *giảm*, nên giá/lợi
**âm**; ngưỡng `prev_ratio × 10` vì thế cũng âm (−0.046) và bước 0.25→0.5 (0.193) vượt ngưỡng
ngay. λ=0.5 bị loại không phải vì nó là khuỷu thật — khuỷu thật nằm ở 0.5→0.75, nơi tỉ số nhảy
từ 0.193 lên 2.26. Cần chặn `prev_ratio` ở một mức dương tối thiểu; chi tiết trong
`docs/finetune-results-report-v4-mixed-r16.md` §4.

Ghi chú về màu: vạch ngân sách OOD đổi từ đỏ `#c9435b` sang tím `#8e44ad` (2026-08-17), vì run
này thêm đường `val youtube` **cũng** màu đỏ đứt nét — trùng cả màu lẫn kiểu nét, đọc ra thành
bốn đường sweep. Ảnh lần 2 và 3 sinh trước thay đổi này nên vạch ngân sách của chúng vẫn đỏ.

---

## Số nào ở mức λ nào (lần 3)

| Tập | Base | λ=1.0 (= training curve) | λ=0.5 (= slide, `Reworkwhisper-large-v4`) |
|---|---|---|---|
| val, 250 seg | 5.83% | 1.02% | 1.60% |
| VIVOS OOD, 760 seg | 2.28% | 4.14% | 2.49% |
| test, 426 seg | 4.26% | 1.71% | **1.96%** |
| họp thật, 196 parent-segment | 31.57% | 35.07% (fail) | **29.06%** (INCONCLUSIVE) |

Nguồn: `Outputs/v3-r16/metrics/{lambda_sweep.csv,baseline.json,gate_results.json}` và
`Outputs/v3-r16_lambda0.5/outputs/v3-r16/metrics/gate_results_lambda0.5.json`.

Dòng "họp thật" ở λ=0.5 tốt hơn base nhưng verdict vẫn **INCONCLUSIVE** — 2 bản ghi, CI rộng.
Đừng trình bày như một chiến thắng.
