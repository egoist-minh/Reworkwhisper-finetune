# Kế hoạch v6 pha 2 — curriculum hai pha trên `v6-corpus`

> ⚠ **§4–§7 của file này đã bị thay bởi `docs/v6-density-ladder-plan.md`** (13/09/2026).
> Mục tiêu đổi từ "ngang v5" sang "vượt Scribe v2", và pha 2 đổi từ một lần chạy ở mật độ
> 6,51% thành bốn lần chạy trên thang mật độ 6,5–14,1%. §0–§3 dưới đây vẫn đúng.
>
> File này trước mang tên `docs/v7-curriculum-plan.md` và gọi run là "v7". Sai tên: nó
> chạy tiếp từ adapter v6, trên tập con corpus v6, cùng base model và cùng rank — là pha 2
> của v6, không phải thế hệ mới. Xem `docs/v6-density-ladder-plan.md` §0.

Chạy trên server GPU thuê, theo quy trình chung ở `docs/server-finetune.md`. File này chỉ
ghi phần khác recipe chung. Số liệu đo lại từ artifact của run `v6-corpus-r32`
(`Outputs/v6-corpus-r32.zip`) và manifest của nó, không trích từ bảng tổng hợp.

## 0. Ý tưởng và lý do

`v6-corpus-r32` train 101,24 giờ ở mật độ token ngoại lai **2,48%**, trong khi mọi bộ đo
thật nằm ở 5,8–8,7% và corpus của v5 ở 7,13%. Kết quả: phần tiếng Việt tốt lên (VIVOS
2,82% so với 3,34% của v5; CER trên segment không chứa từ ngoại lai 5,86% so với 6,06%)
nhưng hành vi giữ từ ngoại lai sụp (retention cross-domain 48,6% so với 66,6%).

Bằng chứng đây là prior của decoder chứ không phải lỗ hổng từ vựng: **85,0%** số lần xuất
hiện từ ngoại lai của cross-domain-bench đã có trong nhãn train của v6, và retention của v6
leo theo tần suất (24,4% với từ chưa gặp → 62,0% với từ xuất hiện > 100 lần) trong khi v5
phẳng ~70% ở mọi mức. v6 chỉ giữ được thứ đã thuộc lòng; v5 giữ theo quy tắc.

Curriculum tách hai thứ đó ra hai pha thay vì bắt một phân phối duy nhất phục vụ cả hai:

| Pha | Dữ liệu | Mục đích |
|---|---|---|
| 1 | 101,24 h, mật độ 2,48% | mô hình hoá âm học/tiếng Việt trên khối lượng lớn |
| 2 | 39,76 h lọc, mật độ 6,51% | nắn lại prior về tỉ lệ 1 token ngoại lai trên 15 token, khớp 1:14 của v5 |

Không vứt giờ dữ liệu nào. Khác với phương án lọc thẳng, pha 1 vẫn giữ nguyên 61,5 h tiếng
Việt thuần — chính phần đã mua được VIVOS 2,82%.

## 1. Pha 1 đã chạy xong — không train lại

Pha 1 **chính là** run `v6-corpus-r32`: 1 epoch, 101,24 h, rank 32 / alpha 64, batch 16.
`checkpoints/best/` trong `Outputs/v6-corpus-r32.zip` là checkpoint step 1820/2186
(val CER 0,04646, tốt nhất trong 6 lượt eval). Đây là điểm xuất phát hợp lệ cho pha 2.

⚠ Phải nạp từ **`checkpoints/best/`**, không phải `adapter/`. Thư mục `adapter/` đã bị
`save_with_lambda` nướng λ=0,5 vào `adapter_config.json:lora_alpha`
(`src/lora.py:save_with_lambda`); nạp nó rồi train tiếp là train trên một adapter đã bị
thu nhỏ một nửa.

⚠ Pha 2 kế thừa **rank 32 / alpha 64** từ adapter của pha 1 — `cfg.lora.rank` bị bỏ qua ở
nhánh nạp adapter. Nghĩa là thiết kế này **không** trả lời được câu hỏi rank 16 vs rank 32;
muốn trả lời thì phải chạy lại cả pha 1, tốn thêm ~35 phút GPU. Quyết định trước khi thuê
máy, không phải sau.

## 2. Thay đổi mã nguồn — làm xong trước khi thuê máy, 0 giờ GPU

### 2.1 `training.init_adapter` — nạp adapter có sẵn thay vì tạo LoRA mới

`src/config.py`, trong `class Training`:

```python
init_adapter: str | None = None   # path to an existing PEFT adapter dir to continue
                                   # training from (curriculum phase 2). null = fresh
                                   # LoRA from cfg.lora. Must be an UNSCALED adapter
                                   # (checkpoints/best/), never one save_with_lambda wrote.
```

`src/train.py`, thay dòng 222 (`model = get_peft_model(...)`):

```python
if cfg.training.init_adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, cfg.training.init_adapter,
                                      is_trainable=True)
else:
    model = get_peft_model(base_model, build_lora_config(cfg))
```

Verify: `model.print_trainable_parameters()` phải in đúng số tham số của rank 32, và một
forward pass trước khi train phải cho CER ngang `v6-corpus-r32` chứ không ngang base model.

### 2.2 Sàn retention trong `select_lambda`

`src/pipeline.py:34`. Hiện chỉ ràng buộc val CER + VIVOS, nên sweep của v6 chọn λ=0,5 mà
không hề thấy retention rơi. Thêm `sweep.retention_floor` (mặc định `null` = giữ nguyên
hành vi cũ): loại mọi λ có retention dưới sàn **trước khi** xét elbow.

Verify: chạy `select_lambda` trên `Outputs/v6-corpus-r32/metrics/lambda_sweep.csv` với sàn
0,666 → phải trả về λ khác 0,5 hoặc raise.

### 2.3 Tier gate trên cross-domain-bench

`src/gate.py`. Thêm một tier đo trên `dataset/cross-domain-bench` (299 segment, 1,30 h),
hai ràng buộc:

```
CER       ≤ 8,19%    (mức v5)
retention ≥ 66,6%    (mức v5)
```

Verify: chạy gate mới trên adapter `v6-corpus-r32` hiện có → **phải FAIL** cả hai. Nếu pass
thì gate viết sai, dừng lại sửa.

### 2.4 `scripts/filter_corpus_density.py`

Lọc **chỉ split train**, giữ nguyên val/test để số của gate còn so được với v5 và v6.

```
--src dataset/v6-corpus --out dataset/v6-corpus-dense --min-foreign 1
```

Giữ segment có ≥ 1 token vượt phép thử hình dạng âm tiết tiếng Việt
(`src.normalize.is_vietnamese_shaped`, cùng bộ lọc mà `src.metrics.english_token_retention`
dùng, để ba con số còn so được với nhau). `dataset/v6-corpus-dense/audio` là symlink tới
`../v6-corpus/audio` — không nhân đôi 14,7 GB.

Verify: script in ra **12.880 segment / 39,76 h / mật độ 6,51%**, val 365 và test 654
nguyên vẹn. Lệch số này là lọc sai.

## 3. Dữ liệu trên máy thuê

Ngoài recipe chung của `docs/server-finetune.md` §3:

| Thứ | Nguồn | Ghi chú |
|---|---|---|
| `dataset/v6-corpus` | HF `rework-whisper-v6-org/v6-corpus` (private) | như `docs/v6-finetune-plan.md` §3 |
| `dataset/vivos` | `scripts/fetch_vivos.py` | không nằm trong gói corpus |
| `dataset/cross-domain-bench` | `dataset/cross-domain-bench.zip` (119 MB, 302 entry) | scp từ máy Windows, cần cho tier gate mới |
| `outputs/v6-corpus-r32/checkpoints/best/` | `Outputs/v6-corpus-r32.zip` (220 MB sau giải nén) | adapter pha 1 |

## 4. Lệnh chạy

```bash
export PYTHONPATH=. TRANSFORMERS_AUTO_CONVERSION=0
export HF_TOKEN=<org_download token>
R=v6-curriculum-p2
OV="--override run_id=$R
    --override data.dataset_path=dataset/v6-corpus-dense
    --override data.real_bench_path=null
    --override training.init_adapter=outputs/v6-corpus-r32/checkpoints/best
    --override training.epochs=2
    --override training.learning_rate=1e-4
    --override training.batch_size=16 --override training.grad_accum_steps=1
    --override training.gradient_checkpointing=false
    --override training.eval_steps=400
    --override eval.batch_size=8
    --override sweep.retention_floor=0.666
    --override data.cross_domain_path=dataset/cross-domain-bench
    --override gates.cross_domain_cer_max=0.0819
    --override gates.cross_domain_retention_min=0.666
    --override gates.cross_domain_no_loanword_cer_max=0.0586
    --override hub.repo_id=rework-whisper-v6-org/Reworkwhisper-large-v6-dense"

python -m src.pipeline --stage baseline   $OV
python -m src.pipeline --stage train      $OV
python -m src.pipeline --stage sweep-gate $OV --override hub.push=True
```

Bốn override `gates.cross_domain_*` / `data.cross_domain_path` là bắt buộc: cả bốn mặc
định `null`, và thiếu chúng thì check cross-domain **im lặng không chạy** chứ không báo
lỗi. Ba override còn lại đáng nói:

- **`learning_rate=1e-4`**, một nửa mặc định. Pha 2 tinh chỉnh một adapter đã hội tụ, không
  học từ đầu; LR đầy đủ có nguy cơ xoá phần pha 1 vừa học. Đây là phán đoán, chưa đo — nếu
  pha 2 làm tụt VIVOS hoặc CER phi-code-switch thì đây là nghi can đầu tiên.
- **`eval.batch_size=8`**, không phải 64 như run v6. Batch eval làm đổi CER, và toàn bộ cột
  v5 trong bảng benchmark đo ở batch 8. Chạy batch 64 là tự tạo lại đúng cái nhiễu đã phải
  ghi chú ở run trước.
- **`eval_steps=400`**: 12.880 / 16 = 805 step một epoch, 2 epoch = **1.610 step** →
  4 lượt eval. Run v6 dùng 6 lượt và tốn 23,3 phút eval trên 26,3 phút train.

## 5. Ngưỡng gate — bỏ Scribe v2 khỏi bảng so sánh

| Bộ đo | n | Ngưỡng | Ai đang giữ |
|---|---:|---:|---|
| cross-domain-bench CER | 299 | ≤ 8,19% | v5 |
| cross-domain-bench retention | 1283 từ | ≥ 66,6% | v5 |
| YouTube test | 228 | ≤ 5,93% | v5 (λ=0,75) |
| Tổng hợp TTS | 426 | ≤ 1,61% | v5 (λ=0,75) |
| VIVOS | 760 | ≤ 2,28% | PhoWhisper-large |
| **CER trên segment KHÔNG có từ ngoại lai** | 48 | **≤ 5,86%** | v6 |

Hàng cuối là bảo hiểm, không phải mục tiêu: nó chặn trường hợp pha 2 nắn được prior nhưng
đánh mất chính thứ 101 giờ đã mua. ⚠ Giá trị đo thật của v6 trên 48 segment đó là
**5,8637%**, nên ngưỡng ghi tròn 0,0586 loại chính v6 (và loại cả v5 ở 6,060%). Nghĩa là
pha 2 phải thắng pha 1 đúng trên lát cắt này, không có biên an toàn — mà n = 48 thì khoảng tin
cậy rộng hơn khoảng cách đang xét. Nếu muốn để bảo hiểm đúng nghĩa là bảo hiểm thì đặt
`gates.cross_domain_no_loanword_cer_max=0.0606` (mức v5); giữ 0,0586 là cố ý chấp nhận
một ngưỡng sát rạt. Mọi số cột v5 lấy từ
`Outputs/lambda075-metrics/v4-mixed-r16-lambda0.75/metrics/gate_results.json` (bản đã
publish), **không** lấy từ `Outputs/v4-mixed-r16/metrics/gate_results.json` — file đó là
λ=0,25 mà gate chọn nhầm và chưa bao giờ được phát hành.

Quy đổi: hồi quy 4 điểm (retention, CER) trên cross-domain cho
`CER ≈ 16,95 − 0,1287 × retention`. Đạt 8,19% cần retention **≈ 68%**. Mục tiêu là ngang
v5, không phải đuổi theo 83,6% của Scribe v2.

## 6. Ngân sách

| Bước | Thời gian |
|---|---|
| baseline | ~3 phút |
| train 1.610 step @ 0,728 s/step | ~20 phút tính toán |
| 4 lượt eval (val 365 + VIVOS, batch 8) | ~15 phút |
| sweep-gate (5 λ × val + VIVOS + cross-domain) | ~15 phút |
| **tổng đồng hồ thật** | **~55 phút** |

Rẻ hơn `v6-corpus-r32` (68 phút 50) vì pha 1 đã có sẵn. Cộng thời gian tải corpus và dựng
môi trường thì một lượt thuê máy 2 giờ là đủ, còn dư cho một lần chạy lại.

## 7. Rủi ro và phương án dự phòng

| Rủi ro | Dấu hiệu | Xử lý |
|---|---|---|
| Pha 2 xoá phần pha 1 đã học | VIVOS > 2,82% hoặc CER phi-code-switch > 5,86% | hạ LR còn 5e-5, hoặc giảm còn 1 epoch |
| Retention vẫn dưới 68% | tier cross-domain fail | mật độ 6,51% chưa đủ — trộn thêm phần synthetic (mật độ 8,07%) hoặc tăng epoch pha 2 |
| Không λ nào qua sàn retention | `select_lambda` raise | đúng thiết kế, fail loud. Không hạ sàn để cho qua |
| 13,87 h nhãn `verified: false` gây nhiễu | không có dấu hiệu trực tiếp | chạy lại pha 2 trên bản loại `verified: false` (11.619 seg / 34,17 h / 6,54%), ~17 phút |

Hai hàng ViMedCSS (hard ≤ 18,25 của v5; test ≤ 14,73 của AG) **không** nằm trong phạm vi kế
hoạch này. Chỉ ~30% số lần xuất hiện từ ngoại lai của ViMedCSS có trong từ vựng train — đó
là lỗ hổng từ vựng y khoa thật (`bilirubin`, `aldosterone`, `glucose`, `vaccine`,
`hormone`), không cách sắp xếp lại dữ liệu hiện có nào lấp được. Cần một đợt sinh dữ liệu
hội thoại khám bệnh riêng.

## 8. Sau khi chạy

Theo `docs/server-finetune.md` §6–§7: chép `outputs/v6-curriculum-p2/` về **trước khi trả
máy**, ghi `experiments/task_ledger.md` và `provenance.md` (run `v6-corpus-r32` vẫn còn nợ
hai mục này). Decode adapter pha 2 trên ViMedCSS hard + test trong cùng lượt thuê máy (~15 phút,
2.272 segment) để bảng benchmark đủ 6 hàng, kể cả khi kế hoạch này không nhắm vào hai hàng
đó.
