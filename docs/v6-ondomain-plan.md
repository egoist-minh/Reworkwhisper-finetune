# v6-ondomain — hai lượt fine-tune đối chứng

Hai run, LoRA tươi từ `vinai/PhoWhisper-large`, cùng baseline / val / test / hyperparam.
**Biến đổi duy nhất giữa hai run là corpus train.**

| run id | corpus | giờ | mật độ từ ngoại lai | % segment có từ ngoại lai | cuộc |
|---|---|---:|---:|---:|---:|
| `v6-ondomain-29h` | quota theo cuộc, θ=4,5% | 29,45 | 7,42% | 86,5% | 251 |
| `v6-ondomain-15h` | lọc nguyên cuộc ≥3% | 15,50 | 7,46% | 64,5% | 48 |

Mốc so sánh: `cross-domain-bench` ở 7,41% / 83,9%; corpus v5 (retention 66,6%) ở 7,31% / 55,2%.
Hai run cùng mật độ token, khác nhau ở số giờ, số cuộc và tỉ lệ segment mang từ ngoại lai —
đó là ba trục đang tranh cãi.

Chẩn đoán vì sao lượt v6 101 h hỏng: `docs/v6-analysis-brief.md`. Vì sao không giữ cả 101 h:
corpus đó 2,48% mật độ, và cắt segment kiểu nào cũng không vượt được trần 6,51%.

**Chưa chạy:** lượt 101 h nguyên ở cấu hình công bằng (rank 16, 3 epoch, LoRA tươi) — 6.558
step, ~80 phút compute. Đây là câu hỏi của professor, còn để ngỏ. Nếu lượt thuê còn giờ thì
đó là thứ chạy tiếp theo.

---

## 1. Sửa code (0 phút GPU)

**a. `scripts/filter_corpus_density.py`** — thêm hai chế độ lọc, cùng dùng
`src.metrics.foreign_token_counts` như chế độ hiện có. Chỉ lọc train; val/test đi qua nguyên
trạng (cần `--config` để biết `data.val_meetings`).

- `--min-meeting-density D`: giữ **nguyên cuộc** nào có mật độ ≥ D. Không đụng dao vào trong
  cuộc, nên cuộc giữ lại còn nguyên nhịp xen kẽ Việt/ngoại lai.
- `--meeting-quota T --min-meeting-foreign-density θ`: mỗi cuộc giữ toàn bộ segment **có**
  từ ngoại lai, cộng thêm segment thuần Việt **dài nhất trước** cho tới khi cuộc chạm mật độ
  T. Cuộc nào riêng phần segment có từ ngoại lai đã dưới θ thì bỏ cả cuộc — nó không thể
  chạm T nếu không cắt cả từ ngoại lai.

Chọn dài-nhất-trước để mỗi token ngân sách mua được nhiều audio nhất. Bằng nhau thì xếp theo
`segment_id`, để lượt sau dựng lại corpus ra đúng từng byte.

**b. Đường decode** — thêm `compression_ratio_threshold` + temperature fallback vào
[src/asr.py](../src/asr.py). Hiện `model.generate()` chỉ truyền `task` + `num_beams`, không có
chống lặp: 1 trong 299 segment cross-domain của v5 là vòng lặp, một mình nó cộng 0,82 điểm CER.
**Không** dùng `no_repeat_ngram_size` — mất ~3 điểm CER vì lệch thanh điệu tiếng Việt.

Verify: `pytest tests/`.

---

## 2. Dựng hai corpus (0 phút GPU)

```
python -m scripts.filter_corpus_density \
    --src dataset/v6-corpus --out dataset/v6-ondomain-29h \
    --meeting-quota 0.0741 --min-meeting-foreign-density 0.045 \
    --config configs/experiment.yaml

python -m scripts.filter_corpus_density \
    --src dataset/v6-corpus --out dataset/v6-ondomain-15h \
    --min-meeting-density 0.03 --config configs/experiment.yaml
```

Verify — phải in ra **đúng** các số này, lệch thì dừng:

| | `-29h` | `-15h` |
|---|---:|---:|
| train segment | 11.504 | 7.022 |
| train giờ | 29,45 | 15,50 |
| train mật độ | 7,42% | 7,46% |
| train cuộc | 251 | 48 |
| val | 365 seg / 0,79 h | như trái |
| test | 654 seg / 1,58 h | như trái |

`--dry-run` in bảng mà không ghi gì. Script symlink `<out>/audio` về `dataset/v6-corpus/audio`,
không copy — trên Windows cần Developer Mode.

**Lên máy thuê cần audio thật.** Hai corpus dùng chung kho audio. Quyết định 2026-09-14:
kéo nguyên `v6-corpus.tar` (14,7 GB, private trên HF) về máy thuê rồi chạy lại hai lệnh lọc
ở trên tại chỗ — không đóng gói riêng phần audio được giữ (4,73 GiB cho hợp của hai corpus,
13.623 file), vì upload từ máy cá nhân chậm hơn băng thông datacenter. Phần tải này ăn vào
giờ máy, đã tính trong §7.

Trạng thái 2026-09-14: §1 và §2 đã xong trên máy này. Hai thư mục
`dataset/v6-ondomain-29h` và `dataset/v6-ondomain-15h` tồn tại, số liệu khớp đúng bảng trên
(Windows không cho symlink nên `audio` là junction `mklink /J`, trên Linux script tự symlink).

---

## 3. Đặt cổng (0 phút GPU)

Trong `configs/experiment.yaml`:

```yaml
gates:
  cross_domain_retention_min: <đo ở bước 4.1>
  cross_domain_no_loanword_cer_max: <đo ở bước 4.1>
  cross_domain_cer_max: <đo ở bước 4.1>
sweep:
  retention_floor: null      # val nằm trong corpus, số bị thổi phồng, để null
```

Ba ngưỡng là số của adapter v5 đang production, **đo lại dưới đường decode mới** ở bước 4.1.
Số cũ (0,0819 CER / 0,666 retention / 0,0586 no-loanword CER) đo dưới đường decode chưa chống
lặp; dùng thẳng là để cổng so hai thứ khác nhau.

`cross_domain_no_loanword_cer_max` là cảm biến cho đúng rủi ro của chế độ quota: corpus
`-29h` có 86,5% segment mang từ ngoại lai, nếu model vì thế mà nhả tiếng Anh bừa trên lời
thuần Việt thì lát segment không có từ ngoại lai sẽ tệ đi và cổng chặn.

Verify: chạy lại gate của `v6-corpus-r32` với ba ngưỡng này phải **FAIL**.

---

## 4. Trên máy thuê

`docs/server-finetune.md` cho phần dựng máy, IP, kiểm `torch.cuda.is_available()`.
Cần `dataset/vivos` (`scripts/fetch_vivos.py`), thiếu là `--stage baseline` chết.
Cần giải nén `dataset/cross-domain-bench.zip` — nó là **tập test**, chỉ vào qua
`data.cross_domain_path`, không bao giờ vào train (3 cuộc / 299 segment, giao với `v6-corpus`
bằng 0 meeting_id và 0 sha256).

**4.1 Decode lại v5 trên cross-domain** bằng đường decode mới (~3 phút GPU). Ba số ra được
điền vào `gates.cross_domain_*` ở bước 3.

**4.2 Chạy — rẻ trước, để mỗi lượt xong là ngân hàng thêm một điểm dữ liệu:**

```bash
run () {
  R=$1; D=$2; E=$3
  OV="--override run_id=$R
      --override data.dataset_path=$D
      --override data.real_bench_path=null
      --override data.cross_domain_path=dataset/cross-domain-bench
      --override training.epochs=3
      --override training.learning_rate=2.0e-4
      --override training.batch_size=16 --override training.grad_accum_steps=1
      --override training.gradient_checkpointing=false
      --override training.eval_steps=$E
      --override eval.batch_size=64"
  python -m src.pipeline --stage baseline $OV || exit 1
  python -m src.pipeline --stage train    $OV || exit 1
}

run v6-ondomain-15h dataset/v6-ondomain-15h 220
run v6-ondomain-29h dataset/v6-ondomain-29h 360
```

`eval_steps` khác nhau để mỗi run có 6 vòng eval: 1.317 step / 220 và 2.157 step / 360.

**4.3 Chọn rồi mới sweep.** Decode cross-domain ở λ=0,75 cố định cho cả hai adapter (~3 phút
mỗi cái), so retention, rồi chỉ chạy `sweep-gate` đầy đủ cho bên thắng:

```bash
python -m src.pipeline --stage sweep-gate <OV của run thắng> || exit 1
```

Run đứt: `--stage train --resume`, **đừng xoá `checkpoints/`**.

**4.4 Chép `outputs/v6-ondomain-15h/` và `outputs/v6-ondomain-29h/` về trước khi trả máy.**

---

## 5. Tham số — vì sao đúng những số đó, đừng tự đổi

Giống hệt nhau ở cả hai run; corpus là biến duy nhất.

| | giá trị | lý do |
|---|---|---|
| `lora.rank` / `alpha` | 16 / 32 (mặc định config) | hình dạng đã đạt retention 66,6% ở v5 |
| `learning_rate` | 2.0e-4 | như v5 |
| `epochs` | 3 | lượt v6 trước chạy 1 epoch, loss phẳng từ step 200 — không phải thiếu epoch |
| `batch` × `grad_accum` | 16 × 1 | batch hiệu dụng 16, bằng v5. H200 không cần tích luỹ |
| `gradient_checkpointing` | false | chỉ T4 cần; bật là chậm thêm 30–40% |
| `init_adapter` | null | LoRA tươi. Không kế thừa trọng số v6 |
| `eval.batch_size` | 64 | CER lệch theo batch — `baseline.json` và eval sau train phải cùng số này |
| `data.exclude_manifest` | null | một biến một lượt |
| `sweep.lambdas` | grid đủ (mặc định) | |
| `sweep.ood_cer_budget` | 0.02 | VIVOS giữ trong ngân sách, không đuổi SOTA ở đó |

---

## 6. Đọc kết quả

Hồi quy trên 7 hệ đã đo: `CER = 16,927 − 0,1293 × retention`, R² = 0,979. Retention là biến
cần tối ưu. Vượt v5 cần retention **67,6%**; vượt Scribe v2 cần **82,6%**; trần thực nghiệm
của bench là 87,30%.

| kết quả | đọc là gì |
|---|---|
| `-29h` > `-15h` | thêm giờ ở cùng mật độ có giúp — lượt sau nới θ xuống 4,0% (31,40 h) hoặc 3,5% (33,73 h) |
| `-15h` ≥ `-29h` | giờ không phải đòn bẩy; 86,5% segment mang từ ngoại lai là quá tay — giữ lọc nguyên cuộc, đẩy mật độ |
| cả hai ~56% retention | đòn bẩy dữ liệu cạn. Ngừng xáo corpus, đổi cơ chế (ràng buộc tầng decode, hoặc sinh dữ liệu code-switch đúng miền) |
| bên thắng ≥ 75% | đáng chạy lượt hai ngay trong lượt thuê |
| `no_loanword_cer` xấu đi so với v5 | chế độ quota làm model bịa tiếng Anh — bỏ hướng cắt segment, kể cả khi retention đẹp |

Cổng fail là fail cứng: không chọn adapter, không push HF. Nếu cả hai fail thì bảng summit
17/09 không có mô hình mới — chốt trước với người dùng là bảng đó trình v5 + phân tích.

Số cho bảng trình bày: chạy riêng `scripts/benchmark_run.py --batch-size 8` (cùng cấu hình
cột v5) + paired bootstrap dCER với Scribe trên đúng 299 segment.

---

## 7. Chi phí

| hạng mục | phút |
|---|---:|
| baseline × 2 (mỗi run một lần, ~8 phút mỗi lần) | 16 |
| train `-15h`: 1.317 step × 0,728 s | 16 |
| train `-29h`: 2.157 step × 0,728 s | 26 |
| eval trong train: 6 vòng × 2 run | 46 |
| decode v5 lấy ngưỡng + decode chọn λ=0,75 × 2 | 9 |
| `sweep-gate` cho bên thắng | 30–70 |
| **tổng compute** | **2,4–3,1 h** |

Cộng dựng máy, tải corpus, chép kết quả về: **thuê 5 h**. Nếu chạy thêm lượt 101 h của
professor (+80 phút train, +23 phút eval) thì thuê 7 h.

---

## 8. Kết quả (2026-09-15, H200 thuê, commit `39aab1d`)

Cả hai lượt chạy hết, `EXIT=0`. **Không lượt nào qua cổng.** Artifact:
`Outputs/v6-ondomain-15h.zip`, `Outputs/v6-ondomain-29h.zip`, metrics và audit đã giải nén
sẵn ở `Outputs/v6-ondomain-{15h,29h}/`, log ở `Outputs/v6-ondomain.run.log` và
`Outputs/v6-ondomain-15h.sweep.log`.

### 8.1 Train

| | `-15h` | `-29h` |
|---|---:|---:|
| step | 1.317 | 2.157 |
| ValCER tốt nhất | 0,0354 (step 1.100) | 0,0352 (step 1.800) |

ValCER hai bên gần như trùng nhau — và **không đọc được gì từ nó**. Val nằm trong phân phối
v6 nên không thấy được chênh lệch cross-domain, vốn lệch hẳn (0,0875 vs 0,1101). Lượt sau
đừng dùng ValCER để so hai corpus.

### 8.2 Cross-domain (`dataset/cross-domain-bench`, 299 segment, 1.283 candidate)

| hệ | cer | retention | no_loanword_cer |
|---|---:|---:|---:|
| ngưỡng (v5) | ≤ 0,0740 | ≥ 0,6648 | ≤ 0,0613 |
| v6-100h (`v6-corpus`) | 0,1081 | 0,4864 | 0,0577 |
| `-29h` λ=1,0 | 0,1101 | 0,5947 | 0,1083 |
| `-15h` λ=1,0 | 0,0919 | 0,6157 | 0,0731 |
| **`-15h` λ=0,75** | **0,0875** | **0,6189** | 0,0628 |
| `-15h` λ=0,25 (elbow chọn) | 0,0926 | 0,5012 | 0,0575 |

### 8.3 Đọc theo bảng §6

Trúng dòng **`-15h` ≥ `-29h`**: giờ không phải đòn bẩy, lọc nguyên cuộc thắng quota ở mọi
cột. Dòng `no_loanword_cer xấu đi` cũng nổ cho chế độ quota (`-29h` 0,1083 so với 0,0613)
— bỏ hướng cắt segment, đúng như bảng đã đoán trước.

Retention về ~62%, không phải ~56%, cũng không tới 75%. Lọc corpus theo mật độ từ ngoại lai
đẩy retention từ 48,6% (100 h) lên 61,9% (15,5 h) — **hướng đúng, mức cải thiện lớn, vẫn
thiếu 4,6 pp so với v5**. CER dư 1,35 pp. v5 giữ production.

### 8.4 Hai phát hiện về cơ chế

**λ đi ngược chiều dự đoán.** Chính base PhoWhisper-large mới là thứ nuốt từ ngoại lai
(val retention 0,342 ở λ=0). Fine-tune v6 *dạy* được retention chứ không phá nó, nên λ càng
nhỏ retention càng tệ:

```
λ      val_cer   ood_cer   val_retention
0,0    0,1024    0,0226    0,3422
0,25   0,0544    0,0227    0,6811
0,5    0,0382    0,0257    0,8051
0,75   0,0333    0,0315    0,8461
1,0    0,0364    0,0405    0,8394
```

**`select_lambda` chọn λ=0,25 — tệ hơn λ=0,75 ở cả cer lẫn retention.** Elbow rule tối ưu
val_cer/ood_cer, mù với retention vì `sweep.retention_floor: null`. Cùng đúng lỗi đã buộc
v5 phải override tay sang λ=0,75 (`experiments/v4-mixed-r16-lambda0.75/README.md`). Lượt
sau đặt `sweep.retention_floor` (≈0,80 theo bảng trên) thay vì override tay.

Số λ=0,75 đo bằng cách copy `checkpoints/best` rồi sửa `lora_alpha` 32 → 24 — λ chỉ nằm ở
`lora_alpha`, `lora_B` bit-identical ở mọi λ (`src/lora.py:save_with_lambda`).

### 8.5 In-domain thì mô hình học rất tốt

Gate tier1 và tier2 pass hết ở λ=0,25: in-domain CER 0,0559 (bound 0,0841), OOD 0,0227
(bound 0,0426), cả youtube lẫn synthetic đều `IMPROVED`. Retention in-domain tăng mạnh so
với baseline — youtube 0,365 → 0,716, synthetic 0,562 → 0,733.

**Nên thất bại không nằm ở chỗ model không học được retention. Nó học tốt trên in-domain,
nhưng không chuyển sang cross-domain bench.** Đây là câu hỏi cho lượt sau, không phải câu
hỏi corpus density nữa — đòn bẩy dữ liệu theo hướng này đã cạn.

Top từ bị nuốt trên bench (λ=0,25): `devops` 27, `jd` 26, `vinpearl` 24, `funnel` 15,
`team` 14, `marketing` 14, `loyalty` 12, `acid` 11, `hunt` 10, `digital` 10, `uric` 10.
Danh từ riêng và thuật ngữ ngành — không phải từ mượn phổ thông.
