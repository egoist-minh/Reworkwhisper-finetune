# Bảng điểm — PhoWhisper-large vs các model đã fine-tune

Gọi model bằng **đúng tên đã publish trên HuggingFace**, không dùng nhãn "v1/v2/v3" nữa — nhãn đó
đá nhau với số thứ tự của HF, của run ID và của dataset (xem bảng ánh xạ cuối file).

Hai tập: **VIVOS** (giọng thật đọc chuẩn, kiểm tra quên) · **trong miền** (họp giả lập).
Đơn vị %. Thấp hơn = tốt hơn. Ô trống = chưa đo, không phải 0.

## Bảng chính

Một model một cột, một bộ test một hàng. Bốn nhóm cột = bốn lần fine-tune, mỗi nhóm có cột base
riêng vì mỗi lần dùng dataset riêng:

- **Lần 1** · `dataset-dot1` · trong miền n=500 *(tập này vừa là val vừa là test)* · VIVOS n=760
  theo `docs/archive/dot1-lora-report.md` §5.3 — đo bằng kernel Kaggle `winhkento/benchmark-asr`,
  không lưu file raw nên không tự kiểm lại được
- **Lần 2** · `paid-dataset` · trong miền n=236 · VIVOS n=760
- **Lần 3** · `paid-dataset-v2` · trong miền n=426 · VIVOS n=760
- **Lần 4** · `mixed-noisy-v1` · trong miền n=654 *(426 giả lập + 228 họp YouTube audio thật)* ·
  VIVOS n=760

| Bộ test | paper<br>`vinai/PhoWhisper-large` | L1 base | L1 **`reworkwhisper-large-v1`** | L2 base | L2 `reworkwhisper-large-v3` | L2 **`reworkwhisper-large-v3-0.5lamda`** | L3 base | L3 adapter `v3-r16` *(chưa publish)* | L3 **`Reworkwhisper-large-v4`** | L4 base | L4 adapter `v4-mixed-r16` *(chưa publish)* | L4 **`Reworkwhisper-large-v5`** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| λ | — | — | 1.0 | — | 1.0 | 0.5 | — | 1.0 | 0.5 | — | 1.0 | 0.75 |
| VIVOS · CER | — | 1.78 | 4.42 | 2.16 | 4.32 | **2.21** | 2.28 | 4.14 | **2.49** | 2.28 | 4.26 | **3.34** |
| VIVOS · WER | **4.67** | 4.57 | 8.12 | 4.61 | — | **4.75** | 4.73 | 8.74 | **—** | — | — | **—** |
| Trong miền · CER | | 29.65 | **18.53** | 5.11 | 1.29 | **1.65** | 4.26 | 1.71 | **1.96** | 11.31 | — | **4.22** |
| Trong miền · WER | | 47.02 | **26.92** | 8.29 | 2.56 | **3.19** | 7.40 | 2.50 | **2.97** | — | — | **—** |

**So dọc trong cùng một nhóm cột, không so ngang giữa bốn nhóm** — hai hàng "trong miền" đo trên
bốn dataset khác nhau.

**Hàng "trong miền" của lần 4 khác loại với ba lần trước.** Ba lần đầu đo trên họp giả lập thuần
TTS; lần 4 đo trên hỗn hợp 426 giả lập + 228 đoạn audio họp thật, và 228 đoạn thật chiếm 60% ký tự
tham chiếu. Base 11.31 cao hơn hẳn base của ba lần trước là vì thế, không phải vì model nền tệ đi.
Tách theo nguồn: giả lập 4.26 → **1.61**, YouTube 15.93 → **5.93**.

## Bảng tỉ lệ — cái này mới so được giữa các lần

| Model | λ | VIVOS | Trong miền |
|---|---|---|---|
| `reworkwhisper-large-v1` | 1.0 | ×2.48 ❌ quên | −37.5% |
| `reworkwhisper-large-v3` | 1.0 | ×2.00 ❌ quên | −74.8% |
| **`reworkwhisper-large-v3-0.5lamda`** | 0.5 | **×1.02 ✓** | **−67.6%** |
| adapter `v3-r16` *(chưa publish)* | 1.0 | ×1.81 ❌ quên | −59.8% |
| **`Reworkwhisper-large-v4`** | 0.5 | **×1.09 ✓** | **−54.0%** |
| adapter `v4-mixed-r16` *(chưa publish)* | 1.0 | ×1.86 ❌ quên | — |
| **`Reworkwhisper-large-v5`** | 0.75 | **×1.46 ⚠ đánh đổi** | **−62.7%** |

## Năm câu kết luận

1. **Số của chúng tôi đúng.** Paper nói VIVOS WER 4.67. Chúng tôi đo lại base ra 4.57 / 4.61 / 4.73
   qua ba pipeline. Lệch tối đa 0.10pp.
2. **λ=1.0 luôn làm model quên.** ×2.48 · ×2.00 · ×1.81 · ×1.86. Bốn lần, ba dataset. Không phải
   tai nạn.
3. **λ=0.5 chữa được.** ×1.02 và ×1.09 — gần như bằng base.
4. **`Reworkwhisper-large-v4` sạch nhất về mặt đo.** Giảm hơn nửa CER trong miền, và **giọng test
   không có trong train** — hai model trước đo trên giọng đã nghe.
5. **`Reworkwhisper-large-v5` giảm CER nhiều nhất, nhưng đó là một cuộc đánh đổi có chủ ý.** −62.7%
   trong miền, và là bản duy nhất được đo trên audio họp thật. Đổi lại VIVOS ×1.46 — tệ hơn hẳn
   ×1.09 của `v4`. λ=0.75 do người chọn chứ không phải luật chọn tự động, vì luật không nhìn chỉ số
   giữ từ tiếng Anh; chi tiết ở `docs/finetune-results-report-v4.md` §3–4. Tính voice-disjoint chỉ
   chắc ở nửa giả lập: nửa YouTube chia theo `meeting_id`, chưa ai kiểm người nói có trùng giữa
   train và test không (`SESSIONS.md` mục G2 đo tương đồng giọng within 0.54 vs cross 0.32, nhưng
   cặp `xKDHjUoUN54`×`3nuCdzuyqng` lên 0.49).

## Ba chỗ dễ hiểu sai

- **Paper không có CER, chỉ WER.** Muốn so với cột paper thì phải đọc hàng WER.
- **`reworkwhisper-large-v1` trông cải thiện nhiều nhất (29.65 → 18.53) là ảo.** `dataset-dot1` có
  lỗi nhãn đẩy CER base lên cao bất thường, và tập đo đó vừa là val vừa là test.
- **Cùng một base, ba lần đo VIVOS ra ba số** (1.78 / 2.16 / 2.28) vì ba cách nạp dữ liệu khác
  nhau. Đừng trừ số tuyệt đối giữa các nhóm cột.

---

## Ánh xạ tên — vì sao không dùng "v1/v2/v3"

| HF model | Run ID | Dataset | Nhãn cũ trong slide |
|---|---|---|---|
| `reworkwhisper-large-v1` | `dot1-v1-lora` | `dataset-dot1` | v1 |
| `reworkwhisper-large-v3-0.5lamda` | `v1c-r16-valfix` | `paid-dataset` | v2 |
| `Reworkwhisper-large-v4` | `v3-r16` | `paid-dataset-v2` | v3 |
| `Reworkwhisper-large-v5` | `v4-mixed-r16` | `mixed-noisy-v1` | — (chưa lên slide) |

Bốn hệ số đếm đá nhau: nhãn slide **v2** ứng với HF **v3**; nhãn slide **v3** ứng với HF **v4** và
dùng dataset **v2**; run của slide v2 lại tên `v1c`. Thêm `v1b-r16-lr1e-4` là run trung gian không
có mặt trên slide. Dùng tên HF thì hết ánh xạ để nhớ sai.

## Nguồn

| Khối | Nguồn |
|---|---|
| paper | [VinAIResearch/PhoWhisper README](https://github.com/VinAIResearch/PhoWhisper/blob/main/README.md) · [arXiv:2406.02555](https://arxiv.org/pdf/2406.02555) |
| `reworkwhisper-large-v1` | `phowhisper-finetune-exp/outputs/dot1-v1-lora/benchmark_results.csv` (trong miền) · `docs/archive/dot1-lora-report.md` §5.1 (VIVOS, đo bằng kernel Kaggle `winhkento/benchmark-asr` — không có file raw trong repo) |
| `reworkwhisper-large-v3*` | `outputs/v1c_lambda05_3way_benchmark_base_new_reports/benchmark_results.csv` (base, λ=0.5) · `outputs/v1c-lambda-sweep-valfix/{final_test_once.csv,lambda_sweep.csv}` (λ=1.0) |
| `Reworkwhisper-large-v4` | `Outputs/v3-r16/metrics/{baseline.json,gate_results.json,lambda_sweep.csv}` · `Outputs/v3-r16_lambda0.5/outputs/v3-r16/metrics/gate_results_lambda0.5.json` · WER tính lại từ `audit/predictions_*.csv` bằng `src/metrics.py` |
| `Reworkwhisper-large-v5` | `Outputs/lambda075-metrics/v4-mixed-r16-lambda0.75/metrics/gate_results.json` (λ=0.75, tách theo nguồn) · `Outputs/v4-mixed-r16/metrics/{baseline.json,lambda_sweep.csv}` (base và VIVOS λ=1.0) · `Outputs/v4-mixed-r16/metrics/split_stats.json` (n=654) |

### Hai ô trống, vì sao

- **VIVOS WER của `reworkwhisper-large-v3` và `Reworkwhisper-large-v4`**: `lambda_sweep.csv` chỉ ghi
  CER, không ghi WER. Lấp bằng decode lại VIVOS 760 câu ở λ đó — `scripts/reeval_lambda.py` có sẵn
  đường chạy.
- **`reworkwhisper-large-v3` VIVOS CER dùng 4.32**, không dùng 4.58 của kernel Kaggle. 4.58 pair với
  base 1.78, 4.32 pair với base 2.16 — phải cùng pipeline mới lấy tỉ lệ được.

### Số phụ, nếu thầy hỏi tới

- **Benchmark khác của paper**: PhoWhisper-large WER 8.14 (Common Voice vi) · 13.75 (VLSP 2020
  Task-1) · 26.68 (VLSP 2020 Task-2).
- **Thước định vị**: PhoWhisper-medium 1.85 CER, small 2.22 CER trên cùng VIVOS. Ở λ=1.0 bản large
  fine-tune (4.14–4.42) **tệ hơn cả small 244M chưa fine-tune**. Ở λ=0.5 thì không.
- **`reworkwhisper-large-v3-0.5lamda`, bỏ phần học-viết-số**: trên 130 câu không chữ số
  (`cer_nonum` của `v1c_lambda05_3way_benchmark_base_new_reports/benchmark_results.csv`), base
  2.65 → **1.43**, tức −45.8%. Đây là phần "nghe tốt hơn thật", không phải −67.6%.
  Đừng lẫn với con số của báo cáo cũ (`v1c-paid-lora-report.md` §6.3): bộ lọc ở đó ra **216 câu**,
  base 2.96 → 1.30, và số 1.30 là **λ=1.0**, không phải λ=0.5. Hai bộ lọc khác nhau, không trộn.
- **Tập họp thật** (`real-meetings-bench`, 43.2 phút, 196 segment) không đưa vào bảng này. Cùng một
  base 31.57: `Reworkwhisper-large-v4` → 29.06, verdict **INCONCLUSIVE** (CI của Δ là
  [−0.010, +0.076], chứa 0); `Reworkwhisper-large-v5` → **24.40**, verdict **IMPROVED** (CI của Δ
  là [+0.040, +0.118], không chứa 0). Đây là tập riêng, không phải hàng "trong miền" của lần 4.
