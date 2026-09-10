# Vấn đề đang mở — đường tới v6

2026-09-09. Mục tiêu: ra v6 tốt hơn thật trước 17/09/2026.

Nguồn số: `docs/so-lieu-tong-hop.md`,
`docs/bao-cao-benchmark-elevenlabs-scribe2-vs-reworkwhisper-v5-2026-08-27.md`.

---

## 0. Mục tiêu tiên quyết cho v1 — chốt 2026-09-09

Hai mục tiêu. Mọi việc khác trong tài liệu này xếp sau.

| # | Mục tiêu | Ngưỡng | Chi phí |
|---|---|---|---:|
| **1** | **SOTA trên ViMedCSS** (bộ công khai, arXiv 2602.12911) | CER < 19,25 · WER < 27,56 · CS-WER < 46,69 | $0 |
| **2** | **Vượt ElevenLabs Scribe v2 trên bộ test YouTube** (228 đoạn) | v5 đang ở 5,93% CER; ElevenLabs **chưa đo** | ~$0,25 |

Ba ngôi trên ViMedCSS thuộc ba hệ khác nhau — CER 19,25 (PhoWhisper-Large),
WER 27,56 (VietASR), CS-WER 46,69 (Whisper-Large-v3). Ngôi CER do chính base
model giữ nên dễ giành nhất. Giành được **một** ngôi là đủ tuyên bố SOTA.

**Vì sao ViMedCSS chứ không phải VIVOS:** VIVOS đã bão hoà — nền 4,73 WER, bản
tốt nhất của mình 4,64 @ λ=0,25, chênh lệch nằm trong nhiễu. Trên VIVOS chỉ
chứng minh được "không làm hỏng". ViMedCSS ở 31,24 WER còn rất nhiều dư địa.
VIVOS **giữ lại** nhưng đổi vai thành hàng rào chống quên (đã đo sẵn 4 λ, $0).

**Điều kiện tiên quyết:** dựng lại được dòng `PhoWhisper-Large` 31,24 WER /
19,25 CER trên tập test. Bài báo **không ghi cách chuẩn hoá text lúc chấm**, nên
lệch là chuyện có thể xảy ra — phát hiện sau khi đã chấm v5 thì phải làm lại
tất cả. CS-WER tính **trên token trong vùng chuyển mã**, không phải mức câu;
dùng cột `cs_terms_list` của bộ dữ liệu.

**Lưới an toàn:** zero-shot trượt cả ba ngôi thì LoRA trên 24,31 h train của
chính ViMedCSS — ~$20, 3 giờ H200. Khác hẳn kế hoạch 100 giờ ở mục 4, vốn đã
lùi xuống v2/v3.

---

## 1. `cross-domain-bench`

Ba buổi giữ riêng, nay có tên. Thư mục `dataset/cross-domain-bench/`, trường
`source: cross-domain`. Vai trò: cổng ngoài (thay tier 4a) **và** bảng benchmark 17/09.

| `video_id` | Domain | Đoạn | Thời lượng |
|---|---|---:|---:|
| `IGZYBrbDUEw` | Tuyển dụng / HR | 114 | 30 phút |
| `ySdJ3sg_2lk` | Hội chẩn y khoa | 69 | 18 phút |
| `z-nODwLyhA0` | Webinar marketing | 116 | 30 phút |
| **Tổng** | | **299** | **78 phút** |

Vì sao đáng tin hơn mọi bộ đo khác: buổi chưa train · chủ đề khác train · người nói khác
· **nhãn từ caption Google, không phải Whisper** (nên không thiên vị họ PhoWhisper, khác
`real-meetings-bench`) · thu âm khác · `verified: true` 100% · đã có ElevenLabs Scribe v2
chấm sẵn.

---

## 2. Bảng benchmark

**CER (%)** — thấp hơn là tốt hơn:

| Mô hình | Tổng hợp<br>n=426 · 34,9 phút | YouTube test<br>n=228 · 60,0 phút | **`cross-domain-bench`<br>n=299 · 78,0 phút** | VIVOS<br>n=760 câu |
|---|---:|---:|---:|---:|
| `vinai/PhoWhisper-large` | 4,26 | 15,93 | — | 2,28 |
| `Reworkwhisper-large-v4` | 1,96 | 11,21 / 12,96 | — | 2,49 |
| `Reworkwhisper-large-v5` | **1,61** | **5,93** | 8,436 | 3,34 |
| `Reworkwhisper` v6 | — | — | — | — |
| ElevenLabs Scribe v2 | — | — | **6,535** | — |

**Giữ từ mượn (%)** — cao hơn là tốt hơn:

| Mô hình | YouTube test<br>n=228 · 60,0 phút | **`cross-domain-bench`<br>n=299 · 78,0 phút** |
|---|---:|---:|
| `vinai/PhoWhisper-large` | 30,2 / 41,0 | — |
| `Reworkwhisper-large-v4` | 79,2 / 76,1 | — |
| `Reworkwhisper-large-v5` | **85,7 / 85,3** | 65,71 |
| ElevenLabs Scribe v2 | — | **83,79** |

**`n` = số đoạn**, không phải số buổi họp. Một đoạn = một dòng manifest = một file
`.wav` = một lượt forward của Whisper (đệm lên 30 giây). Thời lượng VIVOS chưa đo —
dataset không có trên máy này.

**Không so ngang giữa các cột** — bốn bộ khác độ khó. Chỉ so dọc.

### Con số quan trọng nhất

| `Reworkwhisper-large-v5` | YouTube test | `cross-domain-bench` | |
|---|---:|---:|---|
| CER | 5,93 | 8,436 | xấu đi 42% |
| Giữ từ mượn | ~85,5 | 65,71 | tụt ~20 điểm |

Cùng mô hình, cùng audio người thật. Ra khỏi corpus quen thì mất gần một nửa thành tích.
Và trên cột đó v5 **thua ElevenLabs cả ba buổi**.

### Mục tiêu v6 trên `cross-domain-bench`

- Tối thiểu: CER < **8,436%** (thắng chính mình)
- Để bảng thuyết phục: CER < **6,535%** (thắng ElevenLabs)

### Chỗ thua nặng nhất — token chưa từng thấy

ElevenLabs giữ **74,6%** (420/563), v5 giữ **44,6%** (251/563). Chênh 30 điểm.
Token v5 mất nhiều nhất: `jd` `vinpearl` `funnel` `hunt` `acid` `uric` `headhunt`
`salesforce` `touchpoint` `linkedin`.

Không phải lỗi âm học mà là **mất từ mượn và tên riêng** — đúng thứ tier `retention`
trong `src/gate.py` được dựng để bắt. **Nếu v6 chỉ sửa được một thứ, sửa cái này.**

---

## 3. Quyết định chốt 2026-09-09

**Bỏ `real-meetings-bench` khỏi vai trò cổng.** Ba khuyết tật (`PROJECT_CORE.md` §4):
reference là output PhoWhisper-small đã người sửa nên thiên vị họ PhoWhisper · sai miền
(bài giảng ML, không phải họp) · ngưỡng `real_cer_regression_pp: 0.0` nằm trong vùng
nhiễu ±1–1,5pp. Thư mục giữ trên đĩa làm hồ sơ lịch sử.

**Cách thay: đổi một dòng config, không bỏ code.** Không bỏ nhánh `tier4a_real`, chỉ trỏ
`data.real_bench_path` sang `dataset/cross-domain-bench/`. `by_meeting`
(`src/gate.py:330`) đã generic, `_rejoin` (`src/gate.py:213`) thành no-op vô hại vì đoạn
3–30 s, `bootstrap_delta_ci` đã có.

**Không gộp vào `dataset/youtube-meetings/`.** Hai lý do: (a) rủi ro rò vào train —
`build_mixed_dataset` mặc định đưa phần không phải `val_meetings` vào train, quên một
lần là mất bộ đo ngoài duy nhất, mất kiểu không ai phát hiện vì số sẽ đẹp lên; (b) số
gộp xoá mất chênh lệch sân nhà ở mục 2.

---

## 4. Hạ tầng H200

### Giá (ảnh chụp `ai.fptcloud.com/pricing` 2026-09-09, Nhật Bản)

| | RAM | CPU | NVMe | $/giờ | $/GPU/giờ |
|---|---|---|---|---:|---:|
| 1× H200 SXM5 141 GB | 192 GB | 24 nhân | 3 TB | 6,60 | 6,60 |
| 2× | 384 GB | 48 nhân | 6 TB | 13,20 | 6,60 |
| 4× | 768 GB | 96 nhân | 12 TB | 26,40 | 6,60 |
| 8× | 1.536 GB | 192 nhân | 24 TB | 52,80 | 6,60 |

Giá tuyến tính, không chiết khấu số lượng. Nhân GPU chỉ mua thời gian đồng hồ; vì hiệu
suất mở rộng dưới 100% nên tổng tiền **tăng**.

### Chi phí tính theo số đoạn, không theo giờ audio

Whisper đệm mọi đầu vào lên 30 giây. Đoạn 4 s và đoạn 25 s tốn GPU như nhau.

| Nguồn | Đang có | Đoạn TB | 50 giờ thêm vào ra |
|---|---|---:|---:|
| Tổng hợp | 4.946 đoạn / 5,89 giờ | 4,29 s | 41.987 đoạn |
| YouTube | 829 đoạn / 3,62 giờ | 15,74 s | 11.439 đoạn |

Cùng 50 giờ audio, phần tổng hợp tốn **3,7 lần** GPU. Đổi tỷ lệ trộn là đổi dự toán.

```
Đang có                   :  5.775 đoạn /   9,5 giờ
Thêm vào                  : 53.426 đoạn / 100,0 giờ
Tổng                      : 59.201 đoạn / 109,5 giờ
Tập train (~82%)          : 48.545 đoạn
Bước mỗi epoch (batch 16) : 3.034
```

Tỷ lệ audio thật **tính theo đoạn** (đơn vị mô hình thực sự học), không theo giờ:

| Kịch bản | Tổng hợp | YouTube | Tỷ lệ thật |
|---|---:|---:|---:|
| Hiện tại | 4.946 | 829 | 14,4% |
| +50h tổng hợp, +50h YouTube | 46.933 | 12.268 | **20,7%** |
| +50h tổng hợp, +25h YouTube | 46.933 | 6.547 | 12,2% |
| +50h tổng hợp, +10h YouTube | 46.933 | 3.116 | 6,2% |
| +50h tổng hợp, +0 YouTube | 46.933 | 829 | 1,7% |

Kế hoạch 100 giờ **đầy đủ** nâng tỷ lệ thật từ 14,4% lên 20,7%. Nguy hiểm nằm ở việc
thực hiện dở dang: phần tổng hợp xong tự động (máy sinh), phần YouTube phụ thuộc người
soát. Tổng hợp về đủ mà soát không kịp thì rơi xuống 1,7–6,2% — đúng cấu hình đã làm
`v3-r16` trượt tier 4a.

**Luật: soát được bao nhiêu YouTube thì mới đổ bấy nhiêu tổng hợp vào.** Tỷ lệ trộn phải
là quyết định, không phải kết quả của việc soát xong đến đâu.

### Cấu hình: **1× H200 SXM5**

| | Có | Cần |
|---|---|---|
| VRAM | 141 GB | ~40–60 GB (LoRA rank 16 chỉ train ~15–20M tham số) |
| RAM | 192 GB | < 60 GB |
| NVMe | 3 TB | ~60 GB |
| CPU | 24 nhân | **điểm nghẽn thật, phải đo** |

Bộ nhớ chưa bao giờ là ràng buộc. Lý do duy nhất để nhân GPU là thời gian đồng hồ, mà
còn 8 ngày thì không thiếu.

### Số giờ

Mốc đo: `v3-r16` chạy **801 bước / 6 h 31 m** trên 1× T4 ⇒ **1,83 s/đoạn**.

```
Train 3 epoch : 48.545 × 3 × 1,83 s = 74 giờ T4 ÷ 10 (hệ số H200) = 7,4 giờ
Đánh giá      : baseline + quét 5 λ + cổng                        = 2,0 giờ
Dôi ra                                                            = 1–3 giờ
──────────────────────────────────────────────────────────────────────────
Một lượt chạy                                                     = 12 giờ
```

Hệ số 10 = 7,6× (bf16 sm_90 so với fp16 sm_75) × 1,33× (bỏ gradient checkpointing).

| | Giờ | Tiền |
|---|---:|---:|
| Phiên hiệu chỉnh | 1 | $6,60 |
| Lượt chính | 12 | $79,20 |
| Lượt dự phòng | 12 | $79,20 |
| **Tổng** | **25** | **$165** |

Làm tròn **$180**.

### Ba điều phải kèm

**Hệ số 10 là suy ra, chưa đo.** Cả dự toán tỷ lệ tuyến tính với một biến: `s/bước` thật
trên H200.

**Phiên hiệu chỉnh là bắt buộc.** Nó thay thế biên an toàn ×2 đã bỏ. Đo hai số:
`s/bước` thật (thay vào công thức) và **mức dùng GPU (%)** — dưới 80% nghĩa là CPU nạp
không kịp, sửa `dataloader_num_workers` trước khi chạy thật.

**Rủi ro ở CPU, không ở GPU.** Log-mel tính bằng numpy STFT trên CPU
(`WhisperCollator`), và `dataloader_num_workers` hiện không xuất hiện ở đâu trong `src/`.
Trên T4 CPU theo kịp; trên H200 nhanh gấp 10 lần thì 24 nhân có thể không, và tiền thuê
trả cho GPU đứng yên.

### Vượt giờ không mất gì

Bốn chặng `smoke` → `baseline` → `train` → `sweep-gate`, mỗi ranh giới một file trên
đĩa, `.pipeline_state.json` cho chạy tiếp. Vượt 12 giờ thì gia hạn, không chạy lại.

### Bốn câu hỏi cho FPT

1. Tính tiền theo giây/phút hay **làm tròn lên giờ**?
2. Dừng máy có **giữ ổ NVMe 3 TB** không, lúc dừng có tính tiền không?
3. Lưu trữ và băng thông ra có tính riêng không?
4. Thời gian cấp máy có bị tính tiền không?

Câu 2 quan trọng nhất — nếu dừng máy mất dữ liệu thì phải nạp lại 100 giờ audio mỗi
lần, và kế hoạch 25 giờ phải viết lại thành một phiên liền mạch.
