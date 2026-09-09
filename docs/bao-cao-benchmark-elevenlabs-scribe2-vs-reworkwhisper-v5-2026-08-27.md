# Benchmark ElevenLabs Scribe v2 vs Reworkwhisper-large-v5 trên youtube-data-pilot

Nguồn: `compare_reworkwhisper_v5_vs_elevenlabs_youtube_data_pilot_summary.csv` và `review_youtube_data_pilot-20260827T054523Z-1-001.zip`, sinh bởi [scripts/compare_reworkwhisper_v5_vs_elevenlabs_colab.py](../viet-speech/scripts/compare_reworkwhisper_v5_vs_elevenlabs_colab.py), chạy trên Google Colab (GPU T4).

## 1. Tóm tắt

Hai model được chấm trên 3 cuộc họp/webinar YouTube, 299 đoạn, ~78 phút audio, ground truth soát tay 100%. Chế độ per-segment: mỗi đoạn đưa vào model riêng lẻ, không chia sẻ ngữ cảnh giữa các đoạn.

ElevenLabs Scribe v2 thắng Reworkwhisper-large-v5 ở cả 3 cuộc. WER gộp toàn bộ 11.6% so với 14.0%, CER 8.1% so với 9.7%. Cả hai model đều yếu nhất ở nội dung code-switch tiếng Anh dày đặc (webinar marketing). Reworkwhisper có thêm một lỗi riêng: cắt cụt câu giữa chừng, xảy ra rõ nhất trên nội dung thuật ngữ y khoa hiếm.

## 2. Dữ liệu và ground truth

| Cuộc họp | Domain | Đoạn | Từ (ref) | Thời lượng |
|---|---|---:|---:|---:|
| `IGZYBrbDUEw` | Tuyển dụng/HR | 114 | 6,809 | 30 phút |
| `ySdJ3sg_2lk` | Hội chẩn y khoa | 69 | 4,010 | 18 phút |
| `z-nODwLyhA0` | Webinar marketing | 116 | 6,520 | 30 phút |

Cả 299 đoạn `verified: true`, người soát 100% (`youtube-meeting-data-pilot/dataset/youtube-meetings/manifest.<meeting_id>.jsonl`). Script chặn cứng nếu gặp đoạn chưa soát.

## 3. Pipeline chạy như nào

Per-segment: mỗi đoạn 3-30 giây đưa vào model riêng, đúng ranh giới đã cắt trong ground truth.

- **ElevenLabs Scribe v2**: gọi API trả phí (~$0.22/giờ audio), không bật diarize riêng, lấy timestamp theo từ.
- **Reworkwhisper-large-v5** (`winhsss/Reworkwhisper-large-v5`): chạy trên GPU T4 Colab qua `transformers`, decode bằng `WhisperForConditionalGeneration.generate`, có sẵn cờ chống hallucination chuẩn Whisper (`no_repeat_ngram_size`, `no_speech_threshold`, `condition_on_prev_tokens=False`).

## 4. Thang đo

Dùng `backend/benchmark/metrics.py` — cùng một bộ chấm cho cả hai model.

- **normalize_vi**: hạ chữ thường, chuẩn hoá Unicode, bỏ dấu câu, giữ chữ cái có dấu và số.
- **WER/CER**: edit-distance trên từ/ký tự đã normalize. CER là thang chính cho tiếng Việt vì ranh giới từ không rõ ràng.
- **aggregate**: micro-average toàn corpus (cộng dồn lỗi và số từ/ký tự trước khi chia), không lấy trung bình cộng theo từng đoạn.
- **RTF** = thời gian xử lý / thời lượng audio.

## 5. Kết quả tổng hợp

| Cuộc họp | Model | WER | CER | RTF | Thời gian xử lý |
|---|---|---:|---:|---:|---:|
| IGZYBrbDUEw | ElevenLabs | 8.1% | 6.0% | 0.162 | 291.8s |
| | Reworkwhisper v5 | 10.3% | 7.5% | 0.335 | 796.5s |
| ySdJ3sg_2lk | ElevenLabs | 16.0% | 12.5% | 0.173 | 187.3s |
| | Reworkwhisper v5 | 18.3% | 13.9% | 0.365 | 415.3s |
| z-nODwLyhA0 | ElevenLabs | 12.5% | 7.7% | 0.136 | 245.6s |
| | Reworkwhisper v5 | 15.1% | 9.4% | 0.346 | 720.5s |
| **Gộp 3 cuộc (299 đoạn)** | **ElevenLabs** | **11.6%** | **8.1%** | | |
| | **Reworkwhisper v5** | **14.0%** | **9.7%** | | |

ElevenLabs thắng đều trên cả 3 domain khác nhau, chênh khoảng 2-2.5 điểm WER mỗi cuộc, không phải ngẫu nhiên theo 1 mẫu. Tốc độ: ElevenLabs RTF 0.14-0.17 (API), Reworkwhisper 0.33-0.37 (GPU T4, free) — cả hai đều real-time-capable, Reworkwhisper chậm hơn ~2x nhưng không tốn phí.

## 6. Phân tích lỗi

Gộp edit-distance operations (thay/xoá/thêm từ) trên cả 3 cuộc, chế độ per-segment:

| | Thay (sub) | Xoá (del) | Thêm (ins) | Tổng |
|---|---:|---:|---:|---:|
| ElevenLabs | 1159 (58%) | 332 (17%) | 517 (26%) | 2008 |
| Reworkwhisper v5 | 1407 (58%) | 541 (22%) | 472 (20%) | 2420 |

Tỉ lệ thay từ (sub) gần bằng nhau ở cả hai model trên bộ dữ liệu này. Khác biệt nằm ở phần còn lại: ElevenLabs thiên về thêm từ hơn (ghi lại nguyên văn từ đệm, từ lặp do nói vấp), Reworkwhisper thiên về xoá từ hơn — phần lớn do lỗi cắt cụt câu ở mục 7 dưới đây kéo `del` của meeting y khoa lên 28%.

### 6.1. Code-switch tiếng Anh — điểm yếu chung của cả hai model

`z-nODwLyhA0` (webinar marketing, mật độ thuật ngữ tiếng Anh cao nhất trong 3 cuộc: "customer journey", "awareness", "consideration", "framework", "channel"...) có các đoạn lỗi nặng nhất toàn bộ benchmark, ở cả hai model:

```
seg_0082 REF: "...awareness ờ rồi media rồi consideration rồi sau đó purchase và loyal..."
ElevenLabs (wer 68%): "...awareness, à, rồi familiar considerations purchase loyal..."
Reworkwhisper (wer 42%): "...awareness à rồi familiar considerations và rồi đó về case và lò dò..."
```

Cả hai model đều nghe nhầm chuỗi thuật ngữ tiếng Anh liên tiếp — không có model nào xử lý tốt hẳn. Đáng chú ý: đây cũng đúng loại lỗi ghi nhận ở lần benchmark trước trên dataset khác ([../viet-speech/docs/bao-cao-danh-gia-reworkwhisper-v5-vs-elevenlabs-scribe2-2026-08-19.md](../viet-speech/docs/bao-cao-danh-gia-reworkwhisper-v5-vs-elevenlabs-scribe2-2026-08-19.md), lỗi `"set longitude"` → `"set latitude"`) — code-switch tiếng Anh mật độ cao là điểm yếu nhất quán, không phải đặc thù của một bộ dữ liệu.

## 7. Phát hiện riêng — Reworkwhisper cắt cụt câu trên thuật ngữ y khoa

3/69 đoạn của Reworkwhisper trên `ySdJ3sg_2lk` (hội chẩn y khoa) bị dừng sinh giữa chừng, mất gần hết phần còn lại của câu dù audio vẫn còn tiếp tục:

```
seg_0012 (wer 94%) REF (49 từ): "đầu tiên ạ vầng kính thưa thầy thành kính thưa các anh chị em đồng nghiệp thì ở bệnh nhân này..."
                    HYP (3 từ):  "đầu tiên ạ."
seg_0010 (wer 86%) REF (35 từ): "đặc biệt từ trên phim này ạ vâng vâng không thấy có nguyên nhân gây giãn đại bể thận..."
                    HYP (6 từ):  "đặc biệt ở trên phim này."
```

ElevenLabs: 0/183 đoạn bị cắt cụt kiểu này trên `IGZYBrbDUEw`+`ySdJ3sg_2lk`, 1/116 trên `z-nODwLyhA0` (một trường hợp đơn lẻ). Reworkwhisper: 0/114 trên `IGZYBrbDUEw`, 0/116 trên `z-nODwLyhA0`, nhưng 3/69 riêng trên meeting y khoa — đúng lúc gặp câu mở đầu formal dài (xưng hô "kính thưa thầy... đồng nghiệp") hoặc thuật ngữ chuyên ngành hiếm. Đây là lỗi độc lập với WER trung bình: một mình 3 đoạn này kéo `max WER` của Reworkwhisper trên meeting này lên 93.9% (so với p90 chỉ 33.3%).

## 8. Hạn chế của benchmark hiện tại

- **Định dạng số**: `ySdJ3sg_2lk` (y khoa) ghi số liệu dạng chữ số (`86`, `3000`, `500ml`), cả hai model đọc ra chữ (`tám sáu`, `ba nghìn`). `normalize_vi` không quy đổi số↔chữ nên mọi con số bị tính sai 100% dù model đọc đúng giá trị — một phần WER cao hơn của meeting này (16-18% so với 8-10% của meeting HR) là artifact chấm điểm, không hoàn toàn phản ánh model yếu hơn trên domain y khoa.
- **normalize_vi bỏ dấu câu/viết hoa**: hợp lý cho việc so khả năng nhận dạng, nhưng nếu mục tiêu cuối là transcript hiển thị cho người dùng, ElevenLabs có lợi thế thêm (chấm câu, viết hoa) mà bảng số liệu này không thể hiện.
- **Cỡ mẫu**: 3 cuộc, 299 đoạn — đủ thấy xu hướng nhất quán (ElevenLabs thắng ở cả 3 domain khác nhau), nhưng số tuyệt đối trên domain khác (ví dụ hội thoại nhiều người chồng tiếng) chưa được kiểm chứng.

## 9. Kết luận

ElevenLabs Scribe v2 chính xác hơn Reworkwhisper-large-v5 nhất quán trên cả 3 domain, chênh 2-2.5 điểm WER, tốc độ nhanh hơn ~2 lần, đổi lại là chi phí API (~$0.22/giờ audio). Reworkwhisper chạy free trên GPU tự có nhưng có rủi ro cắt cụt câu khi gặp thuật ngữ chuyên ngành hiếm — cần theo dõi nếu định dùng cho domain có nhiều thuật ngữ đặc thù (y khoa, pháp lý, kỹ thuật...). Cả hai model cùng yếu nhất ở nội dung code-switch tiếng Anh mật độ cao — nếu ứng dụng thật có nhiều nội dung dạng này (marketing, tech), nên cân nhắc thêm bước hậu kiểm cho các đoạn đó bất kể chọn model nào.
