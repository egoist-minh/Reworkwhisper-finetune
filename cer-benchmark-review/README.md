# Benchmark CER — fine-tune PhoWhisper (nhờ review lỗ hổng)

Đây là bản gộp riêng, tách khỏi repo pipeline, chỉ chứa **phần tính CER chạy trên Kaggle sau khi
fine-tune xong** — tức Stage 4 (Evaluation Gate) của pipeline. Mục tiêu: nhờ professor đọc và chỉ
ra lỗ hổng phương pháp luận (methodology), không phải review code style.

## Bài toán đo là gì

Fine-tune LoRA `vinai/PhoWhisper-large` bằng dữ liệu tiếng Việt (họp/webinar tổng hợp + một ít
audio thật). Sau khi train xong một adapter, pipeline chấm nó qua 3 tier CER (xem
[gate-spec-excerpt.md](gate-spec-excerpt.md) để đọc đặc tả gốc):

- **Tier 1** — in-domain, dữ liệu synthetic (giọng TTS): `CER_test ≤ 0.9 × CER_base`.
- **Tier 2** — out-of-domain, tập VIVOS (giọng đọc thật, khác domain train): `CER_ood ≤ CER_ood(base) + 0.20pp`, chặn quên kiến thức cũ (catastrophic forgetting).
- **Tier 4a** — audio họp thật, 264 đoạn từ 2 buổi ghi âm thật (không phải TTS): `CER_real ≤ CER_real(base)`, không được tệ hơn baseline.

Rớt bất kỳ tier nào → dừng pipeline, không publish. Không có fallback mềm.

## Cấu trúc folder

```
src/                     -- code tính CER/WER, chuẩn hoá text, chạy ASR, chạy gate
  metrics.py               CER/WER edit-distance, bootstrap CI, verdict (IMPROVED/REGRESSED/INCONCLUSIVE)
  normalize.py              chuẩn hoá text trước khi so (số, dấu câu, filler) -- áp đối xứng cho hyp và ref
  gate.py                   chạy 3 tier, ghép kết quả thành gate_results.json
  asr.py                    gọi model sinh transcript (transcribe_batch)
  pipeline.py               điều phối các stage, gọi baseline eval + gate
  config.py                 load + validate configs/experiment.yaml
  data.py                   resolve train/val/test split, disjointness check

configs/experiment.yaml   -- config thật của run v4-mixed-r16 (ngưỡng gate, split, normalize)

notebook/fine-tune-workflow.ipynb
                           -- notebook Kaggle đã chạy thật (có execution count + output),
                              không phải template. Đây là bằng chứng lệnh nào thật sự chạy.

run-v4-mixed-r16/          -- kết quả gate của run publish gần nhất (adapter đang production, gọi là v5)
  config.json                config đã freeze lúc chạy
  validated_manifest.jsonl   manifest train/val đã qua disjointness check
  metrics/gate_results.json  kết quả 3 tier: cer, ci, bound, pass, by_meeting, verdict, normalization_check
  metrics/baseline.json      CER của base model (PhoWhisper-large chưa fine-tune) trên cùng 3 split
  metrics/lambda_sweep.csv   kết quả sweep hệ số trộn adapter (λ) trên val + OOD
  metrics/split_stats.json   thống kê từng split (số đoạn, số từ...)
  metrics/training.csv       log loss theo step
  audit/predictions_*.csv    text ref/hyp từng đoạn, 6 file (baseline x3 split, candidate x3 split) --
                              đây là nơi soát tay từng câu sai, không chỉ tin con số tổng

run-v3-r16-rejoin-bug-case/metrics/
                           -- ví dụ một lỗi đã tìm ra và sửa trong chính cơ chế chấm (xem dưới)

gate-spec-excerpt.md      -- trích nguyên văn đặc tả gốc (viết trước khi chạy) mô tả rule
                              từng tier, quy ước chuẩn hoá, và lý do
```

## Việc đã tự phát hiện là lỗ hổng, đã note lại (để professor không tốn công lặp)

1. **~62% cải thiện CER trên run trước** hoá ra là do chuẩn hoá số (`mười lăm` → `15`), không
   phải model nghe tốt hơn. → chuẩn hoá số nay bắt buộc đối xứng (áp cả hai bên) và có audit
   riêng đếm số token bị đổi, để tách phần "chấm điểm" khỏi phần "học được".
2. **Rejoin bug (2026-08-04)** — tier 4a từng chấm ở mức đoạn audio đã bị cắt nhỏ để đưa vào
   model (chunk), thay vì mức đoạn gốc trong ground truth (parent segment). Sửa xong, CER real
   đổi từ 48.01% → 35.07% (baseline 45.76% → 31.57%). Xem chi tiết trong
   [gate-spec-excerpt.md](gate-spec-excerpt.md) mục cuối và file `run-v3-r16-rejoin-bug-case/`.
3. **Tier 4a chỉ có 264 đoạn, không renewable** — không được dùng để tune λ (chọn hệ số trộn
   adapter), chỉ đọc một lần lúc gate, không chia dev/test vì mẫu đã nhỏ.
4. **`INCONCLUSIVE` là kết quả hợp lệ** — bootstrap CI trên tier 4a có thể không đủ mạnh để phân
   biệt candidate với baseline; pipeline không đọc nhầm `INCONCLUSIVE` thành pass hay fail.

## Câu hỏi cụ thể muốn professor nhìn hộ

- Rule chọn ngưỡng (`0.9×`, `+0.20pp`, `≤ baseline`) có chỗ nào tuỳ tiện, chưa có căn cứ thống kê?
- Cỡ mẫu tier 4a (264 đoạn, 2 buổi ghi âm) có đủ để kết luận "không regress" không, hay CI đã
  đủ rộng để nói thẳng là "chưa đủ bằng chứng" (như `run-v4-mixed-r16/metrics/gate_results.json`
  đang tự báo `verdict`)?
- Chuẩn hoá số/dấu câu/filler (bảng trong `gate-spec-excerpt.md`) có che giấu loại lỗi nào khác
  không, ngoài việc đã biết là làm mất phân biệt hoa/thường (không đo được viết hoa tên riêng)?
- Bootstrap CI trong `metrics.py` (paired, giữa baseline và candidate trên cùng đoạn) có đúng
  giả định thống kê không?
- Còn góc nào trong `src/gate.py`/`src/metrics.py` có thể đo sai mà chưa ai phát hiện, tương tự
  kiểu lỗi rejoin ở mục 2 trên?
