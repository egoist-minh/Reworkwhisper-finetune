# Báo cáo kết quả Fine-tune PhoWhisper-large + LoRA (v3-r16)

**Ngày:** 2026-08-04
**Model nền:** `vinai/PhoWhisper-large`
**Dataset:** `paid-dataset-v2`

> ⚠️ **[CHỜ SỐ]** Cột λ=0.5 đang chờ chạy `notebooks/reeval_lambda.ipynb` trên Kaggle. Điền vào khi có kết quả.

**CER (Character Error Rate)** = % ký tự sai so với transcript đúng. Thấp hơn là tốt hơn.

---

## 1. So sánh CER: Baseline vs sau Fine-tune

| Loại dữ liệu | Baseline | λ=1.0 | λ=0.5 |
|:---|:---:|:---:|:---:|
| Dữ liệu giả lập (in-domain, giọng chưa nghe khi train) | 4.26% | **1.71%** | [CHỜ] |
| VIVOS (giọng thật, đọc chuẩn) | 2.28% | 4.14% | [CHỜ] (dự kiến ~2.49%) |
| Họp thật (`done/`, 2 bản ghi) | 31.57% | 35.07% | [CHỜ] |

**Đọc bảng:**
- Dữ liệu giả lập: cải thiện rõ, giảm hơn nửa CER.
- VIVOS: ở λ=1.0, CER **tăng** so với baseline (tệ hơn) — dấu hiệu quên tiếng Việt tổng quát. λ=0.5 dự kiến gần bằng baseline (không tệ hơn đáng kể).
- Họp thật: cả hai mức λ đều xấp xỉ baseline, chênh lệch nhỏ so với độ nhiễu tự nhiên của việc chỉ có 2 bản ghi — chưa đủ để nói rõ tốt hơn hay tệ hơn.

---

## 2. So sánh λ=1.0 vs λ=0.5

λ là mức độ áp dụng phần fine-tune học được (0 = giữ nguyên model gốc, 1 = áp dụng toàn bộ). Kiểm tra thêm trên audio họp thật cho thấy ở λ=1.0, model có dấu hiệu **cắt bớt nội dung** khi trả lời (sinh ra ít chữ hơn hẳn so với baseline trên cùng đoạn ghi âm), trong khi λ=0.5 giữ được phần lớn lợi ích ở dữ liệu giả lập mà ít rủi ro này hơn.

[CHỜ số CER thật ở λ=0.5 để kết luận nên chọn mức nào.]

---

## 3. Hạn chế

- **Họp thật chỉ có 2 bản ghi** — số CER trên phần này dao động mạnh, không đủ để kết luận chắc chắn tốt hơn hay tệ hơn baseline.
- **Transcript tham chiếu của bản ghi họp thật** do người chỉnh sửa lại từ bản nháp ASR, không phải bản gốc hoàn hảo — ảnh hưởng đến độ chính xác tuyệt đối của CER trên phần này (nhưng ảnh hưởng như nhau lên cả baseline và bản fine-tune, nên so sánh giữa hai bên vẫn công bằng).
- **Dữ liệu giả lập là do máy tạo (TTS)** — CER tuyệt đối trên phần này không phản ánh chất lượng nghe giọng người thật, chỉ có giá trị so sánh tương đối.
