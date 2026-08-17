# Báo cáo kết quả Fine-tune PhoWhisper-large + LoRA (v3-r16)

**Ngày:** 2026-08-04
**Model nền:** `vinai/PhoWhisper-large`
**Dataset:** `paid-dataset-v2`

**CER (Character Error Rate)** = % ký tự sai so với transcript đúng. Thấp hơn là tốt hơn.

**λ (lambda) là gì?** Fine-tune tạo ra một "phần điều chỉnh" cộng thêm vào model gốc. λ là mức độ áp dụng phần điều chỉnh đó: λ=0 nghĩa là giữ nguyên model gốc (không áp dụng gì), λ=1 nghĩa là áp dụng toàn bộ phần đã học được, λ=0.5 là áp dụng một nửa. Không cần train lại để đổi λ — chỉ là chỉnh mức độ dùng cái đã học, giống như chỉnh volume của hiệu ứng.

---

## 1. So sánh CER: Baseline vs sau Fine-tune

| Loại dữ liệu | Baseline | λ=1.0 | λ=0.5 |
|:---|:---:|:---:|:---:|
| Dữ liệu giả lập (in-domain, giọng chưa nghe khi train) | 4.26% | 1.71% | **1.96%** |
| VIVOS (giọng thật, đọc chuẩn) | 2.28% | 4.14% | **2.49%** |
| Họp thật (`done/`, 2 bản ghi) | 31.57% | 35.07% | **29.06%** |

**Đọc bảng:**
- Dữ liệu giả lập: cải thiện rõ ở cả hai mức λ, giảm hơn nửa CER.
- VIVOS: ở λ=1.0, CER **tăng** so với baseline (tệ hơn) — dấu hiệu quên tiếng Việt tổng quát. λ=0.5 gần như bằng baseline (2.49% vs 2.28%) — không quên đáng kể.
- Họp thật: λ=1.0 tệ hơn baseline (35.07% > 31.57%); λ=0.5 tốt hơn baseline (29.06% < 31.57%). Nhưng chỉ có 2 bản ghi thật, chênh lệch này chưa đủ để nói chắc chắn — có thể do nhiễu mẫu (xem §3).

---

## 2. So sánh λ=1.0 vs λ=0.5 — vì sao chọn λ=0.5

Ở λ=1.0, model học rất tốt trên dữ liệu giả lập nhưng bắt đầu **quên mất khả năng nghe tiếng Việt tổng quát** — đây gọi là hiện tượng "catastrophic forgetting" khi áp dụng phần điều chỉnh quá mạnh. Ví dụ cụ thể trên VIVOS (giọng đọc chuẩn, không liên quan gì đến dữ liệu train): câu "tiếng cọc cạch khựng lại của những khớp sắt" — model gốc (baseline) nghe đúng hoàn toàn, nhưng model ở λ=1.0 lại nghe thành "tuyến tộc cạch khựng lại cổ những gốc sắt", sai gần hết câu. Đây không phải trường hợp cá biệt: phần lớn các câu VIVOS bị lỗi nặng nhất đều theo mô hình này — λ=1.0 sai nhiều hơn hẳn baseline trên chính những câu mà baseline vốn nghe tốt.

λ=0.5 (áp dụng một nửa phần điều chỉnh) tránh được phần lớn hiện tượng quên này trong khi vẫn giữ được đa số lợi ích học từ dữ liệu train.

Kiểm tra thêm một chỉ số phụ trên audio họp thật — tỷ lệ ký tự model sinh ra so với ký tự transcript đúng (thấp hơn nhiều so với baseline nghĩa là model **cắt bớt nội dung** khi trả lời, không chỉ nghe sai từng chữ):

| | Baseline | λ=1.0 | λ=0.5 |
|:---|:---:|:---:|:---:|
| Tỷ lệ ký tự sinh ra / ký tự đúng | 0.931 | 0.842 | **0.887** |

λ=1.0 cắt bớt nội dung rõ rệt nhất (0.842, thấp nhất). λ=0.5 gần với baseline hơn — ít bị cắt nội dung hơn — trong khi vẫn giữ được phần lớn lợi ích ở dữ liệu giả lập (1.96% so với 1.71% của λ=1.0, chênh không nhiều).

**Kết luận:** λ=0.5 là lựa chọn cân bằng tốt hơn λ=1.0 — cải thiện gần tương đương ở dữ liệu giả lập, không quên tiếng Việt tổng quát (VIVOS), ít cắt nội dung hơn, và số trên audio họp thật tốt hơn baseline (dù chưa đủ mẫu để khẳng định chắc).

---

## 3. Hạn chế

- **Họp thật chỉ có 2 bản ghi** — số CER trên phần này dao động mạnh theo mẫu; chênh lệch λ=0.5 so với baseline (29.06% vs 31.57%) là tín hiệu tốt nhưng chưa đủ dữ liệu để khẳng định chắc chắn tốt hơn hẳn.
- **Transcript tham chiếu của bản ghi họp thật** do người chỉnh sửa lại từ bản nháp ASR, không phải bản gốc hoàn hảo — ảnh hưởng đến độ chính xác tuyệt đối của CER trên phần này (nhưng ảnh hưởng như nhau lên cả baseline và bản fine-tune, nên so sánh giữa hai bên vẫn công bằng).
- **Dữ liệu giả lập là do máy tạo (TTS)** — CER tuyệt đối trên phần này không phản ánh chất lượng nghe giọng người thật, chỉ có giá trị so sánh tương đối.
- **Phần "học viết số" đã được tách riêng để không nhầm với việc nghe tốt hơn**: chênh lệch CER giữa hai cách viết số (chữ số vs chữ viết) chỉ 0.064 điểm phần trăm trên dữ liệu giả lập — nghĩa là phần lớn mức cải thiện đến từ nghe tốt hơn thật, không phải chỉ học đúng cách viết số.
