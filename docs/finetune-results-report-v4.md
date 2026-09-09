# Báo cáo kết quả Fine-tune PhoWhisper-large + LoRA (v4-mixed-r16)

**Ngày:** 2026-08-18
**Model nền:** `vinai/PhoWhisper-large`
**Dataset:** `mixed-noisy-v1` (dữ liệu giả lập + 790 đoạn họp thật lấy từ YouTube)
**Thời lượng train/val/test:** train 4717 đoạn (~7 giờ), val 365 đoạn (~48 phút), test 654 đoạn (~1 giờ 35 phút)
**Mức λ đề xuất:** 0.75 — đã publish thành `Reworkwhisper-large-v5` ngày 2026-08-18

**CER (Character Error Rate)** = % ký tự sai so với transcript đúng. Thấp hơn là tốt hơn.

**λ (lambda) là gì?** Fine-tune tạo ra một "phần điều chỉnh" cộng thêm vào model gốc. λ là mức độ áp dụng phần điều chỉnh đó: λ=0 nghĩa là giữ nguyên model gốc (không áp dụng gì), λ=1 nghĩa là áp dụng toàn bộ phần đã học được, λ=0.5 là áp dụng một nửa. Không cần train lại để đổi λ — chỉ là chỉnh mức độ dùng cái đã học, giống như chỉnh volume của hiệu ứng.

**"Giữ từ tiếng Anh" là gì?** Trong họp, người nói xen rất nhiều từ tiếng Anh (`team`, `design`, `service`). Model có thể nghe đúng âm nhưng viết ra theo âm tiết tiếng Việt — `service` thành "sờ vít", `team` thành "tim". Chỉ số này đếm tỷ lệ từ tiếng Anh được viết đúng nguyên dạng. Cao hơn là tốt hơn. Cần đo riêng vì lỗi kiểu này chỉ chiếm vài ký tự trong câu nên **CER gần như không nhìn thấy nó**.

**Hai mốc so sánh, đừng nhầm lẫn:**
- **Model gốc** = `PhoWhisper-large` chưa fine-tune. Cho biết bản fine-tune học được bao nhiêu.
- **Model đang chạy** = bản đang dùng thật trong hệ thống (`v3-r16` ở λ=0.5, publish tên `Reworkwhisper-large-v4`). Cho biết có nên thay hay không — đây mới là câu hỏi thực tế.

---

## 1. So sánh CER: Model gốc vs Model đang chạy vs Bản mới

| Loại dữ liệu | Model gốc | Model đang chạy | **Bản mới (λ=0.75)** |
|:---|:---:|:---:|:---:|
| Họp thật từ YouTube (228 đoạn test) | 15.93% | 12.05% | **5.93%** |
| Dữ liệu giả lập (426 đoạn test) | 4.26% | 1.96% | **1.61%** |
| VIVOS (giọng thật, đọc chuẩn) | 2.28% | 2.49% | **3.34%** |

Bản mới và model đang chạy được chấm trên **đúng cùng 654 đoạn test**, transcript tham chiếu khớp từng ký tự.

**Đọc bảng:**
- **Bản mới tốt hơn cả hai mốc trên toàn bộ dữ liệu test**: giảm 50.8% so với model đang chạy ở họp YouTube, 17.9% ở dữ liệu giả lập.
- **Họp YouTube là dòng quan trọng nhất** vì đây là audio người thật, nhiễu, nhiều người nói xen nhau — gần với dữ liệu chạy thật nhất trong bảng. Dữ liệu giả lập là giọng máy đọc trong môi trường sạch nên dễ hơn nhiều, thấy ngay ở chênh lệch CER giữa hai dòng.
- **VIVOS là dòng duy nhất tệ đi** (3.34% so với 2.49%). Đây là giọng đọc chuẩn từng câu, không phải hội thoại — model học sâu vào giọng họp thì nghe giọng đọc sách kém đi một chút. Chi tiết ở §4.
- Vì sao cải thiện dồn vào audio họp: model đang chạy được train 100% trên giọng máy (TTS) sạch nên không có gì để học về audio nhiễu. Bản mới có thêm 447 đoạn họp thật trong dữ liệu train.

---

## 2. Giữ từ tiếng Anh — chỉ nhìn CER là không đủ

| Loại dữ liệu | Số lượt từ tiếng Anh / tổng số từ | Model gốc | Model đang chạy | **Bản mới (λ=0.75)** |
|:---|:---:|:---:|:---:|:---:|
| Họp thật từ YouTube | 874 / 11685 từ (7.5%) | 36.4% (318/874) | 77.5% (677/874) | **85.5%** (747/874) |
| Dữ liệu giả lập | 479 / 7482 từ (6.4%) | 55.7% (267/479) | 83.1% (398/479) | **86.2%** (413/479) |
| Gộp 654 đoạn test | 1353 / 19167 từ (7.1%) | 43.2% (585/1353) | 79.5% (1075/1353) | **85.7%** (1160/1353) |

Cột thứ hai là mẫu số của chỉ số: đếm trên transcript tham chiếu, tính theo lượt xuất hiện (một từ nói hai lần tính hai lượt). Từ tiếng Anh chiếm 7.1% tổng số từ của bộ test. Cột cuối ghi kèm số lượt giữ đúng trên tổng số lượt.

**Đọc bảng:** model gốc viết sai gần 2/3 số từ tiếng Anh trong họp YouTube; bản mới giữ đúng 85.5%, và vẫn nhỉnh hơn model đang chạy 8.0 điểm trên chính dòng đó (6.3 điểm nếu tính trên dòng gộp).

Chỉ số này được thêm vào bộ kiểm tra sau khi một bản pass hết mọi bài kiểm CER nhưng vẫn tệ hơn hẳn model đang chạy — vì nó phiên âm hết từ tiếng Anh mà CER không phát hiện được.

Vẫn còn một nhóm nhỏ bản mới giữ kém hơn model đang chạy: tên công cụ ngắn (`mysql`, `aws`, `terminal`, `kafka`, `standup`) — mỗi từ hụt 1–2 lượt. Số nhỏ nhưng thành nhóm mạch lạc, cần theo dõi ở lần đo sau.

---

## 3. Vì sao chọn λ=0.75

Cùng một bản fine-tune, chỉ đổi mức λ rồi chấm lại:

| λ | Giả lập | Họp YouTube | Giữ từ tiếng Anh (YouTube) | VIVOS |
|:---:|:---:|:---:|:---:|:---:|
| 0.25 | 2.58% | 7.64% | 70.9% | 2.26% |
| 0.50 | 1.85% | 6.24% | 82.3% | 2.62% |
| **0.75** | **1.61%** | **5.93%** | **85.5%** | 3.34% |
| 1.00 | 1.73% | 6.10% | 84.9% | 4.26% |

**Đọc bảng:**
- CER **chạm đáy ở λ=0.75 rồi xấu đi ở λ=1.0**, và xấu đi đồng thời ở cả hai loại dữ liệu — đây là dấu hiệu áp dụng phần điều chỉnh quá mạnh, không phải dao động ngẫu nhiên của một cột.
- Giữ từ tiếng Anh tăng đều tới λ=0.75 rồi đứng. Trước khi đo có lo ngại λ càng cao thì model càng bám theo cách viết của nhãn YouTube (vốn viết từ mượn theo âm tiết Việt) nên sẽ giữ kém đi — số liệu bác bỏ lo ngại này.
- VIVOS xấu đi đều theo λ, đúng như quy luật đã thấy ở `v3-r16`: λ càng cao càng quên tiếng Việt tổng quát. λ=0.75 vẫn nằm trong ngưỡng cho phép (4.28%), λ=1.0 thì gần chạm.

**λ=0.75 là quyết định của người, không phải của luật chọn tự động.** Luật chọn λ trong pipeline chọn 0.5, vì nó chỉ cân CER với VIVOS mà **không nhìn chỉ số giữ từ tiếng Anh** — đúng chỗ 0.75 hơn 0.5 tới 3.2 điểm. Ngưỡng của luật **cố ý không được chỉnh** để nó tự ra 0.75, vì làm vậy là sửa thước đo cho vừa một đáp án đã biết. Thay vào đó λ=0.75 được kiểm riêng bằng một script tách rời, để một run bình thường không bao giờ lặng lẽ đi vòng qua luật chọn.

---

## 4. Đánh đổi

Bản mới tốt hơn model đang chạy ở cả hai loại dữ liệu test (18% ở giả lập, 51% ở họp YouTube), đổi lại **tệ hơn ở VIVOS**: 3.34% so với 2.49%, tức tệ hơn 0.85 điểm phần trăm, tương đương +34%.

Con số này vẫn dưới ngưỡng cho phép nên bài kiểm vẫn pass, nhưng nó là **chỉ số duy nhất xấu đi** và nó có thật. Hệ thống production xử lý audio họp chứ không xử lý giọng đọc sách, nên đánh đổi này hợp lý — chỉ cần gọi đúng tên là đánh đổi, đừng trình bày như cải thiện toàn diện.

---

## 5. Hạn chế

- **Chữ hoa tên riêng vẫn hỏng, và đổi λ không sửa được.** Nhãn của dữ liệu YouTube viết thường toàn bộ (178,075 ký tự, 0 chữ hoa — quy ước cố ý khi làm dữ liệu), nên λ càng cao model càng bám theo cách viết đó. Hệ quả trực tiếp: module trích thông tin rơi chức danh vì mất chữ hoa. Muốn sửa phải train lại với nhãn có chữ hoa. Nếu sau khi lên production module đó vẫn rơi chức danh thì đó là kết quả đã dự đoán trước, không phải bằng chứng bản mới hỏng.
- **Dữ liệu giả lập là do máy tạo (TTS)** — CER tuyệt đối trên phần này không phản ánh chất lượng nghe giọng người thật, chỉ có giá trị so sánh tương đối.
- **Nhãn của dữ liệu YouTube là bản nháp ASR đã qua người soát, một người soát.** Cả 7/7 buổi họp (790/790 đoạn) đã được soát tay, nhưng điểm xuất phát là phụ đề tự động của Google chứ không phải chép mới từ đầu, và chỉ 2 buổi (`3nuCdzuyqng` ở test, `rCd8DSMk3-c` ở val) được soát lượt thứ hai. Lỗi còn sót sẽ ảnh hưởng như nhau lên mọi bản được chấm, nên so sánh giữa chúng vẫn công bằng.
- **Phần "học viết số" đã được tách riêng để không nhầm với việc nghe tốt hơn**: chênh lệch CER giữa hai cách viết số chỉ 0.161 điểm phần trăm trên dữ liệu test — phần lớn mức cải thiện là nghe tốt hơn thật.
- **Một đoạn duy nhất bóp méo số của model gốc.** Ở một đoạn YouTube, model gốc rơi vào vòng lặp sinh chữ (`long long long …`, CER 2544%), tự nó đẩy CER model gốc từ 12.645% lên 15.93%. Khi báo cáo mức giảm so với model gốc, con số nên dùng là −53.1% (đã loại đoạn đó) chứ không phải −62.8%. Hiện tượng này hiếm: 1/228 đoạn.
- **Đã publish thành `Reworkwhisper-large-v5` (2026-08-18, λ=0.75)** sau khi bản λ=0.25 publish trước đó gây hồi quy production và bị rollback — xem `docs/kiem-diem-v5-production-regression.md`.

---

## Phụ lục — nguồn số

| Khối | Nguồn |
|:---|:---|
| Bài kiểm λ=0.75, CER theo loại dữ liệu, giữ từ tiếng Anh | `Outputs/lambda075-metrics/v4-mixed-r16-lambda0.75/metrics/gate_results.json` |
| Số của model đang chạy (chấm lại ở λ=0.5) | `Outputs/lambda075-metrics/v3-r16-lambda0.5/predictions_tier1_in_domain.csv` (654 dòng) |
| Bảng λ (0.5 / 0.75 / 1.0) | `Outputs/v4-mixed-r16-lambda0.5/summary.json` và output notebook |
| CER model gốc, khoảng tin cậy, kiểm tra cách viết số | `Outputs/v4-mixed-r16/metrics/gate_results.json` · `metrics/baseline.json` |
| Chia dữ liệu train/val/test | `Outputs/v4-mixed-r16/validated_manifest.jsonl` |
| Nhật ký điều tra | [`SESSIONS.md`](../SESSIONS.md), mục "Hồi quy Reworkwhisper-large-v5 trong production" |

Cấu hình train, đường huấn luyện và cách chọn learning rate của run này nằm trong git history của file này (bản báo cáo kỹ thuật đầy đủ, trước khi rút gọn theo mẫu báo cáo v3).
