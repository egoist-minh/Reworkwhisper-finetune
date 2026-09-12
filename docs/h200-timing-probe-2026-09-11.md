# Đo chi phí train PhoWhisper-large trên H200

**Ngày đo:** 11/09/2026 · **Máy:** NVIDIA H200 143 GB, 24 vCPU, 188 GB RAM
**Mã nguồn:** `Reworkwhisper-finetune` @ `0aa0569` · **Run:** `v6-probe`

## Kết quả

**0,728 giây mỗi step** (batch 16, LoRA rank 32, tắt gradient checkpointing). Run v5 trên T4 mất 29,34 giây mỗi step ở cùng batch hiệu dụng — **nhanh gấp 40 lần**.

| | Trần cứng | Điển hình |
|---|---|---|
| Giây mỗi step (median) | 0,731 | **0,728** |
| Phần chờ dữ liệu | 0,001 | 0,001 |
| VRAM đỉnh | 74,9 GB | 73,2 GB |

## Phương pháp

Cần định giá một lượt train 100 giờ trước khi cam kết chạy nó. Dựa vào một tính chất của Whisper: **chi phí một step không phụ thuộc kích thước corpus** — model cố định, batch cố định, mọi clip được pad về cửa sổ 30 giây. Chỉ số lượng step thay đổi. Vậy đo giá một step trên corpus 9,35 giờ đang có rồi nhân lên.

Ba lượt đo, dưới 20 phút GPU: một lượt với 640 segment có nhãn dài nhất corpus (trần cứng), một lượt 640 segment ngẫu nhiên (điển hình), một lượt đo tốc độ đọc và giải mã dữ liệu.

## Vì sao con số đáng tin

**Tách chi phí lặp lại khỏi chi phí một lần.** Lấy `train_runtime` chia số step là cách trộn lẫn eval, nạp model và step thật: trên một lượt thử, cách đó cho 1,515 s/step trong khi giá thật là 0,127 s — sai 11,9 lần, vì eval chiếm 157 trong 205 giây. Phép đo ở đây đặt mốc quanh từng step, bỏ 10 step khởi động.

**Trần bằng đúng điển hình (1,00×).** Batch xấu nhất corpus tạo được cũng không đắt hơn batch trung bình. Con số 0,728 không có batch nào vượt qua.

**Dữ liệu không phải nút cổ chai.** Chờ dữ liệu 0,001 giây mỗi step; đường nạp dữ liệu dư 83,9 lần so với tốc độ GPU tiêu thụ. Nếu ngược lại, con số đo được sẽ là giây chờ và phép ngoại suy sẽ sụp khi corpus lớn hơn.

**Cấu hình đã kiểm chứng.** 74,9 GB trên 143 GB xác nhận tắt gradient checkpointing là chạy được — kỹ thuật này đánh đổi 30–40% thời gian tính toán lấy bộ nhớ đang dư một nửa.

**Môi trường khớp lượt chạy thật.** Cùng repo, cùng pipeline. Corpus verify checksum và cho split 4717/365/654, trùng khớp corpus đã train ra v5.

## Kích thước batch khi chấm điểm

`eval.batch_size` đang là 8, giá trị chọn cho T4. Đo lại trên 400 segment, đã trừ 14,1 giây chi phí cố định:

| `eval.batch_size` | Decode | Nhanh hơn batch 8 | `cer_test` |
|---|---|---|---|
| 8 | 67,0 s | — | 0,163113 |
| 32 | 28,0 s | 2,4× | 0,162613 |
| **64** | **19,0 s** | **3,5×** | 0,162319 |
| 128 | 14,7 s | 4,5× | 0,162024 |

Chọn **64** — lên 128 chỉ thêm 1,3 lần trong khi bộ nhớ gấp đôi.

**CER thay đổi theo batch:** `cer_test` lệch 0,11 điểm phần trăm giữa batch 8 và 128, `cer_ood` lệch 4% tương đối. Nguyên nhân là cách pad và thứ tự cộng dồn, không batch nào đúng hơn. Hệ quả bắt buộc: `baseline.json` và eval sau train phải cùng một `eval.batch_size`, nếu không một phần chênh lệch đo được là do batch chứ không do model học được gì.

## Ngân sách cho corpus 100 giờ

1 epoch, batch 16, ước tính 2 730 step, `eval.batch_size` 64, corpus 80 GB. Không chạy `sweep-gate`.

| Khoản | Thuận lợi | Bi quan | Giả định của cột bi quan |
|---|---|---|---|
| Dựng máy | ~10 phút | ~30 phút | gặp vài vướng phiên bản như lần này |
| Tải corpus 80 GB | ~26 phút | ~67 phút | mạng 20 MB/s thay vì 51 MB/s đo được |
| `--stage baseline` | ~8 phút | ~12 phút | decode chậm hơn 1,5 lần |
| `--stage train` | 33 phút | ~61 phút | 3 670 step thay vì 2 730, và 1,0 giây mỗi step |
| Eval | ~5 phút | ~46 phút | eval theo step, 7 lượt thay vì 1, và cùng mức decode chậm 1,5 lần |
| **Tổng** | **~1,4 giờ** | **~3,6 giờ** | |

Cột bi quan không phải cộng dồn nỗi sợ mà là từng giả định cụ thể: corpus chia thành nhiều segment ngắn hơn ước tính (3 670 step nếu segment synthetic trung bình 3 giây thay vì 4,3); máy thuê lần sau chậm hơn hoặc phải bật lại gradient checkpointing (1,0 giây mỗi step thay vì 0,728); và eval chạy theo step qua `training.eval_steps` (thêm 12/09/2026) nên có 7 lượt chấm điểm thay vì 1 — đổi lại là thấy CER giữa chừng và có checkpoint để chạy tiếp nếu run đứt.

**Nên thuê 6 giờ chứ không phải 2.** Cột bi quan là 3,6 giờ cho một lượt chạy trót lọt; lượt đầu hỏng phải chạy lại là chuyện thường, và chỉ riêng phần GPU chạy lại đã là hơn một giờ.

Hạ corpus về 16 kHz mono trước khi đóng gói thì còn ~15 GB, tổng xuống ~1,0 giờ ở cột thuận lợi và ~2,7 giờ ở cột bi quan. Pipeline dù sao cũng resample về 16 kHz, nên phần dôi ra không mang thêm thông tin cho model.

Nếu chạy thêm `sweep-gate`: 5 giá trị λ chấm trên val và OOD, cộng một lượt gate trên test. Khoảng **30 phút** ở cột thuận lợi, **70 phút** ở cột bi quan.

## Giới hạn

- **2 730 step là ước lượng** từ thời lượng trung bình, chưa đếm trên manifest thật vì corpus 100 giờ chưa tồn tại. Giá một step thì chuyển sang không cần đo lại.
- **Chi phí baseline và eval là ngoại suy** từ tốc độ decode đo trên 400 segment.
- **Mọi số đo I/O lấy trên corpus 16 kHz.** Với corpus 80 GB ở định dạng nặng hơn, khoản dư 83,9 lần hẹp lại — ước chừng còn 8–17 lần, chưa đo.

## Hệ quả

GPU không còn là ràng buộc. Một epoch trên 100 giờ tốn 33 phút tính toán và card dư một nửa bộ nhớ; phần lớn thời gian còn lại là tải dữ liệu và chấm điểm, không phải học.

Ràng buộc thật là dữ liệu: 100 giờ đòi hỏi khoảng **10 600 segment YouTube mới phải duyệt tay**, vì `build_mixed_dataset.py` từ chối mọi bản ghi chưa có `verified: true`. Repo chưa có số liệu nào về tốc độ duyệt, nên chưa quy ra được ngày hoàn thành.
