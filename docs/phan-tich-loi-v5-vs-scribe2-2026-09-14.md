# Vì sao Reworkwhisper-large-v5 thua ElevenLabs Scribe v2

Nguồn: `Outputs/review_youtube_data_pilot-20260827T054523Z-1-001.zip` (per-segment thật, không phải số tổng hợp), đối chiếu lại với `docs/bao-cao-benchmark-elevenlabs-scribe2-vs-reworkwhisper-v5-2026-08-27.md`.

## Tóm tắt

299 đoạn, 3 domain (HR, y khoa, marketing). v5 thua Scribe v2 ở cả 3 domain (CER gộp 9.7% so với 8.1%), nhưng khoảng cách này không dàn đều — tập trung ở 2 lỗi cụ thể:

1. **Cắt cụt câu giữa chừng** — chỉ ở domain y khoa, 2/69 đoạn, nhưng CER 2 đoạn này tới 84-95% trong khi Scribe cùng đoạn chỉ 10-12%. Một mình 2 đoạn này kéo lệch cả domain.
2. **Garble từ vay mượn tiếng Anh/thuật ngữ kỹ thuật** — v5 phiên âm sai thành âm tiết Việt vô nghĩa (`devops` → "đếp ốp"), Scribe giữ nguyên chữ Latin đúng. Lỗi này lặp lại ở cả 3 domain, không phải ngẫu nhiên.

## Mẫu CER cao nhất (toàn bộ 299 đoạn)

| Domain | Đoạn | v5 CER | Scribe CER | Loại lỗi |
|---|---|---:|---:|---|
| Y khoa | seg_0012 | 94.9% | 10.3% | cắt cụt câu |
| Y khoa | seg_0010 | 83.8% | 11.7% | cắt cụt câu |
| Y khoa | seg_0058 | 62.6% | 12.9% | cắt cụt câu |
| Y khoa | seg_0014 | 41.8% | 39.5% | định dạng số (cả hai model đều lỗi ngang nhau — artifact chấm điểm, không phải model yếu) |
| Marketing | seg_0112 | 27.9% | 13.7% | garble loanword |

### seg_0012 (y khoa) — cắt cụt câu, lỗi nặng nhất benchmark
```
REF: "đầu tiên ạ vầng kính thưa thầy thành kính thưa các anh chị em đồng nghiệp thì ở bệnh nhân này..."
v5:  "đầu tiên ạ."                    ← dừng sau 3 từ, audio còn dài ~40 từ nữa
Scribe: dịch đủ cả câu, đúng gần hết
```

### seg_0010 (y khoa) — cùng lỗi cắt cụt
```
REF: "đặc biệt từ trên phim này ạ vâng vâng không thấy có nguyên nhân gây giãn đại bể thận..."
v5:  "đặc biệt ở trên phim này."      ← cắt tương tự
```

### seg_0112 (marketing) — garble loanword, đại diện lỗi phổ biến nhất
```
REF: "...pull audience càng cao... audience sight càng hẹp... cách funnel chúng ta..."
v5:  "...pull audience càng cao... audience size càng hẹp... cái phân hồ chúng ta..."
Scribe: "...pool audience... audience size... funnel..." — giữ đúng thuật ngữ
```
`funnel` → v5 nghe thành "phân hồ" (vô nghĩa), Scribe giữ nguyên chữ Latin.

## Kết luận

Không phải v5 yếu đều khắp — 2 lỗi cụ thể, đo lặp lại được (cắt cụt câu ở câu dài/thuật ngữ hiếm; garble loanword tiếng Anh) gánh phần lớn khoảng cách CER với Scribe v2. Domain y khoa bị thổi phồng thêm bởi artifact chấm điểm số liệu (chữ số vs chữ viết), ảnh hưởng cả hai model như nhau.
