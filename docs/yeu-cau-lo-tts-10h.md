# Yêu cầu dữ liệu — lô TTS 10 giờ bổ sung cho v6

Viết cho người đặt hàng và người viết kịch bản lô TTS tiếp theo.

Mọi ngưỡng trong file này neo vào số đo thật của hai lô đã có: corpus đã tạo ra
adapter `v5` (lô tốt) và drop `50h-elevenlab-data` trong `v6-corpus` (lô hỏng).
Ngưỡng nào không phân biệt được hai lô đó thì không đưa vào đây.

## 0. Chú thích thuật ngữ

- **token** — một lần xuất hiện của từ. "deadline deadline deadline" = 3 token.
- **type** — một từ khác nhau. "deadline deadline checklist" = 2 type.
- **token tiếng Anh** trong tài liệu này = token tiếng Anh **viết thường** nằm
  trong câu tiếng Việt. **Không tính** acronym viết hoa (`ISO`, `KPI`, `SLA`,
  `ESG`, `HSE`) và tên riêng tiếng Việt (`Masan`, `Vinamilk`, `Techcombank`,
  `Vinpearl`). Lô 50 giờ đạt chỉ tiêu "2,81% từ ngoại lai" chủ yếu bằng hai
  loại này — 58% số token của nó là acronym và tên thương hiệu — nên chỉ tiêu
  đó không phản ánh chuyện gì về khả năng chuyển mã.

## 1. Vì sao cần lô này

Lô 50 giờ trước có mật độ chuyển mã 108 token/giờ, 98 type, mỗi type lặp trung
bình 48 lần. Nó làm loãng corpus xuống 229 token/giờ so với 851 của corpus v5,
và mô hình học phiên âm thay vì giữ nguyên từ tiếng Anh: `devops` thành
`để báo`, `customer journey` thành `cách sử dụng`. Retention trên
cross-domain-bench tụt từ 66,6% (v5) xuống 48,6% (v6).

Lô 10 giờ này không sửa được 100 giờ đó — phép chia không cho phép. Việc của nó
là nâng **lõi dữ liệu đậm** từ 17 giờ lên 27 giờ, đủ để train riêng.

## 2. Quy mô

- 10,0 giờ audio, khoảng 70 cuộc họp, mỗi cuộc 8–10 phút.
- Khoảng 8.500 segment, độ dài lượt nói trung vị 4–5 giây.

## 3. Ngưỡng nhận hàng

Đo trên nhãn text của manifest, trước khi thanh toán.

| Tiêu chí | Lô tốt (corpus v5) | Lô hỏng (50 giờ) | Ngưỡng |
|---|---:|---:|---|
| token tiếng Anh mỗi giờ | 851 | 108 | **≥ 900** |
| mật độ trung vị mỗi cuộc | 5,48% | 0,84% | **≥ 5%** |
| số giờ nằm ở cuộc dưới 2% | 5% | 100% | **≤ 10%** |
| số type mỗi giờ | 153 | 2 | **≥ 120** (≥ 1.200 type cho cả lô) |
| token trung bình mỗi type | 5,5 | 48,0 | **≤ 8** |
| tỷ trọng type phổ biến nhất | 1,59% | 5,25% | **≤ 2,5%** tổng token |
| câu mở đầu trùng nhau | — | 196/200 cuộc | **≤ 5%** số cuộc |

Ba tiêu chí **không** dùng, vì chính lô tốt cũng trượt:

- Sàn cứng mật độ từng cuộc (lô tốt có cuộc xuống 0,53%). Dùng trung vị và tỷ
  lệ giờ loãng thay thế.
- "Mỗi type xuất hiện 5–20 lần": phân bố tự nhiên là Zipf, lô tốt chỉ có
  215/863 type nằm trong dải đó.
- Trần tỷ lệ hapax (type chỉ xuất hiện một lần): lô tốt 37%, lô hỏng 4%. Chỉ
  tiêu này ngược dấu — ít hapax là dấu hiệu từ điển nghèo. Đã thay bằng
  "token mỗi type ≤ 8".

## 4. Kịch bản

- Mỗi cuộc một spec riêng, có kế hoạch chuyển mã ghi rõ, theo mẫu pool dot2
  (25 mức dễ / 25 mức khó).
- Không dùng chung câu mở đầu và câu kết. Lô 50 giờ có 200 cuộc dùng đúng 4 câu
  mở đầu, câu phổ biến nhất lặp nguyên văn 52 lần.
- Văn phong **nói suồng sã**: từ đệm, ngắt lời, nói chồng, lượt nói ngắn. Không
  viết theo giọng biên bản trang trọng — đó là nguyên nhân gốc của lô 50 giờ:
  trong 547.318 từ, `team` xuất hiện 0 lần, `user` 0, `customer` 0, `content` 0,
  `app` 0.
- Miền chủ đề: tuyển dụng và headhunt, marketing, devops và hạ tầng, phát triển
  sản phẩm, y tế. Chọn theo miền, **không** đặt theo danh sách từ của bộ đo —
  làm vậy thì bộ đo hết đo được khả năng tổng quát.

## 5. Giọng đọc

- Dùng lại đúng 10 giọng train hiện có.
- Không dùng giọng của tập val và test, nếu không sẽ mất tính tách biệt của bộ đo.

