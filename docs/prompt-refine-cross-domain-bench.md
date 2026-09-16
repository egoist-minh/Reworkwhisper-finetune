# Prompt: dựng lại reference của `cross-domain-bench` (soát mù)

Prompt để đưa cho một phiên Claude khác. Dán nguyên phần trong khung dưới.

## Vì sao làm mù

`dataset/cross-domain-bench` là bộ 299 segment quyết định mọi con số retention của dự án
(`experiments/task_ledger.md`). Nhãn của nó là `google_asr+human`, `verified: true` — bản nháp
Google ASR đã qua một lượt người soát. Sai sót còn lại, nếu có, nằm đúng chỗ đau nhất: một từ
ngoại lai bị phiên âm thành âm tiết Việt ngay trong reference sẽ **phạt model giữ nguyên chữ**
và **thưởng model phiên âm** — đảo ngược dấu của phép đo.

Nhưng loại lỗi đó chính là loại một người đọc đã chấp nhận một lần. Đưa reference ra rồi hỏi
"có sai không" thì người đọc thứ hai bị neo vào nó — đọc thấy `đi vốp` trong câu nói về CI/CD
và trôi qua, vì câu vẫn đọc được. Nên lượt này **không ai được nhìn reference của chính segment
đang xét**. Việc đó được cắt bằng dữ liệu, không bằng lời dặn: file input đơn giản là không
chứa nó.

## Dựng dữ liệu trước

```bash
PYTHONPATH=. python scripts/build_refine_input.py
```

Ra hai file:

| file | nội dung | ai được xem |
|---|---|---|
| `Outputs/cross-domain-refine-input.jsonl` | 299 bản ghi: 2 segment trước, 2 segment sau, 5 bản đọc gắn nhãn A–E | đưa cho phiên soát |
| `Outputs/cross-domain-refine-key.json` | reference thật + bản đồ chữ cái → tên hệ | **giữ lại**, chỉ dùng ở bước đối chiếu |

Nhãn A–E xáo lại riêng cho từng segment (114 hoán vị khác nhau trên 299 segment), nên không hệ
nào mang theo danh tiếng của nó sang bản ghi kế tiếp. Người soát không biết đâu là v5, đâu là
Scribe, đâu là model đang được chấm.

---

```
Bạn đang dựng lại transcript của một bộ benchmark ASR tiếng Việt, mù — bạn KHÔNG có transcript
hiện tại của segment đang xét, và đó là chủ ý. Mọi con số retention và CER của dự án đo trên
bộ này; một lỗi trong transcript không làm model tệ đi, nó làm phép đo sai dấu. Lỗi loại đó đã
sống sót qua một lượt người soát, nên lượt này không được neo vào bản cũ.

## Dữ liệu

`Outputs/cross-domain-refine-input.jsonl`, 299 bản ghi, mỗi dòng một segment:

- `segment_id`, `meeting_id`, `position` (thứ tự trong cuộc), `duration_s`
- `truoc` — transcript của 2 segment liền trước
- `sau` — transcript của 2 segment liền sau
- `ban_doc` — 5 bản đọc của segment này do 5 hệ ASR khác nhau sinh ra, gắn nhãn A–E

Ba cuộc họp, người Việt nói chuyện tự nhiên, code-switch dày: `IGZYBrbDUEw` (tuyển dụng, nghề
tech), `z-nODwLyhA0` (digital marketing, SEO), `ySdJ3sg_2lk`.

## Những gì bạn không có, và phải xử sự đúng với việc đó

**Không có audio.** Bạn không nghe được câu này. Mọi phán đoán dựa trên: ngữ cảnh hai đầu, sự
khác biệt giữa 5 bản đọc, và kiến thức của bạn về tiếng Việt cùng thuật ngữ của ngành đang nói.

**Không có transcript cũ.** Bạn không được đi tìm nó trong repo — không mở
`dataset/cross-domain-bench/`, không mở `Outputs/cross-domain-refine-key.json`, không mở các file
`*.persegment.jsonl` (chúng chứa cột `ref`). Nếu bạn vô tình thấy nó, dừng lại và báo, đừng tiếp
tục segment đó.

**Nhãn A–E vô nghĩa.** Chúng xáo lại ở mỗi segment. "Hệ A đáng tin" là một câu vô nghĩa ở đây.

## Việc phải làm, với mỗi segment

Viết ra bản đọc mà bạn tin là ĐÚNG với những gì người nói đã nói. Không phải bản hay nhất,
không phải bản dễ đọc nhất — bản đúng nhất.

Cách làm: đọc `truoc` và `sau` trước để nắm mạch chuyện và chủ đề. Rồi đọc cả 5 bản trong
`ban_doc`, tìm chỗ chúng khác nhau. Chỗ 5 bản giống nhau thì gần như chắc chắn đúng, chép lại.
Chỗ chúng khác nhau là nơi cần bạn quyết, và căn cứ để quyết là:

- **Mạch chuyện**: cuộc này đang nói về gì, câu trước dẫn tới đâu, câu sau nối tiếp thế nào.
- **Thuật ngữ ngành**: trong một câu về triển khai phần mềm, `devops` là thật và `đi vốp` là
  bản phiên âm của một hệ nghe không ra. Trong một câu về tuyển dụng, `jd` là thật.
- **Ngữ pháp và độ trôi chảy của tiếng Việt nói**.

Không bỏ phiếu theo đa số. Bốn hệ cùng sai một chỗ là chuyện thường — chúng cùng học từ dữ
liệu giống nhau. Một bản lẻ loi đúng thì nó đúng.

## Từ ngoại lai — trọng tâm của cả lượt này

Bộ benchmark này tồn tại để đo một thứ: model có giữ nguyên chữ của từ ngoại lai hay phiên âm
nó thành âm tiết Việt. Nên chỗ 5 bản đọc khác nhau ở một từ ngoại lai là chỗ quan trọng nhất
trong toàn bộ công việc này.

Quy tắc: **viết từ ngoại lai bằng chính chữ của nó** khi người nói đang dùng chính từ đó —
`devops`, `content`, `jd`, `marketing`, `cloud`, `seo`, `apply`, `follow`. Chỉ viết dạng phiên
âm khi từ đó đã thành từ tiếng Việt thật sự và người nói đang dùng nó như tiếng Việt.

Cẩn thận hai chiều. Có những chuỗi vừa là từ tiếng Anh vừa là âm tiết tiếng Việt hợp lệ —
`seo` là một ví dụ, `ba` (business analyst) là một ví dụ khác. Chỉ ngữ cảnh mới phân biệt được;
đừng máy móc theo một hướng nào.

Đây là phép thử hình dạng âm tiết, không phải danh sách từ tiếng Anh. Dự án đã đo được tỷ lệ
báo động giả 72% khi dùng danh sách trắng — đừng quay lại cách đó.

## Quy ước viết, bắt buộc theo

Bộ này có quy ước riêng. Lệch quy ước sẽ bị đếm thành lỗi ở bước đối chiếu, làm nhiễu kết quả
thật:

- **Chữ thường toàn bộ.** Không viết hoa tên riêng, không viết hoa đầu câu.
- **Không dấu câu.** Không phẩy, không chấm, không hỏi chấm.
- **Giữ nguyên từ đệm và lặp**: `ờ`, `à`, `á`, `cái cái cái` là lời nói thật. Các bản đọc A–E
  có thể đã bỏ chúng — bạn thì không được bỏ, nếu ngữ cảnh cho thấy người nói có nói.
- **Số viết bằng chữ số**: `4 5 năm`, `30 giây`, `1 phút`. (Khâu chuẩn hoá của dự án sẽ quy về
  một dạng, nên đây không phải chỗ quyết định — đừng dừng lại lâu ở nó.)

## Đầu ra

Một file JSONL `Outputs/cross-domain-refine-output.jsonl`, mỗi segment một dòng, đủ 299 dòng,
đúng thứ tự file input:

{"segment_id": "...", "ban_dung": "<transcript bạn tin là đúng>",
 "do_tin_cay": "cao|vua|thap",
 "cho_phan_van": ["<từ hoặc cụm bạn không chắc, nếu có>"],
 "ghi_chu": "<chỉ khi có gì đáng nói, để trống nếu không>"}

`do_tin_cay` nói về **cả segment**:
- `cao` — 5 bản gần như thống nhất, hoặc chỗ khác nhau được ngữ cảnh giải quyết dứt khoát.
- `vua` — có chỗ bạn phải chọn, và bạn chọn được có lý do, nhưng không loại trừ hẳn khả năng khác.
- `thap` — các bản đọc phân kỳ nặng, ngữ cảnh không cứu được, cần nghe audio mới biết.

`thap` là kết luận hợp lệ và hữu ích. Một bản `thap` trung thực đáng giá hơn một bản `cao` đoán
liều — bước sau sẽ lọc theo chính trường này.

## Đừng làm

- Đừng đi tìm transcript cũ. Đừng mở `dataset/cross-domain-bench/`,
  `Outputs/cross-domain-refine-key.json`, hay bất kỳ file `*.persegment.jsonl` nào.
- Đừng bỏ qua segment nào. Đủ 299 dòng, kể cả những segment bạn thấy tầm thường — segment nhìn
  tầm thường vẫn có thể chứa đúng một từ ngoại lai bị phiên âm, và đó là thứ cần tìm.
- Đừng "làm sạch" câu nói: không sửa ngữ pháp người nói, không bỏ lặp, không rút gọn.
- Đừng viết một bản dung hoà giữa 5 bản đọc. Chọn cái đúng, không trộn.
```

---

## Bước đối chiếu, sau khi có output

Ghép `Outputs/cross-domain-refine-output.jsonl` với `Outputs/cross-domain-refine-key.json` theo
`segment_id`, rồi đọc theo ba lớp:

1. **Segment bản dựng khớp reference** — bench đúng ở đó, không cần làm gì.
2. **Segment khác nhau, `do_tin_cay` = `cao`** — ứng viên lỗi reference mạnh nhất. Đây là tập
   cần người nghe lại audio để chốt. Đọc kỹ những chỗ khác nhau ở một từ ngoại lai trước.
3. **Segment khác nhau, `do_tin_cay` = `thap`** — nhiều khả năng là giới hạn của việc soát mù
   không có audio, không phải lỗi bench. Để riêng.

Hai điều cần biết khi đọc kết quả:

**Bản dựng mù không phải chân lý.** Nó được dựng từ output của chính những hệ đang được chấm,
cộng ngữ cảnh. Chỗ nó khác reference là chỗ **đáng nghe lại**, không phải chỗ reference sai.
Chốt bằng tai người, trên audio, rồi mới sửa.

**Đừng sửa `manifest.*.jsonl` tại chỗ.** Bench cũ phải giữ nguyên để so được với mọi số đã công
bố. Sau khi chốt, áp patch ra một bản bench riêng rồi chấm lại **cả hai bản** cho đủ 5 hệ bằng
`scripts/benchmark_report.py` (chuẩn hoá `N3` — đừng chấm tay, `src.metrics.score` trên text thô
phạt nặng hệ có dấu câu và chữ hoa, đã làm sai một bảng trong phiên 16/09).

Thứ cần đọc là **delta retention của từng hệ**, không phải CER tuyệt đối. Nếu bản sửa nâng
retention của mọi hệ gần như nhau thì nó chỉ dọn nhiễu. Nếu nó nâng riêng những hệ giữ nguyên
chữ ngoại lai mà không nâng hệ phiên âm, thì lỗi reference đang che một phần khoảng cách thật.
