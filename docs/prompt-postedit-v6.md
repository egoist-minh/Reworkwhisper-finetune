# Prompt: hậu xử lý output của v6 bằng LLM

Đo một câu hỏi: **một model ngôn ngữ, chỉ đọc thứ model ASR đã viết ra, có sửa được đủ nhiều để
kéo CER xuống và retention lên không** — không train lại, không nghe audio. Nếu có, đây là đòn
bẩy dùng được ngay; nếu không, biết sớm còn hơn tiêu thêm một lượt thuê GPU.

Khác hẳn [prompt soát mù reference](prompt-refine-cross-domain-bench.md): ở đó ta nghi ngờ
**reference**; ở đây ta coi reference là đúng và sửa **hypothesis**.

## Dựng dữ liệu

```bash
PYTHONPATH=. python scripts/build_postedit_input.py --system v6_l075
```

| file | nội dung | ai được xem |
|---|---|---|
| `Outputs/v6_l075-postedit-input.jsonl` | 299 bản ghi: `ban_may` (bản v6 đọc ra) + 2 segment trước/sau, cũng của chính v6 | đưa cho phiên hậu xử lý |
| `Outputs/v6_l075-postedit-key.json` | reference + hypothesis gốc | **giữ lại**, chỉ dùng khi chấm |

Hai chỗ rò đã chặn từ khâu dữ liệu, không bằng lời dặn:

- **Reference không có trong input.** Thấy ref là lộ đáp án, con số sau đó vô nghĩa.
- **Ngữ cảnh hai đầu là hypothesis của chính v6, không phải reference của hàng xóm.** Một bộ
  hậu xử lý chạy thật không có reference nào cả; lấy ref làm ngữ cảnh là rò đáp án theo đường
  vòng và thổi phồng kết quả.

Chỉ một hệ vào input. Đưa thêm output của v5/Scribe vào là đo **ensemble**, không phải hậu xử
lý — và ensemble tốn bốn lượt decode lúc chạy thật, một trong đó là API trả tiền. Muốn đo
ensemble thì chạy `--ensemble`, và gọi đúng tên nó khi trình bày.

---

```
Bạn đang sửa lỗi cho output của một hệ nhận dạng tiếng nói tiếng Việt. Mỗi bản ghi là một
đoạn audio khoảng 15 giây đã được máy chuyển thành chữ; bạn đọc bản máy viết ra cùng hai đoạn
liền trước và liền sau, rồi viết lại đoạn đó cho đúng hơn.

## Dữ liệu

`Outputs/v6_l075-postedit-input.jsonl`, 299 bản ghi:

- `ban_may` — bản máy đọc ra cho đoạn này. Đây là thứ bạn sửa.
- `truoc` / `sau` — bản máy đọc ra cho 2 đoạn liền trước và liền sau. Ngữ cảnh, không sửa.
- `segment_id`, `meeting_id`, `position`

Ba cuộc họp, người Việt nói chuyện tự nhiên, code-switch dày: `IGZYBrbDUEw` (tuyển dụng, nghề
tech), `z-nODwLyhA0` (digital marketing, SEO), `ySdJ3sg_2lk`.

Bạn KHÔNG có audio và KHÔNG có transcript đúng. Đừng đi tìm chúng trong repo — không mở
`dataset/cross-domain-bench/`, không mở file `*-postedit-key.json`, không mở file
`*.persegment.jsonl` nào (chúng chứa cột `ref`).

## Cách chấm bạn, và vì sao nó đổi cách bạn nên làm

Bản bạn viết ra sẽ được so ký tự với một transcript tham chiếu mà bạn không được thấy. Mỗi ký
tự khác là một lỗi, kể cả khác về cách viết.

Hệ quả quan trọng: **mỗi lần sửa là một lần đặt cược.** Sửa đúng thì bớt lỗi; sửa sai thì thêm
lỗi — và sửa một chỗ vốn đã đúng thì chắc chắn thêm lỗi. Bản máy đã đúng phần lớn. Nên nguyên
tắc là: **chỉ sửa khi ngữ cảnh làm cho từ đúng gần như chắc chắn.** Ngờ ngợ thì để nguyên. Để
nguyên một chỗ sai không làm bạn tệ đi so với bản gốc; sửa hỏng một chỗ đúng thì có.

## Quy ước viết của bộ tham chiếu — bắt buộc theo

Lệch quy ước bị đếm là lỗi ký tự, nên đây không phải chuyện thẩm mỹ:

- **Chữ thường toàn bộ.** Bản máy có thể viết hoa (`YouTube`, `Viettel`, `BA`) — hạ hết xuống
  chữ thường.
- **Không dấu câu.** Bản máy có thể có phẩy và chấm — bỏ hết.
- **Giữ nguyên từ đệm và lặp**: `ờ`, `à`, `á`, `cái cái cái`. Đây là lời nói thật và transcript
  tham chiếu có giữ chúng. Bỏ chúng đi là tự thêm lỗi.
- **Số viết bằng chữ số**: `4 5 năm`, `30 giây`, `1 phút`.

## Sửa cái gì

1. **Từ ngoại lai bị phiên âm thành âm tiết Việt.** Đây là loại quan trọng nhất và cũng là loại
   dễ nhận nhất. Máy nghe không ra một từ tiếng Anh rồi viết thành âm tiết Việt nghe na ná:
   `cái cd đó` trong câu về tuyển dụng gần như chắc chắn là `cái jd đó`; `copy từ kinh tây nước
   ngoài` là `copy từ content nước ngoài`; `mình mượn sờ` là `mình search`. Ngữ cảnh chuyên
   ngành quyết định, không phải độ giống âm.

2. **Từ tiếng Việt bị nghe nhầm thành từ khác**, khi ngữ cảnh chỉ cho phép một cách đọc:
   `vô vàng` → `vô vàn`, `cái đây 4 năm năm` → `cách đây 4 5 năm`.

3. **Chữ hoa và dấu câu** theo quy ước ở trên. Đây là loại sửa an toàn nhất — không cần đoán gì.

## Đừng sửa

- **Đừng làm sạch câu nói.** Người nói lặp, ngập ngừng, nói sai ngữ pháp — transcript tham chiếu
  ghi đúng như vậy. Sửa cho trôi chảy là thêm lỗi.
- **Đừng diễn đạt lại.** Không đổi trật tự từ, không rút gọn, không thay từ đồng nghĩa.
- **Đừng dịch.** Từ tiếng Anh giữ nguyên tiếng Anh, từ tiếng Việt giữ nguyên tiếng Việt.
- **Đừng nối hay cắt đoạn.** Mỗi bản ghi là một đoạn, biên của nó cố định.
- **Đừng sửa những chỗ chỉ nghi ngờ mơ hồ.** Xem lại phần "mỗi lần sửa là một lần đặt cược".

## Đầu ra

`Outputs/v6_l075-postedit-output.jsonl`, mỗi segment một dòng, đủ 299 dòng, đúng thứ tự input:

{"segment_id": "...", "ban_sua": "<bản đã sửa, hoặc nguyên bản máy nếu không sửa gì>",
 "da_sua": true|false,
 "cho_sua": [{"tu": "cd", "thanh": "jd", "vi_sao": "câu đang nói về mô tả công việc"}]}

`ban_sua` luôn phải có, kể cả khi không sửa gì — khi đó chép nguyên `ban_may`. `cho_sua` để
trống nếu không sửa.

## Hiệu chuẩn

Bản máy này có CER khoảng 9% — tức khoảng 9 trong 100 ký tự sai. Phần lớn câu đọc gần đúng.
Nếu bạn thấy mình đang sửa quá nửa số segment, gần như chắc chắn bạn đang viết lại thay vì sửa
lỗi — dừng lại, đọc lại phần "Đừng sửa".
```

---

## Chấm

```bash
PYTHONPATH=. python scripts/score_postedit.py \
    --edited Outputs/v6_l075-postedit-output.jsonl \
    --key Outputs/v6_l075-postedit-key.json
```

In ra CER/WER/retention của bản gốc và bản đã sửa, delta, và **paired bootstrap dCER** —
phép so có cặp, vì hai bản chấm trên đúng cùng một tập segment. Khoảng tin cậy không cắt 0 thì
mới gọi là cải thiện thật; chênh 0,3 điểm với khoảng tin cậy `[-0,9, +0,4]` là nhiễu, không
phải kết quả.

Script cũng in số segment bị đụng vào — đọc kèm delta. Sửa 200/299 segment mà CER chỉ nhích
0,2 điểm nghĩa là phần lớn chỗ sửa vô ích hoặc triệt tiêu nhau.

## Khi đưa số này lên bảng

Đây là **model cộng một lượt LLM hậu xử lý**, không phải CER của model. Hai cột khác nhau, và
cột hậu xử lý phải ghi đúng tên vì nó tốn thêm một lệnh gọi LLM cho mỗi segment lúc chạy thật —
khác hẳn về chi phí triển khai so với các cột còn lại. Trình nó như CER của v6 là sai sự thật.
