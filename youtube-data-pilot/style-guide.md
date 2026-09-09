# Style guide — soát nháp YouTube pilot

Áp cho `dataset/youtube-meetings/manifest.*.jsonl` khi soát qua `scripts/review_youtube.py`.
Versioned cùng dataset — sửa quy tắc ở đây thì ghi rõ ngày đổi, đừng xoá quy tắc cũ mà không giải thích (giống cách README §5 ghi "Đổi 2026-08-11 — bản trước ghi...").

## Quy trình chạy (mỗi meeting lặp lại 5 bước)

7 `meeting_id`: `dGT3YW0AdD8`, `rIFrrmm8ILY`, `rCd8DSMk3-c`, `iyeFAuuEBl4`,
`7B24A9GfHAo`, `xKDHjUoUN54`, `3nuCdzuyqng`. Làm tuần tự, không cần làm hết 7 trong
một buổi — `--check` chấp nhận `--meeting-id` để xác nhận từng meeting riêng.

1. **Sinh worksheet:**
   ```
   python -m scripts.review_youtube --emit <meeting_id>
   ```
   → `youtube-data-pilot/review/review.<meeting_id>.html`

2. **Mở file đó bằng browser** (double-click, hoặc `start <path>` trên Windows).
   Nghe từng `<audio>`, sửa `<textarea>` theo 4 quy ước dưới + phần Casing cuối
   trang này. **Nghe hết, sửa hết** — không bỏ trống ô nào, kể cả khi nháp đã đúng
   (gõ lại nguyên nháp cũng được, nhưng ô không được trống).

3. **Bấm "Xuất file sửa"** ở đầu trang → browser tải `corrections.<meeting_id>.json`
   (thường vào `Downloads/`).

4. **Di chuyển file đó vào repo rồi apply:**
   ```
   mv ~/Downloads/corrections.<meeting_id>.json youtube-data-pilot/review/
   python -m scripts.review_youtube --apply <meeting_id> \
       --corrections youtube-data-pilot/review/corrections.<meeting_id>.json \
       --reviewed-by "<ten-nguoi-soat>"
   ```
   Thiếu correction cho segment nào, hoặc ô nào để trống → lệnh **raise**, nêu
   đúng `meeting_id/segment_id` — quay lại bước 2, sửa/bổ sung, xuất lại.

5. **Xác nhận xong đúng meeting đó:**
   ```
   python -m scripts.review_youtube --check --meeting-id <meeting_id>
   ```
   Muốn biết còn thiếu bao nhiêu trên **toàn bộ** 7 meeting: chạy `--check`
   không kèm `--meeting-id`.

**Quyết định đã chốt, đừng tự động hoá:** đã cân nhắc tự động bước 3-4 (server
local, File System Access API, `--watch`) và **chọn giữ tay** — friction chỉ 7
lần (1/meeting), tự động hoá tốn code hơn công tiết kiệm, và một số cách (server
auto-apply) phá nguyên tắc "`--apply` đòi đủ correction, không cho ghi dở dang"
đang có. Không mở lại quyết định này khi không có dữ kiện mới.

## Bốn quy ước nội dung

Bốn quy ước dưới đây chuyển thẳng từ `youtube-data-pilot/README.md` bước 4 — đọc bước 4 để có lý do đầy đủ, ở đây chỉ ghi phần hành động cụ thể lúc gõ sửa.

## 1. Không xoá filler

`ạ à ừ ơ dạ vâng nhé nhỉ` (`src.config.LEXICAL_PARTICLES`) là từ có nghĩa — tiểu từ lịch sự/nghi vấn/xác nhận, không phải rác. **Giữ nguyên khi nghe thấy**, kể cả khi nghe hơi thừa. Bằng chứng: bản webinar `viet-speech` đo được 107 lỗi `filler_convention` trên ~380 lỗi đếm được — gần 1/4 lỗi là ảo vì thiếu quy ước này.

## 2. Số: chữ số cho giá trị rõ ràng là số

Tiền, %, thời gian, số lượng → gõ **chữ số** (`"200ms"`, `"95%"`, `"24 tiếng"`). Phần còn lại (số đếm mơ hồ như "một bộ", "năm nay" nghĩa *year*) → giữ chữ, đừng tự đổi. Đổi 2026-08-11 từ quy ước cũ "viết bằng chữ" — xem README §5 để có lý do đầy đủ (khớp corpus hiện tại, cả hai nguồn nháp đều xuất chữ số).

## 3. Chồng lấn: chép người nói trội

Khi hai người nói chồng lên nhau, chép lời của người nói **trội hơn** (nghe rõ hơn), đặt field `quality: overlap` cho record đó. Phần chồng lấn **có** trong train, **không** tính vào CER công bố.

## 4. Token tiếng Anh giữ đúng chính tả người nói

Người nói code-switch chèn từ tiếng Anh thì gõ đúng từ đó, đúng chính tả tiếng Anh — không phiên âm, không Việt hoá. Đây là ground truth cho gate code-switch (README bước 7b) — gõ sai chính tả ở đây làm hỏng phép đo tỉ lệ giữ token tiếng Anh.

## Casing (nháp hiển thị đã tự xử lý, không phải quy ước phải nhớ)

`--emit` tự hạ toàn bộ nháp về chữ thường trước khi hiển thị — khớp đúng convention `dataset/real-meetings-bench` (real speech, đo được: toàn chữ thường, không viết hoa đầu câu), khác `paid-dataset-v2` (synthetic, viết hoa chuẩn câu) vì đây là dữ liệu speech thật, không phải văn bản viết sẵn.

Lý do phải làm tự động, không để người soát tự gõ: caption Google ASR viết hoa ngẫu nhiên giữa câu, không theo quy tắc ngữ pháp nào — đo được thật trên `7B24A9GfHAo/seg_0000`: *"...thì họ có đánh giá làm sao **Là sao Bạn** có muốn làm rõ lại câu hỏi này..."* — "Là sao Bạn" viết hoa giữa câu, không sau dấu câu, không phải danh từ riêng. Sửa tay từng chữ hoa như vậy trên 790 segment là phí công không cần thiết.

**Không cần gõ lại chữ hoa** khi sửa — cứ gõ chữ thường tự nhiên, đúng convention corpus. Nếu nghe được danh từ riêng thật (tên người, tên công ty) thì viết hoa theo ý bạn, `--apply` không ép lại thành chữ thường.
