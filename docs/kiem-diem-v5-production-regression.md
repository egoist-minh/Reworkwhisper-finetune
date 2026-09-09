# Kiểm điểm: hồi quy `Reworkwhisper-large-v5` trên production

> **Phạm vi:** một model qua đủ ba cổng kiểm, được publish, đưa vào production, và làm
> pipeline ASR tệ đi. Tài liệu này hỏi *vì sao quy trình cho phép điều đó*, không phải
> *model sai chỗ nào* — phần sau nằm ở [`finetune-results-report-v4.md`](finetune-results-report-v4.md)
> mục 4–5 (bản rút gọn của `finetune-results-report-v4-mixed-r16.md`, báo cáo kỹ thuật đầy đủ
> không còn trên đĩa), và nhật ký điều tra đầy đủ ở [`SESSIONS.md`](../SESSIONS.md).
>
> **Kết luận ngắn:** weights chưa bao giờ hỏng. Cùng bộ adapter đó, đặt đúng λ, nó tốt hơn
> model production **47,6% CER tương đối** trên 654 segment dùng chung. Toàn bộ sự cố là lỗi
> quy trình, và phần lớn đã được viết ra giấy **trước khi** nó gây hậu quả.

---

## 1. Trình tự

| Thời điểm | Việc |
|---|---|
| 2026-08-16 → 17 | `v4-mixed-r16` train xong, gate `overall_pass: true` ở λ=0.25, publish thành `winhsss/Reworkwhisper-large-v5` |
| 2026-08-17 11:50 | Báo cáo kết quả run được viết. **Mục 4 chẩn đúng lỗi `select_lambda`**, tính đúng cả tỉ số 11,7×, kết luận đúng rằng elbow thật nằm ở bước 0.5→0.75, và đề xuất đúng hai việc cần làm |
| 2026-08-17 | v5 vào pipeline production ở `d:\viet-speech`. Kết quả tệ hơn: `team`→`tim`, `build`→`bill`, chữ hoa 15→1, module trích thực thể mất chức danh |
| 2026-08-17 → 18 | Điều tra, sửa lỗi, dựng chỉ số mới, gate lại ở λ=0.75, rollback production về v4 |

Khoảng cách giữa hàng 2 và hàng 3 là trọng tâm của bản kiểm điểm này: **phân tích đúng đã tồn
tại, viết ra rõ ràng, kèm việc cần làm — và model vẫn lên production.**

---

## 2. Lỗi quy trình

### 2.1 Cổng kiểm chưa bao giờ so với model đang chạy production

Đây là lỗi gốc, và là lỗi đúng bất kể nguyên nhân hành vi là gì.

Mọi tier chỉ so ứng viên với baseline base-model **của chính nó**. Không có phép so nào giữa
ứng viên và thứ đang phục vụ người dùng. Hậu quả đo được: `v3-r16` và `v4-mixed-r16` cách nhau
**51% tương đối** về CER trên đúng 426 segment synthetic dùng chung, `ref` khớp từng byte — mà
**cả hai đều `overall_pass: true`**.

Không có cơ chế nào trong pipeline có thể phát hiện điều này. Cổng đo "ứng viên có tốt hơn base
model không", còn câu hỏi thật là "ứng viên có tốt hơn cái đang chạy không". Hai câu hỏi khác
nhau, và chỉ câu thứ hai quyết định có nên deploy.

**Đã sửa:** `check_no_regression_vs_production` trong [`scripts/merge_and_push.py`](../scripts/merge_and_push.py) —
join theo `(meeting_id, segment_id)`, raise nếu `ref` lệch giữa hai file, raise nếu ứng viên
hồi quy trên tập chung, chạy **trước** bước merge tốn thời gian.

### 2.2 Phân tích đúng đã có sẵn nhưng không thành hành động

Báo cáo ngày 2026-08-17 11:50 viết nguyên văn:

> *Bước đầu làm OOD CER **giảm**, nên tỉ lệ đầu tiên **âm**. Ngưỡng cho bước sau vì thế cũng
> âm (−0,046), và bất kỳ tỉ lệ dương nào cũng vượt ngay lập tức. λ=0.5 bị loại không phải vì
> nó là elbow thật, mà vì `prev_ratio` âm làm ngưỡng vô nghĩa.*

Chẩn đoán này **chính xác từng chi tiết**, kể cả kết luận elbow thật nằm ở bước 0.5→0.75. Nó
còn liệt kê đúng hai việc cần làm: chặn `prev_ratio` ở mức dương, và đo λ=0.5 trên họp thật.

Cả hai đều không được làm. Model lên production và hồi quy.

**Đây là lỗi nặng nhất về mặt quy trình**, vì nó không phải lỗi phân tích — phân tích đã đúng
và kịp thời. Lỗi nằm ở chỗ **một phát hiện viết trong tài liệu không có cơ chế nào buộc nó
phải được xử lý trước khi deploy**. Nó là một đoạn văn trong mục 4 của một báo cáo 380 dòng,
ngang hàng với mọi đoạn văn khác.

### 2.3 "Qua cổng" bị đọc thành "đủ tốt"

Cùng mục 4 của báo cáo đó kết luận: *"Kết quả cuối vẫn tốt — λ=0.25 qua cả ba cổng"*.

Câu này biến **pass** thành **đủ tốt**. Nhưng pass chỉ có nghĩa "không tệ hơn base model quá
ngưỡng" — nó không nói gì về việc có tốt hơn bản đang chạy hay không (§2.1), và cả ba cổng lúc
đó đều chỉ đo CER (§2.4).

Đo lại đầy đủ thì λ=0.25 là mức **tệ nhất trong bốn mức λ**, không phải "đủ tốt": λ=0.75 hơn nó
22% CER tương đối và 14,5pp retention trên YouTube.

**Luật rút ra:** pass là **sàn**, không phải trần. Một run pass mọi cổng vẫn có thể là lựa chọn
tệ nhất trong các lựa chọn sẵn có.

### 2.4 Chỉ số không bắt được kiểu lỗi mà người dùng báo

Triệu chứng production là **mất từ mượn tiếng Anh** (`team`→`tim`). Cổng kiểm chỉ có CER.

Một từ mượn là vài ký tự trong một segment hàng trăm ký tự. Mất sạch từ mượn chỉ đẩy CER lên
một phần nhỏ của điểm phần trăm — nằm gọn trong bề rộng khoảng tin cậy, không phân biệt được
với nhiễu. Thay `team` bằng `tim` tốn đúng 2 ký tự edit distance nhưng **phá huỷ token đó cho
mọi thứ downstream đọc thực thể ra khỏi transcript**.

Tệ hơn: `style-guide.md` quy ước 4 tuyên bố chính tả tiếng Anh là "ground truth cho gate
code-switch", nhưng grep `code_switch|english` trong `src/gate.py` ra **0 kết quả**. Cổng đó
chưa từng được xây. Tài liệu mô tả một cơ chế không tồn tại.

**Đã sửa:** `english_token_retention` trong [`src/metrics.py`](../src/metrics.py) + tier
`retention`/`retention_pass` trong [`src/gate.py`](../src/gate.py).

### 2.5 Chuẩn hoá xoá mất một nửa triệu chứng trước khi đo

`normalization.lowercase: true` áp cho **cả `hyp` lẫn `ref`**. Chữ hoa bị mất đóng góp **đúng
0** vào CER. Nửa còn lại của triệu chứng production — chữ hoa 15→1, module trích thực thể mất
chức danh — không cổng nào nhìn thấy được, và vẫn không nhìn thấy được cho tới giờ.

Nhãn train YouTube có **0 chữ hoa trên 178.075 ký tự** ([`scripts/review_youtube.py:78`](../scripts/review_youtube.py#L78)
là `text.lower()` áp cho toàn bộ), và đó là **quy ước cố ý**, ghi rõ trong `style-guide.md`.
Hợp lý cho CER, nhưng chính nó làm module 8 mất thực thể. Một quyết định đúng cho chỉ số lại
sai cho sản phẩm, và không có chỗ nào trong quy trình đặt hai thứ đó cạnh nhau.

**Chưa sửa** — cần train lại (row D1).

### 2.6 Tham chiếu production sai λ, suýt đánh trượt ứng viên đúng

Khi đã có `check_no_regression_vs_production`, tham chiếu hiển nhiên là
`Outputs/v3-r16/audit/predictions_tier1_in_domain.csv`. File đó **được chấm ở λ=1.0**, trong
khi bản publish chạy production là λ=0.5 — run v3-r16 gate bằng luật cũ "λ lớn nhất còn trong
ngân sách", còn 0.5 là lựa chọn của người sau đó.

Trên synthetic, λ=1.0 cho 1,71% còn λ=0.5 thật sự cho **1,96%**. Ứng viên λ=0.75 đạt 1,61% —
**đậu** so với model thật, nhưng nếu dùng file cũ thì check đã **raise và từ chối push**.

Nghĩa là: vừa sửa xong lỗi "so sai đối tượng" thì suýt lặp lại chính nó ở dạng khác — cơ chế
đúng, dữ liệu nạp vào sai. Chỉ tránh được vì có người soát lại nhãn λ của file, không phải vì
thiết kế ngăn được.

### 2.7 Mọi con số của repo này được đo trên một đường decode production không chạy

Trục này nằm dưới cả §2.1 lẫn §2.4, và cho tới 2026-08-19 chưa được nêu ở đâu. §2.1 sửa việc so
sai **đối tượng**; §2.4 sửa việc đo sai **thứ cần đo**. Nhưng cả hai đều thực hiện phép đo bằng
[`src/asr.py:transcribe_batch`](../src/asr.py#L57) — và module 4 của `d:\viet-speech`, thứ thật
sự phục vụ người dùng, decode bằng một thuật toán khác:

| Hạng mục | Cổng kiểm | Module 4 (production) |
|---|---|---|
| `generate()` | `language`, `task`, `num_beams=1` | thêm `return_timestamps`, `condition_on_prev_tokens=True`, `no_speech_threshold=0.6`, `logprob_threshold=-1.0`, `compression_ratio_threshold=2.4`, `temperature=(0.0 … 1.0)` |
| Độ dài audio | `truncation=True`, cắt cứng 30 s | `truncation=False` + `padding="longest"` + `attention_mask`, long-form tuần tự |
| dtype | fp16 trên T4 | `torch_dtype=torch.float32` ép tường minh |
| Dạng weights | base + `PeftModel` | checkpoint đã merge tải từ HF |
| Batch | 8 clip pad chung | một clip mỗi lần gọi |

Hai hệ quả có thật:

1. **Thang temperature** khiến CER production không suy ra được từ CER cổng. Khi một cửa sổ trượt
   ngưỡng logprob hoặc compression-ratio, production decode lại ở temperature cao dần — tức là
   lấy mẫu, không phải greedy. Đây là thuật toán khác, không phải cùng thuật toán cộng nhiễu.
2. **`condition_on_prev_tokens=True`** tạo phụ thuộc chéo cửa sổ. Cổng chấm từng segment độc lập,
   và `rejoin_real_chunks` ghép *văn bản* sau khi đã decode rời, nên không cơ chế nào của cổng
   quan sát được việc một lỗi kiểu `team`→`tim` được prompt lại vào cửa sổ sau và tự củng cố.

Nói cách khác: `check_no_regression_vs_production` — cơ chế §2.1 dựng ra để chặn đúng sự cố này
— đang xếp hạng hai model trên một đường decode mà **không model nào chạy khi phục vụ người
dùng**. Cơ chế đúng, dữ liệu nạp vào lại sai lần thứ hai, cùng một dạng lỗi với §2.6.

---

## 3. Lỗi của tôi trong phiên điều tra

Ghi lại để không lặp, không phải để tự trách.

| Lỗi | Chi tiết | Phát hiện thế nào |
|---|---|---|
| **Gán sai nhãn λ** | Gọi `Outputs/v3-r16/audit/*.csv` là "λ=0.5", thực tế λ=1.0. Đã viết vào `SESSIONS.md` trước khi phát hiện | Tự phát hiện khi kiểm `adapter_config.json` (`lora_alpha=32.0` → scaling 8.0 → λ=1.0) và đối chiếu `tier2_ood.cer` với row sweep — trùng từng chữ số |
| **Ngoại suy từ 2 điểm** | Khẳng định "retention tăng theo độ mạnh adapter" dựa trên base(λ=0) và v5(λ=0.25), rồi dùng nó làm lập luận. Điểm thứ ba (v3@λ=0.5) lại là adapter khác, bị nhiễu | Tự nêu giới hạn, nhưng chỉ **sau khi** đã dùng nó để lập luận một lượt |
| **Dự đoán sai** | Lo λ cao kéo model về nhãn 0 chữ hoa làm retention **giảm**. Đo ra ngược lại: tăng đều tới 0.75 rồi phẳng | Đo λ=0.75 và 1.0 |
| **Giả thuyết bị bác** | Đề xuất trục "từ đồng âm ngắn thông dụng vs thuật ngữ dài hiếm" giải thích 4 bộ đo. Chỉ giải thích được 2/4, bị `NZiW4QH83CI` bác thẳng | Tự kiểm bằng cách phân loại token, tự bác |
| **Suýt bỏ sót** | Bước chấm v3-r16 ban đầu dùng manifest của chính nó — chỉ có 426 synthetic, không có YouTube. Production check sẽ **lặng lẽ chạy trên nửa bộ test** và báo như thể đủ | Bắt được khi soát lại, không phải do thiết kế ngăn |

Mẫu chung của 5 dòng trên: **bốn cái tự bắt được bằng cách đo lại, một cái bắt được bằng cách
đọc lại.** Không cái nào bị phát hiện bởi một cơ chế tự động — cùng đúng vấn đề mà §2.2 nêu.

---

## 4. Đã sửa

| Việc | Ở đâu |
|---|---|
| So ứng viên với model đang chạy production | `check_no_regression_vs_production`, [`scripts/merge_and_push.py`](../scripts/merge_and_push.py) |
| Chỉ số bắt được mất từ mượn | `english_token_retention`, [`src/metrics.py`](../src/metrics.py) |
| Đưa chỉ số đó vào cổng chặn push | tier `retention_pass` trong `_score_by_source`, [`src/gate.py`](../src/gate.py) |
| Lỗi dấu `select_lambda` | commit `6b0b12b`, kèm test dùng đúng sweep row thật của run này |
| Gate lại ở λ người chọn, có ghi vết | [`scripts/gate_at_lambda.py`](../scripts/gate_at_lambda.py), ghi `_lambda_source` vào `gate_results.json` |
| Tham chiếu production đúng λ, đủ 654 segment | `Outputs/lambda075-metrics/v3-r16-lambda0.5/predictions_tier1_in_domain.csv` |
| **Đo bằng chính module 4, không mô phỏng lại nó** (§2.7) | [`scripts/eval_module4.py`](../scripts/eval_module4.py) gửi audio tới `PhoWhisperASR` thật với `gen_params: null`; chấm bằng `src/metrics.py` + `src/normalize.py` của repo này |
| **Mã module 4 nằm sẵn trong repo** | [`vendor/viet_speech/`](../vendor/viet_speech/VENDORED.md) — bản sao **byte-identical**, trùng đúng file trong `viet-speech-remote.zip` đang chạy trên máy GPU. Không viết lại 6 kwargs decode: một bản sao viết tay là định nghĩa thứ hai, và nó sẽ lệch |
| **Chặn push bằng bằng chứng module 4, cả hai split** | `check_module4_evidence` trong [`scripts/merge_and_push.py`](../scripts/merge_and_push.py); `--module4-dir` và `--module4-production-dir` là **bắt buộc**. `check_no_regression_vs_production` không đổi một dòng — chỉ đổi dữ liệu nạp vào |
| **Retention cũng thành điều kiện chặn push** | `check_retention_vs_production`, dùng `gates.max_retention_regression_pp` — luật §2.4 đã có ở tier 1 so với base model, nay áp thêm so với model đang chạy |
| **Chặn lệch âm thầm của chính bản sao** | [`tests/test_module4_profile.py`](../tests/test_module4_profile.py): hash từng file so với `d:\viet-speech` khi có repo đó, và assert từng default decode khi không có |
| **Đếm chữ hoa trong `hyp` chưa chuẩn hoá** | `capitalization` trong `module4_results.json` — không cần `ref`, nên đo được **nửa triệu chứng mà §2.5 nói "vẫn không nhìn thấy được"**, ở dạng so hai model với nhau |

Lưu ý phạm vi: những dòng trên là **cơ chế đã dựng xong và đã test**, chưa phải số đo. Bộ bằng
chứng thật cần khoảng 4,6 h GPU (138 phút audio × 2 model) và chưa chạy — xem §5.

---

## 5. Chưa sửa

| Việc | Vì sao còn |
|---|---|
| **Casing** | Nhãn vẫn 0 chữ hoa, và λ=0.75 kéo mạnh hơn về phân phối đó nên casing nhiều khả năng **tệ hơn** λ=0.25. Không đo được vì mọi `ref` đã hạ chữ thường. Cần train lại (D1) |
| **OOD tệ hơn production** | 3,34% so với 2,49%, **+34% tương đối**. Trong ngân sách nhưng là thứ duy nhất xấu đi. Đánh đổi có ý thức, không phải bỏ sót |
| **Nhóm từ công cụ/hạ tầng** | `mysql`, `aws`, `terminal`, `kafka`, `standup` — v5@0.75 mất nhiều hơn production 1–2 lượt mỗi từ. Nhỏ nhưng thành nhóm mạch lạc, chưa rõ cơ chế |
| **Luật `select_lambda` vẫn hụt** | Nó tối ưu val CER so với OOD CER, không thấy retention, nên vẫn chọn 0.5 thay vì 0.75. **Cố ý không chỉnh `elbow_ratio_threshold`** để luật ra 0.75 — đó là fit ngưỡng theo đáp án đã biết |
| **Không có cơ chế buộc xử lý phát hiện** | §2.2 vẫn có thể lặp lại. Hiện chỉ có `SESSIONS.md` và tài liệu — không có gì chặn deploy khi một nghi vấn đã ghi nhận còn để ngỏ |
| **Chưa có số đo module 4** | Cơ chế §2.7 đã xong và đã test, nhưng chưa chạy lần nào trên GPU. Đến khi chạy, **độ lệch giữa cổng và production vẫn là con số chưa ai biết** — đó chính là câu hỏi mà `scripts/eval_module4.py` sinh ra để trả lời |
| **Bằng chứng module 4 bỏ qua module 1–3** | Production chạy đủ 8 module, nên module 4 nhận splice **đã tách nguồn và trim im lặng**; `scripts/eval_module4.py` gửi audio real-bench thô. Phần dư có ý thức: đây là bằng chứng về *đường decode*, không phải về toàn pipeline. Ghi ra để không bị hiểu nhầm là đã bao phủ nhiều hơn thực tế |

---

## 6. Luật rút ra

1. **Ứng viên phải so với model đang chạy, không phải baseline của chính nó.** Cổng đo sai đối
   tượng thì mọi con số nó in ra đều không trả lời câu hỏi cần trả lời. *(đã thành code)*

2. **Chỉ số phải bắt được kiểu lỗi mà người dùng báo.** Nếu triệu chứng là mất từ mượn mà cổng
   chỉ có CER, thì cổng đang đo thứ khác với thứ đang hỏng. *(đã thành code)*

3. **"Pass" là sàn, không phải trần.** Pass đủ ba cổng vẫn có thể là lựa chọn tệ nhất trong các
   lựa chọn sẵn có — λ=0.25 chính là ví dụ.

4. **Một phát hiện viết trong báo cáo chưa phải là một việc đã xử lý.** Chẩn đoán đúng nằm trong
   tài liệu 12 tiếng trước khi model hỏng production. *(chưa có cơ chế)*

5. **Tham chiếu phải được xác minh là đúng thứ đang chạy**, không phải file trùng tên trong thư
   mục trùng tên. Kiểm bằng artifact (`adapter_config.json`, đối chiếu số với sweep), không kiểm
   bằng trí nhớ.

6. **Không chỉnh ngưỡng cho khớp một đáp án đã biết.** Nếu luật ra kết quả sai, hoặc sửa cơ chế,
   hoặc override có ghi vết — không lùi ngưỡng tới khi nó ra con số mình muốn.

7. **Đọc số từ artifact, không từ tài liệu mô tả artifact.** Áp dụng cả cho tài liệu do chính
   mình viết ra tuần trước.

8. **Phép đo phải chạy trên đúng đường code sẽ ship, không phải một đường tương đương trên
   giấy.** Cổng đúng đối tượng (luật 1) và đúng chỉ số (luật 2) vẫn vô nghĩa nếu nó chạy một
   thuật toán decode khác với thứ phục vụ người dùng. *(đã thành code — `scripts/eval_module4.py`)*

9. **Muốn đo code của người khác thì gọi chính code đó, đừng chép lại hành vi của nó.** Một bản
   viết tay của cùng tham số là định nghĩa thứ hai, và nó sẽ lệch — lệch đúng vào con số mà phép
   đo sinh ra để tìm. Chép nguyên văn thì được, nhưng phải kèm cơ chế phát hiện lệch.
   *(đã thành code — `vendor/viet_speech/`, `tests/test_module4_profile.py`)*

---

## 7. Điều đáng ghi nhận

Không phải mọi thứ đều hỏng.

- **Nguyên tắc fail-loud có tác dụng.** `check_provenance` đối chiếu λ đã bake với sweep row và
  với `tier2_ood` của gate. Khi gate lại ở λ=0.75, sai lệch ra **đúng 0,0** — giả định "greedy
  decode trên cùng split cho cùng số" được **xác nhận** chứ không phải được tin.
- **Kỷ luật đọc artifact bắt được nhiều lỗi.** Nhãn λ sai, tham chiếu sai λ, split thiếu nửa bộ
  — cả ba bắt được bằng cách mở file ra đối chiếu.
- **Cổng đọc kết quả được đặt ra trước khi đo.** "Cần CER giữ được **và** retention tăng" viết
  trong notebook trước khi chạy, nên khi λ=0.75 thắng cả 6 ô thì đó là kết quả vượt một tiêu
  chuẩn có sẵn, không phải một tiêu chuẩn dựng sau để vừa với số.
- **Weights không hỏng.** Dữ liệu train thật, nhãn đã soát, adapter học đúng thứ cần học. Sự cố
  này không phủ nhận giá trị của `mixed-noisy-v1` — nó tốt hơn `paid-dataset-v2` trên mọi lát
  đo được, chỉ là bị ship ở sai cường độ.
