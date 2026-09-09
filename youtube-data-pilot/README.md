# Pilot dữ liệu YouTube — họp online có code-switch

Ngày chốt: **2026-08-11**. Nguồn: một phiên grill đầy đủ về kế hoạch lấy dữ liệu
họp từ YouTube. Mọi quyết định dưới đây đã được cân nhắc và chốt — đừng mở lại
mà không có dữ kiện mới.

---

## 1. Phạm vi

**Đây là một pilot, không phải một dataset.**

Mục đích: kiểm chứng quy trình gán nhãn có chạy được không, và đo *dấu* của hiệu
ứng khi thêm audio thật vào hỗn hợp huấn luyện — **trước** khi đầu tư 10–30 giờ
theo `D:/viet-speech/docs/meeting-asr-finetune/04-data-plan.md`. Data plan đó vẫn
là kế hoạch chủ đạo cho giai đoạn sau; pilot này không thay thế nó.

**Phạm vi âm học: chỉ cuộc gọi online.** Mục tiêu cuối là họp nói chung, gồm cả
phòng họp vật lý — nhưng audio far-field của họp thật **không tồn tại trên
YouTube** (video hội nghị đều qua mixer, micro gần miệng). Chặng far-field sẽ cần
một nguồn dữ liệu khác, không phải mở rộng nguồn này.

Ghi "pilot, phạm vi online" vào `provenance.md` của run. Kết luận của pilot không
được suy rộng ra far-field.

## 2. Kích cỡ

**6–8 cuộc họp × 15–20 phút = 1,5–2,4 giờ.** Không lấy 1–2 giờ từ một cuộc.

| Split | Số cuộc | Thời lượng | Segment (~15 s) |
|---|---:|---:|---:|
| train | 5 | ~1,5 h | ~360 |
| val | 1 | ~18 ph | ~72 |
| test | 2 | ~36–40 ph | ~130–160 |

Công người: **2,5–5,5 giờ** (soát bản nháp ASR ở mức 1,5–2× thời lượng thật).

**Vì sao không lấy 1–2 giờ từ một cuộc họp:** một cuộc = một phòng, một bộ micro,
4–8 người nói → đa dạng âm học gần bằng 0, đúng lời phê mà `PROJECT_CORE.md` §4
dành cho bench `done/`. Thêm nữa, `meeting_id` là đơn vị chia split, nên một cuộc
họp **không chia được** thành train/val/test tách rời; cần tối thiểu 3, thực tế 6+.

Lấy đoạn **giữa** cuộc họp. Bỏ đầu (chào hỏi, chờ người vào) và cuối (kết thúc
lỏng lẻo). Phần giữa là nơi có tranh luận thật, chồng lấn, và code-switch dày nhất.

## 3. Tiêu chí chọn video

Họp online thật (Meet/Zoom), chủ đề code-switch dày, nhiều người nói, có phần hỏi
đáp tự phát.

**Đã loại:**

| Loại | Vì sao loại |
|---|---|
| Webinar / hội thảo một người trình bày | Bản chất là bài nói, lặp lại property 5 (lệch miền) của bench hiện tại |
| Podcast / talkshow studio | Âm thanh quá sạch, luân phiên lượt nói quá gọn |
| Đại hội cổ đông / họp Quốc hội | Gần như thuần Việt → **xói mòn code-switch**, đúng thứ cần bảo vệ |

---

## 4. Các bước

### Bước 1 — Thu

`yt-dlp` + `ffmpeg`, xuất **mono 16 kHz** ngay lúc tách. Audio YouTube mặc định là
stereo, và `load_audio_16k` ([src/data.py:96](../src/data.py#L96)) **raise** khi
gặp stereo — "stereo là data error".

`yt-dlp` ở `requirements.txt`, **không pin version** (YouTube đổi phía server, bản cũ
vỡ). `ffmpeg` là **system dep, không pip** — `fetch_youtube.py` raise với thông báo nêu
tên nó khi thiếu, thay vì để yt-dlp đổ trace khó đọc.

**Giữ file gốc chưa cắt** của 2 cuộc test, cùng transcript đầy đủ → tier 4b
(long-form) trở thành miễn phí. `PROJECT_CORE.md` §6 ghi tier 4b "gần như miễn
phí" và đang bị hoãn vì thiếu dữ liệu.

*Verify:* `soundfile.read` từng file → `data.ndim == 1` và `sr == 16000`.

### Bước 2 — Nháp nhãn

**Phụ đề do Google sinh tự động, định dạng json3.** Miễn phí. Đây là lựa chọn hiện tại,
không phải lựa chọn cuối — xem "đổi nguồn" dưới.

Lý do chọn nó thay ElevenLabs Scribe: json3 mang **mốc thời gian từng từ**, biết chắc,
không phải kiểm. Bench `done/` chỉ có mốc cấp segment nên `ingest_real_bench.py` phải
chia text **theo tỉ lệ thời lượng** — một phép xấp xỉ, và chính nó sinh ra
`real_0002/seg_0074` (475,7 ký tự/giây). Có mốc từng từ thì cả lớp bug đó biến mất.

**Một video YouTube có thể có hai bộ phụ đề khác nhau, và phải lấy đúng bộ.**

`yt-dlp` trả về hai dict riêng biệt, đó là tên thật của hai thứ này:

| Field của `yt-dlp` | Là gì | Cờ CLI |
|---|---|---|
| `automatic_captions` | Google chạy nhận dạng tiếng nói rồi sinh ra. **Đây là bộ cần lấy.** | `--write-auto-subs` |
| `subtitles` | Chủ kênh tự gõ hoặc tự tải lên | `--write-subs` |

Bộ `subtitles` **có thể không phải bản chép lời**. Đúng video `E5dAymt68-0`,
[`viet-speech/experiments/real-domain-gap-diagnosis/ground_truth.py:20-26`](../../viet-speech/experiments/real-domain-gap-diagnosis/ground_truth.py)
ghi: mật độ từ tụt từ 169 wpm (5 phút đầu) xuống 32 wpm (5 phút cuối) — "nó loãng thành
một bản tóm tắt". Máy nhận dạng tiếng nói không tóm tắt, con người thì có. Lấy nhầm bộ
này thì nhãn của cả dataset là bản tóm tắt, không phải lời nói.

**Bẫy thứ hai, nặng hơn: `automatic_captions` chứa cả bản dịch máy.** YouTube tự dịch
phụ đề tự động ra khoảng 100 thứ tiếng. Nên một video **nói tiếng Anh** vẫn có mục `vi`
trong `automatic_captions`, và đó là bản dịch — text tiếng Việt ghép với audio tiếng Anh.
Sự tồn tại của khoá `vi` **không chứng minh** video nói tiếng Việt. Dấu hiệu: dict
`automatic_captions` có 100+ khoá thì phần lớn là bản dịch.

Ba luật loại video, kiểm hết **trước khi tải audio**:

1. Không có mục tiếng Việt trong `automatic_captions` → loại.
2. Mục tiếng Việt là **bản dịch máy**, không phải bản nhận dạng gốc → loại.
3. wpm tụt dần theo các mốc 5 phút → loại, và điều tra.

Đó là việc của [scripts/probe_youtube_captions.py](../scripts/probe_youtube_captions.py)
(row **F1** trong `SESSIONS.md`). URL ứng viên do người dùng cấp; script không tự đi liệt
kê kênh.

**Cách phân biệt bản gốc với bản dịch, đo trên output thật 2026-08-11 (yt-dlp
2026.07.04):** bản nhận dạng gốc là **một khoá riêng** hậu tố `-orig`, và entry của nó có
field `name` kết thúc bằng `" (Original)"`:

```
vi-orig -> {"ext": "json3", "name": "Vietnamese (Original)"}   <- bản nhận dạng
vi      -> {"ext": "json3", "name": "Vietnamese"}              <- đích của bản dịch
ja      -> {"ext": "json3", "name": "Japanese"}                <- đích của bản dịch
```

Nên ngôn ngữ được nói chính là ngôn ngữ của khoá `(Original)`. Video nói tiếng Anh có
`en-orig` cộng một `vi` là bản dịch — luật 2 loại đúng nó. Video **không có khoá
`(Original)` nào** cũng loại: khi đó không gì trong response chứng minh mục `vi` không
phải bản dịch.

**Endpoint phụ đề không tất định.** Sáu lần extract lại `vi-orig` của `dGT3YW0AdD8` cho
bản thô 5 lần (1196 event, 4866 từ, 4 dấu phẩy, `"mãng"`, `"Zo 11"`) và bản có dấu câu 1
lần (1714 event, 5218 từ, 209 dấu phẩy, `"embedded"`, `"Zero 11"`) — cùng khoá, cách nhau
vài phút. Gọi lại chính URL đã ký thì không đổi bản, và nhanh chóng nhận **HTTP 429**. Hai
hệ quả: mọi số trong `caption-probe.md` là ảnh của **một** lần fetch, và **bước 1 phải lưu
đúng file json3 đã dùng, bước 3 parse file đã lưu, không fetch lại.**

**Nếu một video có cả hai bộ phụ đề:** vẫn cắt segment theo `automatic_captions` để giữ
mốc từng từ, còn `subtitles` dùng để **đo mức đồng thuận** — nối text mỗi bộ thành một
chuỗi rồi tính CER giữa hai chuỗi (`src.normalize.Normalizer` + `src.metrics.score`,
không cần căn chỉnh gì). Đồng thuận cao = Google nghe tốt trên audio đó = công soát nhẹ.
Đọc kèm bảng wpm, vì một bộ `subtitles` dạng tóm tắt cũng cho CER cao nhưng vì lý do
khác hẳn.

*Đã xét và loại:* dùng text của `subtitles` làm nháp (mất mốc từng từ, về lại bẫy
`seg_0074`); căn chỉnh hai bộ để lấy text người gõ ghép mốc thời gian của máy (tốt nhất
về lý thuyết, nhưng là việc thật, không phải một cờ — mở lại nếu pilot cho thấy chất
lượng nháp là điểm nghẽn).

**Đổi nguồn nháp:** `scripts/draft_sources.py` định nghĩa một dạng trung gian duy nhất —
`Word(text, start, end, speaker=None)` — và hai parser cùng trả `list[Word]`:
`parse_json3` (Google) và `parse_scribe` (ElevenLabs). Bước cắt segment và bước ghi
manifest chỉ nhận `list[Word]`, không biết nguồn là gì, nên đổi sang Scribe là một cờ
`--draft-source scribe`, không sửa dòng nào của phần cắt.

`parse_scribe` **raise nếu thiếu mốc thời gian cấp từ** — không có đường rơi về phép
chia theo tỉ lệ, vì chính đường đó là bug `seg_0074`. Đây là chỗ câu "đừng giả định —
kiểm" của bản trước được cơ chế hoá thành một exception.

`label_source` và `asr_draft_model` ghi **theo từng record**, nên corpus trộn hai nguồn
vẫn truy được nguồn của từng segment.

### Bước 3 — Cắt

Dùng lại machinery của
[scripts/ingest_real_bench.py](../scripts/ingest_real_bench.py): cắt theo
khoảng lặng RMS, `MAX_SEGMENT_SEC = 30.0` (cửa sổ Whisper),
`MAX_CHARS_PER_SEC = 60.0` (phát hiện segment lệch nhãn).

**json3 chỉ có mốc bắt đầu của từng từ, không có mốc kết thúc** (đo 2026-08-11):
`dDurationMs` của event là thời lượng *hiển thị* phụ đề và thường tràn qua event kế tiếp
(582/597 event kề nhau chồng lấn thời gian), nên nó chỉ là chặn trên. `draft_sources.Word.end`
vì thế là **suy ra**, không phải đo. Hệ quả cho bước cắt: cắt theo **khoảng cách giữa hai
mốc bắt đầu** (`next.start − cur.start`), không phải theo `next.start − cur.end` — đại lượng
sau bằng 0 ở trong một event và do đó không mang thông tin khoảng lặng.

Nhắm **10–25 giây**. Whisper pad mọi input lên 30 s bất kể ngắn dài, nên segment
3 s và 20 s tốn GPU như nhau — segment quá ngắn là lãng phí. Tránh lệch phân bố
như `done/` (median 3–4 s nhưng max 212 s), vì vài segment sẽ chiếm phần lớn tổng
ký tự và làm sai bootstrap (§4 property 4).

Lọc bỏ: segment không có tiếng nói (nhạc hiệu, vỗ tay, im lặng).

*Verify:* không segment nào > 30 s; `_check_speech_rate` không raise.

### Bước 4 — Soát tay

Nhãn train **và** test đều là **nháp ASR (bước 2), người soát toàn bộ**. Không có
lựa chọn lọc tự động.

**Vì sao không lọc bằng đồng thuận hai ASR:** bộ lọc đồng thuận sẽ loại đúng
những segment cần nhất — segment mà hai model không đồng ý chính là segment nhiễu
nặng, xa micro, nhiều người nói, tức chính điều kiện âm học mà dataset này tồn tại
để dạy. Giữ phần dễ, bỏ phần khó, rồi huấn luyện trên phần dễ. Lọc đồng thuận chỉ
đáng dùng khi có hàng trăm giờ và bỏ 40% vẫn còn thừa.

**Vì sao không để nhãn ASR thô:** với tập test, chỉ số thu được sẽ là "mức độ
PhoWhisper đồng thuận với model nháp", có sàn bằng tỉ lệ lỗi của model nháp; khi
PhoWhisper sửa đúng chỗ nháp nghe sai, phép đo ghi nhận đó là lỗi. Với tập train, nhãn
sai là một bước gradient dạy model nghe sai — tệ hơn thiếu dữ liệu.

Bốn quy ước, ghi vào `style-guide.md` versioned cùng dataset:

1. **Không xoá filler.** `ạ à ừ ơ` là từ có nghĩa trong tiếng Việt (`ạ` là tiểu từ
   lịch sự, `à` là tiểu từ nghi vấn), không phải rác. Bằng chứng định lượng: bản
   webinar bên `viet-speech` có **107 lỗi `filler_convention`** trên ~380 lỗi đếm
   được, cộng 86 từ đệm trong 267 từ "thêm thừa" — khoảng một phần tư số lỗi là ảo
   vì thiếu quy ước.
2. **Quy ước số: chữ số cho giá trị rõ ràng là số** (tiền, %, thời gian, số lượng);
   chữ cho phần còn lại. Xem §5. *Đổi 2026-08-11 — bản trước ghi "viết bằng chữ".*
3. **Chồng lấn: chép người nói trội**, đặt `quality: overlap`. Có trong train,
   **không** tính vào CER công bố.
4. **Token tiếng Anh giữ đúng chính tả người nói.** Đây là ground truth cho gate
   code-switch.

### Bước 5 — Manifest

**Bốn field bắt buộc** — thiếu là crash, không phải cảnh báo:

| Field | Đọc ở đâu |
|---|---|
| `text` | [src/data.py:107](../src/data.py#L107) |
| `audio_filepath` | [src/data.py:105](../src/data.py#L105) |
| `split` | [src/data.py:62](../src/data.py#L62) |
| `meeting_id` | [src/data.py:63](../src/data.py#L63) |

**`split` chỉ nhận `"demo"` hoặc `"test"`.** Mọi giá trị khác bị raise. Train/val
**không nằm trong dữ liệu** — nó suy ra từ `data.val_meetings`:

```
split == "test"                                  -> test
split == "demo" và meeting_id ∈ val_meetings     -> val
split == "demo" và ngược lại                     -> train
```

**`meeting_id` định danh sự kiện, không phải video.** `resolve_splits` có guard
chống một `meeting_id` rơi vào hai split, nhưng guard đó **chỉ bảo vệ khi ID đặt
đúng**. Hai bản đăng của cùng buổi họp (bản đầy đủ và bản highlight, hoặc hai kênh
đăng lại) phải dùng **chung một** `meeting_id`; nếu đặt hai ID khác nhau, guard im
lặng và một bản có thể nằm ở train, bản kia ở test.

`video_id` là field provenance riêng.

**Field provenance — bắt buộc, không phải tuỳ chọn:**

`video_id`, `video_url`, `yt_start`, `yt_end`, `download_date`, `sha256`,
`source: "youtube"`, `label_source: "google_asr+human"` (hoặc `"scribe+human"` nếu
đổi nguồn — bước 2), `asr_draft_model`, `reviewed_by`, `review_date`

`label_source` + `asr_draft_model` ghi **theo từng record**, không theo dataset: đó là
cặp field làm việc đổi nguồn nháp truy được về sau.

Lý do ở §6.

**Dùng tên field đã có trong `04-data-plan.md`, không đặt tên mới:**

- `has_code_switch: true | false`
- `quality: clean | noisy | overlap | unclear`

(Ghi chú: `paid-dataset-v2` dùng `overlap` là field boolean riêng. Hai schema đang
lệch; pilot này theo tên của `04-data-plan.md` để không tạo thêm nợ.)

### Bước 6 — `dataset/mixed-noisy-v1/`

```
dataset/
├── paid-dataset-v2/     ← giữ nguyên, không chạm
├── youtube-meetings/    ← dữ liệu mới tải về
└── mixed-noisy-v1/      ← pipeline đọc thư mục này
    ├── manifest.*.jsonl (copy của cả hai nguồn)
    └── audio/           (copy của cả hai nguồn)
```

Config đổi một dòng: `dataset_path: dataset/mixed-noisy-v1`. **Không sửa code.**

Lý do không đổ thẳng vào `paid-dataset-v2`: run v3-r16 đã chạy trên thư mục đó.
Thêm file vào nó nghĩa là cùng một tên thư mục chỉ hai nội dung khác nhau ở hai
thời điểm, và không so sánh được run mới với run cũ.

Copy thật, không symlink — toàn bộ audio 16 kHz mono chỉ khoảng **900 MB** (5,9 h
tổng hợp ≈ 680 MB, 2 h YouTube ≈ 230 MB).

Muốn thử tỉ lệ trộn khác thì tạo `mixed-noisy-v2/`. Cả hai cùng tồn tại, mỗi run ghi rõ
đọc thư mục nào — tỉ lệ trộn trở thành **dữ liệu**, không phải tham số phải đoán
lại về sau.

**`val_meetings` phải chứa ít nhất một `meeting_id` YouTube.** Nếu val thuần tổng
hợp, `select_lambda()` chọn λ\* dựa trên val tổng hợp cộng OOD VIVOS — **mù hoàn
toàn** với phần dữ liệu thật vừa thêm, tức chọn λ tối ưu cho đúng thứ không quan
tâm.

### Bước 7 — Sửa code TRƯỚC khi train

Ba việc, không tốn một giờ gán nhãn nào.

**7a. [src/asr.py:80](../src/asr.py#L80) — `forced_decoder_ids` đã bị bỏ.**

Đã kiểm trên transformers **5.13.1**:

```
forced_decoder_ids in generate signature:            False
forced_decoder_ids attr on GenerationConfig:         False
_validate_model_kwargs raises on unused kwargs:      True
```

Nên dòng hiện tại sẽ **fail loud**:

```
ValueError: The following `model_kwargs` are not used by the model: ['forced_decoder_ids']
```

`processor.get_decoder_prompt_ids(...)` ở dòng 76 vẫn tồn tại — chỉ cách truyền
kết quả vào `generate()` là vỡ. Sửa: truyền `language="vi", task="transcribe"`
trực tiếp, đúng như `viet-speech`'s `backend/adapters/asr/phowhisper.py` đã làm.

Thời điểm cắn phụ thuộc phiên bản transformers trên Kaggle.

**7b. Gate code-switch — tier mới.**

Chuyển `loanword_dropped_segments`
([scripts/inspect_errors.py:97](../scripts/inspect_errors.py#L97)) từ script
chạy tay thành một tier trong [src/gate.py](../src/gate.py).

- Gate trên **tỉ lệ giữ token tiếng Anh**, không phải CER của lát code-switch
- Field config mới: `gates.code_switch_retention_drop_pp: 0.0` (dung sai 0, cùng
  triết lý `real_cer_regression_pp` của tier 4a)
- Báo cáo thêm CER của lát đó, nhưng **không** gate trên nó
- Tính trên **cả** test tổng hợp **và** test YouTube, báo cáo tách hai lát

**Vì sao không gate bằng CER của lát code-switch:** một segment code-switch phần
lớn vẫn là tiếng Việt. `hiện tại mình đang là software engineer ở Microsoft Việt
Nam` — bỏ trắng cả `software engineer` chỉ làm CER câu đó nhích vài phần trăm.
Model có thể đánh rơi **mọi** token tiếng Anh mà gate vẫn PASS.

**Về bộ phát hiện từ tiếng Anh:** nó đã biết là không chính xác — E5 phải nới regex
vì pattern nguyên bản trong §4 gắn cờ ~99% từ Việt thật (`không`, `được`) là ngoại
lai. Nhưng nó **không phá được gate này**, vì gate so baseline với candidate bằng
**cùng một** bộ phát hiện: sai lệch của detector triệt tiêu trong phép so cặp. Đó
là lý do gate đo *tỉ lệ thay đổi*, không đo *giá trị tuyệt đối*.

**7c. Eval decode khớp cấu hình sản xuất.**

`Fine_tune_wf` eval bằng short-form greedy (`num_beams: 1`, không gate). Sản xuất
bên `viet-speech` chạy long-form với thang temperature 6 bậc, `no_speech_threshold`,
`logprob_threshold`, `compression_ratio_threshold`, `condition_on_prev_tokens=False`,
`no_repeat_ngram_size`. Xem §7 để biết vì sao.

Hệ quả: **λ\* hiện đang được chọn dưới một cấu hình decode mà sản xuất không bao
giờ dùng.** Thêm một tier eval long-form dùng đúng `gen_kwargs` đó — cùng lúc đó
là tier 4b.

*Verify:* baseline chạy lại sạch sau khi sửa 7a; `gate_results.json` có
`tier5_code_switch` với `delta_pp`.

### Bước 8 — Train + gate

Dự đoán, ghi lại để đối chiếu sau khi chạy:

| Tier | Dự đoán | Cơ sở |
|---|---|---|
| test YouTube | **2–6 pp** tốt hơn (10–20% tương đối) | 1–2 h là đầu thấp của vùng có tác dụng; dữ liệu thật là 16–27% hỗn hợp, 3 epoch → ~3–6 h gradient thật trên ~22 h |
| code-switch | **thắng rõ nhất** | Dữ liệu dày code-switch, nhãn người sửa nên chính tả tiếng Anh đúng; đã đo 59/266 segment mất trắng từ vay mượn trên test set v3-r16 |
| tier 1 in-domain | **xấu đi, có thể FAIL** | Thêm 16–27% audio nhiễu thật vào model đang đo trên TTS sạch; gate đòi `min_improvement_pct: 10.0` trên CER ~1,3% |
| λ\* | **thấp hơn 0,5**; có thể **không λ nào vừa** `ood_cer_budget: 0.02` → fail cứng | Dữ liệu tự phát nhiễu ở rất xa VIVOS (giọng đọc, phòng sạch) → mức quên tăng |
| tier 4a (`done/`) | **không kỳ vọng khỏi** | Bench là bài nói học thuật thu trong phòng vật lý; dữ liệu mới không nhắm vào nó |

**Rủi ro lớn nhất — không phân giải được.** 130–160 segment cho CI cỡ **±1,5–2 pp**
(so với ±1–1,5 pp mà §4 tính cho bench 196 segment). Phần dưới của khoảng dự đoán
2–6 pp **nằm dưới ngưỡng đo được**: bạn có thể thắng mà không chứng minh được, và
verdict ra `INCONCLUSIVE`.

Một điểm nghiêng về phía có lợi: model nháp (Google ASR, hoặc Scribe) khác họ với
PhoWhisper, nên lỗi nháp còn sót lại sau khi soát sẽ **phạt** PhoWhisper chứ không tha.
Con số đo được là **sàn**, không phải trần — đo được cải thiện thì cải thiện đó thật.
(Điều này chỉ đúng vì bước 4 có người soát. Với nhãn thô, chiều lệch đảo lại: lỗi của
nháp làm CER đo được **phồng lên**, nên CER cao không chứng minh audio khó.)

**Bốn lớp lỗi không lượng dữ liệu nào dịch chuyển được** (đọc từ cột `reads as`
của `viet-speech`, arm `mixed`):

| Lớp | Số lỗi | Sửa bằng gì |
|---|---:|---|
| Danh từ riêng | 20 | Danh sách tên lúc inference — **không khả thi như can thiệp benchmark** (xem §8) |
| Bịa nội dung khi im lặng | 0 ở `mixed`, 68 ở `track` | `no_speech_threshold`, `temperature` — chỉ áp dụng ở long-form |
| Quy ước từ đệm | 107 | Quy ước chép tham chiếu (bước 4, quy tắc 1) |
| Chồng lấn thật hai luồng | — | Giới hạn kiến trúc Whisper: một luồng đầu ra |

---

## 5. Quy ước số: chữ số cho giá trị rõ ràng là số

*Đổi 2026-08-11. Bản trước chốt "viết bằng chữ"; ba dữ kiện dưới đây đảo quyết định.*

Config hiện tại:

```yaml
normalization:
  number_convention: word_to_digit    # đối xứng; áp cho CẢ hyp và ref
```

**CER không phụ thuộc quy ước.** Normalizer áp cho **cả hai phía**: ref viết
`mười lăm phút`, model xuất `15 phút`, cả hai về `15 phút` → khớp, không tính lỗi. Nên
lựa chọn này không mua được điểm CER nào — nó chỉ quyết định **model học xuất ra dạng
nào** (nhãn train không đi qua normalizer) và **người soát phải gõ bao nhiêu**.

Ba lý do chọn chữ số:

1. **Khớp corpus hiện tại.** Đo `dataset/paid-dataset-v2` (4.946 segment): 529 (10,7%)
   chứa chữ số, 1.022 (20,7%) chứa số viết chữ, 183 chứa cả hai. Ví dụ thật: `"tăng từ
   200ms lên gần 3 giây"`, `"tầm 95% ạ"`, `"delay 24 tiếng"`. Tức corpus đang theo đúng
   luật §Numbers của `05-annotation-guideline.md`. Tập YouTube viết bằng chữ sẽ là tập
   duy nhất lệch.
2. **Cả hai nguồn nháp đều xuất chữ số.** Giữ "viết bằng chữ" nghĩa là người soát gõ
   chuyển đổi bằng tay trên từng segment — lý lẽ cũ ("gõ dễ nhất, gõ y như nghe") đúng
   khi gõ từ đầu, đảo chiều khi có nháp.
3. **Không cần code mới.** `src/normalize.py` chỉ có `words_to_digits`, **không có**
   `digits_to_words`. Chọn chữ số thì không phải viết nó.

**Rủi ro vẫn còn, nhỏ hơn nhưng không bằng không.** §4 property 2: `một` thường là mạo
từ ("một bộ trọng số" = *a set of weights*, không phải *1 set*), và `năm` nghĩa là cả
*five* lẫn *year*. Quy ước chữ số **giảm** số lần hai từ này đi qua bộ chuyển đổi ở phía
ref, nhưng phía **hyp** thì model vẫn xuất chữ và vẫn đi qua đủ. Thêm nữa,
`normalize.py:words_to_digits` **đã từng có bug vòng lặp vô hạn** trên một từ zero-filler
đứng lẻ (tên riêng "Linh") — SESSIONS.md bug #7.

**Bắt buộc:** tier YouTube mới phải có `normalization_check` — chạy lại cùng dữ
liệu với convention `as_written` và ghi `delta_pp`. Tier 1 và tier 4a đã có block
này; **tier 2 OOD thì chưa** (SESSIONS.md **E3**, còn `todo`).

Không có nó, không phân biệt được phần cải thiện nào là **nghe tốt hơn** và phần
nào chỉ là **quy ước số**. Con số đã đo trong quá khứ: **~62% cải thiện CER
in-domain đến từ chuẩn hoá chữ số, không phải học âm học.**

## 6. Công bố: chưa, có thể trong tương lai

Audio lưu local, không phân phối lại. Adapter đẩy lên HF Hub không bị ảnh hưởng —
chỉ audio bị ràng buộc.

Quyết định "có thể trong tương lai" biến nhóm field provenance từ *nên có* thành
**bắt buộc**, vì con đường công bố duy nhất về sau là: **phát hành manifest +
checksum + URL, không phát hành audio.** Người khác tái tạo dataset từ metadata.

Thiếu mấy field đó bây giờ thì con đường ấy đóng lại, và **không cách nào bù lại**
— video có thể bị xoá khỏi YouTube.

`sha256` là bắt buộc chặt nhất: nó là thứ duy nhất chứng minh file người khác tải
về giống file đã dùng để đo.

---

## 7. Dữ kiện từ `D:/viet-speech` (đã xác minh, 2026-08-11)

`Fine_tune_wf` sản xuất **module 4** của pipeline 8 module bên `viet-speech`. Bằng
chứng trong `config/models.yaml`:

```yaml
- id: reworkwhisper-large-v3-0.5lamda
  params:
    model_size: winhsss/reworkwhisper-large-v3-0.5lamda
```

Đó chính là adapter λ=0.5 từ λ sweep của repo này. `winhsss/Reworkwhisper-large-v4`
là model mặc định hiện tại.

**Dây chuyền:** 1 VAD (`silero-vad`) → 2 phát hiện chồng tiếng
(`pyannote-segmentation-osd`) → 3 tách nguồn (`pyannote-separation-ami`) →
`trim_silence` → **4 ASR** → 5 embedding (`ecapa-voxceleb`) → 6 nhận dạng người
nói → 7 ghép biên bản → 8 Meeting AI (LLM).

### Module 4 nhận cái gì

Không phải audio thô, và cũng không phải output mạng tách tiếng thuần:

1. Module 3 tạo track dài bằng cả cuộc họp, chỉ giữ tín hiệu trong cửa sổ của
   người đó, còn lại là zero số.
2. **Nội dung trong các cửa sổ:** crop từ `mixed.wav` gốc cho đoạn solo, output
   mạng tách tiếng chỉ cho đoạn Module 2 báo overlap thật. Đây là hướng A, đã vào
   code chính 2026-08-10 (`backend/adapters/separation/pyannote_separation.py`).
3. `trim_silence` cắt zero rồi **nối tất cả thành một waveform duy nhất**.

**Đo được bằng chạy code** (`trim_silence` với `max_len=25.0`):

| Input | Output |
|---|---|
| 3 span × 20 s trong track 225 s | **1 track, 61,00 s**, 3 mapping entry |
| 1 span 90 s | **1 track, 90,20 s**, 4 mapping entry |

Nên `max_len` **chỉ cắt `mapping`, không cắt audio**, và audio đưa vào ASR routinely
vượt 30 s → **long-form sequential decoding engage** (ngưỡng:
`is_shortform = total_input_frames <= num_segment_frames`,
`generation_whisper.py:658`).

Docstring của `trim_silence` viết long-form *"never engages"* — bị chính hành vi của
nó phản bác. Đây là phát hiện gửi sang repo kia, không sửa từ đây.

### Vì sao điều đó quan trọng với repo này

| | `Fine_tune_wf` train/eval | `viet-speech` sản xuất |
|---|---|---|
| Đơn vị đầu vào | 1 segment ≤30 s, liên tục | 1 track ghép nối, 70 s+ |
| Đường decode | **short-form** | **long-form sequential** |
| Chiến lược | greedy, `num_beams=1`, không gate | thang temperature 6 bậc + 3 ngưỡng gate + `no_repeat_ngram_size` |
| `condition_on_prev_tokens` | không áp dụng | `False` |

**Dữ liệu huấn luyện giữ nguyên: audio thô, segment ≤30 s.** Whisper vốn được
huấn luyện trên cửa sổ 30 s; long-form chỉ là cách *decode* ghép nhiều cửa sổ 30 s.
Không ai train trên input 70 s.

**Thứ phải sửa là eval, không phải data** — bước 7c.

### Cảnh báo về con số 87% / 12,6%

`module3-separation-findings.md` báo 87% thời lượng chỉ một người nói, 12,6% chồng
lấn. **Hai con số này đo trên `paid_meeting_0001` — TTS ElevenLabs tổng hợp**, không
phải họp thật. Chúng là thuộc tính của bộ sinh script tổng hợp.

**Tỉ lệ chồng lấn trong họp online thật vẫn chưa ai đo.** Cơ chế (solo → crop
`mixed.wav` gốc) đã xác minh trong code; **tỉ lệ** thì không.

### SI-SDR của module 3

| Cách làm | SI-SDR |
|---|---|
| Không làm gì — trả nguyên `mixed.wav` | −4,09 dB |
| Module 3 hiện tại | −0,13 dB |
| Cắt `mixed.wav` theo lượt nói đúng, không separation | **+13,83 dB** |

Mạng separation hơn "không làm gì" khoảng 4 dB, nhưng cách **14 dB** so với chỉ cần
biết đúng ranh giới lượt nói. Nó nhắm sai chỗ đang mất điểm, không phải vô dụng.

### Chữ ký lỗi trên webinar YouTube (arm `mixed`, model không qua module 1–3)

CER **27,3%** / WER **35,3%** trên 1617 từ tham chiếu: 1313 khớp / 213 sai từ /
91 bỏ mất / **267 thêm thừa** (trong đó **86 là từ đệm** mà transcript lược đi).

Ví dụ lỗi code-switch và danh từ riêng:

| Ground truth | ASR xuất ra |
|---|---|
| `align với team khác` | `dealight với tim khác` |
| `Engineer Pro` | `sdbr`, `mcw` |
| `Software Engineer ở Microsoft Việt Nam` | *(mất trắng)* |

**Lưu ý về tham chiếu:** phụ đề YouTube đã sửa tay theo hướng verbatim, chưa qua
nguồn kiểm định thứ hai. File mốc của repo đó tự thừa nhận: *"reference too weak
for an absolute CER, because every arm is scored against that same reference."*
Nên so sánh 27,3% (webinar) với 8,2% (bản ghi nội bộ) **không** kết luận được audio
webinar khó gấp 3,3 lần — hai con số khác nhau về cả độ khó audio lẫn chất lượng
tham chiếu, và dữ liệu hiện có không tách được hai thành phần.

---

## 8. Các phương án đã xét và loại

| Phương án | Vì sao loại |
|---|---|
| **Augmentation (nhiễu mô phỏng)** cho train | Người dùng yêu cầu nhiễu thật. Đã ghi nhận: augmentation bảo toàn code-switch tuyệt đối và nhân được corpus lên, nhưng không tái tạo hiệu ứng Lombard |
| **Replay âm học** (phát TTS qua loa trong phòng thật, thu lại) | Cùng lý do. Nó cho chuỗi âm học thật với nhãn miễn phí, nhưng vẫn là cách phát âm trong môi trường yên tĩnh |
| **VIVOS vào train** | VIVOS **là** gate tier2 (`data.ood_eval_path`) và là tiêu chí chọn λ\* qua `ood_cer_budget`. Train trên nó là mất cả gate chống quên lẫn tiêu chí chọn λ trong một lần |
| **Dữ liệu tiếng Anh thuần** | Code-switch là câu tiếng Việt có token tiếng Anh chèn vào — hiện tượng khác. Whisper nhận language token cưỡng chế: nạp audio tiếng Anh dưới token `vi` là dạy model xuất tiếng Anh khi nghe tiếng Việt; dưới token `en` thì là task khác |
| **`prompt_ids` / initial prompt** cho danh từ riêng | Tên riêng chỉ biết sau khi đã biết cuộc họp nói về ai → nạp để đo trên chính bản ghi đó là rò rỉ. `prompt_ids` cũng áp cho cả batch, phá `batch_size: 8`. **Khả thi như tính năng sản phẩm** (lúc triển khai biết trước ai dự họp qua calendar), không khả thi như can thiệp benchmark |
| **Train trên output module 3** | Module 4 nhận crop `mixed.wav` gốc cho đoạn solo (§7), nên audio thô đã khớp phần lớn phân phối sản xuất. Phương án này còn buộc `Fine_tune_wf` phụ thuộc `viet-speech` khi ingest |
| **Nhân bản dữ liệu thật lên 2–3×** | Đổi hai biến cùng lúc (thêm miền mới **và** đổi trọng số miền) → kết quả xấu thì không quy được nguyên nhân. λ sweep đã làm việc điều chỉnh ảnh hưởng, đo được và hoàn lại được; nhân bản thì mù và phải train lại ~9 h GPU |
| **Đổ YouTube vào `paid-dataset-v2`** | Run v3-r16 đã chạy trên thư mục đó; thêm file vào là mất điểm tham chiếu lịch sử |
| **Cho `dataset_path` nhận danh sách** | Đúng hơn về thiết kế, nhưng phải sửa `config.py`, `data.py`, validation — để giải quyết việc mà một thư mục giải quyết xong |

---

## 9. Còn treo

**`val_meetings` (video nào vào val) — đã quyết 2026-08-12: `rCd8DSMk3-c`.** Chọn
theo số đo thật của `youtube-data-pilot/caption-probe.md` (F1), không đoán — lúc
quyết mới có 4 video, và đây là video khó nhất trong 4 theo cả ba chỉ số đo được:

| `meeting_id` | EN words/min | tỉ lệ non-VN-shaped | lexical-particle rate |
|---|---:|---:|---:|
| dGT3YW0AdD8 | 12.1 | 6.84% | 1.60% |
| rIFrrmm8ILY | 8.1 | 3.75% | 1.11% |
| **rCd8DSMk3-c** | **15.3** | **8.08%** | **0.58%** |
| iyeFAuuEBl4 | 13.0 | 6.76% | 1.67% |

Cao nhất mật độ code-switch (EN words/min, tỉ lệ non-VN-shaped) và thấp nhất
particle rate (caption ít tự nhiên nhất trong 4) — đúng thứ §6 cần: chọn λ\* phải
"thấy" đúng phần khó, không phải phần dễ.

**Cập nhật 2026-08-12 (cùng ngày):** thêm 3 URL nữa (`7B24A9GfHAo`, `xKDHjUoUN54`,
`3nuCdzuyqng`, người dùng cung cấp), tổng nay **7 meeting** — khớp mục tiêu 6–8 của
§2. Quyết định val ở trên **không đổi** (`rCd8DSMk3-c` vẫn là video khó nhất đã đo
được, 3 video mới chưa đo lại theo 3 chỉ số này).

**Cập nhật 2026-08-12 (lần 3) — train/test cho 6 meeting còn lại: đã quyết.**
Người dùng chọn theo độ khó: **test = 2 meeting khó nhì/ba** (`7B24A9GfHAo`,
`3nuCdzuyqng`), **train = 4 còn lại** (`iyeFAuuEBl4`, `dGT3YW0AdD8`, `xKDHjUoUN54`,
`rIFrrmm8ILY`) — xếp theo cùng 3 chỉ số `caption-probe.md` đã dùng chọn val, tính
trên cả 7 video (không chỉ 4). Bảng xếp hạng độ khó đầy đủ, thấp = khó hơn:

| Hạng | `meeting_id` | Vai trò |
|---:|---|---|
| 1 | `rCd8DSMk3-c` | val |
| 2 | `7B24A9GfHAo` | test |
| 3 | `3nuCdzuyqng` | test |
| 4 | `iyeFAuuEBl4` | train |
| 5 | `dGT3YW0AdD8` | train |
| 6 | `xKDHjUoUN54` | train |
| 7 | `rIFrrmm8ILY` | train |

**Cùng lúc, người dùng cũng quyết trim mỗi meeting về cửa sổ giữa 30 phút** (không
phải 15–20 như §2 gốc) trước khi cắt segment — giải quyết luôn vấn đề "F3 dùng
nguyên độ dài thô" đã nêu ở trên. Cả hai đã code vào `scripts/ingest_youtube.py`
và chạy thật — xem SESSIONS.md hàng F3, "Cập nhật 2026-08-12 (lần 3)" để có số đo.

**Chưa wire vào `configs/experiment.yaml`.** `data.val_meetings` hiện tại
(`paid_meeting_0001`, `paid_meeting_0002`, `paid_meeting_0011`) áp cho
`dataset_path: dataset/paid-dataset-v2` — thêm `rCd8DSMk3-c` vào đó bây giờ không
có tác dụng, vì `meeting_id` này không tồn tại trong thư mục đó. Wire thật xảy ra
ở bước 6 (`dataset/mixed-noisy-v1/`, ngoài phạm vi vòng thu thập này) cùng lúc đổi
`dataset_path`.

Ràng buộc sẽ áp khi wire: val phải là **cuộc họp riêng**, không phải segment cắt
ra từ cuộc họp train — điều này tự động thoả vì `rCd8DSMk3-c` là một `meeting_id`
riêng biệt, chưa cắt từ meeting nào khác. `resolve_splits` raise khi một
`meeting_id` rơi vào hai split, nhưng **không** bắt được trường hợp cắt một cuộc
thành hai `meeting_id` khác nhau.
