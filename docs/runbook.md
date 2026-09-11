# Runbook — chạy pipeline dữ liệu YouTube và chạy fine-tune

Tài liệu thao tác. Mỗi bước có: lệnh thật, file nó ghi ra, và **cổng kiểm** phải đạt
trước khi sang bước sau. Lý do đằng sau các quyết định nằm ở `PROJECT_CORE.md` và
`youtube-data-pilot/README.md` — ở đây chỉ có cách chạy.

Quy ước đọc: 🔴 = fail loud, dừng pipeline, không có fallback mềm.

---

## 0. Luật chung, đọc một lần

**Chạy từ gốc repo, và gọi script bằng `python -m scripts.<tên>`, không phải
`python scripts/<tên>.py`.** Sáu script import script anh em (`from scripts.X import ...`)
và chỉ resolve được khi chạy ở dạng module: `fetch_youtube`, `ingest_youtube`,
`review_youtube`, `build_mixed_dataset`, `probe_youtube_captions`, `plot_youtube_stats`.
Dạng `-m` đúng cho mọi script nên cứ dùng nó, khỏi phải nhớ script nào thuộc nhóm nào.

**Notebook Kaggle clone GitHub `main`, không đọc cây làm việc local.** Sửa
`configs/experiment.yaml`, `src/`, `scripts/` xong mà chưa commit + push thì phiên Kaggle
chạy code cũ, và `config.json` đóng băng của run ghi lại giá trị bạn không hề định dùng.
Push trước, luôn luôn.

**`ffmpeg` là system dependency, không phải pip.** `fetch_youtube` raise và nêu đúng tên
nó khi thiếu. `yt-dlp` ở `requirements.txt` và **cố ý không pin version** — YouTube đổi
phía server, bản cũ vỡ.

**Không hardcode đường dẫn nền tảng vào `src/`.** Mọi path đi qua `--override`. Notebook
là nơi duy nhất `/kaggle/input/...` được phép xuất hiện.

**`--override` nhận đúng các field có thật trong `configs/experiment.yaml`**, dạng
`--override data.dataset_path=...`, lặp lại được nhiều lần. Không phải một mặt config
riêng.

---

# Phần A — Pipeline dữ liệu YouTube

Bảy bước, chạy tuần tự. Đầu ra cuối là `dataset/mixed-noisy-v1/` để
`data.dataset_path` trỏ vào.

## A1. Gom URL ứng viên — làm tay

Ghi vào `youtube-data-pilot/urls.txt`, mỗi URL một dòng (dòng không phải URL bị bỏ qua,
nên ghi chú thoải mái).

Ba tiêu chí chọn video ở `youtube-data-pilot/README.md` §3. Phạm vi âm học đã chốt:
**chỉ cuộc gọi online** — audio far-field của phòng họp vật lý không tồn tại trên YouTube.

## A2. Sàng lọc phụ đề trước khi tải một byte audio

```bash
python -m scripts.probe_youtube_captions --urls-file youtube-data-pilot/urls.txt
```

Ghi ra `youtube-data-pilot/caption-probe.md`.

Ba luật loại, tất cả kiểm ở đây chứ không phát hiện sau khi đã tải:

1. Không có mục tiếng Việt trong `automatic_captions` → loại.
2. Mục tiếng Việt là bản **dịch máy** chứ không phải bản nhận dạng gốc → loại. YouTube tự
   dịch phụ đề tự động ra ~100 thứ tiếng nên có key `vi` không chứng minh được gì; bản gốc
   là key riêng hậu tố `-orig` (`vi-orig`), `name` kết thúc bằng " (Original)".
3. Mật độ từ/phút sụt theo các khối 5 phút → loại và điều tra. Bản chép lời không loãng
   dần; bản tóm tắt do người viết thì có.

⚠ `--attempts` > 1 lấy nhiều revision rồi giữ bản giàu nhất, nhưng probe mạnh tay thì
endpoint trả **HTTP 429**. Mặc định là 1, để yên.

**Đừng nhầm hai field của `yt-dlp`:** `automatic_captions` là thứ cần lấy;
`subtitles` là do chủ kênh tải lên và **có thể không phải bản chép lời**.

## A3. Viết `sources.jsonl` — làm tay

Hai field mỗi dòng, không hơn:

```json
{"meeting_id": "dGT3YW0AdD8", "video_url": "https://www.youtube.com/watch?v=dGT3YW0AdD8"}
```

`meeting_id` dùng đúng video id của YouTube, không đặt tên mới.

🔴 Video chưa qua A2 mà đưa thẳng vào đây thì `fetch_youtube` raise, nêu tên
`scripts/probe_youtube_captions.py`.

## A4. Tải audio + phụ đề

```bash
python -m scripts.fetch_youtube
```

Ghi ra `dataset/youtube-meetings/raw/<meeting_id>/{audio.wav, captions.json3, provenance.json}`.

- Xuất **mono 16 kHz ngay lúc tách**, không phải bước sau: `load_audio_16k`
  ([src/data.py](../src/data.py)) **raise** khi gặp stereo, mà audio YouTube mặc định stereo.
- Lưu đúng track `json3` đã dùng. Endpoint phụ đề trả revision khác nhau mỗi lần extract,
  nên bước A5 **phải** parse file đã lưu, không được fetch lại.
- File raw luôn là bản **đầy đủ chưa cắt** cho mọi meeting — vật liệu cho tier 4b.
- Idempotent: meeting đã có đủ 3 file và sha256 khớp thì skip. 🔴 Sha256 lệch thì raise —
  file trên đĩa đã đổi từ lần fetch trước, phải điều tra chứ không ghi đè im lặng.

**Cổng kiểm:** `soundfile.read` từng wav → `data.ndim == 1` và `sr == 16000`.

## A5. Cắt segment + sinh manifest

```bash
python -m scripts.ingest_youtube
# đổi meeting test:
python -m scripts.ingest_youtube --test-meetings 7B24A9GfHAo,3nuCdzuyqng
# ép cửa sổ cho một meeting (lặp lại được):
python -m scripts.ingest_youtube --window-override <meeting_id>:<start_sec>:<end_sec>
```

Ghi ra `dataset/youtube-meetings/audio/<meeting_id>/seg_*.wav` và
`manifest.<meeting_id>.jsonl`. Mặc định `--test-meetings` là `7B24A9GfHAo,3nuCdzuyqng`.

Mọi record ra đời với `verified: false` — lật cờ đó là việc của A6, không phải của bước này.

**`split` chỉ nhận `"demo"` hoặc `"test"`**, giá trị khác bị raise. Train/val không nằm
trong dữ liệu mà suy ra từ `data.val_meetings`:

```
split == "test"                                → test
split == "demo" và meeting_id ∈ val_meetings   → val
split == "demo" và còn lại                     → train
```

**Cắt theo khoảng cách giữa hai mốc *bắt đầu* của từ**, không phải `next.start − cur.end`:
json3 chỉ có mốc bắt đầu, `Word.end` là suy ra và bằng 0 khoảng cách trong cùng một event.
`MAX_SEGMENT_SEC = 30.0` (cửa sổ Whisper), `MAX_CHARS_PER_SEC = 60.0`.

**Cổng kiểm:** không segment nào > 30 s; `_check_speech_rate` không raise. Nhắm 10–25 s —
Whisper pad mọi input lên 30 s bất kể ngắn dài, nên segment 3 s tốn GPU đúng bằng segment 20 s.

⚠ Chạy lại `ingest_meeting` với số segment ít hơn: script xoá hết `seg_*.wav` cũ trước khi
ghi (sửa 2026-08-12). Trước bản sửa đó, `audio/` phình lên 889 MB vì file mồ côi.

## A6. Soát tay — 5 bước, lặp cho từng meeting

Không có lựa chọn tự động. Nhãn train **và** test đều là nháp ASR do người soát toàn bộ.
Bốn quy ước nội dung + phần casing ở `youtube-data-pilot/style-guide.md`, đọc trước khi soát.

```bash
# 1 — sinh worksheet
python -m scripts.review_youtube --emit <meeting_id>
#    → youtube-data-pilot/review/review.<meeting_id>.html

# 2 — mở file HTML bằng browser, nghe từng <audio>, sửa từng <textarea>.
#     Nghe hết, sửa hết. Không bỏ trống ô nào, kể cả khi nháp đã đúng.

# 3 — bấm "Xuất file sửa" ở đầu trang → browser tải corrections.<meeting_id>.json

# 4 — chuyển file vào repo rồi apply
mv ~/Downloads/corrections.<meeting_id>.json youtube-data-pilot/review/
python -m scripts.review_youtube --apply <meeting_id> \
    --corrections youtube-data-pilot/review/corrections.<meeting_id>.json \
    --reviewed-by "<tên-người-soát>"

# 5 — xác nhận meeting đó xong
python -m scripts.review_youtube --check --meeting-id <meeting_id>
```

🔴 Thiếu correction cho segment nào, hoặc ô nào để trống → `--apply` raise và nêu đúng
`meeting_id/segment_id`. Quay lại bước 2, không có đường ghi dở dang.

`--check` không kèm `--meeting-id` thì kiểm toàn bộ và raise khi còn bất kỳ record nào
`verified: false`.

**Quyết định đã chốt, đừng mở lại:** đã cân nhắc tự động hoá bước 3–4 (server local,
File System Access API, `--watch`) và chọn giữ tay. Friction chỉ 1 lần/meeting; tự động
hoá tốn code hơn công tiết kiệm, và phá nguyên tắc "`--apply` đòi đủ correction".

**Bốn quy ước, tóm tắt** (bản đầy đủ ở `style-guide.md`):
1. Không xoá filler — `ạ à ừ ơ` là từ có nghĩa, không phải rác.
2. Số: chữ số cho giá trị rõ ràng là số (tiền, %, thời gian, số lượng); chữ cho phần còn lại.
3. Chồng lấn: chép người nói trội, đặt `quality: overlap`.
4. Token tiếng Anh giữ đúng chính tả người nói — đây là ground truth cho gate code-switch.

## A7. Trộn thành corpus huấn luyện

```bash
# chạy khô trước, luôn luôn
python -m scripts.build_mixed_dataset \
    --sources dataset/paid-dataset-v2 dataset/youtube-meetings \
    --out dataset/mixed-noisy-v1 --dry-run

# thật
python -m scripts.build_mixed_dataset \
    --sources dataset/paid-dataset-v2 dataset/youtube-meetings \
    --out dataset/mixed-noisy-v1
```

🔴 Bốn cổng chạy **trước khi copy một byte**, tất cả fail loud:

1. `meeting_id` phải rời nhau giữa các nguồn.
2. Tên thư mục cấp cao nhất dưới `audio/` của mỗi nguồn phải rời nhau — đây là thứ cho
   phép `audio_filepath` giữ nguyên không phải viết lại.
3. Nguồn nào có field `verified` thì **mọi** record phải `verified: true`. Đây là cổng duy
   nhất đứng giữa lệnh này và việc train trên nhãn chưa soát.
4. Thư mục đích không được tồn tại sẵn và khác rỗng.

`--dry-run` chạy đủ 4 cổng và in cùng bảng train/val/test mà không copy gì.

ℹ `configs/experiment.yaml:data.dataset_path` đang trỏ `dataset/mixed-noisy-v1`, nhưng thư
mục đó **không có trên máy này** — run v4-mixed-r16 trộn corpus ngay trên Kaggle (B2), không
trộn ở local. Chạy `--stage smoke` ở local thì phải dựng nó trước bằng lệnh trên, hoặc
`--override data.dataset_path=` sang một corpus có sẵn.

**Copy thật, không symlink.** Đừng đổ thẳng vào `paid-dataset-v2` — run v3-r16 đã chạy
trên thư mục đó, thêm file vào nghĩa là cùng một tên thư mục chỉ hai nội dung khác nhau ở
hai thời điểm, và hai run hết so được với nhau. Muốn tỉ lệ trộn khác thì tạo
`mixed-noisy-v2/`; tỉ lệ trộn trở thành **dữ liệu**, không phải tham số phải đoán lại về sau.

🔴 **`data.val_meetings` phải chứa ít nhất một `meeting_id` YouTube.** Val thuần tổng hợp
thì `select_lambda()` chọn λ\* dựa trên val tổng hợp + OOD VIVOS — mù hoàn toàn với phần
dữ liệu thật vừa thêm, tức tối ưu λ cho đúng thứ không quan tâm.

## A8. Đóng băng checksum

```bash
python -m scripts.checksum_dataset --mode generate   # sinh mới
python -m scripts.checksum_dataset                   # mặc định --mode verify
```

`dataset/CHECKSUMS.txt` là file duy nhất dưới `dataset/` được track trong git, và là bằng
chứng duy nhất rằng hai nền tảng (local vs Kaggle) chạy trên đúng cùng bytes.

## A9. Biểu đồ corpus — tuỳ chọn

```bash
python -m scripts.plot_youtube_stats
```

Sinh 6 PNG vào `docs/youtube-data-charts/`. Chỉ cần `numpy` + `matplotlib`, không cần torch.
Kết quả đo overlap/speaker-similarity nhúng thành hằng số đầu file kèm ngày đo — vì
`dataset/` bị `.gitignore` chặn nên không nhúng thì chạy lại sau này sẽ lỗi ở 2 biểu đồ cuối.

---

# Phần B — Chạy fine-tune

Bốn stage, chạy theo thứ tự. `--stage` chọn điểm vào, **không phải** chạy cô lập — mỗi
stage giả định artifact của stage trước đã có trên đĩa.

```bash
python -m src.pipeline --stage <smoke|baseline|train|sweep-gate> \
    [--config configs/experiment.yaml] [--override key.path=value ...]
```

Mọi ranh giới stage là một file dưới `outputs/{run_id}/` — không có gì truyền trong bộ
nhớ giữa các stage, nên stage nào cũng chạy lại độc lập được và pipeline resume từ
`.pipeline_state.json`.

## B0. Chuẩn bị dữ liệu phụ trợ

```bash
python scripts/fetch_vivos.py --out dataset/vivos --smoke   # xem schema in ra trước
python scripts/fetch_vivos.py --out dataset/vivos           # rồi mới lấy full test split
```

🔴 `data.ood_eval_path` phải tồn tại trước `--stage train`: tier 2 là phép đo quên duy
nhất trong gate, và `select_lambda` không có λ nào để chọn khi thiếu cột `ood_cer` —
`src/config.py:validate` từ chối giá trị null.

`data.real_bench_path` (`dataset/real-meetings-bench`) là tier 4a, **không tuỳ chọn**.
`data.real_clip_path` thì có, `null` = bỏ qua.

## B1. `smoke` — chạy ở local trước, luôn luôn

```bash
python -m src.pipeline --stage smoke
```

CPU-only, không tải model, không GPU. Chứng minh config load, merge manifest, resolve
split, normalization và bản vá compat của `peft` đều chạy được trước khi tiêu một giây GPU.

**Cổng kiểm:** đọc `split_stats` in ra. Với `mixed-noisy-v1` con số đúng là
`{'train': 4717, 'val': 365, 'test': 654}`. Khác đi nghĩa là bước trộn hoặc
`data.val_meetings` không phải thứ bạn nghĩ.

## B2. Kaggle — mount và override

Notebook: [notebooks/fine-tune-workflow.ipynb](../notebooks/fine-tune-workflow.ipynb).

Cell 2 đặt hai thứ, **đừng bỏ cái nào**:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          # phiên T4 x2: Trainer tự bọc DataParallel
os.environ["TRANSFORMERS_AUTO_CONVERSION"] = "0"  # cũng phải lặp inline ở lệnh --stage baseline
```

Cell `!ls -la /kaggle/input` tồn tại để xác nhận đường dẫn mount **trước** khi đặt
`DATASET_PATH`/`REAL_BENCH_PATH`. Đường dẫn mount thật lồng sâu hơn tên dataset:
`/kaggle/input/datasets/<user>/<slug>/<slug>`. Chuyện này đã cắn dự án một lần — chạy lại
cell đó mỗi phiên, đừng giả định.

Trộn corpus ngay trên Kaggle (`--out` tuyệt đối, ngoài repo đã clone, ~1,4 GB;
`/kaggle/working` giới hạn 20 GB và còn phải chứa `outputs/` + checkpoints).

## B3. Pre-flight: độ dài nhãn so với giới hạn decoder

`WhisperCollator` tokenize nhãn với `padding=True` và **không truncation**, nên nhãn dài
quá `max_target_positions` chỉ lộ ra dưới dạng lỗi phía CUDA giữa chừng buổi train. Cell
17 của notebook kiểm bằng tokenizer, vài giây CPU, không tải weights. Nhãn YouTube dài hơn
nhãn tổng hợp nhiều (426 ký tự so với 275) nên bước này không bỏ được.

## B4. `baseline`

```bash
python -m src.pipeline --stage baseline --override run_id=<RUN_ID> <OVERRIDES>
```

Ghi `metrics/baseline.json` và `audit/predictions_baseline_*.csv`. Chỉ chạy sau khi B1 sạch.

⚠ **Kiểm `eval.limit` trước khi tin bất kỳ con số nào.** `src/gate.py:_eval_split` áp nó
cho **mọi** lần eval — baseline lẫn từng tier của sweep-gate. Config ship `null`; giá trị
`20` từng là rác smoke-test còn sót lại.

## B5. `lr_probe` — tuỳ chọn, chạy ở phiên riêng

```bash
python -m scripts.lr_probe --lrs 5e-5,2e-4,6e-4 --source-run outputs/<RUN_ID> \
    --limit 1600 --epochs 1 --out outputs/lr-probe
```

HuggingFace Trainer không có LR finder. `scripts/lr_probe.py` là bản thay thế thủ công:
gọi `--stage train` một lần cho mỗi learning rate ứng viên, `training.limit` chặn split,
rồi chồng các đường loss lên nhau.

Đọc ba hình dạng: quá thấp là đường xuống chậm và thẳng; đúng là tụt nhanh rồi phẳng ra;
quá cao thì nảy hoặc đi lên. Cần `--stage baseline` xong trước (nó dùng lại
`validated_manifest.jsonl` của run). Mỗi probe ghi cả cây `outputs/{RUN_ID}-lr{lr}/`; probe
nào đã có `training.csv` thì skip, nên phiên bị ngắt resume được.

## B5b. Đo thời gian trước khi cam kết corpus lớn

Trả lời câu "corpus N giờ thì train mất bao lâu". Chi phí một step **không** phụ thuộc kích
thước corpus — model cố định, batch cố định, Whisper luôn pad audio về cửa sổ 30 s — nên đo
trên corpus nhỏ rồi nhân với số step của corpus lớn. Chỉ số step thay đổi:

```
số step mỗi epoch = segment_train / (training.batch_size * training.grad_accum_steps)
```

### Chuẩn bị

```bash
R=v6-probe
OV="--override run_id=$R --override lora.rank=32 --override lora.alpha=64  --override training.batch_size=16 --override training.grad_accum_steps=1  --override training.gradient_checkpointing=false  --override training.val_limit=32 --override data.real_bench_path=null"

python -m src.pipeline --stage baseline $OV --override eval.limit=40
```

`eval.limit` chỉ cắt phần chấm điểm; `write_validated_manifest` chạy trước mọi eval nên
manifest vẫn đầy đủ. `training.val_limit=32` để lần eval cuối epoch không nuốt mất phép đo.

**Phải dùng `data.real_bench_path=null`, không phải `=None`.** `apply_override` đẩy giá trị
qua `yaml.safe_load`, mà YAML đọc `None` thành chuỗi `'None'` — truthy — nên pipeline sẽ đi
tìm thư mục tên `None`.

### Probe 1 — trần cứng

```bash
python scripts/make_probe_manifest.py --run-dir outputs/$R --mode worst --steps 40 --batch 16
python -m src.pipeline --stage train $OV --override training.epochs=1
```

`--mode worst` thay split train bằng 640 segment có nhãn dài nhất, nên mọi batch đều pad tới
mức đắt nhất corpus có. Không batch nào trong lượt chạy thật vượt được con số này, và VRAM
đỉnh cũng là trường hợp xấu nhất — trả lời luôn câu rank/batch có OOM không.

Đừng dùng `training.limit` ở đây: manifest đã bị cắt sẵn rồi.

### Probe 2 — chi phí điển hình

```bash
cp outputs/$R/validated_manifest.full.jsonl outputs/$R/validated_manifest.jsonl
python scripts/make_probe_manifest.py --run-dir outputs/$R --mode random --steps 40 --batch 16 --force
python -m src.pipeline --stage train $OV --override training.epochs=1
```

Khoảng cách giữa hai probe cho biết trần lỏng bao nhiêu.

### Probe 3 — dữ liệu có kịp nuôi GPU không

```bash
sync && sudo sysctl -w vm.drop_caches=3     # bỏ nếu không có quyền root
python scripts/probe_io.py --dataset <DATASET_PATH> --files 300 --target-hours 100     --workers 8 --batch 16 --seconds-per-step <đo được ở probe 2>     --feature-extractor vinai/PhoWhisper-large
```

Probe train chỉ đọc vài trăm file nằm sẵn trong page cache; một epoch 100 giờ đọc ~14 GB qua
~53.000 file rời và resample phần lớn từ 24 kHz bằng CPU. Dòng `feed` dưới 1.0x nghĩa là GPU
phải chờ dữ liệu — lúc đó tăng `dataloader_num_workers` hoặc resample sẵn corpus về 16 kHz,
thuê card mạnh hơn không giúp gì.

### Đọc kết quả

`outputs/$R/metrics/timing.json`:

| Trường | Ý nghĩa |
|---|---|
| `steady_state.cycle_seconds.median` | giây mỗi step, đã loại 10 step warmup — **số đem nhân** |
| `steady_state.dataloader_gap_seconds.median` | phần chờ dữ liệu trong mỗi step |
| `steady_state.compute_seconds.median` | phần tính toán thật |
| `eval_seconds` | tổng thời gian eval trong lúc train |
| `model_load_seconds` | nạp checkpoint ~6,2 GB |
| `dead_seconds` | phần ngoài `trainer.train()`: nạp model, dựng dataset, ghi checkpoint |
| `peak_vram_reserved_gb` | con số so với dung lượng card, khớp với `nvidia-smi` |

Ngân sách tổng:

```
T = T_upload + T_baseline + T_train + T_sweepgate
T_train = số_step × cycle_seconds + số_lần_eval × T_eval + dead_seconds
```

`T_baseline` đo riêng bằng một lần `--stage baseline` không cắt `eval.limit`, chia cho số
segment test + OOD để ra chi phí sinh mỗi segment, rồi nhân lên theo corpus mới. Khoản này
không rút ngắn được: gate so metric sau train với chính `metrics/baseline.json`, cắt baseline
là hỏng gate.

### Dọn sau khi đo

```bash
cp outputs/$R/validated_manifest.full.jsonl outputs/$R/validated_manifest.jsonl
```

Manifest của run probe đang là bản đã cắt. Chạy thật trên nó thì train chỉ thấy 640 segment.

## B6. `train`

```bash
python -m src.pipeline --stage train --override run_id=<RUN_ID> <OVERRIDES>
```

LoRA SFT → `checkpoints/best/`. `RobustEvalTrackingCallback` tự lưu `checkpoints/best/`
ngay khoảnh khắc `eval_val_cer` cải thiện (thay cho `EarlyStoppingCallback` /
`load_best_model_at_end`), và 🔴 raise nếu không bao giờ quan sát được `eval_val_cer` thay
vì âm thầm ship sai checkpoint.

**Cổng kiểm:** xác nhận `checkpoints/best/` xuất hiện giữa chừng, không phải chỉ ở cuối.

**Chi phí thật để định liệu:** PhoWhisper-large, `lora.rank=16`, 3 epoch, 801 step,
batch 2 × grad_accum 8 → **6 h 31 m** trên một T4 (29,34 s/step), không OOM. Con số
9,75 GiB @ batch 8 trong `CLAUDE.md` là của `-small`, không phải model này — nâng batch
lên là thay đổi chưa được kiểm.

## B7. `sweep-gate`

```bash
python -m src.pipeline --stage sweep-gate --override run_id=<RUN_ID> <OVERRIDES> \
    --override hub.push=False
```

Sweep λ → gate tier 1/2/4a → push lên HF **chỉ khi** `overall_pass` và `hub.push: true`.

🔴 **Không λ nào trong lưới giữ được OOD CER trong ngân sách → pipeline fail cứng.** Không
adapter nào được chọn, không gì được push. Không bao giờ tự lùi về λ=0 hay "λ gần nhất".

`select_lambda` đi lên theo λ tăng dần, dừng ở khuỷu cost/benefit
(`sweep.elbow_ratio_threshold`, mặc định 10.0). Chỉ bước có tỉ số **dương** mới thành mốc
so cho bước kế — một bước mua được val CER mà OOD CER không xấu đi có tỉ số ≤ 0, và
`threshold × (số không dương)` là cái bar mà mọi chi phí thật đều vượt. Chính lỗi đó dừng
`v4-mixed-r16` ở λ=0.25 và ship adapter yếu hơn sweep của chính nó cho phép.

⚠ Tier 1 chấm **theo từng nguồn** lẫn gộp chung (từ 2026-08-16). Test split là 59,6% ký tự
YouTube từ 34,9% số segment, nên một bound gộp duy nhất cho phép phần YouTube thắng lớn
giữ tier mở trong khi phần tổng hợp tụt. Tier chỉ pass khi bound gộp **và** mọi slice pass.

**In bảng sweep ra tay.** Trên run v3-r16, tqdm vẽ đè mất dòng cuối nên bảng sweep và λ\*
đã chọn không xuất hiện ở đâu ngoài file CSV.

## B8. Kéo bằng chứng về trước khi phiên Kaggle chết

```bash
find outputs/{RUN_ID} -type f | sort
zip -r -q outputs_{RUN_ID}.zip outputs/{RUN_ID}
```

Toàn bộ `outputs/{run_id}/` là bằng chứng của run: `metrics/baseline.json`,
`metrics/lambda_sweep.csv`, `metrics/gate_results.json`, và mọi `audit/predictions_*.csv`.
**Không được lưu ở đâu khác.**

---

# Phần C — Sau khi gate pass

## C1. λ do người chọn, khi luật chọn sai

Luật `select_lambda` cân val CER với OOD CER và **không nhìn thấy**
`english_token_retention` — đúng trục mà production đã hồi quy. Khi cần một λ khác λ luật
chọn, không sửa ngưỡng cho vừa đáp án đã biết; chạy script riêng:

```bash
python -m scripts.gate_at_lambda --run-dir outputs/<RUN_ID> --lam 0.75 \
    --audio-root <path> --ood-eval-path <path> --real-bench-path <path> \
    --out-dir <path>
```

Chạy đủ gate ở λ đó và nướng adapter ở λ đó, sinh ra một run dir mà
`scripts/merge_and_push.py` nhận nguyên không cần sửa. Ghi lý do override vào `SESSIONS.md`
và vào model card — để một run bình thường không bao giờ lặng lẽ đi vòng qua luật chọn.

## C2. Publish

```bash
python -m scripts.merge_and_push --run-dir <dir> --adapter <dir> \
    --repo-id <user>/<name> --out <dir> \
    --module4-dir <dir> --module4-production-dir <dir> [--confirm]
```

Merge adapter vào base weights, push thành model độc lập (load không cần `peft`). Mặc định
private; `--public` là hành động hướng ra ngoài, chỉ bật khi đã quyết.

⚠ **Dùng lại tên repo không làm mất hiệu lực cache của ai cả.** Client đã tải bản cũ vẫn
tiếp tục phục vụ bản cũ cho tới khi thư mục cache bị xoá hoặc truyền `force_download=True`.
Phải báo từng bên tiêu thụ; tên repo tự nó không nói cho họ biết.

## C3. Cập nhật tài liệu sau khi publish

Publish xong thì bốn chỗ này thành sai nếu không sửa — đã từng sai thật:

| File | Sửa gì |
|---|---|
| `docs/score-table.md` | thêm nhóm cột cho lần fine-tune mới + dòng bảng tỉ lệ + dòng ánh xạ tên + dòng nguồn |
| `docs/finetune-slides.html` | cùng nội dung, và slide tóm tắt cuối deck |
| `docs/training-curves/README.md` | bảng "Model xuất bản" của lần đó |
| `docs/finetune-results-report-v*.md` | header "Mức λ đề xuất" và bullet trạng thái publish |

Sinh lại biểu đồ:

```bash
python -m scripts.plot_training_curve outputs/<RUN_ID> --out docs/training-curves/<name>_training-curve.png
python -m scripts.plot_lambda_tradeoff outputs/<RUN_ID> --out docs/training-curves/<name>_lambda-tradeoff.png --chosen 0.75
```

---

# Phần D — Những chỗ đã cắn người thật

Danh sách này lấy từ sự cố đã xảy ra, không phải rủi ro giả định.

| Chỗ | Triệu chứng | Cách tránh |
|---|---|---|
| Đường dẫn mount Kaggle | `FileNotFoundError` sau khi đã tải model | Chạy `!ls -la /kaggle/input` mỗi phiên. Path thật lồng: `/kaggle/input/datasets/<user>/<slug>/<slug>` |
| `TRANSFORMERS_AUTO_CONVERSION` | stage baseline hỏng | Đặt cả env var ở Cell 2 **và** inline ở lệnh `--stage baseline` |
| T4 x2 | Bộ nhớ phí, không nhanh hơn | `CUDA_VISIBLE_DEVICES=0` trước khi bất kỳ stage nào chạm CUDA |
| `eval.limit` | Số đẹp một cách vô lý | Kiểm `null` trước khi trích bất kỳ CER nào từ run |
| `no_repeat_ngram_size=3` | Mất ~3 điểm CER, sai thanh điệu | Không bật ban-ngram cho tiếng Việt. Nghi weights hỏng thì diff kwargs decode trước |
| Target modules LoRA | Fail âm thầm | Đúng 6 module: `q_proj, k_proj, v_proj, out_proj, fc1, fc2`, `task_type=SEQ_2_SEQ_LM`. `o_proj`/`gate_proj`/`CAUSAL_LM` của guide LLM không tồn tại trên Whisper |
| Nhãn dài | Lỗi phía CUDA giữa buổi train | Chạy pre-flight tokenizer (B3) |
| `select_lambda` mù retention | Ship adapter yếu hơn sweep cho phép | Đọc `lambda_sweep.csv` bằng mắt; dùng `gate_at_lambda.py` khi cần override |
| Val không có meeting thật | λ\* chọn mù phần dữ liệu vừa thêm | `val_meetings` phải có ≥1 `meeting_id` YouTube |
| Chạy `python scripts/x.py` | `ModuleNotFoundError: scripts` | Luôn `python -m scripts.x` từ gốc repo |
| Code Kaggle nói chung | Lỗi version/API mỗi lần | Feature-detect, cắt bớt dependency, probe trước khi viết. Chưa lần nào chạy đúng ngay lần đầu |

**Trước khi quy một cải thiện CER cho model:** ~62% mức cải thiện CER trong miền của các
run trước truy về chuẩn hoá chữ số, không phải học âm học. Đọc
`gate_results.json:normalization_check` (`word_to_digit` vs `as_written`) trước.

**Trước khi trích một caveat về dataset:** các cảnh báo trong `CLAUDE.md` có thể chỉ đúng
cho một version (v1 vs v2). Kiểm manifest của chính run đó, đừng trích từ trí nhớ.

---

# Phần E — Bảng benchmark nhiều mô hình

Đo nhiều mô hình trên nhiều bộ benchmark rồi ra một bảng CER/WER. Tách hẳn khỏi pipeline
fine-tune: không train, không gate, không đụng `configs/experiment.yaml`.

Bốn file, hai giai đoạn:

| File | Vai trò |
|---|---|
| `scripts/benchmark_suites.py` | Đăng ký bộ benchmark. Thêm bộ mới = thêm một dòng `SUITE_SPECS` |
| `scripts/benchmark_run.py` | Giải mã (GPU hoặc API), ghi hypothesis **thô**, resume theo `segment_id` |
| `scripts/benchmark_report.py` | Chấm điểm trên CPU từ file đã ghi. Chạy lại bao nhiêu lần cũng được |
| `notebooks/benchmark-matrix.ipynb` | Runner Kaggle. Sinh lại bằng `python scripts/_make_benchmark_notebook.py` |

Sáu bộ có sẵn: `vimedcss-test`, `vimedcss-hard` (tải từ Hub), `youtube-test`,
`synthetic-test` (cùng trỏ vào `mixed-noisy-v1`), `vivos`, `cross-domain`.

## E0. Đóng gói corpus để upload lên Kaggle

```bash
python -m scripts.zip_for_kaggle --src dataset/youtube-data-pilot-package     --out Outputs/cross-domain-bench.zip
```

Chỉ lấy `manifest.*.jsonl` + `audio/`. 🔴 **Không dùng `Compress-Archive` hay chuột phải
→ Send to → Compressed folder trên Windows.** Công cụ zip Windows đã từng ghi tên entry
bằng `\` thay vì `/`; giải nén trên Kaggle ra **một** file phẳng tên
`IGZYBrbDUEw\seg_0000.wav`, và lỗi chỉ lộ ra hàng giờ sau, giữa lúc giải mã. Script này
ghi `/` tường minh rồi mở lại archive kiểm tra, từ chối nếu còn `\`.

Kỳ vọng cho `cross-domain-bench`: 302 entry (3 manifest + 299 wav), ~125 MB.

## E1. Giải mã

```bash
# mô hình local, GPU
python -m scripts.benchmark_run --backend hf --model vinai/PhoWhisper-large     --suites vimedcss-test,youtube-test,synthetic-test,vivos,cross-domain     --path youtube-test=/kaggle/working/dataset/mixed-noisy-v1     --path synthetic-test=/kaggle/working/dataset/mixed-noisy-v1     --path vivos=dataset/vivos     --path cross-domain=/kaggle/input/datasets/<user>/cross-domain-bench/cross-domain-bench     --out Outputs/benchmark-<ngày> --batch-size 8

# API trả phí — luôn probe 1 đoạn trước
python -m scripts.benchmark_run --backend elevenlabs --model scribe_v2     --suites cross-domain --path cross-domain=<dir> --out /tmp/_probe --limit 1
```

Ghi ra `<out>/<model_slug>.<suite>.persegment.jsonl` (hypothesis thô, chưa chuẩn hoá) và
`<...>.failures.jsonl`. 🔴 Đoạn nào giải mã hỏng thì ghi vào file failures và **bị loại
khỏi mọi cột** lúc chấm — không có chuyện một cột nhiều đoạn hơn cột kia.

Cổng kiểm: `--limit 8` chạy khô toàn bộ trước khi bỏ giờ GPU hoặc tiền API vào.

## E2. Chấm điểm

```bash
python -m scripts.benchmark_report --dir Outputs/benchmark-<ngày>     --variant N3 --baseline vinai_phowhisper_large
```

`N3` = chuẩn hoá mặc định của repo, cùng thang với mọi số trong `docs/so-lieu-tong-hop.md`.
`N0` = biến thể thô đã dựng lại được dòng base Bảng 4 của ViMedCSS; **so với bài báo phải
dùng N0**. Hai biến thể là hai lần chạy, không bao giờ trộn trong một bảng.

Ghi ra `benchmark-table.<variant>.md` và `scores.json` (có CI bootstrap, paired bootstrap
so với `--baseline`, và danh sách đoạn bị loại).

## E3. Đừng trả tiền / đốt GPU hai lần

- Resume theo `segment_id`: phiên chết ở 80% thì lần chạy sau chỉ làm 20% còn lại.
- ViMedCSS đã giải mã sẵn cho base + v5 (`Outputs/vimedcss_output/`, 2026-09-09, cùng
  đường decode, đã track trong git). Cell `REUSE_VIMEDCSS` của notebook copy 4 file
  `.persegment.jsonl` vào thư mục run và đổi tên `<model>.test` → `<model>.vimedcss-test`,
  tiết kiệm ~4,2 giờ T4. Mặc định `False` — đo lại từ đầu; đặt `True` để dùng lại.
- Hypothesis đã trả tiền cho API nằm ở `Outputs/benchmark-2026-09-10/`, track bằng
  `git add -f` vì `Outputs/` bị gitignore. Cell ngay sau đó copy chúng vào thư mục run.
- 🔴 **Không so ngang giữa các cột.** Sáu bộ, sáu độ khó. Chỉ đọc dọc.
