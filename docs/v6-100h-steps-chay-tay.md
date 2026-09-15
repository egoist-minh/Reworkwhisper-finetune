# v6-100h-steps — hướng dẫn chạy tay

Bảng lệnh theo thứ tự cho lượt thuê máy. Quyết định và lý do nằm ở
[docs/v6-100h-steps-plan.md](v6-100h-steps-plan.md); quy trình chung nằm ở
[docs/server-finetune.md](server-finetune.md). File này chỉ là thứ tự gõ lệnh.

Đặc điểm lượt này khác quy trình mặc định: **không chạy `--stage baseline`**, không chạy
`sweep-gate`, không cần `dataset/vivos`, không chạy `scripts/select_val_meetings.py`.

**Corpus lượt này là `v6-corpus` cộng add-on 9,07 h TTS.** `v6-corpus.tar` trên Hub là bản
12/09 và không chứa hai lô `paid-meeting-vi-0247-0296` / `paid-meeting-vi-0297-0339`;
`scripts/build_v6_corpus_addon.py` đóng riêng phần mới (1,57 GB) để giải nén đè lên. Train
thành **42.794 segment / 106,69 h / density 3,55%** (trước khi thêm: 2,34%). Lý do ở plan
§2.1. Đừng lẫn với `dataset/v6-ondomain-15h-tts10h` — thư mục khác, dòng thí nghiệm khác.

---

## 0. Trên máy Windows, trước khi thuê máy (bắt buộc)

Máy thuê lấy mã nguồn bằng `git clone`, nên mọi commit chưa push đều không tồn tại với nó.
Mọi thứ lượt này cần đã push tới `c208eed` (val 7 cuộc, `on_train_end`, watcher đẩy
checkpoint). Xác nhận không còn gì kẹt lại:

```bash
git push origin main
git rev-parse HEAD        # ghi lại, lát nữa đối chiếu trên máy thuê
```

Chuẩn bị sẵn hai thứ để chuyển sang máy thuê:

- `dataset/cross-domain-bench.zip` (119 MB) — scp từ máy này, không có trên HF.
- `HF_TOKEN` có quyền đọc org `rework-whisper-v6-org` (corpus là repo **private**).

---

## 1. Máy thuê — kết nối và khảo sát

IP đổi mỗi lần thuê. Hỏi người dùng IP mới rồi sửa host `speech-agent-gpu`:

```bash
sed -i 's/HostName .*/HostName <IP mới>/' ~/.ssh/config

ssh -o StrictHostKeyChecking=accept-new speech-agent-gpu \
  'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
   nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
   nproc; free -g | head -2; df -h /; python3 -V'
```

Hai điều phải xác nhận trước khi đi tiếp: **không có job của người khác đang chiếm GPU**
(mọi số đo thời gian sẽ vô nghĩa), và **còn ít nhất 60 GB đĩa trống** (corpus 14,7 GB tar
cộng add-on 1,57 GB, cộng phần giải nén, cộng ~1,7 GB checkpoint `step-*`).

## 2. Mã nguồn và môi trường

```bash
ssh speech-agent-gpu
mkdir -p ~/speech && cd ~/speech
git clone -b main https://github.com/egoist-minh/Reworkwhisper-finetune.git Fine_tune_wf
cd Fine_tune_wf && git rev-parse HEAD      # phải trùng commit đã ghi ở §0

python3 -m venv .venv && source .venv/bin/activate
pip install -q --upgrade pip && pip install -q huggingface_hub hf_transfer
```

**Bật lượt tải corpus ngay tại đây**, trước `pip install -r requirements.txt` — 14,7 GB là
khoản dài nhất của cả lượt thuê, và phần cài đặt bên dưới không cần đợi nó:

```bash
export HF_TOKEN=<token `org_download`, đuôi HTgk — KHÔNG phải `write_dataset`>
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p ~/speech/Fine_tune_wf/dataset

cat > fetch_corpus.sh <<'EOF'
set -e
cd ~/speech/Fine_tune_wf
source .venv/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download rework-whisper-v6-org/v6-corpus v6-corpus.tar \
    --repo-type dataset --local-dir .
tar -xf v6-corpus.tar -C dataset
rm -f v6-corpus.tar
huggingface-cli download rework-whisper-v6-org/v6-corpus v6-corpus-addon.tar \
    --repo-type dataset --local-dir .
tar -xf v6-corpus-addon.tar -C dataset/v6-corpus
rm -f v6-corpus-addon.tar
echo FETCH_OK
EOF

tmux new-session -d -s fetch "bash fetch_corpus.sh > fetch.log 2>&1"
```

Viết ra file rồi mới `tmux` chạy, không nhét lệnh vào chuỗi lồng nhau — `tmux` + `bash -lc`
+ `python -c` là ba tầng nháy, sai một dấu là tải nhầm hoặc im lặng không chạy.
`HF_TOKEN` đã export ở shell hiện tại nên script kế thừa được.

Phải đúng token **`org_download`** (đuôi `HTgk`): nó có `repo.write` trên org
`rework-whisper-v6-org`. Token `write_dataset` (đuôi `Wlax`) là fine-grained chỉ scope
user, mọi thao tác ghi dưới org trả **403 Forbidden** — kể cả watcher ở §5 và
`push_run_adapters` ở §8, chứ không riêng lượt tải này.

`HF_HUB_ENABLE_HF_TRANSFER=1` đổi tầng tải sang backend Rust nhiều luồng; trên đường truyền
nhanh nó là khác biệt lớn nhất của cả bảng ngân sách. `rm` ngay sau khi giải nén để không
giữ 14,7 GB tar cạnh 14,7 GB đã bung.

Trong lúc nó chạy, cài nốt phần còn lại:

```bash
pip install -r requirements.txt
```

Kiểm torch trước mọi thứ khác — `pip install torch` hay lấy bản CUDA mới hơn driver:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Nếu `False` và báo *"The NVIDIA driver on your system is too old"*, cài lại theo driver đọc
được ở §1 (ví dụ driver 560.x = CUDA 12.6):

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
```

Rồi chạy test suite, phải **0 failed** (số skip nhiều hơn máy local là đúng):

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q
```

## 3. Dữ liệu

### 3.1 Corpus — đợi lượt tải ở §2 xong

```bash
tail -2 fetch.log        # phải thấy FETCH_OK
ls dataset/v6-corpus | head
```

Dùng `huggingface-cli`, **không** dùng `curl` — `curl` không xác thực bị bóp xuống
~120 kB/s so với 51 MB/s.

Xác nhận split đúng như plan §2.1 đã đo — **train 42.794 / val 1.179 / test 654** sau khi
có add-on ở §3.1b và `data.val_meetings` ở §3.3:

```bash
PYTHONPATH=. .venv/bin/python -c "
from src.config import load_config
from src.data import load_manifests, resolve_splits
cfg = load_config('configs/experiment.yaml')
cfg.data.dataset_path = 'dataset/v6-corpus'
tr, va, te = resolve_splits(load_manifests(cfg.data.dataset_path), cfg.data.val_meetings)
print(len(tr), len(va), len(te))"
```

### 3.1b Add-on TTS — kiểm đã nằm đúng chỗ

Add-on là file thứ hai trong chính repo `v6-corpus`, đóng **không có thư mục gốc**, nên nó bung thẳng vào `dataset/v6-corpus/` và
trộn cùng manifest sẵn có. Kiểm:

```bash
ls dataset/v6-corpus/manifest.paid_meeting_0297.jsonl      # phải tồn tại
ls dataset/v6-corpus/manifest.*.jsonl | wc -l              # phải là 392
ls dataset/v6-corpus/audio/paid_meeting_0297/raw_turns | head -2
```

392 = 299 cuộc của corpus gốc cộng 93 cuộc của hai lô TTS. Sai số này thì **dừng** — 
`resolve_splits` sẽ im lặng ra một train nhỏ hơn và mọi số bước dưới đây lệch theo.

### 3.2 Bench cross-domain (scp từ máy Windows)

Chạy **ở máy Windows**:

```bash
scp dataset/cross-domain-bench.zip speech-agent-gpu:'~/speech/Fine_tune_wf/dataset/'
```

Rồi trên máy thuê:

```bash
cd ~/speech/Fine_tune_wf && unzip -q dataset/cross-domain-bench.zip -d dataset/
ls dataset/cross-domain-bench | head
```

Không cần `scripts/fetch_vivos.py`: lượt này không chạy `--stage baseline` và
`data.ood_eval_path=null`.

### 3.3 Mở rộng val bằng 3 cuộc họp YouTube

Val mặc định của v6-corpus là 4 cuộc họp, 365 segment, 0,79 h, và chủ đề trùng test gần
như hoàn toàn (`paid_meeting_0001` deploy, `0011` review experiment mô hình,
`rCd8DSMk3-c` nghề tech). Thêm 3 cuộc họp YouTube đang nằm trong train:

| meeting_id | chủ đề | seg | giờ | density |
|---|---|---|---|---|
| `coteccons-agm-2025` | đại hội cổ đông Coteccons — xây dựng, chủ tịch nói tiếng Anh có phiên dịch (`backlog`, `global`, cả câu tiếng Anh) | 265 | 1,19 | 9,16% |
| `jB1P4bqLwDY` | tọa đàm đầu tư khởi nghiệp (NIC, ThinkZone, Vietnam Venture Summit) — `startup`, `investor`, `angel` | 301 | 1,33 | 7,85% |
| `zlKBfNzfh50` | "chuyện người làm nghề" — digital marketing, SEO | 248 | 1,09 | 2,41% |

Hai cái đầu là code-switching mức câu ngoài domain công nghệ; `zlKBfNzfh50` giữ mặt
marketing. Ba cuộc họp marketing còn lại (`W71X_KzfJg8`, `Kltz8ZJ9HOw`, `lSlkfTiWKBk`) **ở
lại train**: density 1,3–1,8%, và 3,0 h của chúng chiếm 87% khối token của val nếu đưa
vào, kéo density gộp xuống 2,80%.

Không cần dựng file gì. Train/val không nằm trong corpus — `src.data.resolve_splits` suy
ra từ `data.val_meetings` lúc nạp, nên thêm meeting_id vào danh sách đó là vừa đưa vào
val vừa rút khỏi train, một lần. Giá trị đầy đủ nằm trong `$OV` ở §5.

Kiểm trước khi train:

```bash
PYTHONPATH=. .venv/bin/python -c "
from src.config import load as load_config
from src.data import load_manifests, resolve_splits, split_stats
cfg = load_config('configs/experiment.yaml')
cfg.data.val_meetings += ['zlKBfNzfh50','coteccons-agm-2025','jB1P4bqLwDY']
print(split_stats(resolve_splits(load_manifests('dataset/v6-corpus'), cfg.data.val_meetings)))"
```

Phải ra train 42.794 / val 1.179 / test 654 (đã có add-on ở §3.1b).

| | trước | sau |
|---|---|---|
| train | 43.608 seg, 110,30 h, density 3,64%, 5.452 step | 42.794 seg, 106,69 h, density **3,55%**, **5.350 step** |
| val | 4 mtg, 365 seg, 0,79 h, density 8,89%, 253 seg có loanword | 7 mtg, **1.179 seg**, 4,40 h, density **7,12%**, **791** seg có loanword |

Val to gấp 3,2 lần mà density gần như đứng yên: 8,89% xuống 7,12%, và **test là 7,06%** —
val giờ đo trên cùng nồng độ code-switch với bộ sẽ chấm ở §7. Số segment có loanword tăng
253 lên 791, đủ để ValCER thôi trồi sụt vài điểm giữa các vòng.

Hai cột trên đều đo trên corpus **đã có add-on**; chúng tách riêng tác dụng của việc mở
val, không phải của add-on. Density train tụt 3,64% xuống 3,55% vì ba cuộc bị rút đều đậm
hơn trung bình — mất 0,09 điểm trên 106,7 h, không đáng kể so với cái được ở val. Step rời
5.452 xuống 5.350.

**ValCER lượt này vẫn không so được với lượt trước** (4 cuộc họp so với 7) — xem §5.

## 4. Lượt train tí hon — kiểm `checkpoints/step-*` (~3 phút GPU)

Đây là mục còn nợ của plan §1.1: code đã sửa và `pytest` xanh, nhưng chưa lượt nào thật sự
chạy vì máy Windows không có torch. Chạy trước lượt 100 h, vì nếu sai thì cả lượt thuê
không có gì để chấm:

```bash
export PYTHONPATH=. TRANSFORMERS_AUTO_CONVERSION=0
source .venv/bin/activate
SMOKE="--override run_id=smoke-steps
    --override data.dataset_path=dataset/v6-corpus
    --override data.real_bench_path=null
    --override data.ood_eval_path=null
    --override data.val_meetings=[paid_meeting_0001,paid_meeting_0002,paid_meeting_0011,rCd8DSMk3-c,zlKBfNzfh50,coteccons-agm-2025,jB1P4bqLwDY]
    --override data.cross_domain_path=dataset/cross-domain-bench
    --override training.limit=20 --override training.val_limit=20
    --override training.eval_steps=5 --override training.epochs=1
    --override training.batch_size=2 --override training.grad_accum_steps=1
    --override eval.batch_size=8"

python -m src.pipeline --stage prepare $SMOKE
python -m src.pipeline --stage train   $SMOKE
ls outputs/smoke-steps/checkpoints/
```

**Ba cổng, phải qua cả ba:**

1. Nhiều hơn một thư mục `step-<số>` cạnh `best/`. Chỉ thấy `best/` là `on_evaluate` chưa
   lưu vô điều kiện.
2. Có một `step-<số>` ứng với **bước cuối**, không phải bội số của `eval_steps` — đó là
   `on_train_end` ([src/train.py:369](../src/train.py#L369)). Thiếu nó thì checkpoint
   "last" không tồn tại và §8 không có gì để đẩy lên.
3. Watcher đẩy được lên Hub — chạy một lượt quét trên chính thư mục smoke:

   ```bash
   PYTHONPATH=. .venv/bin/python -m scripts.push_checkpoints_live \
       --run-dir outputs/smoke-steps \
       --repo-id rework-whisper-v6-org/smoke-steps-checkpoints \
       --settle 0 --once
   ```

   Phải in ra ít nhất một dòng `step-<số>: https://…`. In `upload failed` là token thiếu
   quyền ghi hoặc sai org — biết ở đây tốn 1 phút, biết sau lượt 96 phút thì mất cả lượt.
   Xoá repo `smoke-steps-checkpoints` trên Hub sau khi qua cổng.

Sai cổng nào thì dừng, sửa xong mới chạy lượt thật.

Dọn trước khi vào lượt thật: `rm -rf outputs/smoke-steps outputs/smoke-steps.zip`.

## 5. Lượt 100 h (~80 phút)

**2 epoch, không phải 3.** 3 epoch là 8.025 bước cộng 20 vòng eval = 164 phút tính từ lúc
train chạy, tức phải tải xong 16 GB trong 16 phút để vừa trần 3 giờ. Không xảy ra. Đây là
chỗ duy nhất lượt này lệch khỏi cấu hình v5 — nói đúng tên khi trình số.

Kèm theo: `lr_scheduler_type` mặc định là linear và warmup 0,1 tính trên tổng số bước, nên
2 epoch có warmup 535 bước và LR về 0 ở 5.350. Một lượt 3 epoch sau này **không chồng
đường cong lên lượt này được** — mọi điểm đều ở LR khác, không riêng phần đuôi.

```bash
export PYTHONPATH=. TRANSFORMERS_AUTO_CONVERSION=0
R=v6-100h-steps
OV="--override run_id=$R
    --override data.dataset_path=dataset/v6-corpus
    --override data.real_bench_path=null
    --override data.ood_eval_path=null
    --override data.val_meetings=[paid_meeting_0001,paid_meeting_0002,paid_meeting_0011,rCd8DSMk3-c,zlKBfNzfh50,coteccons-agm-2025,jB1P4bqLwDY]
    --override data.cross_domain_path=dataset/cross-domain-bench
    --override training.epochs=2
    --override training.learning_rate=2.0e-4
    --override training.batch_size=16 --override training.grad_accum_steps=1
    --override training.gradient_checkpointing=false
    --override training.eval_steps=400
    --override training.val_limit=null
    --override training.early_stopping_patience=null
    --override eval.batch_size=64"

tmux new-session -d -s run "bash -lc '
  cd ~/speech/Fine_tune_wf && source .venv/bin/activate
  export PYTHONPATH=. TRANSFORMERS_AUTO_CONVERSION=0
  python -m src.pipeline --stage prepare $OV &&
  python -m src.pipeline --stage train   $OV
' > run.log 2>&1; echo EXIT=\$? >> run.log"
```

**Bật ngay watcher đẩy checkpoint lên Hub**, cùng lúc với lệnh trên — máy thuê có thể tắt
giữa chừng (hết hạn mức, bị thu hồi), và `outputs/` đi theo máy:

```bash
tmux new-session -d -s push "bash -lc '
  cd ~/speech/Fine_tune_wf && source .venv/bin/activate
  export PYTHONPATH=. HF_TOKEN=$HF_TOKEN
  python -m scripts.push_checkpoints_live \
      --run-dir outputs/$R \
      --repo-id rework-whisper-v6-org/$R-checkpoints \
      --log run.log
' > push.log 2>&1"

tail -5 push.log      # sau vòng eval đầu phải thấy dòng step-400
```

Cứ 120 giây nó quét `checkpoints/`, thấy `step-<N>` mới thì đẩy lên một thư mục con cùng
tên trong repo private đó. Một thư mục chỉ được đẩy khi có đủ `adapter_config.json` +
`adapter_model.safetensors` và 60 giây không ai chạm vào — đẩy giữa lúc PEFT còn đang ghi
là đưa lên Hub một adapter cụt dưới cái tên đọc như bản hoàn chỉnh. Upload hỏng thì thử
lại vòng sau; tiến trình này không bao giờ raise ra ngoài, nên nó chết âm thầm là rủi ro
duy nhất — vì vậy `tail push.log` mỗi lần ngó `run.log`.

Không cần đẩy `checkpoints/best/`: `on_evaluate` ghi `best/` và `step-<N>` từ cùng một
model trong cùng một lệnh ([src/train.py:362](../src/train.py#L362)), nên mọi `best/` từng
tồn tại đều trùng byte với một `step-<N>` đã lên.

`--log run.log` đẩy kèm log mỗi vòng quét, ghi đè bản cũ. Lịch sử ValCER chỉ nằm trong
`run.log` và `trainer_state.json`, cả hai đều chết cùng máy — thiếu nó thì checkpoint cứu
được về sau không có đường cong eval nào để đọc cùng.

Máy chết ở bước 3.000 thì cái mất là 3.000 bước GPU sau đó, không phải toàn bộ lượt: các
điểm `step-400`/`800`/`1.600` mà §6 đặt cược đã nằm trên Hub. Tải về bằng
`huggingface_hub.snapshot_download(repo_id, allow_patterns='step-800/*')`.

`lora.rank` giữ mặc định 16 của config (`alpha` 32) — đó là hình dạng của v5, và là lý do
lượt này so được với v5. Không thêm override rank.

`training.val_limit=null`, không phải 100. `ManifestDataset.limit()` lấy `records[:n]`
không xáo ([src/data.py:151](../src/data.py#L151)), nên để 100 thì ValCER mỗi vòng chỉ
được chấm trên 92 segment của `paid_meeting_0001` cộng 8 segment của `0002` — một cuộc
họp, toàn synthetic, và **không segment YouTube nào**, tức 3 cuộc họp thêm ở 3.3 không
được chấm lần nào. Bỏ cap thì mỗi vòng chấm đủ 1.179 segment, khoảng 70 giây ở batch 64,
13 vòng là ~15 phút — đã tính trong 80 phút ở tiêu đề mục.

**ValCER lượt này không so được với lượt trước.** `Outputs/v6-corpus-r32.run.log` và mọi
run cũ đo trên 4 cuộc họp; từ đây là 7. So hai con số là so hai bộ đo khác nhau.

Theo dõi, đừng poll liên tục:

```bash
tr "\r" "\n" < run.log | grep -E "^progress |ValCER|^ *[0-9]+\.[0-9] " | tail -12
ls outputs/$R/checkpoints/                 # sau vòng eval đầu phải có step-400
```

Mốc nhận ra bất thường: **0,728 giây/bước** trên H200. Quá 2 giây/bước là có gì sai —
`gradient_checkpointing` còn bật, job khác chiếm GPU, hoặc dataloader nghẽn.

`ceil(42794/16)` = **2.675 bước/epoch** ([src/train.py:239](../src/train.py#L239) dùng
`ceil`, không phải chia lấy nguyên), nên 2 epoch là **5.350 bước**: 13 vòng eval ở 400,
800, …, 5.200, cộng `step-5350` do `on_train_end` lưu. Trên đĩa sẽ có **14 thư mục
`step-*`** cộng `best/`.

**Run đứt, máy còn sống:** `python -m src.pipeline --stage train --resume $OV`. **Đừng xoá
`checkpoints/`** — chạy lại không có `--resume` khi đã có checkpoint là `FileExistsError` cố
ý, xoá là mất số giờ GPU đã trả.

**Máy chết hẳn:** `--resume` không dùng được. Nó cần `checkpoints/checkpoint-<N>` — optimizer
state của Trainer, `save_total_limit=2` ([src/train.py:189](../src/train.py#L189)), nằm trong
`outputs/` nên chết cùng máy, và 332 MB mỗi cái là quá nặng để watcher đẩy mỗi vòng. Trên
máy mới chỉ còn hai đường:

1. **Chạy lại từ bước 1.** Sạch, so sánh được, nhưng trả lại toàn bộ số giờ đã mất.
2. **Khởi động ấm từ adapter đã lên Hub:** tải `step-<N>` cao nhất về rồi
   `--override training.init_adapter=<đường dẫn>`. **Đây không phải resume** — mất optimizer
   state và lịch LR, warmup chạy lại từ đầu, nên lượt mới là một đường cong khác chứ không
   phải phần đuôi của đường cũ. Dùng để cứu lấy một adapter dùng được, không dùng để lấy số
   cho bảng so.

Dù đường nào thì `step-400`/`800`/`1600` đã nằm trên Hub — §6 vẫn chấm được, và §6 mới là
deliverable chính.

## 6. Sàng cross-domain — 4 checkpoint (~10 phút)

Đây là deliverable chính. Chạy được bước này là lượt thuê có kết quả, dù không checkpoint
nào thắng v5.

`checkpoints/best/` là "tốt nhất trên ValCER", mà ValCER không nhìn thấy retention. Bản đem
đi benchmark ở §7 và đẩy lên Hub ở §8 được chọn bằng retention đo ở đây, không phải bằng
`best/`. Đừng đọc `best/` như bản thắng.

```bash
for S in 400 800 1600 5350; do
  PYTHONPATH=. .venv/bin/python -m scripts.score_cross_domain_model \
      --model vinai/PhoWhisper-large \
      --adapter outputs/v6-100h-steps/checkpoints/step-$S \
      --cross-domain-path dataset/cross-domain-bench \
      --batch-size 64 \
      --out outputs/v6-100h-steps/metrics/cross_domain.step-$S.json || exit 1
done

grep -h retention outputs/v6-100h-steps/metrics/cross_domain.step-*.json
```

Mốc so, v5 dưới đúng đường decode này: **0,0740 CER / 0,6648 retention / 0,0613
no-loanword CER**.

Nếu bốn điểm lộ ra một đỉnh, chấm thêm hàng xóm của nó (2,5 phút mỗi điểm, các checkpoint
kia vẫn còn trên đĩa). Nếu đồng hồ chạy chậm, bỏ `step-1600` trước — **không bao giờ cắt cả
bước 6**. Và chấm theo thứ tự trên: `step-400` rồi `step-800` trước, đừng đợi đủ bốn điểm
mới đọc kết quả, vì đó là hai điểm plan đặt cược.

Ghi lại tên thư mục thắng — §8 cần nó làm `--best-retention`.

## 7. Benchmark đầy đủ — chỉ bản thắng (~30 phút)

```bash
WIN=step-800                      # thay bằng bản retention cao nhất ở §6
PYTHONPATH=. .venv/bin/python -m scripts.benchmark_run --backend hf     --model vinai/PhoWhisper-large     --adapter outputs/v6-100h-steps/checkpoints/$WIN     --suites cross-domain,youtube-test,synthetic-test,vimedcss-test,vimedcss-hard     --path cross-domain=dataset/cross-domain-bench     --path youtube-test=dataset/v6-corpus     --path synthetic-test=dataset/v6-corpus     --batch-size 8     --out Outputs/bench-v6-100h-steps
```

`--batch-size 8` ở đây, **không phải 64** — cột v5 đã đo ở batch 8, CER lệch theo batch, đổi
batch là làm hỏng khả năng so cột.

Bỏ suite `vivos`: ref viết HOA toàn bộ còn hyp viết thường, 81% CER là hiện vật của quy ước
chữ hoa, không phải phép đo.

Muốn thêm cột λ=0,75 thì bake λ vào một bản sao rồi trỏ `--adapter` vào đó (λ scale
`lora_alpha`, đúng cơ chế `src.lora.save_with_lambda` dùng):

```bash
cp -r outputs/v6-100h-steps/checkpoints/$WIN /tmp/lam075
.venv/bin/python -c "
import json; p='/tmp/lam075/adapter_config.json'
c=json.load(open(p)); c['lora_alpha']=c['lora_alpha']*0.75; json.dump(c,open(p,'w'),indent=2)
print(c['lora_alpha'])"
```

Nếu thiếu thời gian: bỏ `vimedcss-test` (1.612 segment, −15 phút), giữ `vimedcss-hard`.

## 8. Cất kết quả trước khi trả máy

Đã mất một lượt đo vì chép muộn. Làm ngay khi §6 xong, đừng đợi §7.

Watcher ở §5 đã đẩy sẵn mọi `step-*` lên repo `-checkpoints` trong lúc train; mục này
tách riêng **ba bản đáng giữ** sang ba repo có tên đọc được, để sau này không ai phải tra
lại xem `step-1600` là cái gì. Trước khi chạy, quét nốt phần watcher chưa kịp (bước cuối
vừa sinh ra và chưa qua vòng quét nào):

```bash
PYTHONPATH=. .venv/bin/python -m scripts.push_checkpoints_live \
    --run-dir outputs/v6-100h-steps \
    --repo-id rework-whisper-v6-org/v6-100h-steps-checkpoints --once
```

Máy Windows hết chỗ, nên **15 adapter không chép về**. Chạy **trên máy thuê**:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.push_run_adapters \
    --run-dir outputs/v6-100h-steps \
    --repo-prefix rework-whisper-v6-org/v6-100h-steps \
    --best-retention step-1600            # thay bằng bản thắng ở §6
```

Ba repo private: `-best-valcer` (`checkpoints/best/`), `-best-retention` (bản §6 chọn),
`-last` (`step-5350`, script tự tìm số lớn nhất). `eval_val_cer` không nhìn thấy retention,
nên hai cái đầu là hai checkpoint khác nhau. `-last` cách vòng eval cuối 150 bước: nó là
bước train thật sự dừng, và trên một lượt đứt giữa chừng thì đó là thứ duy nhất có. Thêm `--dry-run` để kiểm ba đường
dẫn trước khi đẩy 333 MB.

Adapter đẩy lên **chưa nhân λ**. λ là lựa chọn lúc triển khai, bake sau bằng
`src.lora.save_with_lambda`; `training.init_adapter` từ chối adapter đã bake.

Còn lại đủ nhẹ để scp, chạy **ở máy Windows**:

```bash
scp speech-agent-gpu:'~/speech/Fine_tune_wf/run.log' 'Outputs/v6-100h-steps.run.log'
scp -r speech-agent-gpu:'~/speech/Fine_tune_wf/outputs/v6-100h-steps/metrics' \
    'Outputs/v6-100h-steps-metrics'
scp -r speech-agent-gpu:'~/speech/Fine_tune_wf/Outputs/bench-v6-100h-steps' Outputs/
```

`metrics/` là `training.csv`, `timing.json` và các `cross_domain.step-*.json` của §6 —
vài trăm KB, và là thứ duy nhất dựng lại được đường cong retention-theo-step. Nếu vẫn muốn
`outputs/v6-100h-steps.zip` thì nhớ nó ~1,85 GB vì ôm cả 16 `checkpoints/step-*` (bộ lọc
của `archive_run` chỉ bỏ `checkpoint-<số>` của Trainer).

Thu hồi `HF_TOKEN` đã đưa lên máy — nó có quyền ghi vào org nếu vừa dùng để push.

## 9. Chấm bảng và ghi sổ (máy Windows, 0 phút GPU)

Không cần decode lại các hệ khác — `Outputs/benchmark-2026-09-10 (3)/` đã có per-segment của
`winhsss_reworkwhisper_large_v5`, `scribe_v2`, `vinai_phowhisper_large` trên đủ 6 suite.
`scripts/benchmark_report.py` nối theo `segment_id`; gộp thư mục rồi chạy:

```bash
python -m scripts.benchmark_report <thư mục gộp>
```

Rồi ghi `experiments/task_ledger.md` và `provenance.md` — đó là toàn bộ hệ thống lưu vết của
project này.

Cách đọc bốn khả năng của đường cong retention: plan §6. Không có cổng tự động — người đọc
số rồi quyết định push.
