# v6-100h-steps — hướng dẫn chạy tay

Bảng lệnh theo thứ tự cho lượt thuê máy. Quyết định và lý do nằm ở
[docs/v6-100h-steps-plan.md](v6-100h-steps-plan.md); quy trình chung nằm ở
[docs/server-finetune.md](server-finetune.md). File này chỉ là thứ tự gõ lệnh.

Đặc điểm lượt này khác quy trình mặc định: **không chạy `--stage baseline`**, không chạy
`sweep-gate`, không cần `dataset/vivos`, không chạy `scripts/select_val_meetings.py`.

**Lô TTS +5h (commit `aa51276`, 15/09) không đụng tới `dataset/v6-corpus`.** Nó gộp vào
`dataset/v6-ondomain-15h`, ra `dataset/v6-ondomain-15h-tts5h` — thư mục khác, local-only,
gitignored, `scripts/merge_ondomain15h_tts_batch.py` không gọi HF upload nào nên chắc chắn
**chưa push** lên đâu cả. `dataset/v6-corpus` (mtime 12/09, không đổi) là corpus HF private
`rework-whisper-v6-org/v6-corpus` mà lượt này dùng — số liệu 34.972/365/654 ở plan §2 vẫn
đúng. Đừng lẫn hai tên `v6-ondomain-15h-tts5h` và `v6-corpus`.

---

## 0. Trên máy Windows, trước khi thuê máy (bắt buộc)

Máy thuê lấy mã nguồn bằng `git clone`, nên mọi commit chưa push đều không tồn tại với nó.
Hiện có **18 commit chưa push**, trong đó có đúng hai commit lượt này phụ thuộc
(`2058e24` lưu checkpoint mỗi vòng eval + `--adapter` cho benchmark, `ac5183c` eval_steps=400):

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
cộng phần giải nén, cộng ~1,85 GB checkpoint `step-*`).

## 2. Mã nguồn và môi trường

```bash
ssh speech-agent-gpu
mkdir -p ~/speech && cd ~/speech
git clone -b main https://github.com/egoist-minh/Reworkwhisper-finetune.git Fine_tune_wf
cd Fine_tune_wf && git rev-parse HEAD      # phải trùng commit đã ghi ở §0

python3 -m venv .venv && source .venv/bin/activate
pip install -q --upgrade pip && pip install -r requirements.txt
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

### 3.1 Corpus (private, cần token)

```bash
export HF_TOKEN=<token org rework-whisper-v6-org — cần quyền GHI, §8 đẩy adapter lên>
mkdir -p dataset
.venv/bin/python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('rework-whisper-v6-org/v6-corpus','v6-corpus.tar',
                      repo_type='dataset', local_dir='.'))"
tar -xf v6-corpus.tar -C dataset
```

Dùng `huggingface_hub`, **không** dùng `curl` — `curl` không xác thực bị bóp xuống
~120 kB/s so với 51 MB/s. Đây là khoản lâu nhất của cả lượt (30–45 phút).

Xác nhận split đúng như plan §2 đã đo (train 34.972 / val 365 / test 654):

```bash
PYTHONPATH=. .venv/bin/python -c "
from src.config import load_config
from src.data import load_manifests, resolve_splits
cfg = load_config('configs/experiment.yaml')
cfg.data.dataset_path = 'dataset/v6-corpus'
tr, va, te = resolve_splits(load_manifests(cfg.data.dataset_path), cfg.data.val_meetings)
print(len(tr), len(va), len(te))"
```

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

Phải ra train 34.158 / val 1.179 / test 654.

| | trước | sau |
|---|---|---|
| train | 34.972 seg, 101,23 h, density 2,48%, 6.555 step | 34.158 seg, 97,62 h, density **2,34%**, **6.402 step** |
| val | 4 mtg, 365 seg, 0,79 h, density 8,89%, 253 seg có loanword | 7 mtg, **1.179 seg**, 4,40 h, density **7,12%**, **791** seg có loanword |

Val to gấp 3,2 lần mà density gần như đứng yên: 8,89% xuống 7,12%, và **test là 7,06%** —
val giờ đo trên cùng nồng độ code-switch với bộ sẽ chấm ở §7. Số segment có loanword tăng
253 lên 791, đủ để ValCER thôi trồi sụt vài điểm giữa các vòng.

Density train tụt 2,48% xuống 2,34% vì ba cuộc họp bị rút đều đậm hơn trung bình — mất
0,14 điểm trên 97,6 h, không đáng kể so với cái được ở val. Step rời 6.555 xuống 6.402.

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

## 5. Lượt 100 h (~96 phút)

```bash
export PYTHONPATH=. TRANSFORMERS_AUTO_CONVERSION=0
R=v6-100h-steps
OV="--override run_id=$R
    --override data.dataset_path=dataset/v6-corpus
    --override data.real_bench_path=null
    --override data.ood_eval_path=null
    --override data.val_meetings=[paid_meeting_0001,paid_meeting_0002,paid_meeting_0011,rCd8DSMk3-c,zlKBfNzfh50,coteccons-agm-2025,jB1P4bqLwDY]
    --override data.cross_domain_path=dataset/cross-domain-bench
    --override training.epochs=3
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
      --repo-id rework-whisper-v6-org/$R-checkpoints
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
16 vòng là ~19 phút — đã tính trong 96 phút ở tiêu đề mục.

**ValCER lượt này không so được với lượt trước.** `Outputs/v6-corpus-r32.run.log` và mọi
run cũ đo trên 4 cuộc họp; từ đây là 7. So hai con số là so hai bộ đo khác nhau.

Theo dõi, đừng poll liên tục:

```bash
tr "\r" "\n" < run.log | grep -E "^progress |ValCER|^ *[0-9]+\.[0-9] " | tail -12
ls outputs/$R/checkpoints/                 # sau vòng eval đầu phải có step-400
```

Mốc nhận ra bất thường: **0,728 giây/bước** trên H200. Quá 2 giây/bước là có gì sai —
`gradient_checkpointing` còn bật, job khác chiếm GPU, hoặc dataloader nghẽn.

Tổng **6.402 bước** (sau khi 3.3 rút 814 segment khỏi train), 16 vòng eval ở 400, 800,
…, 6.400. Cộng `on_train_end`, trên đĩa sẽ có 17 thư mục `step-*`: `step-400` … `step-6400`
và **`step-6402`** — bước cuối, không trùng vòng eval nào.

**Run đứt:** `python -m src.pipeline --stage train --resume $OV`. **Đừng xoá
`checkpoints/`** — chạy lại không có `--resume` khi đã có checkpoint là `FileExistsError` cố
ý, xoá là mất số giờ GPU đã trả.

## 6. Sàng cross-domain — 4 checkpoint (~10 phút)

Đây là deliverable chính. Chạy được bước này là lượt thuê có kết quả, dù không checkpoint
nào thắng v5.

`checkpoints/best/` là "tốt nhất trên ValCER", mà ValCER không nhìn thấy retention. Bản đem
đi benchmark ở §7 và đẩy lên Hub ở §8 được chọn bằng retention đo ở đây, không phải bằng
`best/`. Đừng đọc `best/` như bản thắng.

```bash
for S in 400 800 1600 6402; do
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

Nếu bốn điểm lộ ra một đỉnh, chấm thêm hàng xóm của nó (2,5 phút mỗi điểm, 13 checkpoint
kia vẫn còn trên đĩa). Nếu đồng hồ chạy chậm, bỏ `step-1600` trước — **không bao giờ cắt cả
bước 6**.

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

Watcher ở §5 đã đẩy sẵn cả 17 `step-*` lên repo `-checkpoints` trong lúc train; mục này
tách riêng **ba bản đáng giữ** sang ba repo có tên đọc được, để sau này không ai phải tra
lại xem `step-1600` là cái gì. Trước khi chạy, quét nốt phần watcher chưa kịp (`step-6402`
vừa sinh ra ở cuối lượt):

```bash
PYTHONPATH=. .venv/bin/python -m scripts.push_checkpoints_live \
    --run-dir outputs/v6-100h-steps \
    --repo-id rework-whisper-v6-org/v6-100h-steps-checkpoints --once
```

Máy Windows hết chỗ, nên **18 adapter không chép về**. Chạy **trên máy thuê**:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.push_run_adapters \
    --run-dir outputs/v6-100h-steps \
    --repo-prefix rework-whisper-v6-org/v6-100h-steps \
    --best-retention step-1600            # thay bằng bản thắng ở §6
```

Ba repo private: `-best-valcer` (`checkpoints/best/`), `-best-retention` (bản §6 chọn),
`-last` (`step-6402`, tự tìm số lớn nhất). `eval_val_cer` không nhìn thấy retention, nên
hai cái đầu là hai checkpoint khác nhau. `-last` lần này chỉ cách vòng eval cuối 2 bước
(6.402 so với 6.400) — gần như trùng, nhưng vẫn đẩy: nó là bước train thật sự dừng, và
trên một lượt đứt giữa chừng thì đó là thứ duy nhất có. Thêm `--dry-run` để kiểm ba đường
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
