# Chạy fine-tune trên server GPU thuê

Quy trình đầy đủ từ một máy trắng vừa thuê đến adapter đã train, cộng những chỗ đã bị vấp
một lần. Khi người dùng nói "tiến hành fine tune", đây là tài liệu cần theo.

Số liệu thời gian và các quyết định cấu hình lấy từ lần đo trên H200 ngày 11/09/2026 —
xem `docs/h200-timing-probe-2026-09-11.md`.

## 0. Trước khi gõ lệnh gì

**IP đổi mỗi lần thuê.** Host `speech-agent-gpu` trong `~/.ssh/config` trên máy Windows
trỏ tới lần thuê *trước*, gần như chắc chắn đã chết. Hỏi người dùng IP mới, rồi sửa:

```bash
sed -i 's/HostName .*/HostName <IP mới>/' ~/.ssh/config
```

Cấu hình host giữ nguyên phần còn lại (`User ubuntu`, `Port 22`,
`IdentityFile C:/Users/ADMIN/.ssh/id_ed25519`). Nếu file không còn host này thì dựng lại
theo đúng mẫu đó.

**Khảo sát máy trước, đừng giả định.** Driver và số vCPU quyết định bước cài torch ở §2:

```bash
ssh -o StrictHostKeyChecking=accept-new speech-agent-gpu \
  'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
   nproc; free -g | head -2; df -h /; python3 -V; sudo -n true && echo SUDO_YES || echo SUDO_NO'
```

**Máy có thể dùng chung với người khác.** Kiểm `nvidia-smi --query-compute-apps=pid,used_memory
--format=csv,noheader` và `ls ~` trước khi chiếm GPU. Không đụng thư mục của người khác.
Nếu có job khác đang chạy, mọi số đo thời gian đều vô nghĩa — hẹn giờ trống trước.

## 1. Lấy mã nguồn

```bash
ssh speech-agent-gpu 'mkdir -p ~/speech && cd ~/speech &&
  git clone -b <branch> https://github.com/egoist-minh/Reworkwhisper-finetune.git Fine_tune_wf &&
  cd Fine_tune_wf && git rev-parse HEAD'
```

Repo là public nên không cần token. Đối chiếu commit vừa in với `git rev-parse HEAD` ở máy
local — phải trùng, nếu không thì chưa push.

## 2. Môi trường

```bash
cd ~/speech/Fine_tune_wf
python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -r requirements.txt
```

**Rồi gần như chắc chắn phải cài lại torch.** `pip install torch` lấy bản dựng cho CUDA mới
nhất; nếu driver trên máy cũ hơn thì `torch.cuda.is_available()` trả `False` với thông báo
*"The NVIDIA driver on your system is too old"*. Chọn index-url khớp driver đọc được ở §0
(driver 560.x = CUDA 12.6):

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
```

Không nâng driver trên máy dùng chung. Xác nhận trước khi đi tiếp:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
PYTHONPATH=. .venv/bin/python -m pytest -q
```

Test suite phải **0 failed**. Số test bị skip sẽ nhiều hơn ở máy local (khoảng 22 so với 1)
vì máy thuê không có `dataset/youtube-meetings/`, artifact các run cũ, hay repo tiền nhiệm —
đó là skip đúng, không phải lỗi.

## 3. Dữ liệu

Corpus nằm trên HuggingFace, public: `winhsss/mixed-noisy-v1`.

```bash
mkdir -p dataset
.venv/bin/python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('winhsss/mixed-noisy-v1','mixed-noisy-v1.tar',repo_type='dataset',local_dir='.'))"
tar -xf mixed-noisy-v1.tar -C dataset
PYTHONPATH=. .venv/bin/python scripts/checksum_dataset.py --root dataset/mixed-noisy-v1 --mode verify
```

Phải thấy `OK -- dataset matches CHECKSUMS.txt`.

**Dùng `huggingface_hub`, không dùng `curl`.** `curl` không xác thực bị bóp xuống ~120 kB/s;
`hf_hub_download` (có `hf_xet`) đạt 51 MB/s trên cùng file. Chênh hơn 400 lần.

**Bộ OOD phải dựng riêng** — nó không nằm trong tar, và thiếu nó thì `--stage baseline` dừng
ngay với `FileNotFoundError: no manifest.*.jsonl under dataset/vivos`. Tier 2 của gate đo
trên chính bộ này:

```bash
PYTHONPATH=. .venv/bin/python scripts/fetch_vivos.py --out dataset/vivos   # 760 segment, ~45 s
```

## 4. Chạy

Tham số đã chốt cho v6. Đặt thành biến rồi dùng cho cả hai stage — baseline và train **phải**
cùng một bộ, đặc biệt là `eval.batch_size`:

```bash
export PYTHONPATH=. TRANSFORMERS_AUTO_CONVERSION=0
R=v6-<mô tả>
OV="--override run_id=$R
    --override data.dataset_path=dataset/mixed-noisy-v1
    --override data.real_bench_path=null
    --override lora.rank=32 --override lora.alpha=64
    --override training.batch_size=16 --override training.grad_accum_steps=1
    --override training.gradient_checkpointing=false
    --override training.epochs=1
    --override eval.batch_size=64"
```

Giải thích từng lựa chọn:

| Override | Lý do |
|---|---|
| `gradient_checkpointing=false` | H200 đo được 74,9 GB đỉnh trên 143 GB. Bật nó là trả 30–40% thời gian tính toán để tiết kiệm bộ nhớ đang dư một nửa. OOM thì bật lại và đo lại. |
| `batch_size=16`, `grad_accum_steps=1` | Thay cho 2 × 8 của v5; batch hiệu dụng giữ nguyên 16 nên `learning_rate` vẫn là `2.0e-4`. |
| `epochs=1` | Quyết định cho corpus lớn. |
| `eval.batch_size=64` | Giá trị 8 trong config chọn cho T4. Trên H200, 64 nhanh hơn 3,5 lần; 128 chỉ thêm 1,3 lần trong khi bộ nhớ gấp đôi. |
| `real_bench_path=null` | Tier 4a đã bỏ khỏi phạm vi. **Phải là `null` hoặc `~`, không được là `None`** — `apply_override` đẩy giá trị qua `yaml.safe_load`, và YAML đọc `None` thành chuỗi `'None'`, truthy, nên pipeline đi tìm thư mục tên `None`. |
| `lora.rank=32` | Đề xuất, chưa có bằng chứng so với rank 16. Nếu chưa có ai chạy so sánh thì nói rõ điều này. |

Chạy trong `tmux`, không chạy tiền cảnh — SSH đứt là mất run:

```bash
tmux new-session -d -s run "bash -lc '
  cd ~/speech/Fine_tune_wf && source .venv/bin/activate
  export PYTHONPATH=. TRANSFORMERS_AUTO_CONVERSION=0
  python -m src.pipeline --stage baseline $OV
  python -m src.pipeline --stage train    $OV
' > run.log 2>&1; echo EXIT=\$? >> run.log"
```

Theo dõi bằng cách chờ session kết thúc rồi đọc log, đừng poll liên tục:

```bash
ssh speech-agent-gpu 'while tmux has-session -t run 2>/dev/null; do sleep 30; done;
  cd ~/speech/Fine_tune_wf && grep -o "EXIT=[0-9]*" run.log | tail -1'
```

`run.log` rất nặng vì progress bar. Hai dòng đáng đọc, cả hai đều là text thường nên
`tail` được thẳng: `progress step N/M ... eta=...m` (khoảng 50 dòng mỗi run) và bảng
`Epoch Step TrainLoss ValLoss ValCER ValWER OOD_CER` (một hàng mỗi lượt eval).

```bash
ssh speech-agent-gpu 'cd ~/speech/Fine_tune_wf && tr "\r" "\n" < run.log |
  grep -E "^progress |ValCER|^ *[0-9]+\.[0-9] " | tail -12'
```

**Cổng kiểm sau train:** `checkpoints/best/` phải xuất hiện *giữa chừng*, không phải chỉ ở
cuối, cùng với `checkpoints/checkpoint-<step>` (giữ 2 bản mới nhất). Nếu sau lượt eval đầu
tiên vẫn chưa thấy thì `training.eval_steps` chưa được truyền, hoặc `eval_val_cer` không có
trong `metrics` và `RobustEvalTrackingCallback` đang bỏ qua mọi lượt.

### Run đứt giữa chừng

`--stage train` giờ ghi `checkpoints/checkpoint-<step>` mỗi `training.eval_steps` step (giữ
2 bản mới nhất) và chạy lại được từ đó:

```bash
python -m src.pipeline --stage train --resume $OV
```

Resume phục hồi optimizer state, LR schedule, thứ tự dữ liệu, và lịch sử eval đứng sau
early stopping — nên lượt eval đầu sau khi chạy lại không bị tính là "cải thiện" và không
ghi đè `checkpoints/best/` bằng adapter tệ hơn. Phần replay thứ tự dữ liệu tốn thêm một
lượt đọc qua số batch đã tiêu, đo được nhanh gấp 83,9 lần tốc độ GPU tiêu thụ, tức vài giây.

Chạy lại `--stage train` mà **không** có `--resume` trong khi đã có checkpoint là lỗi, không
phải ghi đè: thông báo nói rõ hai lựa chọn. Đây là chặn cố ý — xoá một run đã trả tiền GPU
vì gõ lại lệnh cũ là mất tiền thật.

**Checkpoint nằm trên đĩa máy thuê.** Máy tắt là mất, và resume không cứu được gì. Muốn
chống cả trường hợp mất máy thì phải chép ra ngoài trong lúc chạy — mỗi checkpoint khoảng
700 MB (adapter cộng optimizer state ở rank 32), tức khoảng 15 giây ở tốc độ đo được:

```bash
# chạy ở máy local, song song với run
while true; do
  rsync -az --delete speech-agent-gpu:'~/speech/Fine_tune_wf/outputs/'"$R"'/' "Outputs/$R/"
  sleep 600
done
```

Chạy lại trên **máy mới** thì phải đẩy toàn bộ `outputs/$R/` lên trước, không chỉ thư mục
`checkpoints/`: `--stage train` cần `validated_manifest.jsonl` của `--stage baseline`, và
`--stage sweep-gate` cần `metrics/baseline.json`.

## 5. Lấy kết quả về trước khi trả máy

Mỗi lần chạy stage, pipeline tự đóng gói `outputs/<run_id>/` thành **một file zip** ngay
cạnh nó — kể cả khi gate trượt, kể cả khi stage ném lỗi giữa chừng. Dòng cuối trong log nói
rõ tên file và dung lượng:

```
archive: outputs/v6-x.zip (214 files, 243.7 MB) -- scp this off the box
```

Trong zip có toàn bộ `metrics/`, `audit/`, `config.json`, `validated_manifest.jsonl`,
`checkpoints/best/` và `adapter/`. **Không** có `checkpoints/checkpoint-<step>`: đó là
optimizer state để `--resume` trên chính máy đó, vài trăm MB mỗi bản và vô dụng khi máy đã
trả.

Máy thuê tắt là mất sạch, và đã xảy ra một lần: phiên 11/09/2026 mất `probe.log` cùng toàn
bộ `timing.*.json` vì chép về muộn. Chép ngay khi stage xong, đừng đợi hết phiên:

```bash
scp speech-agent-gpu:'~/speech/Fine_tune_wf/outputs/<run_id>.zip' Outputs/
scp speech-agent-gpu:'~/speech/Fine_tune_wf/run.log' Outputs/
```

Muốn giữ được cả khả năng resume trên máy khác thì phải chép thêm `checkpoints/` bằng
`rsync` như ở §4 — zip cố tình không mang phần đó.

Rồi ghi vào `experiments/task_ledger.md` và `provenance.md` — đó là toàn bộ hệ thống lưu vết
của project này, không có DVC hay W&B.

Nếu có lúc nào đưa token HF lên máy: thu hồi token khi trả máy.

## 6. Thời gian dự kiến

Corpus hiện tại (`mixed-noisy-v1`, 9,35 giờ, 4 717 segment train): 295 step, dưới 5 phút
train. Corpus 100 giờ: khoảng 1,4 giờ nếu mọi thứ thuận lợi, 3,6 giờ theo kịch bản bi quan —
**thuê 6 giờ**. Chi tiết từng khoản trong `docs/h200-timing-probe-2026-09-11.md`.

Mốc để nhận ra bất thường: **0,728 giây mỗi step** trên H200 ở rank 32, batch 16, không
checkpointing. Quá 2 giây mỗi step là có gì sai — checkpointing còn bật, hoặc job khác đang
chiếm GPU, hoặc dataloader bị nghẽn (xem `steady_state.dataloader_gap_seconds` trong
`outputs/<run>/metrics/timing.json`).

## 7. Muốn đo lại chi phí trước khi cam kết một corpus mới

`scripts/probe_timing.sh` chạy trọn bộ phép đo trong dưới 20 phút GPU:

```bash
PY=.venv/bin/python RUN_ID=v6-probe DATASET=dataset/mixed-noisy-v1 \
HF_REPO=winhsss/mixed-noisy-v1 HF_PROBE_FILE=mixed-noisy-v1.tar CORPUS_GB=<GB> \
PYTHONPATH=. bash scripts/probe_timing.sh 2>&1 | tee probe.log
```

Phương pháp và cách đọc kết quả: `docs/runbook.md` §B5b.
