# Kế hoạch v6 pha 2 — thang mật độ, mục tiêu vượt Scribe v2 trên cross-domain

Thay thế `docs/v6-curriculum-plan.md` §4–§7. Phần §0–§3 của file đó (chẩn đoán v6, bốn
thay đổi mã nguồn, dữ liệu cần có trên máy thuê) vẫn đúng và không lặp lại ở đây.

Khác biệt với bản trước: mục tiêu không còn là ngang v5 mà là **vượt Scribe v2**, và pha 2
không còn là một lần chạy duy nhất mà là **bốn lần chạy** trên bốn mức lọc mật độ, để đo
đường cong thay vì đoán một điểm.

Mọi số trong file này đo lại từ `Outputs/v6-corpus-r32.zip` (manifest và
`checkpoints/best/adapter_config.json`) và `Outputs/bench-cross-v6/benchmark-table.N3.md`.

---

## 0. Tên gọi — đây vẫn là v6, không phải một thế hệ mới

Toàn bộ kế hoạch này chạy tiếp từ adapter của `v6-corpus-r32`, trên tập con của chính
corpus đó, cùng base model `vinai/PhoWhisper-large`, cùng rank 32 / alpha 64. Không có
corpus mới, không có base model mới, không có siêu tham số LoRA mới. Gọi nó là "v7" sẽ hứa
một thế hệ mà thực tế không có.

| Thứ | Tên |
|---|---|
| run pha 1 (đã chạy xong) | `v6-corpus-r32` |
| bốn nấc thang, giai đoạn A | `v6-dense-1` … `v6-dense-4` |
| nấc thắng chạy chuẩn bảng, giai đoạn B | `v6-dense-<N>-bench` |
| corpus lọc | `dataset/v6-dense-<N>` |

⚠ Tên repo trên HuggingFace là một quyết định riêng, **chốt lúc đẩy chứ không chốt ở đây**.
Tên repo của project này vốn lệch một nhịp so với `run_id` — `Reworkwhisper-large-v5` là
run `v4-mixed-r16`, `Reworkwhisper-large-v4` là run `v3-r16`. Các lệnh dưới đây để tạm
`Reworkwhisper-large-v6-dense`; đổi trước khi chạy nếu muốn theo nhịp lệch cũ.

---

## 1. Đích cần đạt, quy về một con số

Bảng cross-domain hiện tại (299 segment, 1,30 h, 1.283 lần xuất hiện từ ngoại lai):

| Hệ | CER | 95% CI | Retention |
|---|---:|---|---:|
| `scribe_v2` | **6,25%** | [5,72 – 6,91] | **83,63%** |
| `winhsss_reworkwhisper_large_v5` | 8,19% | [7,03 – 10,17] | 66,64% |
| `rework_whisper_v6` | 10,85% | [9,67 – 12,45] | 48,56% |
| `vinai_phowhisper_large` | 13,32% | [12,35 – 14,21] | 27,83% |

Hồi quy 4 điểm (retention, CER) trên chính bảng này: `CER ≈ 16,945 − 0,1287 × retention`.
Đặt CER = 6,24% vào, ra **retention ≈ 83,2%**.

⚠ Hồi quy chỉ để ước lượng mục tiêu trung gian. Điều kiện nghiệm thu là **CER đo thật**,
không phải giá trị hồi quy dự đoán.

⚠ Khoảng tin cậy của Scribe kéo xuống 5,72%. Một điểm ước lượng 6,24% là **thắng trên giấy,
chưa phải thắng có ý nghĩa thống kê**. Muốn nói "vượt Scribe" trước hội đồng thì phải chạy
paired bootstrap delta CER giữa adapter pha 2 và Scribe trên cùng 299 segment và cho thấy khoảng tin
cậy không chứa 0 — đúng cách `Outputs/bench-cross-v6` đã làm với PhoWhisper. Ghi rõ trong
báo cáo là "thắng điểm ước lượng" hay "thắng có ý nghĩa", đừng nhập nhằng hai thứ.

### 1.1 Retention đó phải lấy từ đâu

1.283 lần xuất hiện chia hai nhóm: **1.090 lần đã có trong nhãn train của v6** (85,0%) và
**193 lần chưa từng xuất hiện** trong nhãn train nào. Quy mọi kịch bản về cùng đích 1.067
lần giữ được:

| Nhóm 193 từ chưa gặp | Nhóm 1.090 từ đã có trong train phải đạt |
|---|---:|
| 38,3% — mức v5 đang có | **91,2%** |
| 50% | 89,1% |
| 61,7% — mức Scribe | 87,0% |

Hiện tại: v5 giữ 71,7% nhóm đã-có-trong-train, v6 giữ 52,8%.

Kết luận quan trọng: **gánh nặng nằm ở nhóm đã có trong train**, vì nó chiếm 85% số lần xuất
hiện. Ngay cả khi nhóm từ lạ leo lên bằng Scribe, nhóm kia vẫn phải đạt 87%. Đây là nhóm mà
dữ liệu train **có** tác động trực tiếp — nên thang mật độ là đòn bẩy đúng, không phải đòn
bẩy duy nhất nhưng là đòn bẩy rẻ nhất và chưa ai kéo.

### 1.2 Ràng buộc thứ hai, dễ quên

15,7% ký tự của bộ cross-domain nằm ở các segment **không** chứa từ ngoại lai (n=48). Muốn
CER gộp về 6,24% trong khi lát đó giữ ở 5,86% (mức v6, tốt nhất repo có), thì lát **có** từ
ngoại lai phải xuống **6,31%** — v5 đang 8,58%, v6 đang 11,78%. Nếu lát sạch trượt lên 7%
thì đích gộp không còn với tới được nữa, dù retention có tốt đến đâu.

Đây chính là rủi ro của các nấc thang mỏng. Gate phải đo nó, không được suy đoán.

---

## 2. Điểm xuất phát: checkpoint pha 1

Đã kiểm trên đĩa. `Outputs/v6-corpus-r32.zip` chứa `v6-corpus-r32/checkpoints/best/`
(220 MB trọng số + `adapter_config.json`):

```
r: 32, lora_alpha: 64, use_rslora: true, lora_dropout: 0.05
base_model_name_or_path: vinai/PhoWhisper-large
target_modules: q_proj k_proj v_proj out_proj fc1 fc2
task_type: SEQ_2_SEQ_LM
```

`lora_alpha` = 64 = 2 × rank, tức **chưa nhân λ** — dùng làm điểm xuất phát hợp lệ. Đối
chiếu: `v6-corpus-r32/adapter/adapter_config.json` có `lora_alpha: 32.0`, tức λ=0,5 đã bị
`save_with_lambda` nướng vào. Trỏ `training.init_adapter` vào `checkpoints/best`, **không**
vào `adapter`.

**Train lại từ đầu là sai lựa chọn**, ba lý do theo thứ tự:

1. 101 giờ của pha 1 đã mua được phần tiếng Việt — CER lát sạch 5,86%, tốt hơn v5 (6,06%) và
   PhoWhisper gốc (7,06%). Đó là con số tốt nhất repo có trên lát đó, và §1.2 cho thấy nó là
   ràng buộc không được mất.
2. Tập lọc không phải dữ liệu mới, nó là tập con của chính train v6. Train từ đầu trên 8–40
   giờ dữ liệu lệch thì không đủ để học lại tiếng Việt, chỉ đủ để học thói quen từ ngoại lai.
3. Chi phí: đi tiếp tốn 3–20 phút mỗi nấc; train lại pha 1 tốn ~35 phút mỗi nấc, nhân bốn.

⚠ Trong zip **không có** `checkpoints/checkpoint-<step>`, tức không có optimizer state. Nghĩa
là `--stage train --resume` không dùng được; chỉ có đường `training.init_adapter`. Đúng thiết
kế, nhưng đừng gõ nhầm lệnh rồi mất lượt.

⚠ Cả bốn nấc đều kế thừa **rank 32 / alpha 64** từ adapter pha 1; `cfg.lora` bị bỏ qua ở
nhánh nạp adapter và `_attach_adapter` in cảnh báo mỗi lần chạy. Thiết kế này không trả lời
được câu hỏi rank 16 hay rank 32.

---

## 3. Thang mật độ — bốn nấc

`scripts/filter_corpus_density.py --min-foreign N` giữ lại segment train nào có **ít nhất N
lần xuất hiện** từ ngoại lai trong nhãn (đếm lần, không đếm từ khác nhau: "chào team team"
là 2 lần). val 365 và test 654 đi qua nguyên vẹn ở mọi nấc, nên số của gate còn so được
giữa các nấc và với v5, v6.

Đo lại từ `validated_manifest.jsonl` của run v6:

| N | segment | giờ | mật độ | step/epoch @ bs16 | tổng lần xuất hiện từ ngoại lai |
|---:|---:|---:|---:|---:|---:|
| — (v6 nguyên) | 34.972 | 101,23 | 2,484% | 2.186 | 31.707 |
| 1 | 12.880 | 39,76 | 6,515% | 805 | 31.707 |
| 2 | 7.699 | 25,20 | 8,759% | 482 | 26.526 |
| 3 | 4.188 | 15,81 | 10,442% | 262 | 19.504 |
| 4 | 2.028 | 8,13 | 14,050% | 127 | 13.024 |

Hai điều đọc được từ bảng này:

- **N=1 không mất một lần xuất hiện nào** (31.707 y nguyên): nó chỉ bỏ các segment có số
  không. Mọi nấc cao hơn đổi độ đậm lấy số lần tiếp xúc — N=4 giữ 14% mật độ nhưng chỉ còn
  41% số lần tiếp xúc.
- Mốc cần tới nằm ở **giữa nấc 3 và nấc 4**. Ước lượng: hai điểm fine-tune hiện có cho độ
  dốc 4,07 điểm retention nhóm đã-có-trong-train trên mỗi điểm mật độ (52,8% ở 2,48% lên
  71,7% ở 7,13%). Từ 71,7% lên 87–91% cần mật độ **10,9–11,9%**.

⚠ Ngoại suy từ hai điểm, và hai điểm đó khác nhau cả về nội dung corpus, số epoch lẫn rank —
không chỉ mật độ. Đừng trích độ dốc này như một hằng số. Nó chỉ dùng để chọn phạm vi thang,
và chính thang là phép đo thay thế nó.

Vì mốc rơi vào giữa hai nấc, chạy cả bốn. Thang là phép đo, không phải lựa chọn ra quyết
định trước.

---

## 4. Chạy — hai giai đoạn, đừng gộp

Chi phí eval lớn hơn chi phí train một bậc, nên tách làm hai:

**Giai đoạn A — thang (thăm dò).** Câu hỏi là "mật độ nào", một câu hỏi **so sánh tương
đối** giữa bốn nấc. Không cần số chuẩn bảng benchmark. Do đó: `eval.batch_size=64` (nhanh
gấp 3,5 lần batch 8 trên H200), λ ghim cứng ở 0,5 để bốn nấc so được trực tiếp, hai lượt
eval giữa chừng.

**Giai đoạn B — ứng viên (chuẩn bảng).** Chỉ nấc thắng. `eval.batch_size=8` đúng như toàn
bộ cột v5 trong bảng benchmark, lưới λ đầy đủ 5 điểm, sàn retention, đủ ngưỡng gate, đẩy
lên HF. Chạy như một run riêng từ đầu (baseline + train + sweep-gate) chứ không chạy lại
stage trên thư mục cũ — `outputs/<run_id>/config.json` là bản đóng băng mà các stage đọc,
đổi batch giữa chừng là tự tạo một run nửa nọ nửa kia.

### 4.1 Chuẩn bị dữ liệu trên máy thuê

Theo `docs/server-finetune.md` §0–§2 cho IP, môi trường và torch. Rồi:

```bash
# corpus 14,7 GB -- cho chạy nền NGAY từ phút đầu, nó là đường găng
.venv/bin/python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('rework-whisper-v6-org/v6-corpus','v6-corpus.tar',
                      repo_type='dataset', local_dir='.'))"
tar -xf v6-corpus.tar -C dataset

PYTHONPATH=. .venv/bin/python scripts/fetch_vivos.py --out dataset/vivos
unzip cross-domain-bench.zip -d dataset/          # scp từ máy Windows trước
mkdir -p outputs/v6-corpus-r32 && unzip v6-corpus-r32.zip -d /tmp/ \
  && cp -r /tmp/v6-corpus-r32/checkpoints outputs/v6-corpus-r32/
```

Dựng bốn corpus lọc — CPU, vài phút, `audio/` là symlink nên không nhân đôi 14,7 GB:

```bash
for N in 1 2 3 4; do
  PYTHONPATH=. .venv/bin/python -m scripts.filter_corpus_density \
    --src dataset/v6-corpus --out dataset/v6-dense-$N --min-foreign $N
done
```

**Cổng kiểm:** bốn lần in ra phải khớp đúng bảng §3 (12.880 / 7.699 / 4.188 / 2.028 segment;
39,76 / 25,20 / 15,81 / 8,13 giờ; 6,52 / 8,76 / 10,44 / 14,05%), và val 365 / test 654 ở cả
bốn. Lệch là lọc sai hoặc corpus tải về khác corpus đã train v6 — dừng lại, đừng chạy tiếp.

### 4.2 Giai đoạn A — bốn nấc

```bash
export PYTHONPATH=. TRANSFORMERS_AUTO_CONVERSION=0

run_rung () {   # $1 = N, $2 = steps/epoch
  R=v6-dense-$1
  OV="--override run_id=$R
      --override data.dataset_path=dataset/v6-dense-$1
      --override data.real_bench_path=null
      --override training.init_adapter=outputs/v6-corpus-r32/checkpoints/best
      --override training.epochs=2
      --override training.learning_rate=1e-4
      --override training.batch_size=16 --override training.grad_accum_steps=1
      --override training.gradient_checkpointing=false
      --override training.eval_steps=$2
      --override eval.batch_size=64
      --override sweep.lambdas=[0.5]
      --override data.cross_domain_path=dataset/cross-domain-bench"
  python -m src.pipeline --stage baseline   $OV
  python -m src.pipeline --stage train      $OV
  python -m src.pipeline --stage sweep-gate $OV
}

run_rung 1 805
run_rung 2 482
run_rung 3 262
run_rung 4 127
```

Đặt đoạn trên vào một file `ladder.sh` rồi chạy trong `tmux` như
`docs/server-finetune.md` §4 — SSH đứt là mất run:

```bash
tmux new-session -d -s ladder "bash -lc 'cd ~/speech/Fine_tune_wf && source .venv/bin/activate && bash ladder.sh' > ladder.log 2>&1; echo EXIT=\$? >> ladder.log"
```

Giải thích các lựa chọn khác `docs/server-finetune.md` §4:

| Override | Lý do |
|---|---|
| `learning_rate=1e-4` | Một nửa mặc định. Pha 2 tinh chỉnh một adapter đã hội tụ, không học từ đầu; LR đầy đủ có nguy cơ xoá phần pha 1 vừa học. Đây là phán đoán, chưa đo — nếu lát sạch tụt thì đây là nghi can đầu tiên. |
| `epochs=2` | Nấc N=4 chỉ có 127 step/epoch; một epoch là 127 bước cập nhật, quá ít để đổi thói quen. Hai epoch cho 254. |
| `eval_steps` = số step một epoch | Đúng 2 lượt eval mỗi nấc. Mỗi lượt decode lại val + VIVOS, tốn phút thật. |
| `eval.batch_size=64` | Giai đoạn A là so sánh tương đối giữa bốn nấc, cả bốn cùng batch nên so được với nhau. Batch 8 chậm gấp 3,5 lần và chỉ cần ở giai đoạn B. |
| `sweep.lambdas=[0.5]` | Ghim λ đúng giá trị v6 đã phát hành, để khác biệt giữa bốn nấc là mật độ chứ không phải λ. Cũng cắt chi phí sweep từ 5 lượt decode xuống 1. |
| bốn `gates.cross_domain_*` bỏ trống | Giai đoạn A **đọc số**, không gác cổng. Để `null` thì check vẫn chạy và vẫn ghi vào `gate_results.json`, chỉ không fail run. |
| `sweep.retention_floor` bỏ trống | Cùng lý do; λ đã ghim nên không có gì để sàn lọc. |

⚠ `data.cross_domain_path` **bắt buộc phải có**. Thiếu nó thì check cross-domain im lặng
không chạy chứ không báo lỗi, và cả lượt thuê máy không thu được con số cần đo.

⚠ λ=0,5 có thể rơi ra ngoài ngân sách OOD ở nấc đậm (`ood_cer_budget=0.02` tính **cộng dồn
trên** baseline, xem `src/pipeline.py:select_lambda`). Khi đó `select_lambda` fail loud và
nấc đó không có số. Đó là kết quả có ý nghĩa, không phải sự cố — ghi lại rồi chạy tiếp nấc
sau. Không hạ ngân sách để cho qua.

### 4.3 Đọc kết quả giai đoạn A

Với mỗi nấc, lấy ba số từ `outputs/v6-dense-$N/metrics/gate_results.json`, khoá
`cross_domain_bench`:

| Trường | Ý nghĩa | Mốc |
|---|---|---|
| `retention` | thước đo chính của thang | v6 48,56%, v5 66,64%, đích 83,2% |
| `cer_no_loanword` | phần tiếng Việt còn nguyên không | **≤ 5,86%**, đây là ràng buộc §1.2 |
| `cer` | kết quả gộp | v5 8,19%, Scribe 6,25% |

Quy tắc chọn, theo thứ tự:

1. Loại mọi nấc có `cer_no_loanword` > 5,86%. Nấc đó đã đánh đổi mất thứ 101 giờ mua được,
   và §1.2 cho thấy không bù lại được bằng retention.
2. Trong số còn lại, chọn nấc có `cer` thấp nhất.
3. Nếu đường cong retention **vẫn đang lên** ở nấc 4, tức mật độ chưa bão hoà: dựng thêm nấc
   giữa (lọc theo tỉ lệ từ ngoại lai trên độ dài câu thay vì theo số tuyệt đối) hoặc tăng
   epoch ở nấc 4. Ghi nhận và quyết ở lượt thuê sau.
4. Nếu đường cong **đã phẳng** dưới 83% mà lát sạch vẫn tốt: mật độ đã hết tác dụng. Đích
   Scribe không tới được bằng cách sắp xếp lại dữ liệu hiện có, và hai đường còn lại là đổi
   base model đa ngữ hoặc điều kiện hoá lúc giải mã. Xem §7.

### 4.4 Giai đoạn B — nấc thắng, chuẩn bảng

```bash
R=v6-dense-<N thắng>-bench
OV="--override run_id=$R
    --override data.dataset_path=dataset/v6-dense-<N thắng>
    --override data.real_bench_path=null
    --override training.init_adapter=outputs/v6-corpus-r32/checkpoints/best
    --override training.epochs=2
    --override training.learning_rate=1e-4
    --override training.batch_size=16 --override training.grad_accum_steps=1
    --override training.gradient_checkpointing=false
    --override training.eval_steps=<step/epoch của nấc đó>
    --override eval.batch_size=8
    --override sweep.retention_floor=0.666
    --override data.cross_domain_path=dataset/cross-domain-bench
    --override gates.cross_domain_cer_max=0.0624
    --override gates.cross_domain_retention_min=0.836
    --override gates.cross_domain_no_loanword_cer_max=0.0586
    --override hub.repo_id=rework-whisper-v6-org/Reworkwhisper-large-v6-dense"

python -m src.pipeline --stage baseline   $OV
python -m src.pipeline --stage train      $OV
python -m src.pipeline --stage sweep-gate $OV --override hub.push=True
```

---

## 5. Ngưỡng gate — mức Scribe v2

| Bộ đo | n | Ngưỡng | Ai đang giữ |
|---|---:|---:|---|
| cross-domain CER | 299 | ≤ 6,24% | Scribe v2 (6,25%) |
| cross-domain retention | 1.283 lần | ≥ 83,6% | Scribe v2 |
| **CER lát KHÔNG có từ ngoại lai** | 48 | ≤ 5,86% | v6 |
| YouTube test | 228 | ≤ 5,93% | v5 (λ=0,75) |
| Tổng hợp TTS | 426 | ≤ 1,61% | v5 (λ=0,75) |
| VIVOS | 760 | ≤ 2,82% | v6 |

Ba hàng đầu là ba trường `gates.cross_domain_*`; ba hàng sau nằm ngoài phạm vi check đó, đo
bằng đường benchmark riêng như `Outputs/bench-cross-v6` đã làm.

Hai sửa so với `docs/v6-curriculum-plan.md` §5:

- **VIVOS 2,28% thành 2,82%.** 2,28% là số của PhoWhisper-large gốc, mà cả v5 (3,34%) lẫn v6
  (2,82%) đều không đạt. Đặt ngưỡng ở mức không adapter nào từng đạt là đặt một ngưỡng
  không có nghĩa. 2,82% là mức v6 — thứ kế hoạch này phải giữ, không được để tụt.
- **Ngưỡng lát sạch 5,86% là sát rạt và cố ý.** Giá trị đo thật của v6 là 5,8637%, nên
  0,0586 loại chính v6, và loại cả v5 (6,060%). n=48 nên khoảng tin cậy rộng hơn khoảng cách
  đang xét. Giữ nguyên vì đây là ràng buộc §1.2, không phải bảo hiểm — nhưng phải hiểu rằng
  fail hàng này ở biên là tín hiệu yếu, không phải bằng chứng.

Mọi số cột v5 lấy từ `Outputs/lambda075-metrics/v4-mixed-r16-lambda0.75/metrics/gate_results.json`
(bản đã phát hành), **không** lấy từ `Outputs/v4-mixed-r16/metrics/gate_results.json` — file
đó là λ=0,25 mà gate chọn nhầm và chưa bao giờ được phát hành.

---

## 6. Ngân sách

Mốc `0,728 giây/step` trên H200 ở rank 32, batch 16, không gradient checkpointing
(`docs/h200-timing-probe-2026-09-11.md`). Eval batch 64 đo được ~3,9 phút cho 1.125 segment
(val 365 + VIVOS 760) từ run v6; batch 8 chậm hơn ~3,5 lần.

| Khoản | Thời gian |
|---|---|
| tải corpus 14,7 GB + dựng môi trường + lọc 4 nấc | ~45 phút (chạy nền song song được) |
| giai đoạn A — train 4 nấc (1.610+964+524+254 step) | ~41 phút |
| giai đoạn A — baseline + 2 lượt eval + sweep 1λ + gate, mỗi nấc | ~20 phút × 4 = 80 phút |
| giai đoạn B — nấc thắng, batch 8, lưới λ đầy đủ | ~2 giờ |
| **tổng** | **~4,5 giờ** |

**Thuê 6 giờ.** Bản kế hoạch trước ghi "một lượt 2 giờ là đủ" — sai, con số đó bỏ quên chi
phí eval, vốn lớn hơn chi phí train một bậc. Nếu chỉ thuê được 3 giờ thì làm giai đoạn A
thôi, chép kết quả về, và để giai đoạn B sang lượt sau; nhưng khi đó phải tải lại 14,7 GB.

---

## 7. Rủi ro

| Rủi ro | Dấu hiệu | Xử lý |
|---|---|---|
| Nấc đậm xoá phần pha 1 đã học | `cer_no_loanword` > 5,86% hoặc VIVOS > 2,82% | hạ LR còn 5e-5, hoặc còn 1 epoch. Nấc 4 (8,13 h) là nghi can chính |
| Retention phẳng dưới 83% | đường cong 4 điểm bão hoà | mật độ đã hết tác dụng, xem hai đường ở dưới |
| λ=0,5 rơi khỏi ngân sách OOD | `select_lambda` raise ở một nấc | ghi nhận, chạy tiếp nấc sau. Không hạ ngân sách |
| Thắng điểm ước lượng nhưng CI chồng lấn Scribe | paired bootstrap chứa 0 | báo cáo đúng như vậy. Không làm tròn thành "vượt Scribe" |
| 13,87 h nhãn `verified: false` gây nhiễu | không có dấu hiệu trực tiếp | chạy lại nấc thắng trên bản loại `verified: false` |

Nếu thang mật độ bão hoà dưới đích, còn đúng hai đường cho nhóm 193 lần xuất hiện chưa từng
gặp (`headhunt`, `insider`, `publishing`, `helpdesk`, `pmax`, `telesale`, `creatinin`,
`urat`, `febuxostat`…):

1. **Đổi base model đa ngữ.** `openai/whisper-large-v3` hoặc `large-v3-turbo`. Chưa từng đo
   ở repo này, kể cả retention thô chưa fine-tune. Phép đo đó tốn ~10 phút GPU và nên làm
   kèm trong cùng lượt thuê để có số mà quyết, khỏi thuê lần nữa.
2. **Điều kiện hoá lúc giải mã** bằng một danh sách thuật ngữ miền.

⚠ Danh sách thuật ngữ phải dựng **độc lập với bộ benchmark**. Lấy 115 từ chưa gặp trong
cross-domain-bench ra làm từ điển là khớp đường cong, không phải kết quả — và với khán giả
lãnh đạo ngày 17/09 thì đó là con số không bảo vệ được khi bị hỏi.

Cũng đáng thử vì gần như miễn phí: **toàn bộ số hiện có đều là greedy** (`eval.num_beams: 1`
trong `configs/experiment.yaml`, chưa ai đổi). Chưa ai đo beam search trên bộ này.

Hai hàng ViMedCSS vẫn ngoài phạm vi: chỉ ~30% số lần xuất hiện từ ngoại lai của nó có trong
từ vựng train — lỗ hổng từ vựng y khoa thật, cần một đợt sinh dữ liệu riêng.

---

## 8. Sau khi chạy

`docs/server-finetune.md` §5: chép `outputs/<run_id>.zip` và `ladder.log` về **trước khi trả
máy**. Ghi `experiments/task_ledger.md` và `provenance.md` — run `v6-corpus-r32` vẫn còn nợ
hai mục này. Thu hồi token HF khi trả máy.

Decode nấc thắng trên ViMedCSS hard + test trong cùng lượt thuê (~15 phút, 2.272 segment) để
bảng benchmark đủ 6 hàng, kể cả khi kế hoạch này không nhắm vào hai hàng đó.
