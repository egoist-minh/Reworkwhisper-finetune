# Kế hoạch — cắt gọn pipeline + dựng sổ thí nghiệm

Hai việc CPU, làm **trước** khi thuê máy. Không cần GPU, không phụ thuộc câu trả lời
của professor về cấu hình lượt 101 h.

Quyết định nền (chốt trong phiên grill 15/09):

- Đường ống rút gọn còn `prepare` (CPU) → `train` → `benchmark_run` → `benchmark_report`
  → người đọc số → `merge_and_push` bằng tay.
- Bỏ: `smoke`, phần decode của `baseline`, `sweep-gate`, và `ood` trong eval lúc train.
- Không cổng tự động. λ chốt cứng; benchmark decode cả λ=1,0 và λ=0,75 trong một lượt.

**Không** làm trong kế hoạch này: refine LLM, lọc lại corpus `-15h`.

---

## 0. Trạng thái (15/09, phiên grill)

**§1 sửa code xong, đã chạy test và verify.** `pytest tests/` xanh (239 passed, 2
skipped) trước và sau khi thêm test §1.3 (13 passed riêng `test_config.py`).

Đã thay đổi:

| file | thay đổi |
|---|---|
| `src/pipeline.py` | thêm `stage_prepare`; `stage_baseline` gọi nó rồi đọc lại `validated_manifest.jsonl`; đăng ký `"prepare"` vào `STAGES`; sửa thông báo `FileNotFoundError` của `stage_train`; cập nhật docstring module; `load(..., stage=args.stage)` |
| `src/config.py` | `validate(cfg, stage=None)`; luật `ood_eval_path` chỉ nổ khi `stage in (None, "sweep-gate")`; `load(..., stage=None)` |
| `src/train.py` | `TrainingDisplayCallback(has_ood=...)`; điều kiện flush dòng bảng; cột OOD in `-` khi không có ood; `eval_dataset` bỏ `"ood"` khi `ood_ds is None` |
| `tests/test_config.py` | thêm `test_null_ood_eval_path_allowed_for_train_stage`, `test_null_ood_eval_path_still_rejected_for_sweep_gate_stage` |

Verify đã chạy (máy local, CPU-only, không có torch/peft):

1. `pytest tests/` — xanh.
2. Test §1.3 — thêm, xanh.
3. §1.1: gọi `stage_prepare(cfg)` trực tiếp (bypass `main()`'s `compat.apply()` vì máy
   này không cài peft) với `run_id=tmp-prepare` trên `dataset/mixed-noisy-v1` — ghi ra
   `validated_manifest.jsonl` + `metrics/split_stats.json`, **không** tạo
   `metrics/baseline.json`. Khớp verify. Dọn `outputs/tmp-prepare` sau đó.
   **Chưa verify được** `--stage train` tiếp nối không raise `FileNotFoundError` —
   cần torch/transformers, máy này không có (`ModuleNotFoundError: No module named
   'torch'`).
4. §1.2 (lượt train tí hon, `training.limit=20`) — **chưa làm, cần máy GPU thuê**
   (torch không cài local). Làm ở đầu session GPU tiếp theo, trước bất cứ việc gì khác
   trên máy đó.

Còn phải làm, theo thứ tự:

1. ~~**`docs/v6-100h-steps-plan.md` §1.1** — lưu adapter ở mọi vòng eval~~ — **xong 15/09**,
   cùng với §1.3 (tắt được early stopping) và §1.4 (`benchmark_run --adapter`) phát sinh khi
   rà lại plan đó. Xem trạng thái ở chính file kia.
2. §1.2 verify thật (bảng eval in đủ dòng, `training_log.csv` đúng cột) — chờ máy GPU.
   Gộp chung với lượt train tí hon của `v6-100h-steps-plan.md` §1.1, cùng một lệnh.
3. ~~§2 sổ thí nghiệm~~ — **xong**, `experiments/task_ledger.md` (commit `39e0117`).
4. ~~§3 git~~ — **xong**, 4 commit + merge `h200-server-run` vào `main` (`aa51276`).

**Lượt thuê tiếp theo đã chốt: `docs/v6-100h-steps-plan.md`** (15/09). Lượt 101 h ở cấu
hình công bằng professor yêu cầu, `eval_steps=800` giữ mọi checkpoint, sàng bằng
cross-domain rồi benchmark đầy đủ 5 suite cho bản thắng. Plan này (§1) là điều kiện cần
của nó.

---

## 1. Cắt pipeline

### 1.1 Tách `stage_baseline` thành phần CPU và phần GPU

`stage_baseline` ([src/pipeline.py:161](../src/pipeline.py#L161)) đang làm hai việc khác
hẳn nhau về chi phí:

| phần | việc | chi phí |
|---|---|---|
| CPU | `freeze` config, `load_manifests`, lọc `exclude_manifest`, `resolve_splits`, ghi `validated_manifest.jsonl`, ghi `split_stats.json` | vài giây |
| GPU | decode base model trên test / ood / real, ghi `baseline.json` + `audit/predictions_baseline_*.csv` | ~8 phút |

`stage_train` bắt buộc phải có `validated_manifest.jsonl` — nó raise
`FileNotFoundError` nếu thiếu ([src/pipeline.py:285](../src/pipeline.py#L285)). Nên phần
CPU phải giữ, chỉ phần GPU là cắt được.

Lý do cắt được phần GPU: số của base model đã có sẵn trong
`Outputs/benchmark-2026-09-10/`, và `vinai/PhoWhisper-large` không đổi giữa các lượt.

**Thay đổi:**

- Thêm `stage_prepare(cfg)` chứa đúng phần CPU ở trên, kết thúc bằng
  `_write_state(out, "prepare")`.
- `stage_baseline` gọi `stage_prepare(cfg)` rồi làm tiếp phần decode — giữ nguyên hành
  vi cũ cho ai còn cần.
- Đăng ký `"prepare": stage_prepare` vào `STAGES` ([src/pipeline.py:460](../src/pipeline.py#L460)).
- Sửa thông báo lỗi của `stage_train` thành "run stage prepare (hoặc baseline) first".

**Verify:** `python -m src.pipeline --stage prepare --override run_id=tmp-prepare ...`
ghi ra `outputs/tmp-prepare/validated_manifest.jsonl` + `metrics/split_stats.json`,
**không** tải model, **không** tạo `metrics/baseline.json`. Chạy tiếp `--stage train`
trên cùng run_id không raise `FileNotFoundError`.

### 1.2 Bỏ `ood` khỏi eval trong lúc train

Căn cứ: `docs/v6-ondomain-plan.md` §8.1 — ValCER không phân biệt được hai corpus vì val
nằm trong phân phối train; và VIVOS trong lúc train chỉ để ngắm, không tham gia chọn
checkpoint (`checkpoints/best/` chọn theo `eval_val_cer`).

**Thay đổi ở [src/train.py](../src/train.py):**

- `eval_dataset={"val": val_ds}` khi không có `ood_ds` (dòng 463).
- **Bẫy:** `TrainingDisplayCallback.on_evaluate` chỉ in một dòng bảng **khi thấy key
  `eval_ood_*`** (dòng 439). Bỏ `ood` mà không sửa chỗ này thì bảng không bao giờ in.
  Sửa: truyền cờ `has_ood` vào callback; không có ood thì flush ngay khi thấy
  `eval_val_cer`, và cột OOD in `-`.
- `training_log.csv` giữ nguyên cột `eval_ood_cer` (dòng 507), để trống.

**Verify:** `pytest tests/`, cộng một lượt `--stage train` với
`training.limit=20 training.val_limit=20 training.eval_steps=5`: bảng phải in đủ số
dòng eval, `training_log.csv` có `eval_val_cer` ở mọi dòng và `eval_ood_cer` rỗng.

### 1.3 Nới `validate` cho `ood_eval_path` null

Sau khi cắt decode baseline, cắt ood trong train và cắt `sweep-gate`, **không stage nào
còn đọc `data.ood_eval_path`**. Nhưng [src/config.py:249](../src/config.py#L249) vẫn
raise nếu nó null — nghĩa là vẫn phải chạy `scripts/fetch_vivos.py` trên máy thuê cho
một tập không ai dùng, và thiếu nó là chết ngay ở bước nạp config.

**Thay đổi:**

- `validate(cfg, stage: str | None = None)` — luật ood chỉ nổ khi
  `stage in (None, "sweep-gate")`. Mặc định `None` giữ nguyên hành vi cũ, nên test và
  mọi caller khác không đổi.
- `load(path, overrides=None, stage=None)` truyền `stage` xuống `validate`.
- `pipeline.main` gọi `load(args.config, overrides=args.override, stage=args.stage)`
  ([src/pipeline.py:493](../src/pipeline.py#L493)).

Ý định gốc của luật này là *fail sớm thay vì fail sau khi đã trả tiền train* — giữ
nguyên ý định đó: lượt nào sẽ chạy `sweep-gate` thì vẫn nổ ngay ở bước nạp config.

**Hệ quả vận hành:** bỏ được `scripts/fetch_vivos.py` khỏi checklist dựng máy thuê, và
bỏ một cách chết đã ghi trong `CLAUDE.md`.

**Verify:** `pytest tests/` (test cũ phải còn xanh vì mặc định không đổi). Thêm một test:
`load(..., stage="train")` với `ood_eval_path: null` không raise; `stage="sweep-gate"`
vẫn raise.

### 1.4 `smoke`

`stage_smoke` không tốn GPU và không nằm trên đường chạy — để nguyên, không xoá. Chỉ
đơn giản là không gọi nó nữa.

---

## 2. Sổ thí nghiệm `experiments/task_ledger.md`

`CLAUDE.md` gọi đây là một trong bốn trụ versioning. File **không tồn tại**. Hậu quả đã
xảy ra: ba lượt curriculum `v6-dense-{1,3,4}` chạy xong 13/09, kết quả không được ghi ở
đâu ngoài zip, và phiên 15/09 suýt lên kế hoạch chạy lại lần thứ tư.

Kiểm chứng: `grep -rn "ladder-phaseA" docs/ SESSIONS.md` → 0 dòng. `v6-dense` chỉ xuất
hiện trong hai file *kế hoạch*, không file nào ghi *kết quả*.

### 2.1 Cột

Mỗi lượt một dòng:

`run_id | ngày | corpus (giờ, mật độ) | rank/α | epoch | LR | init_adapter | λ báo cáo |
cross-domain CER | cross-domain retention | no_loanword CER | tier2 OOD | artifact | kết luận một câu`

Số **đọc từ artifact**, không chép từ doc — `docs/so-lieu-tong-hop.md` đã được ghi nhận
là lệch khỏi manifest.

### 2.2 Nguồn số cho các lượt điền ngược

| run_id | nguồn |
|---|---|
| `v3-r16` | `Outputs/v3-r16/metrics/gate_results.json` (bản `.pre_rejoin_fix.json` là bản cũ, **không** dùng) |
| `v3-r16` @λ=0,5 | `experiments/v3-r16-lambda0.5/` |
| `v4-mixed-r16` | `Outputs/v4-mixed-r16/metrics/gate_results.json` |
| `v4-mixed-r16` @λ=0,75 (= v5 production) | `Outputs/lambda075-metrics/v4-mixed-r16-lambda0.75/` |
| `v6-corpus-r32` | `Outputs/v6-corpus-r32.zip` → `v6-corpus-r32/metrics/gate_results.json` |
| `v6-dense-1` | `Outputs/ladder-phaseA/v6-dense-1.zip` |
| `v6-dense-3` | `Outputs/ladder-phaseA/v6-dense-3.zip` |
| `v6-dense-4` | `Outputs/ladder-phaseA/v6-dense-4.zip` |
| `v6-ondomain-15h` | `Outputs/v6-ondomain-15h/metrics/{gate_results,cross_domain,cross_domain_lambda075}.json` |
| `v6-ondomain-29h` | `Outputs/v6-ondomain-29h/metrics/cross_domain.json` |

Số cross-domain đã đọc được trong phiên 15/09, dùng để đối chiếu khi điền:

| run | CER | retention | no_loanword |
|---|---:|---:|---:|
| ngưỡng v5 (đường decode chống lặp) | 0,0740 | 0,6648 | 0,0613 |
| `v6-corpus-r32` | 0,1081 | 0,4864 | 0,0577 |
| `v6-dense-1` (hai pha) | 0,1059 | 0,5269 | 0,0572 |
| `v6-dense-4` (hai pha) | 0,0987 | 0,5355 | 0,0553 |
| `v6-dense-3` (hai pha) | 0,0914 | 0,5596 | 0,0555 |
| `v6-ondomain-29h` λ=1,0 | 0,1101 | 0,5947 | 0,1083 |
| `v6-ondomain-15h` λ=1,0 | 0,0919 | 0,6157 | 0,0731 |
| `v6-ondomain-15h` λ=0,75 | 0,0875 | 0,6189 | 0,0628 |

### 2.3 Hai kết luận phải nằm trong sổ, vì chúng đóng hướng đi

1. **Curriculum hai pha thua một pha.** Cả ba lượt `v6-dense` (nối từ
   `v6-corpus-r32/checkpoints/best`, LR 1e-4, 2 epoch) đều thua `v6-ondomain-15h` một
   pha LoRA tươi ở cả CER lẫn retention. Lượt có pha 2 **lớn nhất** (`-1`, 12.880 seg)
   lại cho retention **tệ nhất**. Cơ chế: pha 1 trên corpus loãng 2,48% đặt trần.
2. **Giờ ở cùng mật độ không phải đòn bẩy.** Corpus của v5 chỉ **7,77 h**
   (2,46 h youtube + 5,31 h synthetic) ở 7,3% mật độ → retention 66,6%. `-15h` có
   **16,29 h** ở cùng dải mật độ, gấp đôi số type từ vựng → retention 61,9%. Mật độ đã
   khớp mà vẫn thua, nên biến còn lại nằm ở **thành phần corpus**, không phải mật độ:
   YouTube của v5 là 2,46 h người soát toàn bộ và **0 segment toàn tiếng Anh**; YouTube
   của `-15h` có 105 segment toàn tiếng Anh (4,7%) và phần lớn nhãn là máy.

**Verify:** mỗi dòng trong sổ phải trỏ tới một file có thật; `grep` tên artifact trong
sổ ra đúng đường dẫn tồn tại trên đĩa.

---

## 3. Đóng lại git

Sau khi §1 và §2 xong:

- Commit 4 file đang dở (`src/config.py`, `src/data.py`, `src/pipeline.py`,
  `docs/v6-ondomain-plan.md`) cùng `docs/yeu-cau-lo-tts-10h.md` chưa track.
- Commit thay đổi §1, §2.
- Merge `h200-server-run` (đang đi trước `main` 14 commit) vào `main`.

**Verify:** `git status` sạch; `git log --oneline main..h200-server-run` rỗng.

---

## 4. Thứ tự và chi phí

| bước | thời gian |
|---|---|
| §1.1 tách `prepare` | ~20 phút |
| §1.2 bỏ ood khỏi eval train | ~20 phút |
| §1.3 nới `validate` | ~10 phút |
| `pytest tests/` | ~2 phút |
| §2 sổ thí nghiệm (giải nén 4 zip, đọc 10 file, lập bảng) | ~30 phút |
| §3 git | ~10 phút |

Tổng ~1,5 h, không phút GPU nào.
