# Đặc tả gốc — Stage 4 (Evaluation Gate) + quy ước normalize

Trích nguyên văn từ `PROJECT_CORE.md` §6 của repo (không sửa), phần mô tả đúng cơ chế tính CER
đang được nộp cho professor xem. Đây là tài liệu thiết kế viết **trước khi** pipeline chạy —
không phải diễn giải sau khi có số.

---

### Stage 4: Evaluation Gate (5 checks)

| Tier | Check | Gate Rule |
|:---|:---|:---|
| 1 | In-domain (synthetic) | `CER_test ≤ 0.9 * CER_base` (10% improvement) |
| 2 | OOD | `CER_ood ≤ CER_ood(base) + 0.20pp` |
| 3 | RTF | `RTF ≤ 1.05 * RTF_base` |
| 4a | **Real audio, segmented** | `CER_real ≤ CER_real(base)` — no regression on real meeting audio |
| 4b | **Real audio, long-form** | Whole-meeting decode: repetition rate, dropped-span ratio, and length ratio all within `gates.longform_*` bounds |

If ANY tier fails → **Pipeline halts**, no push to HF.

**Tier 4 is the only tier measured on real speech, and therefore the only tier whose absolute numbers mean anything for production.** Tiers 1–2 are relative signals on synthetic / read speech. Do not report tier 1 as the headline result.

Phạm vi thực tế của run `v4-mixed-r16` đính kèm: chỉ tier 1, 2, 4a đang bật (`gates:` trong
`configs/experiment.yaml`) — tier 3 (RTF) bị bỏ, tier 4b (long-form) hoãn, ghi rõ trong comment
của config.

#### Tier 4 rules of use (`data.real_bench_path`)

The real-audio benchmark is small and non-renewable. It is a **gate, not a tuning signal**:

- **Never trained on.** Never in `validated_manifest.jsonl`.
- **Never used to rank λ.** λ* is selected on synthetic val + OOD only (§6 Stage 3). Sweeping λ against 40 minutes of one or two rooms fits λ to those rooms.
- **Read once per run, at gate time.** Every extra look burns the set's independence.
- **Not split into dev/test.** Splitting a set this small halves its statistical power for no benefit — tuning happens on synthetic data instead.
- **4b is nearly free**: long-form decode of a whole meeting needs only the full reference transcript, no segment-level timestamps.

**Statistical rigor added 2026-08-02:** tier 4a's result carries, alongside the point-estimate `cer`/`ci`/`bound`/`pass`:
- `by_meeting`: CER broken out per real-bench recording — a pooled CER can hide a regression concentrated in one room/speaker-set.
- `delta_ci` / `verdict`: a **paired** bootstrap between baseline's predictions and the candidate's predictions on the identical segments, classified into `IMPROVED`/`REGRESSED`/`INCONCLUSIVE`. `INCONCLUSIVE` means the sample genuinely cannot resolve the difference.
- `normalization_check`: convention-sensitivity check, see below.

None of the three affect `pass`/`overall_pass` — read-only rigor on top of the existing gate rule.

#### Normalization contract (decided 2026-07-31)

`metrics.py` applies **one** normalization function to both hypothesis and reference.

| Convention | Decision | State in the data | Implementation note |
|:---|:---|:---|:---|
| Numbers | **word→digit, symmetric** (`mười lăm` → `15`) | mixed: 111 digit vs 117 number-word tokens, opposite ratios per meeting | Applied to hyp **and** ref |
| Punctuation / casing | **strip both** | already near-absent | Trivial, low risk |
| Fillers | **remove** | present and lexically ambiguous | Restricted list only |
| Code-switch | **keep English as-is** | matches | No-op |
| Speaker labels / timestamps | moot | live in separate JSON fields, never inline in `text` | Nothing to strip |

**Numbers — word→digit, applied symmetrically.** Vietnamese number words are ambiguous with
non-numeric uses (`một` = indefinite article, `năm` = *five* or *year*). Symmetry is what makes
this safe: a conversion on both sides that agrees cancels out and contributes nothing to CER.
Damage only occurs where hypothesis and reference disagree at a converted token.

Required safeguards:
- **Symmetric or nothing.** Same function, same call path, both sides. Never normalize one side.
- **Audit the conversions** (`normalization.audit_conversions`): per-token conversion counts,
  `một`/`năm` broken out — if those two dominate, the number is moved by grammar, not numerals.
- **Baseline reports both**: `metrics/baseline.json` carries `cer_test` (normalized) and
  `cer_test_raw` (as-written).
- **Convention-sensitivity check**: tier 4a is scored a second time under the opposite
  `number_convention`; both numbers + delta land in `gate_results.json:tier4a_real.normalization_check`
  (diagnostic only, does not affect `pass`).

**Fillers — restricted deletion.** Only `ừm`, `ờm`, `ehm`, `uhm`, `hmm`. Explicitly **not**
`ạ`, `à`, `ừ`, `ơ`, `dạ`, `vâng` — politeness/question particles, real words.

**Repetition interacts with tier 4b.** The reference itself transcribes genuine disfluency
(`em nói nói là`). Tier 4b's repetition check must be relative to the reference's own
repetition rate, not absolute.

---

## Bối cảnh thêm — ~62% cải thiện CER quá khứ đến từ chuẩn hoá số, không phải học âm

Ghi trong `CLAUDE.md` của repo: trên run trước (`paid-dataset`), ~62% mức cải thiện CER in-domain
đo được là do chuẩn hoá số (digit normalization), không phải model học tốt hơn về âm thanh. Đây
là lý do bảng trên coi số là nguồn nhiễu điểm số lớn nhất, cần audit riêng thay vì tin CER thô.

## Lỗi đã phát hiện và sửa trong chính cơ chế chấm (rejoin bug, 2026-08-04)

Tier 4a chấm ở mức "chunk" (đoạn audio bị cắt nhỏ để đưa vào model) thay vì mức "parent segment"
(đoạn gốc trong ground truth) trước ngày 2026-08-04. Sau khi ghép lại đúng cấp độ parent-segment
(`rejoin fix`), số tier4a real CER đổi hẳn:

| | pre-fix (chunk-level) | post-fix (parent-segment, đúng) |
|---|---:|---:|
| CER real | 48.01% | 35.07% |
| baseline CER | 45.76% | 31.57% |
| n segments | 146 + 118 | 115 + 81 |

File đính kèm `run-v3-r16-rejoin-bug-case/metrics/gate_results.pre_rejoin_fix.json` (số sai) và
`gate_results.json` (số đã sửa, có field `_note` ghi lại chuyện này) — để professor thấy loại lỗi
đã từng lọt vào và cách nó được phát hiện, phòng khi còn dạng lỗi tương tự chưa bị bắt.
