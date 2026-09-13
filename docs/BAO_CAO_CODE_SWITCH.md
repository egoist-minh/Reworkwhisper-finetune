# Báo cáo đo mật độ code-switch — bộ `mix-train-100h`

Ngày đo: 2026-09-13. Đối tượng: `experiments/paid-tts-meeting-asr-demo/mix-train-100h`
(train 99,28 h / 34.525 segment / 286 meeting; val 0,79 h; test 1,58 h).

## Cách đo

Tách token từ trường `text` của 3 manifest (`manifest.train/val/test.jsonl`).
Token có dấu tiếng Việt → tiếng Việt. Token ASCII thuần thì kiểm tra theo âm vị
học tiếng Việt (âm đầu + vần + âm cuối hợp lệ); không khớp thì coi là ngoại lai.
Token ngoại lai được chia 4 nhóm:

- **english** — từ tiếng Anh thật (checklist, escalate, marketing…)
- **acronym** — viết hoa toàn bộ (KPI, SLA, ISO, API…)
- **brand** — tên thương hiệu (Viettel, FPT, Shopee, Techcombank…)
- **proper?** — viết hoa chữ đầu, phần lớn là tên riêng/tên sản phẩm

Đây là heuristic, **không phải** language-ID chuẩn. Sai số chủ yếu nằm ở nhóm
`proper?` (lẫn tên riêng tiếng Việt không dấu và từ tiếng Anh viết hoa đầu câu).
Con số dùng để kết luận là nhóm **english**.

Lưu ý về cột "giờ": chỉ tính được ở mức segment (tổng thời lượng các segment
**có chứa** ít nhất 1 token EN), không phải thời lượng thực sự nói tiếng Anh —
manifest không có word-level timestamp.

## 1. Train (99,28 h / 34.525 segment)

Nhóm `english` thuần:

| Nguồn | token EN | % token | seg có EN | % seg | giờ chứa EN | % giờ |
|---|---:|---:|---:|---:|---:|---:|
| elevenlabs_tts | 10.536 | **1,69%** | 4.760 | **20,4%** | 9,75 | 19,5% |
| youtube | 10.002 | **1,61%** | 3.611 | **32,4%** | 15,95 | 32,4% |
| **tổng** | **20.538** | **1,65%** | **8.371** | **24,2%** | **25,7** | **25,9%** |

Nếu tính gộp cả acronym + brand + tên riêng: **2,44% token / 36,4% segment**.

Chi tiết theo nhóm:

| Nguồn | english | acronym | brand | proper? |
|---|---:|---:|---:|---:|
| elevenlabs_tts (token / % token) | 10.536 / 1,69% | 3.667 / 0,59% | 1.547 / 0,25% | 2.155 / 0,35% |
| youtube (token / % token) | 10.002 / 1,61% | 575 / 0,09% | 659 / 0,11% | 1.233 / 0,20% |

### Nhận xét

- **286/286 meeting đều có ít nhất 1 token ngoại lai** — không có meeting nào thuần Việt.
- Phân bố rất lệch giữa các meeting:
  - Cao nhất: `9DCiGdqA8-c` (YouTube) 93,9% segment, `paid_meeting_0044` 82,6%,
    `paid_meeting_0043` 79,2%, `EWGnqRZdjqU` 76,4%, `coteccons-agm-2025` 74,7%.
  - Thấp nhất: `fqShV4VBFkc` 2,1%, `bao-hiem-chay-no-toa-dam` 4,4%,
    `paid_meeting_0006` 7,4%, `hqc-hocmon-doi-nha` 7,5%.
- Từ vựng khác nhau rõ giữa hai nguồn:
  - **TTS**: jargon quản trị dự án — `checklist` (278), `escalate` (240),
    `milestone` (229), `dashboard` (227), `benchmark` (223), `compliance` (198),
    `variance` (194), `forecast` (182), `baseline` (180), `audit` (172).
  - **YouTube**: hội thảo/khởi nghiệp — `startup` (289), `marketing` (198),
    `online` (135), `sale` (109), `website` (91).
- Điểm khác biệt về **bản chất** code-switch: phần YouTube lẫn cả function word
  tiếng Anh (`and` 150, `is` 117, `we` 106, `of` 96, `you` 93), tức có đoạn nói
  hẳn câu tiếng Anh chứ không chỉ chèn thuật ngữ. Phần TTS gần như chỉ chèn danh
  từ/thuật ngữ vào câu tiếng Việt.

## 2. Val / test lệch nặng so với train

| Split | Nguồn | % token EN | % seg có EN |
|---|---|---:|---:|
| train | elevenlabs_tts | 1,69 | 20,4 |
| train | youtube | 1,61 | 32,4 |
| **val** | synthetic | **9,11** | **52,0** |
| **val** | youtube | **8,13** | **93,9** |
| **test** | synthetic | **6,40** | **62,4** |
| **test** | youtube | **6,99** | **89,9** |

Hai vấn đề:

1. **Mật độ**: val/test có code-switch gấp **4–5 lần** train theo token, gấp
   2–3 lần theo segment.
2. **Domain**: val/test phần YouTube rơi vào một domain hẹp — phỏng vấn/tuyển
   dụng IT (`leetcode`, `cv`, `interview`, `binary`, `backend`, `junior`,
   `senior`, `bigtech`, `algorithm`) — trong khi train phần YouTube là hội thảo
   doanh nghiệp/khởi nghiệp. Test phần synthetic cũng nghiêng về data
   engineering (`dependency`, `lakehouse`, `cutover`, `rerun`, `mapping`).

Hệ quả: WER đo trên val/test hiện tại **phần lớn phản ánh năng lực xử lý tiếng
Anh chen vào**, không phản ánh phân phối của train. Số WER này không nên đọc như
"chất lượng model trên dữ liệu họp tiếng Việt".

## 3. Đề xuất

Hai hướng, không loại trừ nhau:

1. **Giữ nguyên val/test hiện tại**, nhưng dán nhãn đúng bản chất trong
   `DATASET_CARD.md` là bộ *stress test code-switch + domain IT*, và bổ sung một
   val thứ hai lấy từ phân phối train (~2% token EN) để theo dõi trong lúc train.
2. **Trộn lại val/test** cho mật độ EN về ~2% khớp train.

Khuyến nghị: **hướng 1**. Bộ val/test hiện tại có giá trị riêng (nó đo đúng cái
khó nhất), vấn đề chỉ là nó đang bị đọc nhầm thành "val đại diện". Trộn lại sẽ
mất luôn tín hiệu đó.

## 4. Tái lập

Script đo hiện nằm ở scratchpad của phiên làm việc, chưa commit vào repo.
Cần đưa thành tool chính thức thì đặt dưới
`experiments/paid-tts-meeting-asr-demo/` và chạy với tham số là thư mục bộ data.
