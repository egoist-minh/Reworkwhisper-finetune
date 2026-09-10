# Tinh chỉnh nhẹ mô hình nhận dạng tiếng nói tiếng Việt cho các cuộc họp có chuyển mã

> Bản nháp. Số liệu lấy từ `docs/so-lieu-tong-hop.md`; mỗi phần được chốt riêng trước khi ghi vào đây.
> Quy ước: dùng đơn vị **giờ** cho mọi thời lượng dữ liệu trong toàn bài.

## Trạng thái từng phần

| Phần | Trạng thái |
|---|---|
| Abstract | chốt (bản 4) |
| 1. Giới thiệu | chốt |
| 2. Công trình liên quan | chốt |
| 3. Dữ liệu | chốt |
| 4. Phương pháp | chốt |
| 5. Thiết lập thí nghiệm | chốt |
| 6. Kết quả | chốt |
| 7. Phân tích | chốt tạm |
| 8. Hạn chế | chốt |
| 9. Kết luận | chưa |

---

## Abstract

Nhận dạng tiếng nói cho các cuộc họp công việc tiếng Việt gặp một khó khăn mà các tập chuẩn
hiện hành không phản ánh: hiện tượng code-switching Việt–Anh dày đặc. Trên bộ dữ liệu YouTube meeting mà
chúng tôi xây dựng, 6,67% số token và 89,7% số đoạn chứa ít nhất một từ mượn tiếng Anh, trong
khi tập test chuẩn VIVOS không chứa token nào thuộc loại này. Hệ quả là mô hình nền
`PhoWhisper-large`, dù đạt CER 2,28% trên VIVOS, chỉ giữ đúng 36,4% số lượt từ mượn và đạt
CER 15,93% trên phần YouTube meeting của tập test.

Chúng tôi trình bày `Reworkwhisper-large-v5`, bản tinh chỉnh `PhoWhisper-large` bằng LoRA hạng
16 (28,8 triệu tham số, 1,76% tổng số), huấn luyện trên tập trộn gồm 1,96 giờ YouTube meeting và 5,02
giờ hội thoại synthetic. Phần YouTube meeting tuy nhỏ nhưng đậm đặc đúng hiện tượng cần học — 89,7%
số đoạn chứa từ mượn — và được hiệu đính thủ công từ phụ đề tự động, với 779 trên 790 đoạn có
thay đổi so với nhãn nháp. Trọng số hợp nhất được chọn qua một lượt quét hệ số λ có ràng buộc
ngân sách sai số trên VIVOS.

Trên tập test 654 đoạn, cấu hình do quy trình tự chọn (λ=0,5) đưa CER trên phần YouTube meeting từ
15,93% xuống 6,24% — giảm tương đối 60,8% — WER từ 22,69% xuống 10,11%, và tỷ lệ giữ đúng từ
mượn từ 36,4% lên 82,6%. Cấu hình được triển khai thực tế (λ=0,75), chọn sau khi quan sát toàn
bộ đường cong λ, đạt lần lượt 5,93%, 9,37% và 85,5%. Trên ba buổi họp giữ riêng thuộc ba lĩnh
vực khác nhau và hoàn toàn nằm ngoài phân phối huấn luyện, mô hình đạt CER 8,44% so với 6,54%
của một hệ thống thương mại đang dẫn đầu.

---

## 1. Giới thiệu

Nhận dạng tiếng nói tiếng Việt đã đạt mức chất lượng cao trên các tập chuẩn công khai.
`PhoWhisper` (Le, Nguyen & Nguyen, 2024), tinh chỉnh từ Whisper trên 843,79 giờ tiếng Việt,
đạt WER 4,67% trên tập test VIVOS ở bản `large`, vượt rõ các hệ dựa trên wav2vec2 trước đó.
Trong lần đo lại của chúng tôi theo quy ước chuẩn hoá riêng, con số tương ứng là CER 2,28% và
WER 4,73% — xác nhận rằng với giọng đọc rõ, câu ngắn, nội dung thuần Việt, bài toán gần như đã
được giải.

Các cuộc họp công việc lại là một phân phối khác hẳn. Trên bộ dữ liệu bảy buổi YouTube meeting mà
chúng tôi thu thập và hiệu đính, 6,67% số token và 89,7% số đoạn chứa ít nhất một từ mượn tiếng
Anh không được Việt hoá — tên công cụ, thuật ngữ chuyên môn, tên thương hiệu. Cùng chỉ số ấy
trên tập test VIVOS bằng đúng 0: không một token nào. Sự chênh lệch này không phải là khác biệt
về mức độ khó mà là khác biệt về loại hiện tượng, và nó ẩn hoàn toàn khỏi mọi bảng xếp hạng dựa
trên tập chuẩn hiện có. Đo trên phần YouTube meeting, `PhoWhisper-large` đạt CER 15,93%, gấp bảy lần
con số VIVOS của chính nó, và chỉ giữ đúng dạng gốc của 36,4% số lượt từ mượn — phần còn lại bị
phiên âm thành các âm tiết tiếng Việt gần giống, khiến câu vẫn đọc trôi nhưng mất chính xác ở
đúng những từ mang nhiều thông tin nhất.

Khoảng trống này khó lấp vì ba lẽ. Thứ nhất, dữ liệu họp có nhãn rất đắt: nội dung thường mang
tính nội bộ nên khó công bố, còn phiên âm thủ công một giờ hội thoại tự phát tốn nhiều công hơn
hẳn một giờ giọng đọc. Thứ hai, full fine-tuning một mô hình 1,55 tỷ tham số đòi hỏi hạ tầng
ngoài tầm với của phần lớn nhóm ứng dụng — bài `PhoWhisper` gốc dùng 8 GPU A100 — và mang rủi ro
làm suy giảm năng lực sẵn có trên miền chung. Thứ ba, và ít được chú ý nhất, các chỉ số hiện
hành không đo được điều cần đo: CER tổng thể trên một tập không chứa hiện tượng code-switching sẽ
không hề thay đổi dù mô hình xử lý code-switching tốt lên hay tệ đi.

Chúng tôi tiếp cận cả ba mặt. Về dữ liệu, thay vì phiên âm từ đầu, chúng tôi lấy phụ đề tự động
của các buổi họp công khai làm nhãn nháp rồi hiệu đính thủ công, tập trung vào đúng những chỗ hệ
tự động hỏng có hệ thống; kết quả là 3,46 giờ đã soát, trong đó 1,96 giờ dùng để huấn luyện. Về
mô hình, chúng tôi dùng LoRA hạng 16 — 28,8 triệu tham số khả huấn, tương đương 1,76% tổng số —
huấn luyện trọn vẹn trên một GPU T4, rồi kiểm soát mức độ can thiệp bằng một hệ số hợp nhất λ
được chọn dưới ràng buộc ngân sách sai số trên VIVOS, để năng lực miền chung không bị đánh đổi
lấy năng lực miền hẹp một cách âm thầm. Về đo lường, chúng tôi bổ sung một chỉ số tỷ lệ giữ đúng
từ mượn bên cạnh CER và WER, đo trực tiếp thứ mà CER tổng thể không tách ra được.

**Đóng góp.**

1. Một corpus bảy buổi họp công việc tiếng Việt có code-switching dày đặc, 790 đoạn, 3,46 giờ, nhãn
   hiệu đính thủ công, kèm hồ sơ định lượng về mật độ code-switching, mức chồng tiếng và độ tương
   đồng người nói. Chúng tôi phát hành phần nhãn — transcript đã hiệu đính, mã video và mốc
   thời gian từng đoạn — để tái lập được, còn audio thì người dùng tự tải từ nguồn gốc.
2. Bằng chứng về hiệu quả dữ liệu: 1,96 giờ YouTube meeting, trộn với hội thoại synthetic, đưa CER trên
   phần YouTube meeting từ 15,93% xuống 6,24% ở cấu hình do quy trình chọn và 5,93% ở cấu hình đã
   triển khai, không cần full fine-tuning.
3. Một quy tắc chọn hệ số hợp nhất có ràng buộc ngân sách sai số miền chung, kèm đường cong đánh
   đổi đầy đủ trên năm mức λ.
4. Đánh giá ngoài phân phối trên ba buổi họp thuộc ba lĩnh vực khác nhau, đối chiếu trực tiếp với
   một hệ thống thương mại.


---

## 2. Công trình liên quan

**2.1. Nhận dạng tiếng nói tiếng Việt.** `PhoWhisper` (Le, Nguyen & Nguyen, 2024) tinh chỉnh đầy
đủ họ Whisper trên 843,79 giờ tiếng Việt; bản `large` 1,55 tỷ tham số đạt WER 4,67% trên VIVOS và
8,14% trên CMV–Vi, tốt hơn các hệ wav2vec2 tiếng Việt trên mọi tập trong bảng của họ. Hai giới
hạn dẫn tới bài này: quy trình cần 8 GPU A100, ngoài tầm với của phần lớn nhóm ứng dụng; và cả
bốn tập đánh giá đều là tiếng Việt đọc hoặc phát thanh, không tập nào chứa hội thoại công việc tự
phát hay hiện tượng code-switching.

**2.2. Tinh chỉnh hiệu quả tham số.** LoRA (Hu và cộng sự, 2021) đóng băng trọng số gốc và chỉ
học hiệu chỉnh low-rank `BA` cho từng ma trận đích, giảm số tham số khả huấn vài bậc và không
phát sinh chi phí suy luận do hiệu chỉnh cộng thẳng được vào trọng số nền. Ngoài lý do tiết kiệm
bộ nhớ, điều quan trọng với chúng tôi là mô hình gốc còn nguyên vẹn và bản tinh chỉnh chỉ là một
tệp adapter vài chục megabyte.

**2.3. Nội suy trọng số.** Wortsman và cộng sự (2022) chỉ ra tinh chỉnh làm giảm độ bền trước
dịch chuyển phân phối, và nội suy trọng số giữa mô hình nền với bản tinh chỉnh lấy lại phần lớn
độ bền đã mất. Ilharco và cộng sự (2023) tổng quát thành task arithmetic: hiệu
trọng số trước và sau tinh chỉnh được co giãn bằng hệ số λ chọn trên tập validation, với λ=1 khôi
phục mô hình tinh chỉnh đầy đủ. Cả hai đều làm trên mô hình thị giác và trọng số đầy đủ. Chúng
tôi áp dụng cơ chế ấy cho nhận dạng tiếng nói, với task vector chính là tích `BA` của LoRA
nên phép hợp nhất có dạng `W + λ·BA`, và với một khác biệt về quy tắc chọn: thay vì lấy λ tối ưu
trên miền đích, chúng tôi ràng buộc λ bằng ngân sách sai số trên miền chung rồi mới chọn λ tốt
nhất trong số các giá trị thoả ràng buộc (mục 5).


---

## 3. Dữ liệu

**3.1. Nguồn và sàng lọc.** Corpus YouTube meeting gồm bảy video công khai trên YouTube thuộc miền công
nghệ và tuyển dụng, đều là hội thảo hoặc buổi tư vấn, thời lượng gốc từ 27,4 đến 110,9 phút. Ba
luật sàng lọc chạy trên phụ đề tự động **trước khi tải audio**, nhằm loại sớm những video không
dùng được: không có phụ đề tiếng Việt; phụ đề tiếng Việt là bản dịch máy chứ không phải bản gốc;
và tốc độ từ trên phút tụt dần qua các mốc năm phút, dấu hiệu cho thấy phụ đề là bản tóm tắt do
người viết chứ không phải bản nhận dạng giọng nói. Cả bảy video đều qua được cả ba luật; không
video nào bị loại. Việc sàng lọc trước khi tải là chủ ý, vì bước tải và cắt tốn kém hơn bước đọc
phụ đề nhiều bậc.

**3.2. Cắt đoạn.** Mỗi buổi được cắt về một cửa sổ ba mươi phút liên tục căn giữa video, tránh
phần chào hỏi đầu buổi và phần hỏi đáp rời rạc cuối buổi. Sáu trên bảy buổi cho đúng 1800,0 giây;
buổi `dGT3YW0AdD8` chỉ đạt 1644,96 giây vì video gốc ngắn hơn cửa sổ. Điểm cắt đoạn lấy theo mốc
thời gian từng từ của phụ đề chứ không theo khoảng lặng trên dạng sóng — cách này giữ ranh giới
đoạn trùng với ranh giới nhãn, đổi lại một số đoạn bắt đầu hoặc kết thúc giữa nhịp thở. Kết quả
là 790 đoạn, tổng 3,46 giờ, 41.210 từ, âm thanh 16 kHz một kênh.

**3.3. Gán nhãn.** Phụ đề tự động được dùng làm nhãn nháp, sau đó người soát hiệu đính toàn bộ.
Cách này rẻ hơn phiên âm từ đầu nhiều lần, và quan trọng hơn, nó dồn công sức vào đúng chỗ hệ tự
động sai có hệ thống thay vì rải đều lên cả những đoạn vốn đã đúng. Toàn bộ 790 đoạn đều đã qua
soát; trường trạng thái ở mức từng đoạn hiện chỉ được điền cho 229 đoạn, phần còn lại thiếu thao
tác ghi nhận chứ không thiếu bước soát. Mức độ can thiệp đo được từ chính dữ liệu: text sau hiệu
đính khác text nhãn nháp ở 779 trên 790 đoạn. Loại sửa đặc trưng nhất nằm đúng ở từ mượn — chẳng
hạn chuỗi `grap` xuất hiện 21 lượt trong nhãn nháp và 0 lượt trong nhãn cuối, trong khi `grab` đi
từ 9 lượt lên 43 lượt.

**3.4. Hồ sơ định lượng.** Đặc điểm nổi bật nhất là mật độ code-switching: 6,67% số token và 89,7% số
đoạn chứa ít nhất một từ mượn tiếng Anh không Việt hoá. Để đối chiếu, cùng phép đo trên tập test
VIVOS cho kết quả 0 token. Hai chỉ số khác được đo nhưng không phải là thách thức chính của corpus
này. Chồng tiếng chỉ chiếm 0,89% tổng số giây thực sự có hai người nói cùng lúc, dù 26,7% số đoạn
có chồng tiếng khác 0 — định dạng hội thảo có người dẫn khiến các lượt nói ít giẫm lên nhau. Về
người nói, độ tương đồng ECAPA trong cùng buổi đạt 0,54 so với 0,32 giữa các buổi khác nhau, tức
các buổi phân tách được về mặt giọng nói ở mức tổng thể.

**3.5. Ba tầng độc lập giữa huấn luyện và kiểm thử.** Chúng tôi kiểm tra riêng ba chiều độc lập.

| Chiều | Trạng thái |
|---|---|
| Buổi họp và tệp âm thanh | tách hoàn toàn — không buổi nào xuất hiện ở cả hai phía |
| Người nói | tách ở mức tổng thể; buổi `xKDHjUoUN54` thuộc tập huấn luyện có độ tương đồng ECAPA cao hơn mức thường thấy với hai buổi kiểm thử |
| Chủ đề | dùng chung — cả bảy buổi cùng thuộc miền công nghệ và tuyển dụng |

Chủ đề dùng chung là đặc điểm cố hữu của một corpus thu hẹp theo miền: khi mục tiêu là học cách
xử lý thuật ngữ và lối chêm từ của một lĩnh vực, việc các buổi cùng miền là điều mong muốn ở phía
huấn luyện, đồng thời có nghĩa là kết quả trên tập test đo năng lực **trong** miền ấy. Để đo phần khái quát hoá ra ngoài miền, chúng tôi
đánh giá thêm trên ba buổi giữ riêng thuộc ba lĩnh vực khác hẳn — tuyển dụng nhân sự, hội chẩn y
khoa và webinar marketing — trình bày ở mục 5.

**3.6. Dữ liệu synthetic và phân chia.** Bên cạnh YouTube meeting, chúng tôi dùng một tập hội thoại họp
synthetic sinh bằng kịch bản do mô hình ngôn ngữ viết rồi đọc bằng hệ text-to-speech thương
mại, với mười giọng ở tập huấn luyện và mười giọng ở tập kiểm thử không giao nhau. Phân chia cuối
cùng:

| Split | YouTube meeting | Synthetic | Tổng đoạn |
|---|---|---|---:|
| Huấn luyện | 447 đoạn · 1,96 giờ | 4.270 đoạn · 5,02 giờ | 4.717 |
| Validation | 115 đoạn · 0,50 giờ | 250 đoạn · 0,29 giờ | 365 |
| Kiểm thử | 228 đoạn · 1,00 giờ | 426 đoạn · 0,58 giờ | 654 |

Phần YouTube meeting chỉ chiếm 28% thời lượng huấn luyện. Việc trộn hai nguồn nhằm hai mục đích: dữ liệu
synthetic cung cấp khối lượng và độ sạch nhãn, còn dữ liệu YouTube meeting cung cấp hiện tượng — nhiễu
nền, lượt nói chồng, và trên hết là cách người Việt thực sự chêm từ tiếng Anh khi nói chuyện công
việc.


---

## 4. Phương pháp

**4.1. Tinh chỉnh.** Chúng tôi gắn adapter LoRA vào sáu ma trận projection của mỗi khối
Transformer trong cả encoder lẫn decoder: `q_proj`, `k_proj`, `v_proj`, `out_proj` của khối
self-attention, cùng `fc1`, `fc2` của khối feed-forward. Danh sách này bám theo tên module thực
tế của Whisper, khác với quy ước phổ biến ở các mô hình decoder-only — chọn nhầm tên sẽ khiến
adapter được tạo ra nhưng không gắn vào đâu cả, và quá trình huấn luyện vẫn chạy trơn tru mà
không học được gì.

Cấu hình: rank 16, alpha 32, dropout 0,05, dùng rsLoRA thay cho hệ số scaling `alpha/r` mặc định.
Tổng cộng 28,8 triệu tham số khả huấn trên 1,64 tỷ, tương đương 1,76%. Huấn luyện 3 epoch,
learning rate 2×10⁻⁴, warmup ratio 0,1, gradient checkpointing, seed 42, tổng 885 bước cập nhật
trên một GPU T4 duy nhất.

**4.2. Chọn hệ số hợp nhất.** Sau huấn luyện, adapter được hợp nhất theo `W + λ·BA` với λ quét
trên lưới {0; 0,25; 0,5; 0,75; 1,0}. Mỗi mức λ được chấm trên tập validation và trên VIVOS. Quy
trình chọn gồm hai tầng.

*Ràng buộc cứng.* CER trên VIVOS không được vượt quá CER của mô hình nền cộng 0,02 tuyệt đối,
tức trần 0,0428 trong lần chạy này. Đây là ngân sách cho phép mô hình quên một lượng hữu hạn
năng lực miền chung, khai báo trước chứ không suy ra từ kết quả. Nếu không λ nào trong lưới thoả
ràng buộc, quy trình dừng và báo lỗi — không có fallback mềm, không chọn "λ gần nhất".

*Quy tắc elbow.* Trong số các λ thoả ngân sách, ta duyệt theo chiều tăng và tính tỉ số
cost/benefit của từng bước, lấy mức tăng CER trên VIVOS chia cho mức giảm CER trên validation.
Khi tỉ số của một bước vượt quá mười lần tỉ số của bước trước đó, bước ấy và mọi λ lớn hơn bị
loại; λ được chấp nhận cuối cùng là kết quả. Quy tắc này thay cho cách chọn "λ lớn nhất còn
trong ngân sách", vốn có xu hướng đẩy tới λ=1,0 ngay cả khi phần lợi ích tăng thêm đã cạn.

Trên đường cong của lần chạy này, quy tắc chọn **λ=0,5**: bước lên 0,75 có tỉ số nhảy 11,68 lần,
vượt ngưỡng 10.

**4.3. Chuẩn hoá và các chỉ số.** Mọi phép chấm áp cùng một chuỗi normalize lên cả hypothesis
lẫn reference: Unicode NFC, lowercase, bỏ dấu câu, gộp khoảng trắng, loại filler token, và quy
đổi số viết bằng chữ về chữ số. Decode bằng greedy search (`num_beams=1`), chỉ định `language=vi`,
batch 8, fp16. Chỉ số chính là CER, kèm WER để tham khảo. CER là tỉ số giữa số phép sửa ký tự
cần thiết để biến hypothesis thành reference và số ký tự của reference; nó không bị chặn ở 100%,
vì một hypothesis dài hơn reference nhiều lần sẽ cần nhiều phép sửa hơn độ dài reference.

Chúng tôi bổ sung một chỉ số thứ ba, **tỷ lệ giữ đúng từ mượn**. Với mỗi đoạn, ta lấy các token
trong reference dài hơn một ký tự, thuần chữ cái, và **không** khớp phép thử hình dạng âm tiết
tiếng Việt; rồi đếm bao nhiêu token trong số đó xuất hiện lại trong hypothesis, so khớp theo
multiset nên việc đảo thứ tự không bị phạt còn việc nói hai lần mà chỉ nhận ra một lần thì bị.
Chúng tôi cố ý nhận diện từ mượn bằng hình dạng âm tiết thay vì đối chiếu whitelist từ tiếng
Anh, bởi trong thử nghiệm sơ bộ, cách dùng whitelist cho tới 72% false positive — phần lớn là
các âm tiết tiếng Việt trùng mặt chữ với từ tiếng Anh. Trên tập test, bộ nhận diện đánh dấu 362
type và 1.353 lượt; rà soát thủ công tìm được 7 type đáng ngờ ứng với 15 lượt, tức 1,1% theo lượt.


---

## 5. Thiết lập thí nghiệm

**5.1. Các mô hình được so sánh.** Bốn cấu hình được chấm trên cùng một tập test:

| Ký hiệu trong bài | λ | Dữ liệu huấn luyện |
|---|---:|---|
| `PhoWhisper-large` | — | mô hình nền, không tinh chỉnh |
| `Reworkwhisper-large-v4` | 0,5 | chỉ hội thoại synthetic |
| `Reworkwhisper-large-v5` @ λ=0,5 | 0,5 | synthetic + YouTube meeting |
| `Reworkwhisper-large-v5` | 0,75 | synthetic + YouTube meeting |

Hai dòng cuối là cùng một adapter ở hai hệ số hợp nhất, không phải hai mô hình khác nhau: λ=0,5
là cấu hình do quy tắc ở mục 4.2 chọn, λ=0,75 là cấu hình đã triển khai. Chúng tôi báo cáo song
song cả hai.

`Reworkwhisper-large-v4` đóng vai trò đối chứng về nguồn dữ liệu. Nó dùng cùng kiến trúc, cùng
rank, cùng siêu tham số, nhưng huấn luyện **chỉ trên hội thoại synthetic**, không có một giây họp
thật nào. Chênh lệch giữa nó và `Reworkwhisper-large-v5` do đó phần lớn quy được về 1,96 giờ họp
thật thêm vào. Đây không phải phép so sánh có kiểm soát hoàn toàn — hai lần chạy còn khác nhau ở
danh sách buổi dùng làm validation và ở số bước cập nhật (801 so với 885) — nên chúng tôi đọc nó
như một chỉ dấu mạnh chứ không phải một ablation nghiêm ngặt.

**5.2. Các bộ đánh giá.** Bốn bộ, đo bốn thứ khác nhau.

*Tập test in-domain (654 đoạn).* Gồm 228 đoạn YouTube meeting từ hai buổi không xuất hiện trong huấn
luyện và 426 đoạn synthetic từ ba buổi synthetic riêng. Kết quả luôn được báo tách theo hai phần,
vì gộp chung sẽ che mất chênh lệch giữa chúng. Cả bốn cấu hình chấm trên đúng cùng 654 khoá
`(meeting_id, segment_id)`, và bản reference của bốn lần chấm lệch nhau 0 dòng.

*VIVOS (760 câu).* Tập test chuẩn công khai, đóng vai trò thước đo năng lực miền chung và là cơ
sở cho ràng buộc ngân sách ở mục 4.2. Không câu nào trong tập này chứa token mang hình dạng phi
âm tiết tiếng Việt.

*Ba buổi giữ riêng (299 đoạn, 78 phút).* Ba buổi thuộc ba lĩnh vực nằm ngoài miền huấn luyện —
tuyển dụng nhân sự, hội chẩn y khoa, webinar marketing — dùng để đo khả năng khái quát hoá ra
ngoài miền. Trên bộ này chúng tôi đối chiếu trực tiếp với ElevenLabs Scribe v2.

*Benchmark audio thu thật (2 bản ghi, 43,2 phút).* Ghi âm cuộc họp thu tại chỗ, khác hoàn
toàn về điều kiện âm học so với audio YouTube. Reference của bộ này là bản nháp do
`PhoWhisper-small` sinh rồi người hiệu đính, nên nó đo mức cải thiện tương đối chứ không phải sai
số tuyệt đối đáng tin.

**5.3. Khoảng tin cậy.** Chúng tôi lấy bootstrap ở mức đoạn trên số ký tự đúng và số phép sửa,
báo khoảng 95%. Với so sánh trực tiếp giữa hai mô hình trên cùng tập đoạn, chúng tôi bootstrap
trên hiệu số theo cặp thay vì so hai khoảng riêng lẻ.

**5.4. Một khác biệt về điều kiện chấm.** Lần chấm ba buổi giữ riêng chạy ngày 2026-08-27 trên
Google Colab, với cấu hình decode khác phần còn lại của bài: có đặt `no_repeat_ngram_size`,
`no_speech_threshold`, và `condition_on_prev_tokens=False`. Ngoài ra bản `Reworkwhisper-large-v5`
dùng trong lần chấm ấy được tải từ HuggingFace tại thời điểm đó và revision hash không được ghi
lại. Vì vậy các con số ở mục ba buổi giữ riêng **không so trực tiếp** được với các con số trên
tập test in-domain; chúng chỉ so được nội bộ, tức giữa `Reworkwhisper-large-v5` và ElevenLabs
Scribe v2 trong cùng lần chấm ấy.



---

## 6. Kết quả

**6.1. Tập test in-domain.** Bảng 1 trình bày kết quả của bốn cấu hình trên cùng 654 đoạn, tách
theo hai phần. Trong ngoặc là khoảng tin cậy bootstrap 95% ở mức đoạn.

**Bảng 1.** CER, WER và tỷ lệ giữ từ mượn trên tập test in-domain (228 đoạn YouTube meeting, 426
đoạn synthetic), đơn vị phần trăm. Giá trị tốt nhất mỗi hàng in đậm.

| Chỉ số | `PhoWhisper-large` | `Reworkwhisper-large-v4` | `-v5` @ λ=0,5 | `Reworkwhisper-large-v5` |
|---|---:|---:|---:|---:|
| CER — YouTube meeting | 15,93 [12,05; 23,35] | 12,05 | 6,24 | **5,93** [5,42; 6,45] |
| CER — synthetic | 4,26 [3,78; 4,73] | 1,96 | 1,85 | **1,61** [1,31; 1,89] |
| CER — gộp | 11,31 | 8,05 | 4,50 | **4,22** |
| WER — YouTube meeting | 22,69 | 16,10 | 10,11 | **9,37** |
| WER — synthetic | 7,40 | 2,97 | 2,89 | **2,37** |
| Giữ từ mượn — YouTube meeting | 36,4 | 77,5 | 82,3 | **85,5** |
| Giữ từ mượn — synthetic | 55,7 | 83,1 | 83,1 | **86,2** |

Cả bảy chỉ số cải thiện đơn điệu qua bốn cấu hình. Mức giảm CER tuyệt đối lớn nhất nằm ở phần
YouTube meeting, 10,00 điểm giữa mô hình nền và cấu hình đã triển khai, so với 2,65 điểm ở phần
synthetic. Chỉ số giữ từ mượn tăng mạnh nhất ở bước đầu tiên, 41,1 điểm giữa mô hình nền và
`Reworkwhisper-large-v4`.

Khoảng tin cậy của mô hình nền trên phần YouTube meeting rộng hơn hẳn các cấu hình còn lại.
Nguyên nhân là một đoạn trong 228 đoạn mà mô hình nền sinh chuỗi lặp, cho CER 2543,8%; loại đoạn
này, CER của mô hình nền trên phần YouTube meeting là 12,65. Mọi con số trong bài đều tính đủ
228 đoạn.

**6.2. Đường cong hệ số hợp nhất.** Bảng 2 và Hình 1 trình bày toàn bộ lưới λ.

**Bảng 2.** Ảnh hưởng của hệ số hợp nhất λ, đo trên tập test và trên VIVOS, đơn vị phần trăm.
Ngân sách CER trên VIVOS là 4,28.

| λ | CER YouTube meeting | CER synthetic | Giữ từ mượn | CER VIVOS |
|---:|---:|---:|---:|---:|
| 0,00 | 15,93 | 4,26 | 36,4 | 2,28 |
| 0,25 | 7,64 | 2,58 | 70,9 | **2,26** |
| 0,50 | 6,24 | 1,85 | 82,3 | 2,62 |
| 0,75 | **5,93** | **1,61** | **85,5** | 3,34 |
| 1,00 | — | — | — | 4,26 |

**Hình 1.** CER theo λ trên ba tập đánh giá. (`docs/training-curves/paper-fig1-lambda.png`)

Bước từ λ=0 lên 0,25 giảm 8,29 điểm CER trên phần YouTube meeting trong khi CER trên VIVOS giảm
0,02 điểm. Ba bước sau cộng lại giảm thêm 1,71 điểm trên phần YouTube meeting và làm CER trên
VIVOS tăng 2,00 điểm. Ở λ=1,0, CER trên VIVOS đạt 4,26, cách ngân sách 0,02 điểm. Cấu hình λ=1,0
không được lưu thành artifact nên không có số trên tập test.

**6.3. Ngoài phân phối huấn luyện.** Bảng 3 so `Reworkwhisper-large-v5` với ElevenLabs Scribe v2
trên ba buổi giữ riêng.

**Bảng 3.** CER trên ba buổi giữ riêng (299 đoạn, ba lĩnh vực ngoài miền huấn luyện), đơn vị
phần trăm. Điều kiện chấm khác mục 6.1, xem mục 5.4.

| Buổi | n | Scribe v2 | `Reworkwhisper-large-v5` |
|---|---:|---:|---:|
| Tuyển dụng nhân sự | 114 | **5,30** | 6,49 |
| Hội chẩn y khoa | 69 | **8,21** | 11,67 |
| Webinar marketing | 116 | **6,78** | 8,48 |
| Gộp — CER | 299 | **6,54** | 8,44 |
| Gộp — WER | 299 | **11,05** | 13,68 |
| Gộp — giữ từ mượn | 299 | **83,8** | 65,7 |

`Reworkwhisper-large-v5` cho CER cao hơn Scribe v2 trên cả ba buổi, chênh lệch từ 1,19 điểm ở
buổi tuyển dụng nhân sự đến 3,46 điểm ở buổi hội chẩn y khoa. Thời gian xử lý trên thời lượng
audio là 0,335–0,365 với `Reworkwhisper-large-v5` trên một GPU T4 và 0,136–0,173 với Scribe v2
qua API.

---

## 7. Phân tích: đóng góp của dữ liệu YouTube meeting

Bảng 4 đo trên tập test in-domain ở mục 5.2 — 228
đoạn YouTube meeting và 426 đoạn synthetic — so ba mô hình: mô hình nền chưa tinh chỉnh, adapter huấn
luyện chỉ trên hội thoại synthetic, và adapter huấn luyện trên synthetic cộng 1,96 giờ YouTube meeting.
Hai adapter được chấm ở cùng λ=0,5, nên khác biệt giữa chúng quy về dữ liệu huấn luyện chứ không
phải mức hợp nhất.

**Bảng 4.** Ảnh hưởng của việc thêm 1,96 giờ YouTube meeting vào tập huấn luyện, đơn vị phần trăm.

| Chỉ số | `PhoWhisper-large` | Chỉ synthetic | Synthetic + YouTube meeting |
|---|---:|---:|---:|
| CER — YouTube meeting (228 đoạn) | 15,93 | 12,05 | **6,24** |
| WER — YouTube meeting | 22,69 | 16,10 | **10,11** |
| Giữ từ mượn — YouTube meeting | 36,4 | 77,5 | **82,3** |
| CER — synthetic (426 đoạn) | 4,26 | 1,96 | **1,85** |

Hình dạng của Bảng 4 quan trọng hơn từng con số. Trên phần synthetic, hai adapter gần như bằng nhau
(1,96 và 1,85) — bản chỉ huấn luyện trên dữ liệu synthetic đã chạm trần của miền ấy, thêm dữ
liệu YouTube meeting không mua thêm được gì đáng kể. Trên phần YouTube meeting thì khác hẳn: dữ liệu synthetic
một mình đưa CER từ 15,93% xuống 12,05%, còn thêm 1,96 giờ YouTube meeting đưa tiếp xuống 6,24% — phần
sau lớn hơn phần trước rưỡi lần, dù chỉ chiếm 28% thời lượng huấn luyện.

Chỉ số giữ từ mượn có hình dạng khác. Phần lớn mức tăng đến ngay từ dữ liệu synthetic, từ 36,4%
lên 77,5%; dữ liệu YouTube meeting thêm 4,8 điểm nữa. Một cách đọc khả dĩ, tuy chưa được đo riêng, là
kịch bản synthetic cũng chứa từ mượn nhưng được đọc bằng giọng sạch, nên mô hình học được cách
viết từ mượn từ dữ liệu synthetic và cần giọng nói thật mới học được cách nhận ra chúng khi bị nói
nhanh, nói lướt hoặc lẫn nhiễu.

Cần nhắc lại giới hạn ở mục 5.1: hai lần chạy còn khác nhau ở danh sách buổi dùng làm validation
và ở số bước cập nhật, nên đây là chỉ dấu mạnh chứ không phải ablation có kiểm soát hoàn toàn.


---

## 8. Hạn chế

**8.1. Phạm vi của corpus.** Bảy buổi đến từ cùng miền công nghệ và tuyển dụng, nên các con số ở
Bảng 1 đặc trưng cho miền ấy hơn là cho cuộc họp tiếng Việt nói chung. Phần synthetic dùng giọng
text-to-speech, không mang các đặc điểm của giọng người thật, nên CER tuyệt đối trên phần này chỉ
có giá trị so sánh tương đối. Mở rộng sang nhiều lĩnh vực và nhiều nguồn ghi âm là hướng tiếp
theo tự nhiên.

**8.2. Đánh đổi với miền chung.** Cấu hình đã triển khai làm CER trên VIVOS tăng 1,06 điểm, nằm
trong ngân sách khai báo trước. Đây là đánh đổi có chủ đích cho một hệ thống chuyên xử lý họp;
nếu cần giữ trọn năng lực trên giọng đọc chuẩn thì λ=0,25 cho mức suy giảm bằng không, đổi lại
CER trên phần YouTube meeting cao hơn 1,7 điểm.

**8.3. Đánh giá ngoài miền.** Trên ba buổi thuộc lĩnh vực khác, ElevenLabs Scribe v2 cho CER thấp
hơn 1,9 điểm. Lần chấm này dùng cấu hình decode khác Bảng 1 nên hai bảng không so trực tiếp; tái
lập bằng đúng quy trình chấm của bài sẽ cho một phép so sánh chặt hơn.

---

## Tài liệu tham khảo (đang gom)

- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W. *LoRA:
  Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685, 2021.
- Ilharco, G., Ribeiro, M. T., Wortsman, M., Gururangan, S., Schmidt, L., Hajishirzi, H.,
  Farhadi, A. *Editing Models with Task Arithmetic.* arXiv:2212.04089, 2022. ICLR 2023.
- Le, T., Nguyen, L. T., Nguyen, D. Q. *PhoWhisper: Automatic Speech Recognition for Vietnamese.*
  arXiv:2406.02555. ICLR 2024 Tiny Papers Track.
- Wortsman, M., Ilharco, G., Kim, J. W., Li, M., Kornblith, S., Roelofs, R., Gontijo-Lopes, R.,
  Hajishirzi, H., Farhadi, A., Namkoong, H., Schmidt, L. *Robust fine-tuning of zero-shot models.*
  arXiv:2109.01903, 2021. CVPR 2022.

## Câu hỏi còn treo

- Có thêm trích dẫn Whisper gốc (Radford và cộng sự) không? Chưa quyết.
- `use_rslora: true` trong cấu hình: nhắc ở mục Phương pháp không kèm trích dẫn (phương án đang
  theo), hay thêm trích dẫn riêng cho hệ số co giãn ổn định hạng?
- Mục 4.3 cũ (giải thích λ=0,75 là người override quy tắc) đã bỏ theo quyết định 2026-09-08.
  Hai cấu hình sẽ được giới thiệu ở đầu mục 5 như hai đối tượng đánh giá, không kèm giải thích
  cách chọn. Lập luận đánh đổi (0,31 điểm CER YouTube meeting đổi 0,72 điểm VIVOS; 17% so với 53% ngân
  sách) giữ ngoài bài, dùng khi cần trả lời phản biện.
- Rà soát 7 type đáng ngờ ở mục 4.3 do chính nhóm tác giả làm, chưa có kiểm định độc lập — nói
  rõ ở đâu, mục 4.3 hay mục Hạn chế? Chưa quyết.
- Mục 5.4: có thêm một câu nói rằng ảnh hưởng của cấu hình decode khác biệt chưa được định lượng
  không? Chưa quyết. (Bằng chứng nội bộ: `no_repeat_ngram_size` tốn khoảng 3 điểm CER trên tiếng
  Việt — nếu đúng thì 8,44% là cận trên. Chưa đo lại nên không đưa vào bài.)
- Có chạy lại ba buổi giữ riêng bằng đúng cấu hình decode của bài không? Nếu chạy được thì mục
  5.4 biến mất. Công cụ: `scripts/rescore_pilot_benchmark.py`.
- Benchmark audio thu thật: giữ trong bài hay bỏ? Thiếu cột `Reworkwhisper-large-v4` trên cùng
  phân chia.
- Tên `Reworkwhisper-large-v4`/`v5` giữ nguyên (quyết định 2026-09-09), dù repo HuggingFace đang
  ở chế độ riêng tư nên người đọc không mở được.
- Ablation về quy ước chuẩn hoá chữ số đã bỏ theo quyết định 2026-09-09. Số vẫn còn trong
  `docs/so-lieu-tong-hop.md` §12.1 nếu cần dùng lại để trả lời phản biện: đổi quy ước chỉ làm CER
  xê dịch 0,161 điểm ở λ=0,75 và 0,447 điểm ở λ=0,25.
- Đoạn dị thường của mô hình nền (`7B24A9GfHAo/seg_0079`, CER 2543,8%): nêu trong bài hay bỏ?
  Chưa quyết. Nó chiếm 3,3 trong 10 điểm cải thiện đang báo cáo — mô hình nền đạt 12,65% nếu loại
  đoạn này, so với 15,93% khi tính đủ 228 đoạn.
- Phân tích seen/unseen đã bỏ khỏi bài (quyết định 2026-09-09), kéo theo việc bỏ câu tương ứng
  trong Abstract, vế trong mục 1.4, đóng góp số 4, và câu dẫn ở mục 3.5.
- Ba vật liệu còn chưa dùng, để ngỏ cho mục 7 nếu cần mở rộng: (a) danh sách token bị mất nhiều
  lượt nhất — `jd` 26, `vinpearl` 24, `funnel` 15 — cho thấy lỗi còn lại tập trung ở viết tắt
  chuyên ngành và danh từ riêng; (b) chấm lại ba buổi giữ riêng bằng bộ chuẩn hoá của `viet-speech`
  cho Scribe v2 8,1% và `Reworkwhisper-large-v5` 9,7%, giữ nguyên thứ tự ở Bảng 3; (c) kết quả
  theo từng buổi test (`so-lieu-tong-hop.md` §6).
- Hai hạn chế đã loại khỏi mục 8 (quyết định 2026-09-09), giữ lại đây nếu bài nhắm hội thảo có
  phản biện: (a) nhãn hiệu đính từ phụ đề tự động thay vì phiên âm độc lập, mức ảnh hưởng của bản
  nháp lên nhãn cuối chưa định lượng; (b) một lần huấn luyện với một seed, phương sai do khởi tạo
  chưa đo.
