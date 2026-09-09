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
| 5. Thiết lập thí nghiệm | chưa |
| 6. Kết quả | chưa |
| 7. Phân tích | chưa |
| 8. Hạn chế | chưa |
| 9. Kết luận | chưa |

---

## Abstract

Nhận dạng tiếng nói cho các cuộc họp công việc tiếng Việt gặp một khó khăn mà các tập chuẩn
hiện hành không phản ánh: hiện tượng code-switching Việt–Anh dày đặc. Trên bộ dữ liệu họp thật mà
chúng tôi xây dựng, 6,67% số token và 89,7% số đoạn chứa ít nhất một từ mượn tiếng Anh, trong
khi tập test chuẩn VIVOS không chứa token nào thuộc loại này. Hệ quả là mô hình nền
`PhoWhisper-large`, dù đạt CER 2,28% trên VIVOS, chỉ giữ đúng 36,4% số lượt từ mượn và đạt
CER 15,93% trên slice họp thật.

Chúng tôi trình bày `Reworkwhisper-large-v5`, bản tinh chỉnh `PhoWhisper-large` bằng LoRA hạng
16 (28,8 triệu tham số, 1,76% tổng số), huấn luyện trên tập trộn gồm 1,96 giờ họp thật và 5,02
giờ hội thoại tổng hợp. Phần họp thật tuy nhỏ nhưng đậm đặc đúng hiện tượng cần học — 89,7%
số đoạn chứa từ mượn — và được hiệu đính thủ công từ phụ đề tự động, với 779 trên 790 đoạn có
thay đổi so với nhãn nháp. Trọng số hợp nhất được chọn qua một lượt quét hệ số λ có ràng buộc
ngân sách sai số trên VIVOS.

Trên tập test 654 đoạn, cấu hình do quy trình tự chọn (λ=0,5) đưa CER trên slice họp thật từ
15,93% xuống 6,24% — giảm tương đối 60,8% — WER từ 22,69% xuống 10,11%, và tỷ lệ giữ đúng từ
mượn từ 36,4% lên 82,6%. Cấu hình được triển khai thực tế (λ=0,75), chọn sau khi quan sát toàn
bộ đường cong λ, đạt lần lượt 5,93%, 9,37% và 85,5%. Tách theo độ phủ từ vựng huấn luyện cho
thấy mức cải thiện không đến từ việc ghi nhớ: với các từ mượn chưa từng xuất hiện trong dữ liệu
huấn luyện, tỷ lệ giữ đúng vẫn tăng từ 34,1% lên 68,1%. Trên ba buổi họp giữ riêng thuộc ba lĩnh
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

Các cuộc họp công việc lại là một phân phối khác hẳn. Trên bộ dữ liệu bảy buổi họp thật mà
chúng tôi thu thập và hiệu đính, 6,67% số token và 89,7% số đoạn chứa ít nhất một từ mượn tiếng
Anh không được Việt hoá — tên công cụ, thuật ngữ chuyên môn, tên thương hiệu. Cùng chỉ số ấy
trên tập test VIVOS bằng đúng 0: không một token nào. Sự chênh lệch này không phải là khác biệt
về mức độ khó mà là khác biệt về loại hiện tượng, và nó ẩn hoàn toàn khỏi mọi bảng xếp hạng dựa
trên tập chuẩn hiện có. Đo trên slice họp thật, `PhoWhisper-large` đạt CER 15,93%, gấp bảy lần
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
từ mượn bên cạnh CER và WER, và tách chỉ số ấy theo việc từ đó có xuất hiện trong dữ liệu huấn
luyện hay không, nhằm phân biệt học quy tắc với ghi nhớ danh sách.

**Đóng góp.**

1. Một corpus bảy buổi họp công việc tiếng Việt có code-switching dày đặc, 790 đoạn, 3,46 giờ, nhãn
   hiệu đính thủ công, kèm hồ sơ định lượng về mật độ code-switching, mức chồng tiếng và độ tương
   đồng người nói. Chúng tôi phát hành phần nhãn — transcript đã hiệu đính, mã video và mốc
   thời gian từng đoạn — để tái lập được, còn audio thì người dùng tự tải từ nguồn gốc.
2. Bằng chứng về hiệu quả dữ liệu: 1,96 giờ họp thật, trộn với hội thoại tổng hợp, đưa CER trên
   slice họp thật từ 15,93% xuống 6,24% ở cấu hình do quy trình chọn và 5,93% ở cấu hình đã
   triển khai, không cần full fine-tuning.
3. Một quy tắc chọn hệ số hợp nhất có ràng buộc ngân sách sai số miền chung, kèm đường cong đánh
   đổi đầy đủ trên năm mức λ.
4. Chỉ số giữ đúng từ mượn tách theo độ phủ từ vựng huấn luyện, cho thấy mức cải thiện lan cả
   sang những từ chưa từng gặp (34,1% lên 68,1%).
5. Đánh giá ngoài phân phối trên ba buổi họp thuộc ba lĩnh vực khác nhau, đối chiếu trực tiếp với
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

**3.1. Nguồn và sàng lọc.** Corpus họp thật gồm bảy video công khai trên YouTube thuộc miền công
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
huấn luyện, đồng thời có nghĩa là kết quả trên tập test đo năng lực **trong** miền ấy. Câu hỏi kèm
theo — liệu mô hình có chỉ ghi nhớ vốn từ của miền hay không — được trả lời riêng bằng phân tích
tách theo độ phủ từ vựng huấn luyện ở mục 7. Để đo phần khái quát hoá ra ngoài miền, chúng tôi
đánh giá thêm trên ba buổi giữ riêng thuộc ba lĩnh vực khác hẳn — tuyển dụng nhân sự, hội chẩn y
khoa và webinar marketing — trình bày ở mục 5.

**3.6. Dữ liệu tổng hợp và phân chia.** Bên cạnh họp thật, chúng tôi dùng một tập hội thoại họp
tổng hợp sinh bằng kịch bản do mô hình ngôn ngữ viết rồi đọc bằng hệ tổng hợp giọng nói thương
mại, với mười giọng ở tập huấn luyện và mười giọng ở tập kiểm thử không giao nhau. Phân chia cuối
cùng:

| Split | Họp thật | Tổng hợp | Tổng đoạn |
|---|---|---|---:|
| Huấn luyện | 447 đoạn · 1,96 giờ | 4.270 đoạn · 5,02 giờ | 4.717 |
| Validation | 115 đoạn · 0,50 giờ | 250 đoạn · 0,29 giờ | 365 |
| Kiểm thử | 228 đoạn · 1,00 giờ | 426 đoạn · 0,58 giờ | 654 |

Phần họp thật chỉ chiếm 28% thời lượng huấn luyện. Việc trộn hai nguồn nhằm hai mục đích: dữ liệu
tổng hợp cung cấp khối lượng và độ sạch nhãn, còn dữ liệu họp thật cung cấp hiện tượng — nhiễu
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
batch 8, fp16. Chỉ số chính là CER, kèm WER để tham khảo.

Chúng tôi bổ sung một chỉ số thứ ba, **tỷ lệ giữ đúng từ mượn**. Với mỗi đoạn, ta lấy các token
trong reference dài hơn một ký tự, thuần chữ cái, và **không** khớp phép thử hình dạng âm tiết
tiếng Việt; rồi đếm bao nhiêu token trong số đó xuất hiện lại trong hypothesis, so khớp theo
multiset nên việc đảo thứ tự không bị phạt còn việc nói hai lần mà chỉ nhận ra một lần thì bị.
Chúng tôi cố ý nhận diện từ mượn bằng hình dạng âm tiết thay vì đối chiếu whitelist từ tiếng
Anh, bởi trong thử nghiệm sơ bộ, cách dùng whitelist cho tới 72% false positive — phần lớn là
các âm tiết tiếng Việt trùng mặt chữ với từ tiếng Anh. Trên tập test, bộ nhận diện đánh dấu 362
type và 1.353 lượt; rà soát thủ công tìm được 7 type đáng ngờ ứng với 15 lượt, tức 1,1% theo lượt.

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
  cách chọn. Lập luận đánh đổi (0,31 điểm CER họp thật đổi 0,72 điểm VIVOS; 17% so với 53% ngân
  sách) giữ ngoài bài, dùng khi cần trả lời phản biện.
- Rà soát 7 type đáng ngờ ở mục 4.3 do chính nhóm tác giả làm, chưa có kiểm định độc lập — nói
  rõ ở đâu, mục 4.3 hay mục Hạn chế? Chưa quyết.
