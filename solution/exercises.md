# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Chạy `call_openai` với prompt trên ở cả 4 mức, model đều chọn kể về hang Sơn
> Đoòng (Phong Nha - Kẻ Bàng) chứ không đổi chủ đề, nhưng chi tiết thì vênh
> nhau ngay cả ở temperature = 0.0: bản 0.0 và 1.0 nói hang "dài hơn 5 km" và
> không nêu tên ai phát hiện, còn bản 0.5 và 1.5 lại nói "dài khoảng 9 km".
> Riêng bản 1.5 thêm hẳn tên người phát hiện (Hồ Khanh, năm 1991) và tên tổ
> chức khảo sát (Hiệp hội Hang động Hoàng gia Anh), câu cũng dài và bay bổng
> hơn hẳn 3 bản còn lại (thêm cả chi tiết "có cả đám mây riêng trong hang").
> Tóm lại temperature không chỉ đổi văn phong (ngắn gọn khô khan ở 0.0 dần
> thành kể chuyện chi tiết ở 1.5) mà số liệu cụ thể (số km) cũng không cố định
> theo temperature, kể cả để 0.0 cũng không đảm bảo model trả lời giống nhau
> giữa các lần gọi hay đúng tuyệt đối.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Khoảng 0.2–0.3. Chatbot hỗ trợ khách hàng cần trả lời nhất quán, đúng chính
> sách công ty và ít rủi ro bịa thông tin — quan sát ở Câu 1.1 cho thấy
> temperature cao dễ sinh chi tiết sai lệch, điều không chấp nhận được khi
> khách hàng dựa vào câu trả lời để ra quyết định.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Tính ra (10.000 × 3 × 350 = 10.500.000 token output/ngày): GPT-4o tốn khoảng
> $105/ngày, GPT-4o-mini khoảng $6.30/ngày — GPT-4o đắt hơn **16.67 lần**.
> Dùng GPT-4o khi task cần suy luận phức tạp, độ chính xác cao (tư vấn pháp
> lý, debug code); dùng mini cho các tác vụ đơn giản, lặp lại như trả lời FAQ,
> phân loại ý định — nơi chênh lệch chất lượng không đáng kể so với chênh
> lệch chi phí ở quy mô lớn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Bản "giáo viên tiểu học" ngắn (118 từ), ví von blockchain như cuốn sổ ghi
> điểm cả lớp cùng xem, không dùng thuật ngữ nào. Bản "chuyên gia tài chính"
> dài hơn hẳn (191 từ, bị cắt giữa chừng vì max_tokens=256), chia mục rõ ràng
> và dùng liên tiếp các thuật ngữ (distributed ledger, hash mật mã,
> decentralization, immutability...). Cùng một câu hỏi, chỉ đổi system prompt
> là model đổi hẳn từ vựng, cách trình bày (văn xuôi kể chuyện so với liệt kê
> có tiêu đề) và độ dài phản hồi.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Đoạn văn 131 từ (giới thiệu về Việt Nam): tiktoken đếm 161 token, ước lượng
> từ/0.75 ra 175 token — chênh khoảng 8.7%, ước lượng cao hơn thực tế. Phép so
> sánh đúng nghĩa phải viết cùng một ý bằng hai ngôn ngữ: cùng đoạn đó dịch
> sang tiếng Anh dài tương đương về ký tự (576 so với 560 ký tự) nhưng bản
> tiếng Việt tốn 161 token còn bản tiếng Anh chỉ 103 token, tức nhiều hơn
> khoảng 56%. Lý do: bộ mã hoá BPE của GPT học chủ yếu trên văn bản tiếng
> Anh nên từ tiếng Anh hay gộp gọn thành 1 token, còn các ký tự có dấu thanh
> tiếng Việt (ư, ệ, ề...) không khớp token phổ biến nên bị tách vụn thành
> nhiều token con hơn cho cùng một lượng chữ.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi có người đang chờ trực tiếp trên màn hình —
> chatbot, trợ lý CLI — vì ký tự đầu tiên xuất hiện sau vài trăm mili giây
> thay vì phải chờ hết cả đoạn văn, giúp người dùng cảm thấy hệ thống phản
> hồi nhanh dù tổng thời gian không đổi. Non-streaming phù hợp hơn khi kết
> quả cần được xử lý tiếp bằng code trước khi hiển thị — ví dụ parse JSON,
> chạy qua bộ kiểm duyệt nội dung, hay ghi log — vì lúc đó cần response trọn
> vẹn mới xử lý được, streaming từng mảnh không có lợi.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Với delay cố định, hàng nghìn client bị lỗi cùng lúc sẽ retry đồng loạt sau
> đúng 1 giây, tạo một đợt sóng request mới đập vào server vừa mới hồi phục —
> lặp lại chu kỳ quá tải, gọi là "thundering herd". Exponential backoff giãn
> thời điểm retry ra xa dần theo từng client (đặc biệt khi cộng thêm jitter
> ngẫu nhiên), nên request rải đều theo thời gian thay vì dồn cục, cho server
> đủ thời gian hồi phục thật sự.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona: "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng
> tiếng Việt." Từ "ngắn gọn" giới hạn độ dài output — vừa giữ chi phí thấp
> (output là phần đắt nhất theo bảng giá), vừa hợp với giao diện CLI dạng
> stream nơi câu trả lời quá dài gây khó theo dõi. Từ "bằng tiếng Việt" ép
> model trả lời nhất quán một ngôn ngữ dù người dùng gõ tiếng Anh hay dùng
> teencode, tránh trợ lý đổi ngôn ngữ tuỳ hứng giữa các lượt.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất: history chỉ giữ 3 lượt gần nhất và mất hoàn toàn khi thoát
> chương trình — trợ lý không "nhớ" gì giữa các phiên chat. Cải thiện cụ thể:
> lưu history xuống một file JSON theo session (ví dụ `sessions/<id>.json`)
> sau mỗi lượt, và nạp lại 3 lượt cuối từ file đó khi `run_assistant` khởi
> động thay vì bắt đầu với `history = []`. Việc này giữ nguyên cơ chế cắt
> `history[-6:]` để chi phí mỗi lượt không đổi, chỉ thêm bước đọc/ghi file.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
