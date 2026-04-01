# Script thuyết trình tiếng Việt cho `slide/apr1st`

## Slide 1. Tiêu đề

Nói như sau:

"Em xin trình bày đề tài của em là retrieval-augmented reward modeling cho bài toán thao tác robot dài hạn trong mô phỏng. Ý tưởng chính là không để robot học chỉ từ sparse reward, mà sẽ truy xuất thêm bằng chứng liên quan từ memory bank gồm instruction, stage description và demonstration, sau đó dùng context truy xuất được để ước lượng progress và reward tốt hơn. Hôm nay em sẽ trình bày ngắn gọn phần proposal và cập nhật kết quả implementation của tuần này."

## Slide 2. Outline

Nói như sau:

"Bố cục bài trình bày gồm sáu phần. Đầu tiên là bài toán và động lực. Thứ hai là cách em framing đề tài qua mục tiêu, câu hỏi nghiên cứu và giả thuyết. Thứ ba là phương pháp. Thứ tư là kế hoạch thực nghiệm. Thứ năm là roadmap triển khai. Cuối cùng là cập nhật tiến độ và kết quả thực hiện trong tuần này."

## Slide 3. Problem Setting

Nói như sau:

"Điểm xuất phát của đề tài là bài toán long-horizon manipulation thường có sparse reward. Nếu reward chỉ xuất hiện ở cuối task, policy rất khó học vì vấn đề credit assignment và exploration. Một cách xử lý phổ biến là tự thiết kế dense reward bằng tay, nhưng cách đó tốn công, khó scale sang task mới, và dễ vỡ khi đổi object hoặc layout."

"Ở đây em muốn đổi góc nhìn: thay vì cố viết reward bằng tay, mình dùng retrieval để lấy thêm bằng chứng liên quan từ demonstration và stage description, rồi dùng phần context đó để ước lượng progress dày hơn. Nghĩa là trọng tâm trước tiên không phải policy, mà là reward quality."

## Slide 4. Why Retrieval for Reward Modeling

Nói như sau:

"Lý do em chọn hướng này là vì nó nằm ở giao điểm của ba nhánh đã có nền tảng. Một là embodied retrieval, tức agent biết lấy external context khi cần. Hai là retrieval trên demonstration trong robotics. Ba là reward modeling cho robot. Các nhánh này đều đã có công trình gần đây, nên đề tài của em không đi từ con số 0, mà đứng trên một nền research tương đối rõ."

"Tuy nhiên em giới hạn phạm vi rất hẹp. Em làm trong simulation trước, không claim real robot, không train VLA từ đầu, và cũng không xây một embodied RAG tổng quát. Câu hỏi em muốn kiểm chứng là hẹp hơn nhiều: retrieval có giúp reward quality cho long-horizon manipulation hay không."

## Slide 5. Objectives and Research Questions

Nói như sau:

"Mục tiêu của đề tài có ba lớp. Lớp thứ nhất là build được một pipeline retrieval-conditioned reward modeling trong simulation. Lớp thứ hai là phải có prototype ngắn hạn để chứng minh đường đi memory bank, retrieval, và progress supervision là khả thi. Lớp thứ ba là nghiên cứu: đo xem retrieval có thực sự cải thiện reward estimation và downstream learning hay không."

"Ba câu hỏi nghiên cứu chính là: retrieved context có giúp dự đoán progress tốt hơn không; reward có retrieval có giúp policy learning và OOD robustness tốt hơn không; và retrieval granularity nào là phù hợp nhất, episode, stage hay sub-trajectory."

## Slide 6. Hypotheses and Success Criteria

Nói như sau:

"Giả thuyết thứ nhất của em là retrieval đa modal sẽ giúp phân biệt state trung gian có ý nghĩa với state nhiễu nhưng nhìn có vẻ đúng. Giả thuyết thứ hai là retrieval theo stage hoặc sub-trajectory sẽ tốt hơn retrieval ở mức full episode vì ít loãng thông tin hơn. Giả thuyết thứ ba là lợi ích của retrieval sẽ rõ hơn trong các task dài hạn và các split OOD."

"Về tiêu chí thành công, em không đòi hỏi ngay lập tức phải có policy mạnh. Mốc đầu tiên là progress estimation phải tốt hơn rõ ràng so với no-context và random-context. Nếu bước đó đứng vững, khi đó mới có cơ sở sang reward head và IQL."

## Slide 7. Method Overview

Nói như sau:

"Phương pháp em hình dung là reward tại thời điểm t không chỉ phụ thuộc vào quan sát hiện tại và task condition, mà còn phụ thuộc vào context được retrieve từ memory bank. Memory bank sẽ chứa task instruction, stage description, các clip demonstration hoặc sub-trajectory, cùng với progress label và metadata."

"Từ quan sát hiện tại, retriever sẽ lấy top-k bằng chứng liên quan nhất. Ý tưởng ở đây là reward model không phải dự đoán trong khoảng không, mà được cung cấp thêm reference cases gần với state hiện tại."

## Slide 8. Supervision and Training Path

Nói như sau:

"Về supervision, em ưu tiên simulator-derived labels thay vì human feedback tự do. Target chính là progress regression dựa trên stage progress do simulator hoặc cấu trúc task suy ra. Target phụ là pairwise ranking giữa state sớm và muộn, hoặc giữa trajectory thành công và thất bại."

"Về lộ trình train, em không nhảy ngay vào reward head đầy đủ. Prototype đầu tiên dùng retrieval-based progress proxy để xác nhận rằng retrieval thực sự mang tín hiệu hữu ích. Khi proxy này ổn, em mới chuyển sang reward head retrieval-conditioned, và cuối cùng mới nối nó vào offline RL với IQL."

## Slide 9. Benchmarks and OOD Protocol

Nói như sau:

"Về benchmark, hướng chính của proposal là LIBERO vì nó phù hợp với long-horizon manipulation và transfer setting. ManiSkill được giữ như một benchmark phụ nếu sau này cần thêm throughput hoặc task diversity. Các baseline cốt lõi em muốn giữ là sparse reward, no-context, random-context, và retrieved-context."

"Về protocol, em muốn nói rõ OOD là gì chứ không để mơ hồ. Em tách thành OOD-object và OOD-layout. Tức là hoặc object instance bị giữ ra, hoặc cấu hình scene và pose bị giữ ra. Cách tách như vậy giúp đánh giá lợi ích của retrieval một cách có cấu trúc hơn."

## Slide 10. Ablations and Evaluation Metrics

Nói như sau:

"Proposal đã khóa các ablation quan trọng. Một là no context, random context và retrieved context. Hai là text-only, video-only và multimodal retrieval. Ba là retrieval granularity ở mức episode, stage hay sub-trajectory. Bốn là objective chỉ progress regression so với progress cộng ranking."

"Metric đánh giá cũng đi theo hai tầng. Ở tầng reward quality sẽ có regression error, correlation và ranking accuracy. Ở tầng downstream policy sẽ có success rate, return và độ bền trên các split OOD. Nghĩa là em muốn đánh giá reward trước, rồi mới đánh giá policy."

## Slide 11. Expected Contributions

Nói như sau:

"Đóng góp kỳ vọng của đề tài không phải là xây một hệ VLA lớn, mà là đưa ra một formulation rõ ràng cho retrieval-conditioned reward modeling, một cách tạo supervision rõ ràng từ simulator, và một protocol đánh giá reward quality trước khi bàn sang policy quality. Ngoài ra em cũng muốn nghiên cứu retrieval granularity một cách có hệ thống trong bài toán thao tác dài hạn."

## Slide 12. Execution Roadmap

Nói như sau:

"Roadmap được chia thành P0 đến P5. P0 là dựng prototype một task để chứng minh pipeline retrieval-conditioned progress estimation. P1 là khóa benchmark, split và supervision labels. P2 là build memory bank hoàn chỉnh. P3 là train reward head. P4 là nối reward học được vào IQL. P5 là chạy ablation, phân tích OOD robustness và hoàn thiện phần viết."

"Điểm quan trọng là tuần này em đang ở P0. Mục tiêu của P0 không phải là có kết quả RL, mà là giảm rủi ro implementation cho các phase nghiên cứu tiếp theo."

## Slide 13. This Week's Progress: Implementation

Nói như sau:

"Trong tuần này, phần implementation đã đạt được một prototype chạy được. Repo đã được pin về Python 3.8.20 và thống nhất workflow bằng uv. Em đã tạo file demo chính có hai backend: một backend replay2d để demo ổn định và một backend LIBERO để chạm vào benchmark thật."

"Trên pipeline đó, em đã dựng được mini memory bank, top-k retrieval, retrieval-weighted progress proxy, và xuất ra các artifact trực quan như demo snapshot, retrieval examples, progress plot và GIF. Ý em muốn nhấn mạnh ở slide này là proposal không còn dừng ở ý tưởng, mà đã có một execution path chạy được từ môi trường tới artifact báo cáo."

## Slide 14. This Week's Progress: Metrics

Nói như sau:

"Về kết quả, retrieved context hiện tại cho thấy hiệu quả tốt hơn rõ ràng so với no-context và random-context. Ở backend replay2d, MAE của retrieved context là 0.0797 và correlation là 0.9735. Ở backend LIBERO, MAE là 0.0973 và correlation là 0.9475."

"Điểm em muốn chốt ở slide số liệu là: dù đây mới là progress proxy, retrieved context đã thắng rõ ràng các baseline yếu hơn. Điều này cho thấy retrieval đang mang tín hiệu hữu ích chứ không chỉ thêm nhiễu. Bước tiếp theo là thay progress proxy bằng retrieval-conditioned reward head, rồi mới nối sang IQL."

## Slide 15. This Week's Progress: Progress Plot

Nói như sau:

"Slide này là minh họa trực quan cho phần số liệu vừa rồi. Đường progress dự đoán khi dùng retrieved context bám oracle tốt hơn hẳn so với no-context và random-context. Ý nghĩa của hình này là pipeline retrieval sang progress estimation không chỉ đúng về mặt intuition, mà đã thấy được ngay trên artifact trực quan."

## Slide 16. Takeaway

Nói như sau:

"Thông điệp cuối cùng là đề tài này được framing khá hẹp và có thể kiểm chứng được. Em không claim một embodied RAG tổng quát, mà tập trung vào việc dùng retrieval để cải thiện reward quality cho long-horizon manipulation trong simulation. Tuần này em đã có bằng chứng cho pha prototype. Nếu pha tiếp theo train được reward head và nối được vào IQL, thì đề tài sẽ có một pipeline nghiên cứu rõ ràng, khả thi và đúng hướng với proposal."
