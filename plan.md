# Active Plan

## Mục tiêu active

Chuyển proposal hiện tại thành một chương trình nghiên cứu simulation-first có thể triển khai theo từng pha, không phụ thuộc robot thật trong giai đoạn đầu.

## Pha 1: Thiết kế benchmark

- Chọn benchmark chính: ưu tiên `LIBERO` cho long-horizon transfer; dùng `ManiSkill` khi cần mở rộng sang đánh giá throughput hoặc môi trường đa dạng hơn.
- Xác định tập nhiệm vụ gồm tối thiểu một nhóm in-distribution và một nhóm OOD theo object, layout, hoặc instruction variation.
- Chốt baseline: `sparse reward`, `reward model không retrieval`, `retrieval-only guidance`, và `retrieval + reward model`.

## Pha 2: Memory bank và retrieval

- Thiết kế schema lưu `instruction`, `manual text`, `stage description`, và `video clip/sub-trajectory`.
- Ưu tiên truy xuất theo `stage` hoặc `sub-trajectory` thay vì toàn episode để giảm nhiễu.
- Thiết lập chiến lược scoring đa tiêu chí: tương đồng quan sát hiện tại, mục tiêu task, và tiến độ stage.

## Pha 3: Reward modeling

- Dùng encoder pretrained ở chế độ frozen hoặc fine-tune nhẹ; không huấn luyện foundation model từ đầu.
- Huấn luyện reward head hoặc ranking head dự đoán `progress score`, `pairwise preference`, hoặc `stage completion`.
- Theo dõi cả chất lượng reward và độ tin cậy của retrieval để tránh khuếch đại ngữ cảnh sai.

## Pha 4: Policy learning và đánh giá

- Nối reward model vào offline RL hoặc policy selection.
- Báo cáo ít nhất các chỉ số `success rate`, `normalized return`, `sample efficiency`, `reward-quality correlation`, và `OOD robustness`.
- Chuẩn bị phần future work cho hướng VLA/world model sau khi pipeline reward hoạt động ổn định.
