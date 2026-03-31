# Milestones

## Hoàn thành

- `2026-03-31`: chốt hướng đề tài chính là `retrieval-augmented reward modeling` cho long-horizon robot manipulation trong mô phỏng.
- `2026-03-31`: hoàn thành proposal chi tiết trong `pdf/`, bao gồm đánh giá tính mới, tính khả thi, hướng phát triển hiện có, thiết kế thực nghiệm, và ước lượng tài nguyên tính toán.

## Mốc tiếp theo

- `M1`: chốt benchmark chính giữa `LIBERO` và `ManiSkill`, đồng thời xác định tập task long-horizon và OOD split dùng cho đánh giá.
- `M2`: dựng memory bank đa phương thức từ instruction, manual, và demonstration clip/stage.
- `M3`: huấn luyện reward model có điều kiện theo ngữ cảnh retrieval và kiểm định chất lượng reward bằng correlation/ranking metrics.
- `M4`: nối reward model vào pipeline offline RL hoặc policy selection để đo `success rate`, `sample efficiency`, và `OOD generalization`.
- `M5`: hoàn tất ablation về granularity truy xuất, nguồn ngữ cảnh, và độ trễ suy luận; sau đó mở rộng sang hướng VLA hoặc world model nếu còn tài nguyên.

## Ghi chú

- Repo hiện chưa có cơ chế `VERSION` riêng ngoài version trong `pyproject.toml`; chưa thực hiện version bump ở task này.
- Chưa có mindmap trong repo để cập nhật ở bước proposal.
