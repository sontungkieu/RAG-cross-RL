# Milestones

## Hoàn thành

- `2026-03-31`: chốt hướng đề tài chính là `retrieval-augmented reward modeling` cho long-horizon robot manipulation trong mô phỏng.
- `2026-03-31`: hoàn thành proposal chi tiết trong `pdf/`, bao gồm đánh giá tính mới, tính khả thi, hướng phát triển hiện có, thiết kế thực nghiệm, và ước lượng tài nguyên tính toán.
- `2026-03-31`: refine proposal theo phản biện reviewer, chốt rõ reward supervision, OOD protocol, retrieval sources, baseline, và downstream recipe với `IQL`.
- `2026-04-01`: hoàn thành `M0` prototype simulation demo cho `retrieval-conditioned progress estimation`, gồm memory bank mini, retrieval top-k, progress proxy, và artifact trực quan (`demo_case`, `retrieval_examples`, `progress_plot`, `demo.gif`).

## Mốc tiếp theo

- `M0 follow-up`: mở rộng nhánh `LIBERO` hiện đã chạy được trong workflow `uv` từ random rollout ngắn sang rollout benchmark gọn có stage mapping rõ ràng hơn.
- `M1`: chốt benchmark chính giữa `LIBERO` và `ManiSkill`, đồng thời xác định rõ `ID`, `OOD-object`, và `OOD-layout` split dùng cho đánh giá.
- `M2`: dựng memory bank từ task instruction, stage descriptions, và demonstration clip/stage; `manual text` chỉ là extension nếu benchmark hỗ trợ.
- `M3`: huấn luyện reward model có điều kiện theo ngữ cảnh retrieval với progress pseudo labels từ simulator và pairwise ordering phụ trợ.
- `M4`: nối reward model vào pipeline offline RL với `IQL` để đo `success rate`, `sample efficiency`, và `OOD generalization`.
- `M5`: hoàn tất ablation về granularity truy xuất, nguồn ngữ cảnh, và độ trễ suy luận; sau đó mở rộng sang hướng VLA hoặc world model nếu còn tài nguyên.

## Ghi chú

- Repo hiện chưa có cơ chế `VERSION` riêng ngoài version trong `pyproject.toml`; chưa thực hiện version bump ở task này.
- Chưa có mindmap trong repo để cập nhật ở bước proposal.
- `questions.md/questions.txt`: `N/A`.
- `temp/`, `old_plan/`: `N/A` ở task này.
