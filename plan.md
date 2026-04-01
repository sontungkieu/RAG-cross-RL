# Active Plan

## Mục tiêu active

Chuyển proposal hiện tại thành một chương trình nghiên cứu simulation-first có thể triển khai theo từng pha, không phụ thuộc robot thật trong giai đoạn đầu.

## Immediate Sprint: 5-hour Demo

### Objective

- Dựng một prototype demo cho `retrieval-conditioned progress estimation` từ simulation replay.
- Chốt memory bank mini, retrieval top-k, progress proxy, và artifact trực quan đủ để báo nhanh.
- Giữ hướng downstream `IQL` là mục tiêu chính của research plan, nhưng không ép vào deliverable đầu tiên.

### Scope Lock

- Chỉ làm `1` task và `4` stage pseudo labels cho demo đầu tiên.
- Không làm full reward head, không chạy `IQL`, không benchmark rộng, không OOD đầy đủ.
- `manual text` vẫn là extension; demo M0 chỉ bám `instruction + stage text + frames`.

### Deliverables

- `demo/demo_v0.py`
- `data/demo_memory_bank.jsonl`
- `data/demo_index.pkl`
- `artifacts/demo_case.png`
- `artifacts/retrieval_examples.png`
- `artifacts/progress_plot.png`
- `artifacts/demo.gif`
- `docs/demo_v0_notes.md`

### Fallback Strategy

- `auto` backend thử `LIBERO` trước nếu group optional đã được sync và runtime dùng được.
- Nếu `LIBERO` không import hoặc không khởi tạo env ổn trong gate đầu sprint, chuyển thẳng sang `replay2d`.
- `replay2d` là đường hoàn tất mặc định của sprint để bảo đảm demo luôn chạy được trong repo.

## Next 7-10 Days

- Mở rộng nhánh `LIBERO` hiện đã chạy được trong workflow `uv` từ random rollout ngắn sang rollout có kiểm soát hơn, stage mapping rõ hơn, và query/support split ổn định hơn.
- Khóa benchmark chính, protocol `ID / OOD-object / OOD-layout`, và task subset đầu tiên dùng cho reward modeling thật.
- Map simulator predicates hoặc task progress oracle sang stage-progress labels nhất quán hơn temporal pseudo labels.
- Mở rộng memory bank từ frame-level demo sang stage chunk hoặc sub-trajectory retrieval.

## Medium-Term Research Phases

### Pha 1: Thiết kế benchmark

- Chọn benchmark chính: ưu tiên `LIBERO` cho long-horizon transfer; dùng `ManiSkill` khi cần mở rộng sang đánh giá throughput hoặc môi trường đa dạng hơn.
- Xác định rõ `ID`, `OOD-object`, và `OOD-layout` split; chỉ thêm `OOD-task-composition` nếu benchmark hỗ trợ gọn.
- Chốt baseline: `sparse reward + IQL`, `reward model không retrieval bank`, `reward model với random context`, và `retrieval + reward model`.
- Chốt nguồn pseudo-label cho reward từ simulator predicates hoặc task progress oracle.

### Pha 2: Memory bank và retrieval

- Thiết kế schema lưu `task instruction`, `stage description`, `video clip/sub-trajectory`, và metadata task.
- Chỉ dùng `manual text` như một extension khi benchmark thật sự cung cấp text hữu ích.
- Ưu tiên truy xuất theo `stage` hoặc `sub-trajectory` thay vì toàn episode để giảm nhiễu.
- Thiết lập chiến lược scoring đa tiêu chí: tương đồng quan sát hiện tại, mục tiêu task, và tiến độ stage.

### Pha 3: Reward modeling

- Dùng encoder pretrained ở chế độ frozen hoặc fine-tune nhẹ; không huấn luyện foundation model từ đầu.
- Huấn luyện reward head với mục tiêu chính là `progress regression` từ stage-aware pseudo labels; `pairwise ranking` là loss phụ trợ.
- Đo chất lượng reward độc lập bằng correlation với oracle progress, pairwise accuracy, và stage-level metrics trước khi chạy downstream RL.
- Theo dõi cả chất lượng reward và độ tin cậy của retrieval để tránh khuếch đại ngữ cảnh sai.

### Pha 4: Policy learning và đánh giá

- Nối reward model vào offline RL với `IQL` làm thiết lập chính; policy selection chỉ là phân tích phụ nếu cần.
- Báo cáo ít nhất các chỉ số `success rate`, `normalized return`, `sample efficiency`, `reward-quality correlation`, `pairwise accuracy`, và `OOD robustness`.
- Chuẩn bị phần future work cho hướng VLA/world model sau khi pipeline reward hoạt động ổn định.
