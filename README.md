# RAG-cross-RL

Workspace này hiện được dùng để soạn proposal nghiên cứu cho hướng `retrieval-augmented reward modeling` trong robot manipulation mô phỏng.
Phiên bản proposal hiện tại đã chốt rõ:

- nguồn supervision cho reward từ simulator/task-derived pseudo labels;
- OOD protocol chính theo object split và layout split;
- thuật toán downstream chính là offline RL với `IQL`.

Mốc triển khai gần nhất trong repo là `Sprint M0`: một prototype simulation replay cho `retrieval-conditioned progress estimation`.
Prototype này cố tình đi trước reward head đầy đủ và `IQL`, để khóa nhanh pipeline dữ liệu, memory bank, retrieval, và progress proxy có thể demo được.

## Cấu trúc chính

- `demo/demo_v0.py`: entrypoint cho demo `auto | libero | replay2d`.
- `data/`: memory bank mini, index, và frame dump cho demo.
- `artifacts/`: hình minh họa retrieval, plot progress, và GIF demo.
- `docs/demo_v0_notes.md`: ghi chú ngắn cho demo M0.
- `pdf/`: tài liệu proposal LaTeX và file PDF đầu ra.
- `milestones.md`: các mốc công việc đã chốt và mốc tiếp theo.
- `plan.md`: kế hoạch active cho bước nghiên cứu kế tiếp.
- `pyproject.toml`, `uv.lock`: cấu hình `uv` và dependency lock của repo.

## Near-term Demo Goal

Mục tiêu gần nhất là chạy được một prototype simulation-based để show:

- memory bank mini từ frame replay + stage text + progress pseudo label;
- top-k retrieval cho frame query;
- progress/reward proxy theo ba mode: `no context`, `random context`, `retrieved context`;
- artifact trực quan đủ để báo cáo nhanh.

Artifact kỳ vọng của demo M0:

- `artifacts/demo_case.png`
- `artifacts/retrieval_examples.png`
- `artifacts/progress_plot.png`
- `artifacts/demo.gif`
- `docs/demo_v0_notes.md`

## Build proposal PDF

Luôn build trong thư mục `pdf/` theo đúng quy trình dọn sạch file trung gian:

```bash
cd pdf
latexmk -pdf main.tex
latexmk -c main.tex
find . -maxdepth 1 -type f \( \
  -name "*.aux" -o -name "*.log" -o -name "*.out" -o -name "*.toc" -o \
  -name "*.fls" -o -name "*.fdb_latexmk" -o -name "*.synctex.gz" -o \
  -name "*.nav" -o -name "*.snm" -o -name "*.vrb" -o \
  -name "*.bbl" -o -name "*.blg" \
\) -delete
```

Sau khi build xong, file cần giữ lại là `pdf/main.pdf` cùng các nguồn `.tex`, `.bib`, và `latexmkrc`.

## Python workspace

Repo dùng một workflow `uv` duy nhất và đang pin Python ở `3.8.20` để tương thích với nhánh spike `LIBERO`.
Thiết lập cơ bản:

```bash
uv sync
uv run python --version
```

Nếu muốn thử nhánh `LIBERO` trong chính project này, sync thêm optional group:

```bash
uv sync --group libero
```

Repo hiện kèm một local import shim `libero/` để bridge packaging issue của upstream `LIBERO` khi cài trực tiếp bằng `uv` từ Git source.
Vì vậy, `uv sync --group libero` là đủ để `demo/demo_v0.py --backend auto` thử env `LIBERO` thật trước khi fallback.

Entrypoint demo:

```bash
uv run python demo/demo_v0.py --backend auto
uv run python demo/demo_v0.py --backend replay2d
```

`auto` sẽ thử `LIBERO` trước; nếu import hoặc khởi tạo env không chạy được đủ nhanh, script sẽ fallback sạch sang `replay2d`.
