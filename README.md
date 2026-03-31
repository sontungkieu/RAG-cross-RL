# RAG-cross-RL

Workspace này hiện được dùng để soạn proposal nghiên cứu cho hướng `retrieval-augmented reward modeling` trong robot manipulation mô phỏng.
Phiên bản proposal hiện tại đã chốt rõ:

- nguồn supervision cho reward từ simulator/task-derived pseudo labels;
- OOD protocol chính theo object split và layout split;
- thuật toán downstream chính là offline RL với `IQL`.

## Cấu trúc chính

- `pdf/`: tài liệu proposal LaTeX và file PDF đầu ra.
- `milestones.md`: các mốc công việc đã chốt và mốc tiếp theo.
- `plan.md`: kế hoạch active cho bước nghiên cứu kế tiếp.
- `pyproject.toml`, `uv.lock`: scaffold Python tối thiểu của repo.

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

Repo vẫn giữ scaffold `uv` tối thiểu:

```bash
uv sync
uv run python --version
uv lock
```

Python được ghim ở phiên bản `3.10`.
