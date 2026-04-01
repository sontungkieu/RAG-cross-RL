from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
DOCS_DIR = ROOT / "docs"
FRAME_DIR = DATA_DIR / "demo_frames"
MEMORY_BANK_PATH = DATA_DIR / "demo_memory_bank.jsonl"
INDEX_PATH = DATA_DIR / "demo_index.pkl"
NOTES_PATH = DOCS_DIR / "demo_v0_notes.md"

TASK_ID = "pick_and_place_replay_v0"
TASK_INSTRUCTION = "Pick the blue object and place it inside the green goal zone."
STAGE_TEXT = {
    0: "approach object",
    1: "grasp object",
    2: "transport object",
    3: "place object in goal",
}


@dataclass
class FrameRecord:
    episode_id: str
    frame_path: str
    task_id: str
    instruction: str
    stage_id: int
    stage_text: str
    progress: float
    timestep: int
    source: str
    split: str


@dataclass
class FrameSample:
    record: FrameRecord
    image: np.ndarray
    embedding: np.ndarray


@dataclass
class DatasetBundle:
    backend: str
    support_samples: List[FrameSample]
    query_samples: List[FrameSample]
    fallback_reason: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a simulation replay demo for retrieval-conditioned progress estimation."
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "libero", "replay2d"),
        default="auto",
        help="Execution backend. 'auto' tries LIBERO first, then falls back to replay2d.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of retrieved exemplars.")
    parser.add_argument(
        "--num-stages",
        type=int,
        default=4,
        help="Number of pseudo stages. replay2d currently supports 4 only.",
    )
    parser.add_argument(
        "--seed", type=int, default=7, help="Random seed used for support/query sampling."
    )
    parser.add_argument(
        "--max-libero-steps",
        type=int,
        default=24,
        help="Maximum rollout steps to try when backend=libero.",
    )
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in (DATA_DIR, ARTIFACTS_DIR, DOCS_DIR, FRAME_DIR):
        path.mkdir(parents=True, exist_ok=True)


def stage_id_from_ratio(ratio: float, num_stages: int) -> int:
    clipped = min(max(ratio, 0.0), 0.999999)
    return min(int(clipped * num_stages), num_stages - 1)


def normalized_progress(index: int, total_steps: int) -> float:
    if total_steps <= 1:
        return 1.0
    return index / float(total_steps - 1)


def as_uint8(image: np.ndarray) -> np.ndarray:
    clipped = np.clip(image, 0, 255)
    if clipped.dtype != np.uint8:
        clipped = clipped.astype(np.uint8)
    return clipped


def save_frame_image(image: np.ndarray, backend: str, episode_id: str, timestep: int) -> Path:
    episode_dir = FRAME_DIR / backend / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)
    frame_path = episode_dir / f"frame_{timestep:03d}.png"
    Image.fromarray(as_uint8(image)).save(frame_path)
    return frame_path


def embed_image(image: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(as_uint8(image)).convert("L").resize((24, 24))
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    flat = arr.reshape(-1)
    centered = flat - flat.mean()
    norm = np.linalg.norm(centered)
    if norm < 1e-8:
        return centered
    return centered / norm


def cosine_similarity(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    bank_norm = np.linalg.norm(bank, axis=1)
    query_norm = np.linalg.norm(query)
    denom = np.clip(bank_norm * query_norm, 1e-8, None)
    return bank @ query / denom


def softmax(scores: np.ndarray, temperature: float = 0.15) -> np.ndarray:
    if scores.size == 0:
        return scores
    scaled = scores / max(temperature, 1e-6)
    scaled = scaled - np.max(scaled)
    exp_scores = np.exp(scaled)
    denom = np.sum(exp_scores)
    if denom <= 0:
        return np.full_like(exp_scores, 1.0 / len(exp_scores))
    return exp_scores / denom


def interpolate(start: np.ndarray, end: np.ndarray, frac: float) -> np.ndarray:
    return start + frac * (end - start)


def replay_episode_state(episode_seed: int, timestep: int, total_steps: int) -> Dict[str, np.ndarray]:
    rng = random.Random(episode_seed)
    start = np.array([18 + rng.uniform(-3, 3), 20 + rng.uniform(-4, 4)], dtype=np.float32)
    obj = np.array([54 + rng.uniform(-6, 6), 74 + rng.uniform(-5, 5)], dtype=np.float32)
    goal = np.array([108 + rng.uniform(-4, 4), 34 + rng.uniform(-6, 6)], dtype=np.float32)
    phase_lengths = [6, 6, 6, 6]
    stage_offsets = np.cumsum([0] + phase_lengths)

    if timestep < stage_offsets[1]:
        frac = timestep / max(phase_lengths[0] - 1, 1)
        gripper = interpolate(start, obj + np.array([-2.0, -2.0], dtype=np.float32), frac)
        object_pos = obj.copy()
        attached = False
    elif timestep < stage_offsets[2]:
        frac = (timestep - stage_offsets[1]) / max(phase_lengths[1] - 1, 1)
        gripper = interpolate(obj + np.array([-2.0, -2.0], dtype=np.float32), obj, frac)
        object_pos = obj.copy()
        attached = frac > 0.6
    elif timestep < stage_offsets[3]:
        frac = (timestep - stage_offsets[2]) / max(phase_lengths[2] - 1, 1)
        gripper = interpolate(obj, goal, frac)
        object_pos = gripper.copy()
        attached = True
    else:
        frac = (timestep - stage_offsets[3]) / max(phase_lengths[3] - 1, 1)
        release_target = goal + np.array([16.0, 6.0], dtype=np.float32)
        gripper = interpolate(goal, release_target, frac)
        object_pos = goal.copy()
        attached = False

    return {
        "gripper": gripper,
        "object": object_pos,
        "goal": goal,
        "attached": np.array([1.0 if attached else 0.0], dtype=np.float32),
    }


def render_replay_frame(state: Dict[str, np.ndarray], stage_id: int, episode_id: str) -> np.ndarray:
    canvas = Image.new("RGB", (128, 128), (245, 244, 239))
    draw = ImageDraw.Draw(canvas)

    for offset in range(0, 129, 16):
        draw.line((offset, 0, offset, 128), fill=(224, 224, 220), width=1)
        draw.line((0, offset, 128, offset), fill=(224, 224, 220), width=1)

    goal = tuple(state["goal"])
    goal_box = [goal[0] - 14, goal[1] - 14, goal[0] + 14, goal[1] + 14]
    draw.rounded_rectangle(goal_box, radius=7, fill=(191, 231, 196), outline=(38, 122, 62), width=3)

    object_pos = tuple(state["object"])
    obj_box = [object_pos[0] - 8, object_pos[1] - 8, object_pos[0] + 8, object_pos[1] + 8]
    draw.ellipse(obj_box, fill=(55, 115, 225), outline=(20, 54, 126), width=2)

    gripper = tuple(state["gripper"])
    grip_box = [gripper[0] - 9, gripper[1] - 9, gripper[0] + 9, gripper[1] + 9]
    draw.ellipse(grip_box, fill=(214, 85, 70), outline=(120, 42, 33), width=2)

    if bool(state["attached"][0]):
        draw.ellipse(
            [object_pos[0] - 12, object_pos[1] - 12, object_pos[0] + 12, object_pos[1] + 12],
            outline=(240, 180, 65),
            width=2,
        )

    draw.text((7, 7), f"{episode_id}", fill=(38, 38, 38))
    draw.text((7, 21), f"stage {stage_id}: {STAGE_TEXT[stage_id]}", fill=(38, 38, 38))
    return np.asarray(canvas, dtype=np.uint8)


def collect_replay2d_bundle(num_stages: int, seed: int) -> DatasetBundle:
    if num_stages != 4:
        raise ValueError("replay2d currently supports exactly 4 stages.")

    total_steps = 24
    episode_specs = [
        ("support_ep0", seed),
        ("support_ep1", seed + 17),
        ("query_ep0", seed + 41),
    ]
    support_samples: List[FrameSample] = []
    query_samples: List[FrameSample] = []

    for episode_id, episode_seed in episode_specs:
        split = "support" if episode_id.startswith("support") else "query"
        for timestep in range(total_steps):
            progress = normalized_progress(timestep, total_steps)
            stage_id = stage_id_from_ratio(progress, num_stages)
            state = replay_episode_state(episode_seed, timestep, total_steps)
            image = render_replay_frame(state, stage_id=stage_id, episode_id=episode_id)
            frame_path = save_frame_image(image, backend="replay2d", episode_id=episode_id, timestep=timestep)
            record = FrameRecord(
                episode_id=episode_id,
                frame_path=str(frame_path.relative_to(ROOT)),
                task_id=TASK_ID,
                instruction=TASK_INSTRUCTION,
                stage_id=stage_id,
                stage_text=STAGE_TEXT[stage_id],
                progress=round(progress, 4),
                timestep=timestep,
                source="replay2d",
                split=split,
            )
            sample = FrameSample(record=record, image=image, embedding=embed_image(image))
            if split == "support":
                support_samples.append(sample)
            else:
                query_samples.append(sample)

    return DatasetBundle(backend="replay2d", support_samples=support_samples, query_samples=query_samples)


def extract_rgb_frame(obs: object) -> np.ndarray | None:
    if isinstance(obs, dict):
        preferred_keys = [
            "agentview_image",
            "robot0_eye_in_hand_image",
            "frontview_image",
            "image",
            "rgb",
        ]
        for key in preferred_keys:
            value = obs.get(key)
            if isinstance(value, np.ndarray) and value.ndim == 3:
                array = value
                if array.dtype != np.uint8:
                    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
                if array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
                    array = np.transpose(array, (1, 2, 0))
                if array.shape[-1] >= 3:
                    return array[..., :3]
    if isinstance(obs, np.ndarray) and obs.ndim == 3 and obs.shape[-1] >= 3:
        return obs[..., :3].astype(np.uint8)
    return None


def collect_libero_bundle(num_stages: int, max_steps: int) -> DatasetBundle:
    if num_stages != 4:
        raise ValueError("The current LIBERO spike path expects exactly 4 stages.")

    package_root = prepare_libero_runtime()

    try:
        from libero.libero.envs import OffScreenRenderEnv
    except Exception as exc:
        raise RuntimeError(f"unable to import LIBERO stack: {type(exc).__name__}: {exc}") from exc

    task_name, task_language, bddl_file = pick_libero_task(package_root)

    try:
        env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=128, camera_widths=128)
        obs = env.reset()
    except Exception as exc:
        raise RuntimeError(f"unable to initialize LIBERO environment: {type(exc).__name__}: {exc}") from exc

    support_samples: List[FrameSample] = []
    query_samples: List[FrameSample] = []
    rng = np.random.default_rng(13)

    try:
        for timestep in range(max_steps):
            if timestep > 0:
                action = rng.uniform(low=-0.05, high=0.05, size=7).astype(np.float32)
                obs, _, _, _ = env.step(action.tolist())
            frame = extract_rgb_frame(obs)
            if frame is None:
                raise RuntimeError("LIBERO observation does not expose an RGB image.")
            progress = normalized_progress(timestep, max_steps)
            stage_id = stage_id_from_ratio(progress, num_stages)
            frame_path = save_frame_image(frame, backend="libero", episode_id="libero_query", timestep=timestep)
            record = FrameRecord(
                episode_id="libero_query",
                frame_path=str(frame_path.relative_to(ROOT)),
                task_id=task_name,
                instruction=task_language,
                stage_id=stage_id,
                stage_text=STAGE_TEXT[stage_id],
                progress=round(progress, 4),
                timestep=timestep,
                source="libero",
                split="query" if timestep % 3 == 0 else "support",
            )
            sample = FrameSample(record=record, image=frame, embedding=embed_image(frame))
            if record.split == "support":
                support_samples.append(sample)
            else:
                query_samples.append(sample)
    finally:
        env.close()

    if len(support_samples) < 3 or len(query_samples) < 3:
        raise RuntimeError("LIBERO rollout did not produce enough support/query frames.")

    return DatasetBundle(backend="libero", support_samples=support_samples, query_samples=query_samples)


def prepare_libero_runtime() -> Path:
    try:
        import libero as libero_shim
    except Exception as exc:
        raise RuntimeError(f"unable to import local LIBERO shim: {type(exc).__name__}: {exc}") from exc

    package_root = getattr(libero_shim, "find_upstream_package_root", lambda: None)()
    if not package_root:
        raise RuntimeError("uv installed LIBERO metadata, but the upstream source checkout was not found.")

    package_root_path = Path(package_root)
    benchmark_root = package_root_path / "libero"
    config_dir = Path.home() / ".cache" / "rag-cross-rl" / "libero"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)

    config_values = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(package_root_path / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }
    config_text = "".join(f"{key}: {value}\n" for key, value in config_values.items())
    config_path.write_text(config_text, encoding="utf-8")
    return package_root_path


def pick_libero_task(package_root: Path) -> Tuple[str, str, str]:
    suite_dir = package_root / "libero" / "bddl_files" / "libero_spatial"
    bddl_candidates = sorted(suite_dir.glob("*.bddl"))
    if not bddl_candidates:
        raise RuntimeError(f"no LIBERO BDDL files found under {suite_dir}")

    bddl_file = bddl_candidates[0]
    task_name = bddl_file.stem
    language = grab_language_from_filename(task_name + ".bddl")
    return task_name, language, str(bddl_file)


def grab_language_from_filename(filename: str) -> str:
    if filename[0].isupper():
        if "SCENE10" in filename:
            language = " ".join(filename[filename.find("SCENE") + 8 :].split("_"))
        else:
            language = " ".join(filename[filename.find("SCENE") + 7 :].split("_"))
    else:
        language = " ".join(filename.split("_"))
    return language[: language.find(".bddl")]


def resolve_backend(args: argparse.Namespace) -> DatasetBundle:
    if args.backend == "replay2d":
        return collect_replay2d_bundle(num_stages=args.num_stages, seed=args.seed)
    if args.backend == "libero":
        return collect_libero_bundle(num_stages=args.num_stages, max_steps=args.max_libero_steps)

    try:
        bundle = collect_libero_bundle(num_stages=args.num_stages, max_steps=args.max_libero_steps)
        bundle.fallback_reason = None
        return bundle
    except Exception as exc:
        print(f"[demo] LIBERO unavailable, falling back to replay2d: {exc}")
        bundle = collect_replay2d_bundle(num_stages=args.num_stages, seed=args.seed)
        bundle.fallback_reason = str(exc)
        return bundle


def write_memory_bank(samples: Sequence[FrameSample]) -> None:
    with MEMORY_BANK_PATH.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(asdict(sample.record)) + "\n")


def write_index(samples: Sequence[FrameSample]) -> None:
    payload = {
        "frame_paths": [sample.record.frame_path for sample in samples],
        "records": [asdict(sample.record) for sample in samples],
        "embeddings": np.stack([sample.embedding for sample in samples], axis=0),
    }
    with INDEX_PATH.open("wb") as handle:
        pickle.dump(payload, handle)


def stage_prototypes(samples: Sequence[FrameSample]) -> Dict[int, Tuple[np.ndarray, float]]:
    grouped: Dict[int, List[FrameSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.record.stage_id, []).append(sample)
    prototypes: Dict[int, Tuple[np.ndarray, float]] = {}
    for stage_id, stage_samples in grouped.items():
        matrix = np.stack([sample.embedding for sample in stage_samples], axis=0)
        progress_mean = float(np.mean([sample.record.progress for sample in stage_samples]))
        prototypes[stage_id] = (np.mean(matrix, axis=0), progress_mean)
    return prototypes


def predict_no_context(query_embedding: np.ndarray, prototypes: Dict[int, Tuple[np.ndarray, float]]) -> float:
    stage_ids = sorted(prototypes)
    proto_matrix = np.stack([prototypes[stage_id][0] for stage_id in stage_ids], axis=0)
    sims = cosine_similarity(query_embedding, proto_matrix)
    weights = softmax(sims)
    progress_values = np.array([prototypes[stage_id][1] for stage_id in stage_ids], dtype=np.float32)
    return float(np.sum(weights * progress_values))


def retrieve_top_k(
    query_embedding: np.ndarray, support_samples: Sequence[FrameSample], top_k: int
) -> List[Tuple[int, float]]:
    support_matrix = np.stack([sample.embedding for sample in support_samples], axis=0)
    sims = cosine_similarity(query_embedding, support_matrix)
    top_indices = np.argsort(-sims)[:top_k]
    return [(int(index), float(sims[index])) for index in top_indices]


def predict_from_retrieval(
    retrieved: Sequence[Tuple[int, float]], support_samples: Sequence[FrameSample]
) -> Tuple[float, List[float]]:
    sims = np.array([score for _, score in retrieved], dtype=np.float32)
    weights = softmax(sims)
    progresses = np.array(
        [support_samples[index].record.progress for index, _ in retrieved], dtype=np.float32
    )
    prediction = float(np.sum(weights * progresses))
    return prediction, weights.tolist()


def select_random_context(
    rng: random.Random, support_samples: Sequence[FrameSample], top_k: int
) -> List[Tuple[int, float]]:
    indices = list(range(len(support_samples)))
    rng.shuffle(indices)
    selected = indices[:top_k]
    return [(index, 0.0) for index in selected]


def query_case_indices(query_samples: Sequence[FrameSample]) -> Dict[str, int]:
    total = len(query_samples)
    return {
        "early": max(1, total // 8),
        "mid": total // 2,
        "late": max(total - total // 8 - 1, 0),
    }


def compute_predictions(
    bundle: DatasetBundle, top_k: int, seed: int
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    prototypes = stage_prototypes(bundle.support_samples)
    rng = random.Random(seed)
    rows: List[Dict[str, object]] = []

    for sample in bundle.query_samples:
        retrieved = retrieve_top_k(sample.embedding, bundle.support_samples, top_k=top_k)
        pred_retrieved, retrieved_weights = predict_from_retrieval(retrieved, bundle.support_samples)
        random_context = select_random_context(rng, bundle.support_samples, top_k=top_k)
        pred_random, random_weights = predict_from_retrieval(random_context, bundle.support_samples)
        pred_no_context = predict_no_context(sample.embedding, prototypes)

        rows.append(
            {
                "sample": sample,
                "oracle_progress": sample.record.progress,
                "pred_no_context": pred_no_context,
                "pred_random": pred_random,
                "pred_retrieved": pred_retrieved,
                "retrieved": retrieved,
                "retrieved_weights": retrieved_weights,
                "random_context": random_context,
                "random_weights": random_weights,
            }
        )

    metrics = {
        "mae_no_context": mae([row["oracle_progress"] for row in rows], [row["pred_no_context"] for row in rows]),
        "mae_random": mae([row["oracle_progress"] for row in rows], [row["pred_random"] for row in rows]),
        "mae_retrieved": mae(
            [row["oracle_progress"] for row in rows], [row["pred_retrieved"] for row in rows]
        ),
    }
    return rows, metrics


def mae(targets: Sequence[float], predictions: Sequence[float]) -> float:
    target_arr = np.asarray(targets, dtype=np.float32)
    pred_arr = np.asarray(predictions, dtype=np.float32)
    return float(np.mean(np.abs(target_arr - pred_arr)))


def correlation(targets: Sequence[float], predictions: Sequence[float]) -> float:
    target_arr = np.asarray(targets, dtype=np.float32)
    pred_arr = np.asarray(predictions, dtype=np.float32)
    if len(target_arr) < 2:
        return 0.0
    corr = np.corrcoef(target_arr, pred_arr)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def plot_retrieval_examples(bundle: DatasetBundle, rows: Sequence[Dict[str, object]], top_k: int) -> None:
    case_map = query_case_indices(bundle.query_samples)
    fig, axes = plt.subplots(nrows=3, ncols=top_k + 1, figsize=(4 * (top_k + 1), 10))
    fig.suptitle(f"Retrieval examples ({bundle.backend})", fontsize=15)

    for row_index, (label, sample_index) in enumerate(case_map.items()):
        row = rows[sample_index]
        sample = row["sample"]
        axes[row_index, 0].imshow(sample.image)
        axes[row_index, 0].set_title(
            f"{label} query\nstage {sample.record.stage_id} | p={sample.record.progress:.2f}"
        )
        axes[row_index, 0].axis("off")

        for col_index, (support_index, score) in enumerate(row["retrieved"], start=1):
            support = bundle.support_samples[support_index]
            axes[row_index, col_index].imshow(support.image)
            axes[row_index, col_index].set_title(
                f"top-{col_index}\nscore={score:.3f}\n{support.record.stage_text}\np={support.record.progress:.2f}",
                fontsize=9,
            )
            axes[row_index, col_index].axis("off")

    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "retrieval_examples.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_progress(rows: Sequence[Dict[str, object]], metrics: Dict[str, float]) -> None:
    timesteps = [row["sample"].record.timestep for row in rows]
    oracle = [row["oracle_progress"] for row in rows]
    no_context = [row["pred_no_context"] for row in rows]
    random_ctx = [row["pred_random"] for row in rows]
    retrieved = [row["pred_retrieved"] for row in rows]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(timesteps, oracle, label="oracle progress", color="#111111", linewidth=2.3)
    ax.plot(timesteps, no_context, label=f"no context (MAE={metrics['mae_no_context']:.3f})", linewidth=2.0)
    ax.plot(timesteps, random_ctx, label=f"random context (MAE={metrics['mae_random']:.3f})", linewidth=2.0)
    ax.plot(
        timesteps,
        retrieved,
        label=f"retrieved context (MAE={metrics['mae_retrieved']:.3f})",
        linewidth=2.2,
    )
    ax.set_xlabel("timestep")
    ax.set_ylabel("progress")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Progress proxy over query rollout")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "progress_plot.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_demo_case(bundle: DatasetBundle, rows: Sequence[Dict[str, object]]) -> None:
    sample_index = query_case_indices(bundle.query_samples)["mid"]
    row = rows[sample_index]
    sample = row["sample"]

    fig = plt.figure(figsize=(15, 8))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.5])
    axes = [fig.add_subplot(grid[0, idx]) for idx in range(4)]

    axes[0].imshow(sample.image)
    axes[0].set_title(
        f"query frame\nstage {sample.record.stage_id} | oracle={sample.record.progress:.2f}"
    )
    axes[0].axis("off")

    for plot_index, (support_index, score) in enumerate(row["retrieved"], start=1):
        support = bundle.support_samples[support_index]
        axes[plot_index].imshow(support.image)
        axes[plot_index].set_title(
            f"retrieved #{plot_index}\nscore={score:.3f}\n{support.record.stage_text}",
            fontsize=10,
        )
        axes[plot_index].axis("off")

    text_ax = fig.add_subplot(grid[1, :])
    text_ax.axis("off")
    text_ax.text(
        0.02,
        0.9,
        "\n".join(
            [
                f"backend: {bundle.backend}",
                f"instruction: {sample.record.instruction}",
                f"predicted progress (no context): {row['pred_no_context']:.3f}",
                f"predicted progress (random context): {row['pred_random']:.3f}",
                f"predicted progress (retrieved context): {row['pred_retrieved']:.3f}",
                f"retrieved stage ids: {[bundle.support_samples[index].record.stage_id for index, _ in row['retrieved']]}",
            ]
        ),
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
    )
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "demo_case.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_gif(rows: Sequence[Dict[str, object]], bundle: DatasetBundle) -> None:
    frames = []
    for row in rows:
        image = Image.fromarray(as_uint8(row["sample"].image)).convert("RGB")
        draw = ImageDraw.Draw(image)
        top_stage_id = bundle.support_samples[row["retrieved"][0][0]].record.stage_id
        lines = [
            f"backend: {bundle.backend}",
            f"timestep: {row['sample'].record.timestep}",
            f"oracle progress: {row['oracle_progress']:.2f}",
            f"retrieved progress: {row['pred_retrieved']:.2f}",
            f"random progress: {row['pred_random']:.2f}",
            f"retrieved stage: {top_stage_id}",
        ]
        draw.rounded_rectangle((4, 4, 124, 62), radius=6, fill=(250, 250, 250), outline=(60, 60, 60))
        for idx, text in enumerate(lines):
            draw.text((8, 8 + idx * 9), text, fill=(25, 25, 25))
        frames.append(np.asarray(image, dtype=np.uint8))

    imageio.mimsave(ARTIFACTS_DIR / "demo.gif", frames, duration=0.22, loop=0)


def write_notes(
    bundle: DatasetBundle,
    metrics: Dict[str, float],
    rows: Sequence[Dict[str, object]],
    top_k: int,
) -> None:
    corr_retrieved = correlation(
        [row["oracle_progress"] for row in rows], [row["pred_retrieved"] for row in rows]
    )
    note_lines = [
        "# Demo v0 Notes",
        "",
        "## Objective",
        "",
        "- Build a simulation replay prototype for retrieval-conditioned progress estimation.",
        "- Show that retrieved context gives a better progress proxy than no-context and random-context baselines.",
        "",
        "## Data and Backend",
        "",
        f"- Backend used: `{bundle.backend}`.",
        f"- Task instruction: `{TASK_INSTRUCTION}`." if bundle.backend == "replay2d" else f"- Task instruction source: `{rows[0]['sample'].record.instruction}`.",
        f"- Support memory size: `{len(bundle.support_samples)}` frames.",
        f"- Query rollout size: `{len(bundle.query_samples)}` frames.",
        f"- Retrieval setting: `top-k={top_k}` with cosine similarity over frozen image embeddings.",
    ]
    if bundle.fallback_reason:
        note_lines.extend(
            [
                f"- `auto` backend fell back to `replay2d` because LIBERO initialization failed: `{bundle.fallback_reason}`.",
            ]
        )

    note_lines.extend(
        [
            "",
            "## Pipeline",
            "",
            "- Build a mini memory bank from support frames with task instruction, stage text, and pseudo progress labels.",
            "- Retrieve top-k support entries for each query frame.",
            "- Predict progress with a retrieval-weighted average of retrieved progress labels.",
            "- Compare three modes: `no context`, `random context`, and `retrieved context`.",
            "",
            "## Current Metrics",
            "",
            f"- MAE no context: `{metrics['mae_no_context']:.4f}`.",
            f"- MAE random context: `{metrics['mae_random']:.4f}`.",
            f"- MAE retrieved context: `{metrics['mae_retrieved']:.4f}`.",
            f"- Correlation retrieved vs oracle: `{corr_retrieved:.4f}`.",
            "",
            "## Limitations",
            "",
            "- This sprint uses a lightweight progress proxy, not a learned reward head.",
            "- The demo is single-task and stage labels are pseudo labels from stage windows / timestep progress.",
            "- No IQL training, broad benchmark sweep, or OOD evaluation is included yet.",
            "",
            "## Next Step",
            "",
            "- Replace the retrieval-weighted proxy with a proper retrieval-conditioned reward model, then connect that reward to offline RL with IQL.",
        ]
    )
    NOTES_PATH.write_text("\n".join(note_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dirs()
    bundle = resolve_backend(args)
    write_memory_bank(bundle.support_samples)
    write_index(bundle.support_samples)
    rows, metrics = compute_predictions(bundle, top_k=args.top_k, seed=args.seed)
    plot_retrieval_examples(bundle, rows, top_k=args.top_k)
    plot_progress(rows, metrics)
    plot_demo_case(bundle, rows)
    render_gif(rows, bundle)
    write_notes(bundle, metrics, rows, top_k=args.top_k)

    corr_retrieved = correlation(
        [row["oracle_progress"] for row in rows], [row["pred_retrieved"] for row in rows]
    )
    print(
        json.dumps(
            {
                "backend": bundle.backend,
                "fallback_reason": bundle.fallback_reason,
                "memory_bank_entries": len(bundle.support_samples),
                "query_frames": len(bundle.query_samples),
                "mae_no_context": round(metrics["mae_no_context"], 4),
                "mae_random": round(metrics["mae_random"], 4),
                "mae_retrieved": round(metrics["mae_retrieved"], 4),
                "corr_retrieved": round(corr_retrieved, 4),
                "artifacts": [
                    "artifacts/demo_case.png",
                    "artifacts/retrieval_examples.png",
                    "artifacts/progress_plot.png",
                    "artifacts/demo.gif",
                    "docs/demo_v0_notes.md",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
