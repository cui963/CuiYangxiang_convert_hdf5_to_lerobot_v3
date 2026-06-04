#!/usr/bin/env python3
"""Convert HDF5 robot episodes to LeRobotDataset v3 format.

The script auto-detects common ALOHA-style HDF5 layouts:

  observations/qpos              -> observation.state
  action or actions              -> action
  observations/images/<camera>   -> observation.images.<camera>

For other HDF5 schemas, pass explicit mappings:

  python convert_hdf5_to_lerobot_v3.py \
    --input ./raw_hdf5 \
    --output ./lerobot_dataset \
    --repo-id local/my_dataset \
    --fps 50 \
    --task "pick the cube" \
    --state obs/robot_state=observation.state \
    --action actions=action \
    --image obs/front_rgb=observation.images.front
"""

from __future__ import annotations

import argparse
import inspect
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import h5py
    import numpy as np
except ModuleNotFoundError as exc:
    missing = exc.name or "required package"
    print(
        f"Missing Python package: {missing}\n\n"
        "Install the conversion dependencies into the same Python environment you use to run this script:\n"
        "  python -m pip install -U h5py numpy pillow opencv-python tqdm lerobot\n\n"
        "On your machine, that likely means:\n"
        "  C:\\Users\\32495\\.local\\bin\\python3.14.exe -m pip install -U h5py numpy pillow opencv-python tqdm lerobot",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


LOG = logging.getLogger("hdf5_to_lerobot_v3")


COMMON_STATE_PATHS = (
    "observations/qpos",
    "observations/state",
    "observation/state",
    "obs/state",
    "obs/robot_state",
    "state",
    "states",
    "qpos",
)

COMMON_ACTION_PATHS = (
    "action",
    "actions",
    "control",
    "controls",
)

COMMON_IMAGE_ROOTS = (
    "observations/images",
    "observation/images",
    "obs/images",
    "images",
)

COMMON_EXTRA_LOWDIM = (
    ("observations/qvel", "observation.velocity"),
    ("observations/effort", "observation.effort"),
    ("obs/qvel", "observation.velocity"),
    ("obs/effort", "observation.effort"),
)


@dataclass(frozen=True)
class FeatureMapping:
    hdf5_path: str
    lerobot_key: str
    kind: str
    dtype: str = "float32"


@dataclass(frozen=True)
class EpisodeRef:
    file_path: Path
    group_path: str


def natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def sanitize_feature_leaf(value: str) -> str:
    leaf = value.strip("/").split("/")[-1]
    leaf = re.sub(r"(_rgb|_image|_images)$", "", leaf, flags=re.IGNORECASE)
    leaf = re.sub(r"[^0-9A-Za-z_]+", "_", leaf)
    leaf = leaf.strip("_")
    return leaf or "camera"


def join_hdf5_path(base: str, child: str) -> str:
    base = base.strip("/")
    child = child.strip("/")
    if not base:
        return child or "/"
    return f"{base}/{child}" if child else base


def open_group(h5_file: h5py.File, group_path: str) -> h5py.Group | h5py.File:
    if group_path in ("", "/"):
        return h5_file
    return h5_file[group_path]


def get_node(group: h5py.Group | h5py.File, hdf5_path: str) -> h5py.Dataset | h5py.Group | None:
    path = hdf5_path.strip()
    if not path:
        return None

    attempts = [path]
    if path.startswith("/"):
        attempts.append(path.lstrip("/"))
    else:
        attempts.append("/" + path)

    for candidate in attempts:
        try:
            return group[candidate]
        except KeyError:
            continue
    return None


def get_dataset(group: h5py.Group | h5py.File, hdf5_path: str) -> h5py.Dataset:
    node = get_node(group, hdf5_path)
    if node is None:
        raise KeyError(f"HDF5 path not found: {hdf5_path}")
    if not isinstance(node, h5py.Dataset):
        raise TypeError(f"HDF5 path is not a dataset: {hdf5_path}")
    return node


def iter_datasets(group: h5py.Group | h5py.File) -> Iterable[tuple[str, h5py.Dataset]]:
    def visitor(name: str, node: h5py.Dataset | h5py.Group) -> None:
        if isinstance(node, h5py.Dataset):
            datasets.append((name, node))

    datasets: list[tuple[str, h5py.Dataset]] = []
    group.visititems(visitor)
    return datasets


def format_dataset_listing(group: h5py.Group | h5py.File, limit: int = 80) -> str:
    rows = []
    for name, dataset in sorted(iter_datasets(group), key=lambda item: natural_key(item[0])):
        rows.append(f"  {name} shape={dataset.shape} dtype={dataset.dtype}")
        if len(rows) >= limit:
            rows.append("  ...")
            break
    return "\n".join(rows) if rows else "  <no datasets>"


def collect_hdf5_files(input_path: Path, input_glob: str | None, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    finder = input_path.rglob if recursive else input_path.glob
    if input_glob:
        files = sorted(finder(input_glob), key=lambda path: natural_key(str(path)))
    else:
        files = sorted(
            [*finder("*.hdf5"), *finder("*.h5")],
            key=lambda path: natural_key(str(path)),
        )

    if not files:
        raise FileNotFoundError(f"No .hdf5 or .h5 files found under: {input_path}")
    return files


def group_has_episode_data(group: h5py.Group | h5py.File) -> bool:
    for path in (*COMMON_ACTION_PATHS, *COMMON_STATE_PATHS):
        node = get_node(group, path)
        if isinstance(node, h5py.Dataset) and len(node.shape) >= 1:
            return True
    return False


def discover_episode_paths(h5_file: h5py.File, episode_root: str | None) -> list[str]:
    if episode_root:
        root_path = episode_root.strip("/") or "/"
        root_group = open_group(h5_file, root_path)
        child_groups = sorted(
            [name for name, node in root_group.items() if isinstance(node, h5py.Group)],
            key=natural_key,
        )
        if child_groups and not group_has_episode_data(root_group):
            return [join_hdf5_path(root_path, child) for child in child_groups]
        return [root_path]

    for root in ("data", "episodes", "demos", "trajectories", "trials"):
        node = get_node(h5_file, root)
        if isinstance(node, h5py.Group):
            child_groups = sorted(
                [name for name, child in node.items() if isinstance(child, h5py.Group)],
                key=natural_key,
            )
            if child_groups:
                return [join_hdf5_path(root, child) for child in child_groups]

    top_episode_groups = sorted(
        [
            name
            for name, node in h5_file.items()
            if isinstance(node, h5py.Group) and re.search(r"(demo|episode|traj|trial)", name, re.IGNORECASE)
        ],
        key=natural_key,
    )
    if len(top_episode_groups) > 1:
        return top_episode_groups

    return ["/"]


def collect_episode_refs(files: list[Path], episode_root: str | None) -> list[EpisodeRef]:
    refs: list[EpisodeRef] = []
    for file_path in files:
        with h5py.File(file_path, "r") as h5_file:
            refs.extend(EpisodeRef(file_path=file_path, group_path=path) for path in discover_episode_paths(h5_file, episode_root))
    return refs


def parse_mapping(spec: str, kind: str, default_key: str | None = None, default_dtype: str = "float32") -> FeatureMapping:
    if "=" in spec:
        left, right = spec.split("=", 1)
        parts = right.split(":")
        lerobot_key = parts[0].strip()
        dtype = parts[1].strip() if len(parts) > 1 else default_dtype
    else:
        if default_key is None:
            raise ValueError(f"Mapping must be HDF5_PATH=LEROBOT_KEY: {spec}")
        left = spec
        lerobot_key = default_key
        dtype = default_dtype

    hdf5_path = left.strip()
    if not hdf5_path or not lerobot_key:
        raise ValueError(f"Invalid mapping: {spec}")
    return FeatureMapping(hdf5_path=hdf5_path, lerobot_key=lerobot_key, kind=kind, dtype=dtype)


def find_first_dataset(group: h5py.Group | h5py.File, candidates: Iterable[str]) -> str | None:
    for path in candidates:
        if isinstance(get_node(group, path), h5py.Dataset):
            return path
    return None


def is_image_like_dataset(path: str, dataset: h5py.Dataset, include_depth: bool = False) -> bool:
    lower_path = path.lower()
    if not include_depth and "depth" in lower_path:
        return False

    name_hint = any(token in lower_path for token in ("image", "rgb", "camera", "cam"))
    shape = dataset.shape
    if not shape:
        return False

    if len(shape) == 4:
        dims = shape[1:]
        return name_hint and (dims[0] in (1, 3, 4) or dims[-1] in (1, 3, 4))

    if len(shape) in (1, 2):
        dtype = dataset.dtype
        is_encoded = dtype.kind in ("O", "S", "V") or dtype == np.uint8
        return name_hint and is_encoded

    return False


def discover_image_mappings(group: h5py.Group | h5py.File, include_depth: bool) -> list[FeatureMapping]:
    mappings: list[FeatureMapping] = []

    for root in COMMON_IMAGE_ROOTS:
        node = get_node(group, root)
        if not isinstance(node, h5py.Group):
            continue
        for rel_path, dataset in sorted(iter_datasets(node), key=lambda item: natural_key(item[0])):
            full_path = join_hdf5_path(root, rel_path)
            if not is_image_like_dataset(full_path, dataset, include_depth=include_depth):
                continue
            camera = sanitize_feature_leaf(rel_path)
            mappings.append(FeatureMapping(full_path, f"observation.images.{camera}", "image"))
        if mappings:
            return mappings

    for path, dataset in sorted(iter_datasets(group), key=lambda item: natural_key(item[0])):
        if is_image_like_dataset(path, dataset, include_depth=include_depth):
            camera = sanitize_feature_leaf(path)
            mappings.append(FeatureMapping(path, f"observation.images.{camera}", "image"))

    return mappings


def resolve_mappings(group: h5py.Group | h5py.File, args: argparse.Namespace) -> list[FeatureMapping]:
    mappings: list[FeatureMapping] = []

    state_mapping = parse_mapping(args.state, "lowdim", default_key="observation.state") if args.state else None
    if state_mapping is None:
        state_path = find_first_dataset(group, COMMON_STATE_PATHS)
        if state_path:
            state_mapping = FeatureMapping(state_path, "observation.state", "lowdim")
    if state_mapping is None:
        raise RuntimeError(
            "Could not auto-detect robot state. Pass --state HDF5_PATH=observation.state.\n"
            "Available datasets:\n"
            + format_dataset_listing(group)
        )
    mappings.append(state_mapping)

    action_mapping = parse_mapping(args.action, "lowdim", default_key="action") if args.action else None
    if action_mapping is None:
        action_path = find_first_dataset(group, COMMON_ACTION_PATHS)
        if action_path:
            action_mapping = FeatureMapping(action_path, "action", "lowdim")
    if action_mapping is None:
        raise RuntimeError(
            "Could not auto-detect action. Pass --action HDF5_PATH=action.\n"
            "Available datasets:\n"
            + format_dataset_listing(group)
        )
    mappings.append(action_mapping)

    if args.include_extra_defaults:
        existing_keys = {mapping.lerobot_key for mapping in mappings}
        for hdf5_path, lerobot_key in COMMON_EXTRA_LOWDIM:
            if lerobot_key not in existing_keys and isinstance(get_node(group, hdf5_path), h5py.Dataset):
                mappings.append(FeatureMapping(hdf5_path, lerobot_key, "lowdim"))
                existing_keys.add(lerobot_key)

    for spec in args.feature or []:
        mappings.append(parse_mapping(spec, "lowdim"))

    if args.image:
        mappings.extend(parse_mapping(spec, "image") for spec in args.image)
    else:
        mappings.extend(discover_image_mappings(group, include_depth=args.include_depth_images))

    return mappings


def decode_hdf5_string(value: Any) -> str:
    if isinstance(value, h5py.Dataset):
        if value.shape == ():
            value = value[()]
        else:
            value = value[0]
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif value.size == 1:
            value = value.reshape(-1)[0]
        else:
            value = value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def get_task_text(group: h5py.Group | h5py.File, args: argparse.Namespace, episode_index: int) -> str:
    if args.task_key:
        node = get_node(group, args.task_key)
        if node is None:
            raise KeyError(f"Task key not found: {args.task_key}")
        return decode_hdf5_string(node)

    for attr_name in ("task", "language_instruction", "instruction"):
        if attr_name in group.attrs:
            return decode_hdf5_string(group.attrs[attr_name])

    try:
        return args.task.format(episode_index=episode_index)
    except Exception:
        return args.task


def lowdim_frame(value: Any, dtype: str, flatten: bool) -> np.ndarray:
    array = np.asarray(value)
    if array.shape == ():
        array = array.reshape(1)
    if flatten:
        array = array.reshape(-1)

    if dtype in ("float32", "float"):
        return array.astype(np.float32, copy=False)
    if dtype == "float64":
        return array.astype(np.float64, copy=False)
    if dtype == "int64":
        return array.astype(np.int64, copy=False)
    if dtype == "int32":
        return array.astype(np.int32, copy=False)
    if dtype == "bool":
        return array.astype(np.bool_, copy=False)
    return array.astype(dtype, copy=False)


def decode_encoded_image(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("S", "O"):
            value = value.reshape(-1)[0]
        elif value.dtype == np.uint8:
            value = value.tobytes()
    if isinstance(value, np.void):
        value = bytes(value)
    if isinstance(value, np.bytes_):
        value = bytes(value)
    if not isinstance(value, (bytes, bytearray)):
        raise TypeError(f"Cannot decode encoded image from value with type {type(value)!r}")

    try:
        import cv2  # type: ignore

        raw = np.frombuffer(value, dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("cv2.imdecode returned None")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception:
        from PIL import Image

        import io

        with Image.open(io.BytesIO(value)) as image:
            return np.asarray(image.convert("RGB"))


def normalize_image(value: Any) -> np.ndarray:
    array = np.asarray(value)

    encoded = False
    if array.ndim in (0, 1) and array.dtype.kind in ("S", "O", "V"):
        encoded = True
    if array.ndim == 1 and array.dtype == np.uint8 and array.size > 256:
        encoded = True

    if encoded:
        array = decode_encoded_image(value)

    if array.ndim == 2:
        array = array[:, :, None]

    if array.ndim != 3:
        raise ValueError(f"Expected image frame with 2 or 3 dims, got shape {array.shape}")

    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        chw = array
    elif array.shape[-1] in (1, 3, 4):
        chw = np.transpose(array, (2, 0, 1))
    else:
        raise ValueError(f"Cannot infer channel dimension for image shape {array.shape}")

    if chw.shape[0] == 4:
        chw = chw[:3]
    if chw.shape[0] == 1:
        chw = np.repeat(chw, 3, axis=0)

    if np.issubdtype(chw.dtype, np.floating):
        max_value = float(np.nanmax(chw)) if chw.size else 1.0
        if max_value <= 1.0:
            chw = chw * 255.0
        chw = np.nan_to_num(chw, nan=0.0, posinf=255.0, neginf=0.0)

    if chw.dtype != np.uint8:
        chw = np.clip(chw, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(chw)


def feature_shape_for_lowdim(dataset: h5py.Dataset, dtype: str, flatten: bool) -> tuple[int, ...]:
    sample = lowdim_frame(dataset[0], dtype=dtype, flatten=flatten)
    return tuple(sample.shape)


def get_episode_length(group: h5py.Group | h5py.File, mappings: list[FeatureMapping]) -> int:
    action_mapping = next((mapping for mapping in mappings if mapping.lerobot_key == "action"), mappings[0])
    action_dataset = get_dataset(group, action_mapping.hdf5_path)
    if len(action_dataset.shape) < 1:
        raise ValueError(f"Action dataset must have a time dimension: {action_mapping.hdf5_path}")
    length = int(action_dataset.shape[0])

    for mapping in mappings:
        dataset = get_dataset(group, mapping.hdf5_path)
        if len(dataset.shape) < 1:
            raise ValueError(f"Mapped dataset must have a time dimension: {mapping.hdf5_path}")
        if int(dataset.shape[0]) != length:
            raise ValueError(
                f"Dataset {mapping.hdf5_path} has {dataset.shape[0]} frames, expected {length}."
            )
    return length


def build_features(
    group: h5py.Group | h5py.File,
    mappings: list[FeatureMapping],
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}

    for mapping in mappings:
        dataset = get_dataset(group, mapping.hdf5_path)
        if mapping.kind == "image":
            sample = normalize_image(dataset[0])
            channels, height, width = sample.shape
            features[mapping.lerobot_key] = {
                "dtype": "video" if args.use_videos else "image",
                "shape": (channels, height, width),
                "names": ["channel", "height", "width"],
            }
        else:
            shape = feature_shape_for_lowdim(dataset, dtype=mapping.dtype, flatten=args.flatten_lowdim)
            features[mapping.lerobot_key] = {
                "dtype": mapping.dtype,
                "shape": shape,
                "names": None,
            }

    return features


def print_conversion_summary(mappings: list[FeatureMapping], features: dict[str, dict[str, Any]]) -> None:
    print("Resolved mappings:")
    for mapping in mappings:
        print(f"  {mapping.hdf5_path} -> {mapping.lerobot_key} ({mapping.kind})")
    print("LeRobot features:")
    for key, spec in features.items():
        print(f"  {key}: dtype={spec['dtype']} shape={spec['shape']}")


def import_lerobot_dataset() -> type:
    errors = []
    for module_name in (
        "lerobot.datasets",
        "lerobot.datasets.lerobot_dataset",
        "lerobot.common.datasets.lerobot_dataset",
    ):
        try:
            module = __import__(module_name, fromlist=["LeRobotDataset"])
            return getattr(module, "LeRobotDataset")
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")
    raise ImportError(
        "Could not import LeRobotDataset. Install LeRobot first, for example:\n"
        "  pip install -U lerobot\n\n"
        + "\n".join(errors)
    )


def create_lerobot_dataset(
    lerobot_dataset_cls: type,
    output_root: Path,
    repo_id: str,
    features: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> Any:
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "fps": args.fps,
        "features": features,
        "root": output_root,
        "robot_type": args.robot_type,
        "use_videos": args.use_videos,
        "image_writer_processes": args.image_writer_processes,
        "image_writer_threads": args.image_writer_threads,
        "batch_encoding_size": args.batch_encoding_size,
        "streaming_encoding": args.streaming_encoding,
        "metadata_buffer_size": args.metadata_buffer_size,
        "data_files_size_in_mb": args.data_file_mb,
        "video_files_size_in_mb": args.video_file_mb,
    }

    signature = inspect.signature(lerobot_dataset_cls.create)
    supported_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return lerobot_dataset_cls.create(**supported_kwargs)


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output path already exists. Use --overwrite to replace it: {output_root}")
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)


def write_episode(
    lerobot_dataset: Any,
    episode_ref: EpisodeRef,
    mappings: list[FeatureMapping],
    args: argparse.Namespace,
    episode_index: int,
) -> None:
    with h5py.File(episode_ref.file_path, "r") as h5_file:
        group = open_group(h5_file, episode_ref.group_path)
        length = get_episode_length(group, mappings)
        if length == 0:
            LOG.warning("Skipping empty episode: %s %s", episode_ref.file_path, episode_ref.group_path)
            return

        datasets = {mapping: get_dataset(group, mapping.hdf5_path) for mapping in mappings}
        task = get_task_text(group, args, episode_index)

        for frame_index in range(length):
            frame: dict[str, Any] = {"task": task}

            for mapping, dataset in datasets.items():
                if mapping.kind == "image":
                    frame[mapping.lerobot_key] = normalize_image(dataset[frame_index])
                else:
                    frame[mapping.lerobot_key] = lowdim_frame(
                        dataset[frame_index],
                        dtype=mapping.dtype,
                        flatten=args.flatten_lowdim,
                    )

            lerobot_dataset.add_frame(frame)

    lerobot_dataset.save_episode()


def parse_episode_filter(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    selected: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            selected.update(range(int(start), int(end) + 1))
        else:
            selected.add(int(part))
    return selected


def progress(iterable: Iterable[tuple[int, EpisodeRef]], total: int) -> Iterable[tuple[int, EpisodeRef]]:
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc="Converting episodes")
    except Exception:
        return iterable


def verify_dataset(lerobot_dataset_cls: type, repo_id: str, output_root: Path) -> None:
    dataset = lerobot_dataset_cls(repo_id=repo_id, root=output_root, episodes=[0])
    sample = dataset[0]
    print(f"Verified dataset read. First sample keys: {sorted(sample.keys())}")


def default_repo_id(output_root: Path) -> str:
    name = re.sub(r"[^0-9A-Za-z_.-]+", "_", output_root.name).strip("_")
    return f"local/{name or 'hdf5_dataset'}"


def configure_local_hf_cache(output_root: Path) -> None:
    cache_root = output_root.parent / ".hf-cache"
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--input", required=True, help="Input .hdf5/.h5 file or directory containing files.")
    parser.add_argument("--output", required=True, help="Output LeRobot dataset root directory.")
    parser.add_argument("--repo-id", default=None, help="LeRobot repo id, for example my-org/my-dataset.")
    parser.add_argument("--fps", type=int, required=True, help="Frames per second for the dataset.")
    parser.add_argument("--task", default="unknown task", help="Constant task text. Can include {episode_index}.")
    parser.add_argument("--task-key", default=None, help="HDF5 dataset path containing task text.")
    parser.add_argument("--robot-type", default="unknown", help="Robot type metadata.")

    parser.add_argument("--input-glob", default=None, help="Glob for directory input, for example '*.hdf5'.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search for HDF5 files under --input.")
    parser.add_argument(
        "--episode-root",
        default=None,
        help="Optional HDF5 group whose children are episodes, for example data or episodes.",
    )
    parser.add_argument("--episodes", default=None, help="Episode indices to convert, for example 0,2,5-10.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Convert at most this many episodes.")

    parser.add_argument("--state", default=None, help="Mapping for state: HDF5_PATH=observation.state.")
    parser.add_argument("--action", default=None, help="Mapping for action: HDF5_PATH=action.")
    parser.add_argument(
        "--image",
        action="append",
        default=None,
        help="Image mapping, repeatable: HDF5_PATH=observation.images.camera_name.",
    )
    parser.add_argument(
        "--feature",
        action="append",
        default=None,
        help="Extra low-dimensional mapping, repeatable: HDF5_PATH=LEROBOT_KEY or HDF5_PATH=LEROBOT_KEY:dtype.",
    )
    parser.add_argument("--include-depth-images", action="store_true", help="Allow auto-detected depth image datasets.")
    parser.add_argument(
        "--no-include-extra-defaults",
        dest="include_extra_defaults",
        action="store_false",
        help="Do not auto-include qvel/effort if present.",
    )
    parser.set_defaults(include_extra_defaults=True)

    parser.add_argument(
        "--keep-lowdim-shape",
        dest="flatten_lowdim",
        action="store_false",
        help="Keep mapped low-dimensional arrays in their original per-frame shape instead of flattening.",
    )
    parser.set_defaults(flatten_lowdim=True)

    parser.add_argument("--no-videos", dest="use_videos", action="store_false", help="Store image frames instead of MP4 videos.")
    parser.set_defaults(use_videos=True)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--batch-encoding-size", type=int, default=1)
    parser.add_argument("--streaming-encoding", action="store_true")
    parser.add_argument("--metadata-buffer-size", type=int, default=10)
    parser.add_argument("--data-file-mb", type=int, default=100)
    parser.add_argument("--video-file-mb", type=int, default=500)

    parser.add_argument("--overwrite", action="store_true", help="Delete output directory first if it exists.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect mappings/features without writing output.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip reopening the output dataset after conversion.")
    parser.add_argument("--push-to-hub", action="store_true", help="Call dataset.push_to_hub() after conversion.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    input_path = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    repo_id = args.repo_id or default_repo_id(output_root)
    configure_local_hf_cache(output_root)

    files = collect_hdf5_files(input_path, args.input_glob, args.recursive)
    episode_refs = collect_episode_refs(files, args.episode_root)

    selected = parse_episode_filter(args.episodes)
    if selected is not None:
        episode_refs = [ref for index, ref in enumerate(episode_refs) if index in selected]
    if args.max_episodes is not None:
        episode_refs = episode_refs[: args.max_episodes]
    if not episode_refs:
        raise RuntimeError("No episodes selected for conversion.")

    with h5py.File(episode_refs[0].file_path, "r") as h5_file:
        first_group = open_group(h5_file, episode_refs[0].group_path)
        mappings = resolve_mappings(first_group, args)
        get_episode_length(first_group, mappings)
        features = build_features(first_group, mappings, args)

    print(f"Input files: {len(files)}")
    print(f"Episodes: {len(episode_refs)}")
    print(f"Output root: {output_root}")
    print(f"Repo id: {repo_id}")
    print_conversion_summary(mappings, features)

    if args.dry_run:
        return 0

    prepare_output_root(output_root, overwrite=args.overwrite)
    lerobot_dataset_cls = import_lerobot_dataset()
    lerobot_dataset = create_lerobot_dataset(lerobot_dataset_cls, output_root, repo_id, features, args)

    for episode_index, episode_ref in progress(list(enumerate(episode_refs)), total=len(episode_refs)):
        LOG.info("Converting episode %s: %s [%s]", episode_index, episode_ref.file_path, episode_ref.group_path)
        write_episode(lerobot_dataset, episode_ref, mappings, args, episode_index)

    if hasattr(lerobot_dataset, "finalize"):
        lerobot_dataset.finalize()

    if not args.skip_verify:
        verify_dataset(lerobot_dataset_cls, repo_id, output_root)

    if args.push_to_hub:
        if not hasattr(lerobot_dataset, "push_to_hub"):
            raise AttributeError("This LeRobotDataset version does not expose push_to_hub().")
        lerobot_dataset.push_to_hub()

    print("Conversion complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        LOG.error("%s", exc)
        raise
