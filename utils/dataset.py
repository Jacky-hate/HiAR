from utils.lmdb import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import torch
try:
    import lmdb
except ModuleNotFoundError:
    lmdb = None
import json
import hashlib
import sqlite3
from collections import OrderedDict
import multiprocessing as mp
from pathlib import Path
from PIL import Image
import os


_VERIFIED_SQLITE_INDEXES = set()


class ResumableDistributedSampler(DistributedSampler):
    """DistributedSampler that can skip an already-consumed batch prefix."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_index = 0

    def set_start_batch(self, batch_index: int, batch_size: int):
        self._start_index = int(batch_index) * int(batch_size)

    def __iter__(self):
        indices = list(super().__iter__())
        if self._start_index < 0 or self._start_index > len(indices):
            raise RuntimeError(
                f"sampler start index {self._start_index} is outside "
                f"[0, {len(indices)}]"
            )
        return iter(indices[self._start_index:])


class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class SQLiteT2VVideoDataset(Dataset):
    """Real-video T2V dataset backed by a SQLite manifest.

    Only RGB video and text are consumed. Camera, anchor-frame, pose and depth
    fields are deliberately ignored. Frames are returned as uint8 [C,T,H,W] to
    avoid a large CPU-memory expansion in DataLoader workers; normalization is
    performed immediately before VAE encoding on the GPU.
    """

    _SELECT = """
        SELECT clean_name, video_path, prompt_path, frame_count,
               camera_root, scene_dir
        FROM records
        WHERE scan_error IS NULL
          AND has_prompt = 1
          AND video_path IS NOT NULL AND video_path <> ''
          AND prompt_path IS NOT NULL AND prompt_path <> ''
        ORDER BY clean_name
    """

    @staticmethod
    def _normalize_metadata(value) -> str:
        return "".join(
            character
            for character in str(value or "").casefold()
            if character.isalnum()
        )

    def __init__(
        self,
        index_path: str,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        seed: int = 0,
        target_fps: float = 0.0,
        prompt_cache_size: int = 128,
        decode_video: bool = True,
        align_prompt_segments: bool = True,
        exclude_metadata_patterns=None,
    ):
        if num_frames <= 0 or height <= 0 or width <= 0:
            raise ValueError(
                "num_frames, height and width must all be positive"
            )
        self.index_path = os.path.realpath(index_path)
        self.num_frames = int(num_frames)
        self.height = int(height)
        self.width = int(width)
        self.seed = int(seed)
        self.target_fps = float(target_fps)
        self.decode_video = bool(decode_video)
        self.align_prompt_segments = bool(align_prompt_segments)
        if isinstance(exclude_metadata_patterns, str):
            raise TypeError(
                "exclude_metadata_patterns must be a sequence, not a string"
            )
        self.exclude_metadata_patterns = tuple(
            self._normalize_metadata(pattern)
            for pattern in (exclude_metadata_patterns or ())
        )
        if any(not pattern for pattern in self.exclude_metadata_patterns):
            raise ValueError(
                "exclude_metadata_patterns must not contain empty patterns"
            )
        # Shared memory keeps persistent DataLoader workers in sync when the
        # outer cycle advances to a new DistributedSampler epoch.
        self._epoch = mp.Value("q", 0, lock=False)
        self.prompt_cache_size = int(prompt_cache_size)
        self._prompt_cache = OrderedDict()

        if not os.path.isfile(self.index_path):
            raise FileNotFoundError(f"SQLite video index not found: {index_path}")
        uri = f"file:{self.index_path}?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True) as connection:
            distributed = torch.distributed.is_available() and (
                torch.distributed.is_initialized()
            )
            verify_here = (
                self.index_path not in _VERIFIED_SQLITE_INDEXES
                and (not distributed or torch.distributed.get_rank() == 0)
            )
            if verify_here:
                quick_check = connection.execute(
                    "PRAGMA quick_check"
                ).fetchone()
                if quick_check != ("ok",):
                    raise RuntimeError(
                        "SQLite quick_check failed for "
                        f"{self.index_path}: {quick_check}"
                    )
                _VERIFIED_SQLITE_INDEXES.add(self.index_path)
            records = connection.execute(self._SELECT).fetchall()
            self.records = []
            for record in records:
                if int(record[3]) < self.num_frames:
                    continue
                metadata = tuple(
                    self._normalize_metadata(value) for value in record[4:]
                )
                if any(
                    pattern in value
                    for pattern in self.exclude_metadata_patterns
                    for value in metadata
                ):
                    continue
                self.records.append(record[:4])
        if not self.records:
            raise RuntimeError(
                f"No usable records remain after filtering {self.index_path}"
            )
        record_hasher = hashlib.sha256()
        for record in self.records:
            record_hasher.update(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                .encode("utf-8")
            )
            record_hasher.update(b"\n")
        self.records_digest = record_hasher.hexdigest()

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch: int):
        self._epoch.value = int(epoch)

    @staticmethod
    def _caption(entry) -> str:
        if not isinstance(entry, dict):
            return str(entry).strip()
        start = str(entry.get("start", "")).strip()
        dynamic = str(entry.get("dynamic", "")).strip()
        return " ".join(part for part in (start, dynamic) if part)

    def _load_prompt_segments(self, prompt_path: str):
        cached = self._prompt_cache.pop(prompt_path, None)
        if cached is not None:
            self._prompt_cache[prompt_path] = cached
            return cached
        with open(prompt_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        detailed = payload.get("detailed") if isinstance(payload, dict) else None
        if not isinstance(detailed, dict) or not detailed:
            raise ValueError(
                f"Prompt JSON lacks a non-empty detailed mapping: {prompt_path}"
            )
        frames = sorted((int(key), self._caption(value))
                        for key, value in detailed.items())
        segments = []
        start_frame, previous_frame, caption = (
            frames[0][0], frames[0][0], frames[0][1]
        )
        for frame_index, current_caption in frames[1:]:
            if current_caption == caption and frame_index == previous_frame + 1:
                previous_frame = frame_index
                continue
            segments.append((start_frame, previous_frame, caption))
            start_frame = previous_frame = frame_index
            caption = current_caption
        segments.append((start_frame, previous_frame, caption))
        if self.prompt_cache_size > 0:
            self._prompt_cache[prompt_path] = segments
            while len(self._prompt_cache) > self.prompt_cache_size:
                self._prompt_cache.popitem(last=False)
        return segments

    def _select_prompt(
        self, prompt_path: str, first_frame: int, last_frame: int
    ) -> str:
        segments = self._load_prompt_segments(prompt_path)
        best = max(
            segments,
            key=lambda item: (
                max(0, min(item[1], last_frame) - max(item[0], first_frame) + 1),
                -item[0],
            ),
        )
        if not best[2]:
            raise ValueError(f"Selected empty prompt from {prompt_path}")
        return best[2]

    def _frame_indices(
        self,
        length: int,
        source_fps: float,
        idx: int,
        prompt_path: str = None,
    ):
        if self.target_fps > 0:
            if not np.isfinite(source_fps) or source_fps <= 0:
                raise ValueError(f"Invalid source FPS: {source_fps}")
            stride = source_fps / self.target_fps
        else:
            stride = 1.0
        span = int(np.rint((self.num_frames - 1) * stride))
        max_start = length - span - 1
        if max_start < 0:
            raise ValueError(
                f"Video has {length} frames, insufficient for {self.num_frames} "
                f"samples at stride {stride:.4f}"
            )
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [self.seed, int(self._epoch.value), int(idx)]
            )
        )
        start = None
        if self.align_prompt_segments and prompt_path:
            # Prompt packages describe long videos as frame-indexed temporal
            # segments (typically one caption per 81-frame source clip).  A
            # globally random start almost always straddles two captions.  Draw
            # uniformly from starts whose complete sampled span belongs to one
            # segment instead. T2V has no anchor frame, so the first segment
            # starts exactly at frame zero.
            valid_ranges = []
            total_valid_starts = 0
            for segment_start, segment_end, _caption in (
                self._load_prompt_segments(prompt_path)
            ):
                valid_start = max(0, int(segment_start))
                valid_end = min(
                    int(segment_end) - span,
                    max_start,
                )
                if valid_end < valid_start:
                    continue
                count = valid_end - valid_start + 1
                valid_ranges.append((valid_start, count))
                total_valid_starts += count
            if total_valid_starts > 0:
                selected = int(rng.integers(0, total_valid_starts))
                for valid_start, count in valid_ranges:
                    if selected < count:
                        start = valid_start + selected
                        break
                    selected -= count
        if start is None:
            # Generic fallback for caption formats whose every segment is
            # shorter than the requested temporal span.  _select_prompt still
            # chooses the maximum-overlap caption for this window.
            start = int(rng.integers(0, max_start + 1))
        indices = np.rint(
            start + np.arange(self.num_frames, dtype=np.float64) * stride
        ).astype(np.int64)
        return np.clip(indices, 0, length - 1)

    def __getitem__(self, idx):
        clean_name, video_path, prompt_path, indexed_frames = self.records[idx]
        if not os.path.isfile(prompt_path):
            raise FileNotFoundError(
                f"Prompt JSON missing for {clean_name}: {prompt_path}"
            )

        # The DMD critic consumes text only. A second prompt-only view of this
        # dataset avoids decoding and pinning an 81-frame clip that would be
        # discarded immediately. For the default consecutive-frame policy the
        # indexed frame count is sufficient to select the same deterministic
        # temporal window and its overlapping prompt.
        if not self.decode_video and self.target_fps <= 0:
            actual_frames = int(indexed_frames)
            indices = self._frame_indices(
                actual_frames, 0.0, idx, prompt_path=prompt_path
            )
            prompt = self._select_prompt(
                prompt_path, int(indices[0]), int(indices[-1])
            )
            return {
                "prompts": prompt,
                "clean_name": clean_name,
                "start_frame": int(indices[0]),
                "end_frame": int(indices[-1]),
                "prompt_path": prompt_path,
            }

        try:
            import decord
            if self.decode_video:
                import cv2
        except ImportError as exc:
            raise ImportError(
                "SQLiteT2VVideoDataset requires decord and, when decoding "
                "video, opencv-python"
            ) from exc

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video missing for {clean_name}: {video_path}")
        reader = decord.VideoReader(
            video_path, ctx=decord.cpu(0), num_threads=1
        )
        actual_frames = len(reader)
        if indexed_frames is not None and actual_frames != int(indexed_frames):
            raise RuntimeError(
                f"Frame-count mismatch for {clean_name}: SQLite={indexed_frames}, "
                f"video={actual_frames}"
            )
        indices = self._frame_indices(
            actual_frames,
            float(reader.get_avg_fps()),
            idx,
            prompt_path=prompt_path,
        )
        if not self.decode_video:
            del reader
            prompt = self._select_prompt(
                prompt_path, int(indices[0]), int(indices[-1])
            )
            return {
                "prompts": prompt,
                "clean_name": clean_name,
                "start_frame": int(indices[0]),
                "end_frame": int(indices[-1]),
                "prompt_path": prompt_path,
            }
        frames = reader.get_batch(indices).asnumpy()
        del reader

        source_height, source_width = frames.shape[1:3]
        source_aspect = source_width / source_height
        target_aspect = self.width / self.height
        if source_aspect > target_aspect:
            crop_width = max(1, int(round(source_height * target_aspect)))
            left = (source_width - crop_width) // 2
            frames = frames[:, :, left:left + crop_width]
        elif source_aspect < target_aspect:
            crop_height = max(1, int(round(source_width / target_aspect)))
            top = (source_height - crop_height) // 2
            frames = frames[:, top:top + crop_height, :]
        resized = np.empty(
            (self.num_frames, self.height, self.width, 3), dtype=np.uint8
        )
        for frame_index, frame in enumerate(frames):
            resized[frame_index] = cv2.resize(
                frame,
                (self.width, self.height),
                interpolation=cv2.INTER_LINEAR,
            )
        video = torch.from_numpy(resized).permute(3, 0, 1, 2).contiguous()
        prompt = self._select_prompt(
            prompt_path, int(indices[0]), int(indices[-1])
        )
        return {
            "video": video,
            "prompts": prompt,
            "clean_name": clean_name,
            "start_frame": int(indices[0]),
            "end_frame": int(indices[-1]),
            "video_path": video_path,
            "prompt_path": prompt_path,
        }


class ODERegressionLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        if lmdb is None:
            raise ImportError("LMDB datasets require the optional lmdb package")
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }


class ShardingLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        if lmdb is None:
            raise ImportError("LMDB datasets require the optional lmdb package")
        self.envs = []
        self.index = []

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            env = lmdb.open(path,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, 'latents')
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))

            # print("shard_id ", shard_id, " local_i ", local_i)

        self.max_pair = max_pair

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
            Outputs:
                - prompts: List of Strings
                - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        shard_id, local_idx = self.index[idx]

        latents = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "latents", np.float16, local_idx,
            shape=self.latents_shape[shard_id][1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "prompts", str, local_idx
        )

        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }


class TextImagePairDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        eval_first_n=-1,
        pad_to_multiple_of=None
    ):
        """
        Args:
            data_dir (str): Path to the directory containing:
                - target_crop_info_*.json (metadata file)
                - */ (subdirectory containing images with matching aspect ratio)
            transform (callable, optional): Optional transform to be applied on the image
        """
        self.transform = transform
        data_dir = Path(data_dir)

        # Find the metadata JSON file
        metadata_files = list(data_dir.glob('target_crop_info_*.json'))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        # Extract aspect ratio from metadata filename (e.g. target_crop_info_26-15.json -> 26-15)
        aspect_ratio = metadata_path.stem.split('_')[-1]

        # Use aspect ratio subfolder for images
        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        eval_first_n = eval_first_n if eval_first_n != -1 else len(self.metadata)
        self.metadata = self.metadata[:eval_first_n]

        # Verify all images exist
        for item in self.metadata:
            image_path = self.image_dir / item['file_name']
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        self.dummy_prompt = "DUMMY PROMPT"
        self.pre_pad_len = len(self.metadata)
        if pad_to_multiple_of is not None and len(self.metadata) % pad_to_multiple_of != 0:
            # Duplicate the last entry
            self.metadata += [self.metadata[-1]] * (
                pad_to_multiple_of - len(self.metadata) % pad_to_multiple_of
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary containing:
                - image: PIL Image
                - caption: str
                - target_bbox: list of int [x1, y1, x2, y2]
                - target_ratio: str
                - type: str
                - origin_size: tuple of int (width, height)
        """
        item = self.metadata[idx]

        # Load image
        image_path = self.image_dir / item['file_name']
        image = Image.open(image_path).convert('RGB')

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'prompts': item['caption'],
            'target_bbox': item['target_crop']['target_bbox'],
            'target_ratio': item['target_crop']['target_ratio'],
            'type': item['type'],
            'origin_size': (item['origin_width'], item['origin_height']),
            'idx': idx
        }


class PDPairDataset(Dataset):
    """
    Progressive Distillation dataset: loads .pt files generated by generate_ode_pairs.py.

    Each .pt file format: {prompt_string: tensor[1, num_saves, F, C, H, W]}.
    Version-2 data requires manifest.json and stores points interpolated at the
    student's exact sigma schedule.
    """
    def __init__(self, data_dir: str, require_manifest: bool = True):
        self.data_dir = data_dir
        manifest_path = os.path.join(data_dir, "manifest.json")
        self.manifest = None
        if os.path.isfile(manifest_path):
            import json
            with open(manifest_path, encoding="utf-8") as handle:
                self.manifest = json.load(handle)
            if self.manifest.get("format_version") != 2:
                raise ValueError(
                    "Unsupported PD manifest format_version: "
                    f"{self.manifest.get('format_version')}"
                )
            aligned_sigmas = self.manifest.get("aligned_sigmas")
            if not isinstance(aligned_sigmas, list) or len(aligned_sigmas) < 2:
                raise ValueError("PD manifest must contain aligned_sigmas")
            self.num_saves = len(aligned_sigmas)
        elif require_manifest:
            raise FileNotFoundError(
                f"PD data directory lacks required manifest.json: {data_dir}. "
                "Regenerate pairs with scripts/generate_ode_pairs.py."
            )
        else:
            self.num_saves = None
        self.file_list = sorted([
            f for f in os.listdir(data_dir) if f.endswith('.pt')
        ])
        if len(self.file_list) == 0:
            raise FileNotFoundError(f"No .pt files found in PD data directory: {data_dir}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = os.path.join(self.data_dir, self.file_list[idx])
        data = torch.load(filepath, map_location='cpu', weights_only=True)
        if not isinstance(data, dict) or len(data) != 1:
            raise ValueError(f"Invalid PD pair mapping in {filepath}")
        prompt, latent = next(iter(data.items()))
        # latent shape: [1, num_saves, F, C, H, W] -> squeeze batch dim -> [num_saves, F, C, H, W]

        if latent.dim() != 6 or latent.shape[0] != 1:
            raise ValueError(
                f"PD latent in {filepath} must have shape "
                f"[1, saves, frames, channels, height, width], got {latent.shape}"
            )
        latent = latent.squeeze(0)
        if self.num_saves is not None and latent.shape[0] != self.num_saves:
            raise ValueError(
                f"PD latent in {filepath} has {latent.shape[0]} save points; "
                f"manifest requires {self.num_saves}"
            )
        if not torch.isfinite(latent).all():
            raise ValueError(f"PD latent contains NaN/Inf: {filepath}")
        return {
            "prompts": prompt,
            "ode_latent": latent.float()
        }


def cycle(dl, start_batch: int = 0):
    """Cycle a DataLoader, optionally resuming at a logical batch cursor."""
    start_batch = int(start_batch)
    if start_batch < 0:
        raise ValueError(f"start_batch must be non-negative, got {start_batch}")
    batches_per_epoch = len(dl)
    if batches_per_epoch <= 0:
        raise RuntimeError("DataLoader has no batches")
    epoch, batch_offset = divmod(start_batch, batches_per_epoch)
    while True:
        sampler = getattr(dl, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        sampler_handles_offset = (
            sampler is not None and hasattr(sampler, "set_start_batch")
        )
        if sampler_handles_offset:
            sampler.set_start_batch(batch_offset, dl.batch_size)
        dataset = getattr(dl, "dataset", None)
        if dataset is not None and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        yielded = False
        for batch_index, data in enumerate(dl):
            if not sampler_handles_offset and batch_index < batch_offset:
                continue
            yielded = True
            yield data
        if not yielded:
            raise RuntimeError(
                "DataLoader produced no batches; check dataset size, world size, "
                "batch size, and drop_last"
            )
        epoch += 1
        batch_offset = 0
        if sampler_handles_offset:
            sampler.set_start_batch(0, dl.batch_size)
