"""Dataset adapters: GT loading, detection file paths, frame listing.

Supported datasets:
  - ua-detrac : UA-DETRAC (XML annotations, DETRAC directory layout)
  - mot17     : MOT17 (MOT-format TXT annotations)
  - uavdt     : UAVDT (MOT-like TXT annotations, UAV-benchmark layout)

All adapters return GT in a shared internal format:
  gt_data : {frame_id: [{'id': int, 'bbox': (l,t,w,h), 'class': str}, ...]}
  ignored_regions : [(l, t, w, h), ...]

Detection files (comma-separated, 7-8 cols):
  frame_id, track_id, left, top, width, height, conf[, class]
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_detection_results(det_file: Path) -> dict:
    """Load detections from the project's standard comma-separated format.

    Returns: {frame_id: [(left, top, width, height, conf, cls, track_id), ...]}
    """
    detections: Dict[int, list] = {}
    if not det_file.exists():
        return detections
    with open(det_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            left, top = float(parts[2]), float(parts[3])
            width, height = float(parts[4]), float(parts[5])
            conf = float(parts[6])
            cls = parts[7] if len(parts) >= 8 else "unknown"
            detections.setdefault(frame_id, []).append(
                (left, top, width, height, conf, cls, track_id)
            )
    return detections


class DatasetAdapter:
    """Abstract base; subclasses implement per-dataset I/O."""

    def get_video_split(self, video_name: str) -> str:
        raise NotImplementedError

    def load_gt(
        self, video_name: str, data_root: Path
    ) -> Tuple[Dict[int, List[Dict]], List[Tuple[float, float, float, float]]]:
        raise NotImplementedError

    def get_detection_file(
        self, model_name: str, video_name: str, data_root: Path
    ) -> Path:
        """Return path to the pre-computed detection .txt for this model+video."""
        raise NotImplementedError

    def get_frame_dir(self, video_name: str, data_root: Path) -> Path:
        """Return the directory containing image frames for this video."""
        raise NotImplementedError

    def get_frame_glob(self) -> str:
        """Glob pattern for frame images inside the frame directory."""
        return "img*.jpg"

    def list_videos(
        self, data_root: Path, split: Optional[str] = None
    ) -> List[str]:
        raise NotImplementedError

    def get_image_size(self, video_name: str, data_root: Path) -> Tuple[int, int]:
        """Return (width, height). Subclasses override for dataset-specific sizes."""
        return 960, 540


# ---------------------------------------------------------------------------
# UA-DETRAC
# ---------------------------------------------------------------------------

class UADETRACAdapter(DatasetAdapter):
    """UA-DETRAC (960×540, XML annotations, DETRAC-{split}-Detections layout).

    Expected data_root::

        data_root/
            DETRAC-Images/MVI_xxxxx/imgXXXXX.jpg
            DETRAC-Annos/DETRAC-{Train,Test}-Annotations-XML/MVI_xxxxx.xml
            DETRAC-Detections/DETRAC-{Train,Test}-Detections/{MODEL}/
                MVI_xxxxx_Det_{MODEL}.txt
    """

    TRAIN_IDS = {
        20011, 20012, 20032, 20033, 20034, 20035, 20051, 20052,
        20061, 20062, 20063, 20064, 20065,
        39761, 39771, 39781, 39801, 39811, 39821, 39851, 39861, 39931,
        40131, 40141, 40152, 40161, 40162, 40171, 40172, 40181, 40191,
        40192, 40201, 40204, 40211, 40212, 40213, 40241, 40243, 40244,
        40732, 40751, 40752, 40871, 40962, 40963, 40981, 40991, 40992,
        41063, 41073,
        63521, 63525, 63544, 63552, 63553, 63554, 63561, 63562, 63563,
    }

    def get_video_split(self, video_name: str) -> str:
        return "train" if int(video_name.split("_")[1]) in self.TRAIN_IDS else "test"

    def load_gt(self, video_name: str, data_root: Path):
        split = self.get_video_split(video_name)
        subdir = (
            "DETRAC-Train-Annotations-XML" if split == "train"
            else "DETRAC-Test-Annotations-XML"
        )
        xml_path = data_root / "DETRAC-Annos" / subdir / f"{video_name}.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"GT not found: {xml_path}")

        tree = ET.parse(xml_path)
        root = tree.getroot()

        ignored_regions = []
        ign = root.find("ignored_region")
        if ign is not None:
            for box in ign.findall("box"):
                ignored_regions.append((
                    float(box.get("left", "0")), float(box.get("top", "0")),
                    float(box.get("width", "0")), float(box.get("height", "0")),
                ))

        gt_data: Dict[int, list] = {}
        for frame in root.findall("frame"):
            fnum = frame.get("num")
            if fnum is None:
                continue
            fid = int(fnum)
            targets = []
            tlist = frame.find("target_list")
            if tlist is not None:
                for tgt in tlist.findall("target"):
                    tid = tgt.get("id")
                    box = tgt.find("box")
                    attr = tgt.find("attribute")
                    if tid is None or box is None or attr is None:
                        continue
                    targets.append({
                        "id": int(tid),
                        "bbox": (
                            float(box.get("left", "0")), float(box.get("top", "0")),
                            float(box.get("width", "0")), float(box.get("height", "0")),
                        ),
                        "class": attr.get("vehicle_type", "car"),
                    })
            gt_data[fid] = targets

        return gt_data, ignored_regions

    def get_detection_file(self, model_name: str, video_name: str, data_root: Path) -> Path:
        split = self.get_video_split(video_name)
        mdir = model_name.upper().replace("_", "-")
        return (
            data_root / "DETRAC-Detections"
            / f"DETRAC-{split.capitalize()}-Detections"
            / mdir / f"{video_name}_Det_{mdir}.txt"
        )

    def get_frame_dir(self, video_name: str, data_root: Path) -> Path:
        return data_root / "DETRAC-Images" / video_name

    def list_videos(self, data_root: Path, split: Optional[str] = None) -> List[str]:
        images_dir = data_root / "DETRAC-Images"
        if not images_dir.exists():
            return []
        videos = sorted(d.name for d in images_dir.iterdir() if d.is_dir())
        if split:
            videos = [v for v in videos if self.get_video_split(v) == split]
        return videos

    def get_image_size(self, video_name: str, data_root: Path) -> Tuple[int, int]:
        return 960, 540


# ---------------------------------------------------------------------------
# MOT17
# ---------------------------------------------------------------------------

class MOT17Adapter(DatasetAdapter):
    """MOT17 (1920×1080 default, MOT-format TXT annotations).

    Expected data_root::

        data_root/
            train/
                MOT17-02-DPM/
                    gt/gt.txt
                    img1/000001.jpg ...
                    seqinfo.ini
                MOT17-04-DPM/ ...
            test/
                MOT17-01-DPM/ ...
            Detections/
                {MODEL}/
                    {video_name}_Det_{MODEL}.txt

    Each detector variant (-DPM, -FRCNN, -SDP) is a separate sequence folder.

    GT class IDs kept: 1=pedestrian, 2=person-on-vehicle, 3=car,
    4=motorcycle, 5=bicycle, 6=motorized vehicle, 7=non-motorized vehicle.
    Rows with conf=0 ("don't care") and classes 0, 8-12 are skipped.
    """

    TRAIN_BASES = {
        "MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09",
        "MOT17-10", "MOT17-11", "MOT17-13",
    }
    VALID_CLASSES = {1, 2, 3, 4, 5, 6, 7}

    def get_video_split(self, video_name: str) -> str:
        base = "-".join(video_name.split("-")[:2])  # 'MOT17-02-DPM' → 'MOT17-02'
        return "train" if base in self.TRAIN_BASES else "test"

    def load_gt(self, video_name: str, data_root: Path):
        split = self.get_video_split(video_name)
        gt_file = data_root / split / video_name / "gt" / "gt.txt"
        if not gt_file.exists():
            raise FileNotFoundError(f"GT not found: {gt_file}")

        gt_data: Dict[int, list] = {}
        with open(gt_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 6:
                    continue
                fid = int(parts[0])
                obj_id = int(parts[1])
                left, top = float(parts[2]), float(parts[3])
                w, h = float(parts[4]), float(parts[5])
                conf = float(parts[6]) if len(parts) > 6 else 1.0
                cls = int(parts[7]) if len(parts) > 7 else 1

                if conf == 0 or cls not in self.VALID_CLASSES:
                    continue

                gt_data.setdefault(fid, []).append({
                    "id": obj_id,
                    "bbox": (left, top, w, h),
                    "class": str(cls),
                })

        return gt_data, []  # no separate ignored regions

    def get_detection_file(self, model_name: str, video_name: str, data_root: Path) -> Path:
        mdir = model_name.upper().replace("_", "-")
        return data_root / "Detections" / mdir / f"{video_name}_Det_{mdir}.txt"

    def get_frame_dir(self, video_name: str, data_root: Path) -> Path:
        split = self.get_video_split(video_name)
        return data_root / split / video_name / "img1"

    def get_frame_glob(self) -> str:
        return "*.jpg"

    def list_videos(self, data_root: Path, split: Optional[str] = None) -> List[str]:
        videos = []
        for s in ([split] if split else ["train", "test"]):
            d = data_root / s
            if d.exists():
                videos.extend(sorted(x.name for x in d.iterdir() if x.is_dir()))
        return videos

    def get_image_size(self, video_name: str, data_root: Path) -> Tuple[int, int]:
        split = self.get_video_split(video_name)
        seqinfo = data_root / split / video_name / "seqinfo.ini"
        if seqinfo.exists():
            w = h = None
            with open(seqinfo) as f:
                for line in f:
                    if line.startswith("imWidth"):
                        w = int(line.split("=")[1].strip())
                    elif line.startswith("imHeight"):
                        h = int(line.split("=")[1].strip())
            if w and h:
                return w, h
        return 1920, 1080


# ---------------------------------------------------------------------------
# UAVDT
# ---------------------------------------------------------------------------

class UAVDTAdapter(DatasetAdapter):
    """UAVDT (1080×540, MOT-like TXT annotations, UAV-benchmark layout).

    Expected data_root::

        data_root/
            UAV-benchmark-M/
                M0101/img000001.jpg ...
                M0102/ ...
            UAV-benchmark-MOTD_v1.0/
                GT/
                    M0101_gt_whole.txt
                    M0102_gt_whole.txt ...
            Detections/
                {MODEL}/
                    {video_name}_Det_{MODEL}.txt

    GT columns: frame, id, left, top, width, height, score, in_view, occlusion
    Objects with in_view=0 (outside camera view) are excluded.

    UAVDT has no standard detection-benchmark train/test split.  All sequences
    are reported as "train" by default; use --hardcoded-split or explicit
    --videos lists when a custom split is needed.
    """

    def get_video_split(self, video_name: str) -> str:
        return "train"

    def load_gt(self, video_name: str, data_root: Path):
        gt_dir = data_root / "UAV-benchmark-MOTD_v1.0" / "GT"
        gt_file = gt_dir / f"{video_name}_gt_whole.txt"
        if not gt_file.exists():
            raise FileNotFoundError(f"GT not found: {gt_file}")

        gt_data: Dict[int, list] = {}
        with open(gt_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 6:
                    continue
                fid = int(parts[0])
                obj_id = int(parts[1])
                left, top = float(parts[2]), float(parts[3])
                w, h = float(parts[4]), float(parts[5])
                in_view = int(parts[7]) if len(parts) > 7 else 1

                if in_view == 0:
                    continue

                gt_data.setdefault(fid, []).append({
                    "id": obj_id,
                    "bbox": (left, top, w, h),
                    "class": "vehicle",
                })

        # Load per-frame ignore regions; take the union across all frames
        # so the static ignored_regions interface stays compatible.
        ignored_regions = []
        ignore_file = gt_dir / f"{video_name}_gt_ignore.txt"
        if ignore_file.exists():
            seen = set()
            with open(ignore_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) < 6:
                        continue
                    bbox = (float(parts[2]), float(parts[3]),
                            float(parts[4]), float(parts[5]))
                    if bbox not in seen:
                        seen.add(bbox)
                        ignored_regions.append(bbox)

        return gt_data, ignored_regions

    def get_detection_file(self, model_name: str, video_name: str, data_root: Path) -> Path:
        mdir = model_name.upper().replace("_", "-")
        return data_root / "Detections" / mdir / f"{video_name}_Det_{mdir}.txt"

    def get_frame_dir(self, video_name: str, data_root: Path) -> Path:
        return data_root / "UAV-benchmark-M" / video_name

    def get_frame_glob(self) -> str:
        return "img*.jpg"

    def list_videos(self, data_root: Path, split: Optional[str] = None) -> List[str]:
        images_dir = data_root / "UAV-benchmark-M"
        if not images_dir.exists():
            return []
        return sorted(d.name for d in images_dir.iterdir() if d.is_dir())

    def get_image_size(self, video_name: str, data_root: Path) -> Tuple[int, int]:
        return 1080, 540


# ---------------------------------------------------------------------------
# ImageNet VID (ILSVRC 2015)
# ---------------------------------------------------------------------------

class ImageNetVIDAdapter(DatasetAdapter):
    """ILSVRC 2015 VID task.

    Expected data_root layout::

        data_root/
            Data/VID/
                train/
                    ILSVRC2015_VID_train_0000/
                        ILSVRC2015_train_00000000/
                            000000.JPEG, 000001.JPEG, ...
                val/
                    ILSVRC2015_val_00000000/
                        000000.JPEG, ...
            Annotations/VID/
                train/
                    ILSVRC2015_VID_train_0000/
                        ILSVRC2015_train_00000000/
                            000000.xml, 000001.xml, ...
                val/
                    ILSVRC2015_val_00000000/
                        000000.xml, ...
            Detections/
                YOLO11S/
                    ILSVRC2015_val_00000000_Det_YOLO11S.txt

    video_name = leaf clip directory name (e.g. "ILSVRC2015_val_00000000").
    Per-frame XML annotations use PASCAL VOC bndbox (xmin/ymin/xmax/ymax).
    Object class is a WordNet synset ID (e.g. n02691156); GT matching is IoU-based.
    val clips are treated as the "test" split; train clips as "train".
    """

    def __init__(self):
        self._cache: Dict[str, Dict[str, Tuple[Path, Path]]] = {}

    def _build_cache(self, data_root: Path) -> Dict[str, Tuple[Path, Path]]:
        key = str(data_root)
        if key in self._cache:
            return self._cache[key]

        clip_map: Dict[str, Tuple[Path, Path]] = {}
        for split in ("val", "train"):
            data_split = data_root / "Data" / "VID" / split
            anno_split = data_root / "Annotations" / "VID" / split
            if not data_split.exists():
                continue
            for entry in sorted(data_split.iterdir()):
                if not entry.is_dir():
                    continue
                if "VID_train" in entry.name:
                    # Group directory — one level deeper holds actual clips
                    for clip_dir in sorted(entry.iterdir()):
                        if clip_dir.is_dir():
                            anno_dir = anno_split / entry.name / clip_dir.name
                            clip_map[clip_dir.name] = (clip_dir, anno_dir)
                else:
                    # Direct clip directory (val layout)
                    clip_map[entry.name] = (entry, anno_split / entry.name)

        self._cache[key] = clip_map
        return clip_map

    def get_video_split(self, video_name: str) -> str:
        return "test" if "val" in video_name else "train"

    # Only evaluate on car objects; other synsets (elephant, bicycle, etc.) pollute mAP.
    CAR_SYNSET = "n02958343"

    def load_gt(self, video_name: str, data_root: Path):
        clip_map = self._build_cache(data_root)
        if video_name not in clip_map:
            raise FileNotFoundError(f"Clip not found: {video_name}")
        _, anno_dir = clip_map[video_name]
        if not anno_dir.exists():
            raise FileNotFoundError(f"Annotation dir not found: {anno_dir}")

        gt_data: Dict[int, list] = {}
        for xml_path in sorted(anno_dir.glob("*.xml")):
            fid = int(xml_path.stem)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            objects = []
            for obj in root.findall("object"):
                tid_elem = obj.find("trackid")
                name_elem = obj.find("name")
                bndbox = obj.find("bndbox")
                if None in (tid_elem, name_elem, bndbox):
                    continue
                if name_elem.text != self.CAR_SYNSET:
                    continue
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
                objects.append({
                    "id": int(tid_elem.text),
                    "bbox": (xmin, ymin, xmax - xmin, ymax - ymin),
                    "class": name_elem.text,
                })
            gt_data[fid] = objects

        return gt_data, []

    def get_detection_file(self, model_name: str, video_name: str, data_root: Path) -> Path:
        mdir = model_name.upper().replace("_", "-")
        return data_root / "Detections" / mdir / f"{video_name}_Det_{mdir}.txt"

    def get_frame_dir(self, video_name: str, data_root: Path) -> Path:
        clip_map = self._build_cache(data_root)
        if video_name not in clip_map:
            raise FileNotFoundError(f"Clip not found: {video_name}")
        return clip_map[video_name][0]

    def get_frame_glob(self) -> str:
        return "*.JPEG"

    def list_videos(self, data_root: Path, split: Optional[str] = None) -> List[str]:
        clip_map = self._build_cache(data_root)
        if split:
            return sorted(k for k in clip_map if self.get_video_split(k) == split)
        return sorted(clip_map.keys())

    def get_image_size(self, video_name: str, data_root: Path) -> Tuple[int, int]:
        clip_map = self._build_cache(data_root)
        if video_name not in clip_map:
            return 1280, 720
        _, anno_dir = clip_map[video_name]
        first_xml = next(iter(sorted(anno_dir.glob("*.xml"))), None)
        if first_xml:
            try:
                tree = ET.parse(first_xml)
                size = tree.getroot().find("size")
                if size is not None:
                    return int(size.find("width").text), int(size.find("height").text)
            except Exception:
                pass
        return 1280, 720


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ADAPTERS = {
    "ua-detrac":    UADETRACAdapter,
    "mot17":        MOT17Adapter,
    "uavdt":        UAVDTAdapter,
    "imagenet-vid": ImageNetVIDAdapter,
}

DATASET_NAMES = list(_ADAPTERS)


def get_adapter(dataset_name: str) -> DatasetAdapter:
    key = dataset_name.lower()
    if key not in _ADAPTERS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from: {DATASET_NAMES}")
    return _ADAPTERS[key]()
