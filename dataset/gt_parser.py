"""Parse UA-DETRAC ground truth annotations from XML files."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple


def parse_gt_xml(xml_path: Path) -> Tuple[Dict[int, List[Dict]], List[Tuple[float, float, float, float]]]:
    """
    Parse ground truth XML file for a single video.

    Args:
        xml_path: Path to XML annotation file

    Returns:
        Tuple of:
        - Dictionary mapping frame_id -> list of ground truth objects
          Each object: {'id': int, 'bbox': (left, top, width, height), 'class': str}
        - List of ignored regions (left, top, width, height)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Parse ignored regions (applicable to all frames)
    ignored_regions = []
    ignored_region_elem = root.find('ignored_region')
    if ignored_region_elem is not None:
        for box in ignored_region_elem.findall('box'):
            left = float(box.get('left', '0'))
            top = float(box.get('top', '0'))
            width = float(box.get('width', '0'))
            height = float(box.get('height', '0'))
            ignored_regions.append((left, top, width, height))

    gt_data = {}

    for frame in root.findall('frame'):
        frame_num = frame.get('num')
        if frame_num is None:
            continue
        frame_id = int(frame_num)
        targets = []

        target_list = frame.find('target_list')
        if target_list is not None:
            for target in target_list.findall('target'):
                target_id_str = target.get('id')
                if target_id_str is None:
                    continue
                target_id = int(target_id_str)

                # Parse bbox
                box = target.find('box')
                if box is None:
                    continue
                left = float(box.get('left', '0'))
                top = float(box.get('top', '0'))
                width = float(box.get('width', '0'))
                height = float(box.get('height', '0'))

                # Parse vehicle type
                attribute = target.find('attribute')
                if attribute is None:
                    continue
                vehicle_type = attribute.get('vehicle_type', 'car')

                targets.append({
                    'id': target_id,
                    'bbox': (left, top, width, height),
                    'class': vehicle_type
                })

        gt_data[frame_id] = targets

    return gt_data, ignored_regions


def load_gt_for_video(video_name: str, data_root: Path, split: str = 'train') -> Tuple[Dict[int, List[Dict]], List[Tuple[float, float, float, float]]]:
    """
    Load ground truth for a specific video.
    
    Args:
        video_name: Video name (e.g., 'MVI_20011')
        data_root: Root path to UA-DETRAC dataset
        split: 'train' or 'test'
        
    Returns:
        Tuple of:
        - Dictionary mapping frame_id -> list of GT objects
        - List of ignored regions (left, top, width, height)
    """
    if split == 'train':
        xml_dir = data_root / 'DETRAC-Annos' / 'DETRAC-Train-Annotations-XML'
    else:
        xml_dir = data_root / 'DETRAC-Annos' / 'DETRAC-Test-Annotations-XML'
    
    xml_path = xml_dir / f'{video_name}.xml'
    
    if not xml_path.exists():
        raise FileNotFoundError(f"GT annotation not found: {xml_path}")
    
    return parse_gt_xml(xml_path)


def get_video_split(video_name: str) -> str:
    """
    Determine if video belongs to train or test split.

    Args:
        video_name: Video name (e.g., 'MVI_20011')

    Returns:
        'train' or 'test'
    """
    # Extract video number
    video_num = int(video_name.split('_')[1])

    # Train video IDs (exact list from DETRAC-Train-Annotations-XML)
    # Note: The 407xx-409xx range is NOT continuous - some videos are in test set
    train_video_ids = {
        20011, 20012, 20032, 20033, 20034, 20035, 20051, 20052, 20061, 20062, 20063, 20064, 20065,
        39761, 39771, 39781, 39801, 39811, 39821, 39851, 39861, 39931,
        40131, 40141, 40152, 40161, 40162, 40171, 40172, 40181, 40191, 40192, 40201, 40204, 40211, 40212, 40213, 40241, 40243, 40244,
        40732, 40751, 40752, 40871, 40962, 40963, 40981, 40991, 40992,
        41063, 41073,
        63521, 63525, 63544, 63552, 63553, 63554, 63561, 63562, 63563
    }

    return 'train' if video_num in train_video_ids else 'test'
