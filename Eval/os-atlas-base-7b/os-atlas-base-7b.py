#!/usr/bin/env python3
"""Run OS-Atlas-Base-7B on ScreenSpot and save results to JSON."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info
except ImportError as exc:
    raise ImportError("Missing dependency qwen-vl-utils. Install with: pip install qwen-vl-utils") from exc


NUMBER_PATTERN = r"-?\d+(?:\.\d+)?"
BOX_TOKEN_RE = re.compile(
    rf"<\|box_start\|>\s*\(\s*({NUMBER_PATTERN})\s*,\s*({NUMBER_PATTERN})\s*\)\s*,\s*\(\s*({NUMBER_PATTERN})\s*,\s*({NUMBER_PATTERN})\s*\)\s*<\|box_end\|>"
)
BOX_LIST_RE = re.compile(
    rf"\[\[\s*({NUMBER_PATTERN})\s*,\s*({NUMBER_PATTERN})\s*,\s*({NUMBER_PATTERN})\s*,\s*({NUMBER_PATTERN})\s*\]\]"
)
POINT_TOKEN_RE = re.compile(
    rf"<point>\s*\[\[\s*({NUMBER_PATTERN})\s*,\s*({NUMBER_PATTERN})\s*\]\]\s*</point>"
)
POINT_LIST_RE = re.compile(rf"\[\[\s*({NUMBER_PATTERN})\s*,\s*({NUMBER_PATTERN})\s*\]\]")
POINT_PAREN_RE = re.compile(rf"\(\s*({NUMBER_PATTERN})\s*,\s*({NUMBER_PATTERN})\s*\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OS-Atlas-Base-7B inference on ScreenSpot")
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to OS-Atlas-Base-7B model folder",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing screenspot.json and images folder",
    )
    parser.add_argument(
        "--json-file",
        type=str,
        default="screenspot.json",
        help="JSON annotation file name (relative to --data-root)",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="images",
        help="Image directory name (relative to --data-root)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./results"),
        help="Directory to store results",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=datetime.now().strftime("screenspot_%Y%m%d_%H%M%S"),
        help="Run name used in output file names",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max new tokens for generation",
    )
    parser.add_argument(
        "--coord-space",
        type=str,
        default="auto",
        choices=["auto", "normalized", "pixel"],
        help="Interpretation of model coordinates",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Process only first N samples (0 means all)",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        default=True,
        help="Save visualization images with prediction boxes",
    )
    parser.add_argument(
        "--vis-dir-name",
        type=str,
        default="vis_images",
        help="Subdirectory name for visualization images",
    )
    return parser.parse_args()


def load_model(model_path: Path) -> Tuple[Qwen2VLForConditionalGeneration, AutoProcessor]:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    model.eval()
    return model, processor


def load_annotations(data_root: Path, json_file: str) -> List[Dict[str, Any]]:
    json_path = data_root / json_file
    with json_path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    
    records = []
    for idx, item in enumerate(items):
        row = dict(item)
        row["line_idx"] = idx
        records.append(row)
    return records


def clean_model_text(text: str) -> str:
    return text.replace("", "").strip()


def _need_normalized_conversion(values: List[float], coord_space: str) -> bool:
    if coord_space == "normalized":
        return True
    if coord_space == "pixel":
        return False
    return min(values) >= 0 and max(values) <= 1000


def _norm_to_pixel(x: float, size: int) -> int:
    return int(round((x / 1000.0) * size))


def parse_prediction(
    output_text: str,
    image_width: int,
    image_height: int,
    coord_space: str,
) -> Dict[str, Any]:
    clean_text = clean_model_text(output_text)

    def convert_pair(x: float, y: float, is_normalized: bool) -> Tuple[int, int]:
        if is_normalized:
            return _norm_to_pixel(x, image_width), _norm_to_pixel(y, image_height)
        return int(round(x)), int(round(y))

    box_match = BOX_TOKEN_RE.search(clean_text)
    if box_match:
        vals = [float(box_match.group(i)) for i in range(1, 5)]
        is_norm = _need_normalized_conversion(vals, coord_space)
        x1, y1 = convert_pair(vals[0], vals[1], is_norm)
        x2, y2 = convert_pair(vals[2], vals[3], is_norm)
        x_left, x_right = sorted((x1, x2))
        y_top, y_bottom = sorted((y1, y2))
        center = [int(round((x_left + x_right) / 2.0)), int(round((y_top + y_bottom) / 2.0))]
        return {
            "parsed": True,
            "type": "bbox",
            "coord_source": "normalized" if is_norm else "pixel",
            "position": [x_left, y_top, x_right, y_bottom],
            "center": center,
            "clean_output": clean_text,
        }

    box_list_match = BOX_LIST_RE.search(clean_text)
    if box_list_match:
        vals = [float(box_list_match.group(i)) for i in range(1, 5)]
        is_norm = _need_normalized_conversion(vals, coord_space)
        x1, y1 = convert_pair(vals[0], vals[1], is_norm)
        x2, y2 = convert_pair(vals[2], vals[3], is_norm)
        x_left, x_right = sorted((x1, x2))
        y_top, y_bottom = sorted((y1, y2))
        center = [int(round((x_left + x_right) / 2.0)), int(round((y_top + y_bottom) / 2.0))]
        return {
            "parsed": True,
            "type": "bbox",
            "coord_source": "normalized" if is_norm else "pixel",
            "position": [x_left, y_top, x_right, y_bottom],
            "center": center,
            "clean_output": clean_text,
        }

    point_match = POINT_TOKEN_RE.search(clean_text) or POINT_LIST_RE.search(clean_text)
    if point_match:
        vals = [float(point_match.group(1)), float(point_match.group(2))]
        is_norm = _need_normalized_conversion(vals, coord_space)
        x, y = convert_pair(vals[0], vals[1], is_norm)
        return {
            "parsed": True,
            "type": "point",
            "coord_source": "normalized" if is_norm else "pixel",
            "position": [x, y],
            "center": [x, y],
            "clean_output": clean_text,
        }

    paren_matches = POINT_PAREN_RE.findall(clean_text)
    if len(paren_matches) == 1:
        vals = [float(paren_matches[0][0]), float(paren_matches[0][1])]
        is_norm = _need_normalized_conversion(vals, coord_space)
        x, y = convert_pair(vals[0], vals[1], is_norm)
        return {
            "parsed": True,
            "type": "point",
            "coord_source": "normalized" if is_norm else "pixel",
            "position": [x, y],
            "center": [x, y],
            "clean_output": clean_text,
        }

    return {
        "parsed": False,
        "type": "unknown",
        "coord_source": "unknown",
        "position": None,
        "center": None,
        "clean_output": clean_text,
    }


def build_prompt(instruction: str) -> str:
    return (
        "In this UI screenshot, what is the position of the element corresponding "
        f'to the command "{instruction}" (with bbox)?'
    )


def predict_one(
    model: Qwen2VLForConditionalGeneration,
    processor: AutoProcessor,
    image_path: Path,
    instruction: str,
    device: str,
    max_new_tokens: int,
) -> str:
    prompt = build_prompt(instruction)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0] if output_text else ""


def _clip_xyxy(box: List[int], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def prediction_to_vis_box(parsed: Dict[str, Any], width: int, height: int) -> Optional[List[int]]:
    pred_type = parsed.get("type")
    pos = parsed.get("position")
    center = parsed.get("center")

    if pred_type == "bbox" and pos and len(pos) == 4:
        return _clip_xyxy([int(pos[0]), int(pos[1]), int(pos[2]), int(pos[3])], width, height)

    if pred_type == "point" and center and len(center) == 2:
        px, py = int(center[0]), int(center[1])
        half = max(8, int(round(min(width, height) * 0.015)))
        return _clip_xyxy([px - half, py - half, px + half, py + half], width, height)

    return None


def draw_prediction_on_image(
    image_path: Path,
    save_path: Path,
    parsed: Dict[str, Any],
) -> None:
    YELLOW = (255, 255, 0)
    
    with Image.open(image_path).convert("RGB") as image:
        draw = ImageDraw.Draw(image)
        width, height = image.size

        vis_box = prediction_to_vis_box(parsed, width, height)
        center = parsed.get("center")

        if vis_box is not None:
            draw.rectangle(vis_box, outline=YELLOW, width=4)

        if center and len(center) == 2:
            cx, cy = int(center[0]), int(center[1])
            r = max(5, int(round(min(width, height) * 0.008)))
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=YELLOW, outline=YELLOW)

        pred_type = parsed.get("type", "unknown")
        label = f"pred: {pred_type}"
        label_bg = (0, 0, 0)
        label_text = (255, 255, 255)
        
        text_x, text_y = 10, 10
        bbox = draw.textbbox((0, 0), label)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        draw.rectangle(
            (text_x - 2, text_y - 2, text_x + text_w + 4, text_y + text_h + 4),
            fill=label_bg
        )
        draw.text((text_x, text_y), label, fill=label_text)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(save_path)


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def ensure_paths(args: argparse.Namespace) -> Tuple[Path, Optional[Path]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_json = args.output_dir / f"{args.run_name}_results.json"
    
    vis_dir = None
    if args.save_vis:
        vis_dir = args.output_dir / args.vis_dir_name
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    return result_json, vis_dir


def main() -> None:
    args = parse_args()
    result_json, vis_dir = ensure_paths(args)

    device = resolve_device(args.device)
    image_root = args.data_root / args.image_dir

    records = load_annotations(args.data_root, args.json_file)
    if args.max_samples > 0:
        records = records[: args.max_samples]
    total = len(records)

    model, processor = load_model(args.model_path)

    # 收集所有结果
    all_results = []
    
    for idx, item in enumerate(records, start=1):
        img_name = item.get("img_filename", "")
        instruction = item.get("instruction", "")
        img_path = image_root / img_name

        sample_result: Dict[str, Any] = {
            "idx": idx,
            "img_name": img_name,
            "instruction": instruction,
            "raw_output": "",
            "parsed_type": None,
            "parsed_position": None,
            "parsed_center": None,
            "coord_source": None,
            "success": False,
        }

        if not img_path.exists():
            sample_result["error"] = f"Image not found: {img_path}"
            all_results.append(sample_result)
            print(f"[{idx}/{total}] ERROR: Image not found {img_name}")
            continue

        try:
            with Image.open(img_path) as image:
                width, height = image.size

            raw_output = predict_one(
                model=model,
                processor=processor,
                image_path=img_path,
                instruction=instruction,
                device=device,
                max_new_tokens=args.max_new_tokens,
            )
            parsed = parse_prediction(raw_output, width, height, args.coord_space)

            sample_result["raw_output"] = raw_output
            sample_result["parsed_type"] = parsed["type"]
            sample_result["parsed_position"] = parsed["position"]
            sample_result["parsed_center"] = parsed["center"]
            sample_result["coord_source"] = parsed["coord_source"]
            sample_result["success"] = parsed["parsed"]

            if args.save_vis and vis_dir is not None:
                vis_name = f"{idx:06d}_{img_name}"
                vis_path = vis_dir / vis_name
                draw_prediction_on_image(
                    image_path=img_path,
                    save_path=vis_path,
                    parsed=parsed,
                )

            print(f"[{idx}/{total}] {img_name}: {parsed['type']} -> {parsed['center']}")

        except Exception as exc:
            sample_result["error"] = f"ERROR: {exc}"
            print(f"[{idx}/{total}] ERROR: {exc}")

        all_results.append(sample_result)

    # 保存最终结果到 JSON 文件
    final_output = {
        "run_info": {
            "run_name": args.run_name,
            "model_path": str(args.model_path),
            "data_root": str(args.data_root),
            "json_file": args.json_file,
            "image_dir": args.image_dir,
            "total_samples": total,
            "processed_samples": len(all_results),
            "timestamp": datetime.now().isoformat(),
        },
        "results": all_results,
    }

    with result_json.open("w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"Inference complete!")
    print(f"Results saved to: {result_json}")
    if args.save_vis and vis_dir is not None:
        print(f"Visualizations: {vis_dir}")
    print(f"Total samples: {total}, Success: {sum(1 for r in all_results if r['success'])}")


if __name__ == "__main__":
    main()