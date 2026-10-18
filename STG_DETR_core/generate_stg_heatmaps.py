

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from ultralytics import RTDETR


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate P3/STGF/DAGS heatmaps.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(
            "/home/we/code/yxl/Xrong_CWPS_DAGS_v2_200e/"
            "a7v2_full_dags_v2_s0/weights/best.pt"
        ),
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("/home/we/code/yxl/CWPS_sanitized/images/val"),
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("/home/we/code/yxl/CWPS_sanitized/labels/val"),
    )
    parser.add_argument("--image-id", default="val_000244")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/we/code/yxl/STG_VIS/cwps/heatmaps"),
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="CUDA index such as 0, or cpu.")
    parser.add_argument("--alpha", type=float, default=0.48)
    parser.add_argument("--p3-layer", type=int, default=5)
    parser.add_argument("--stgf-layer", type=int, default=8)
    parser.add_argument("--ggas-layer", type=int, default=9)
    return parser.parse_args()


def resolve_image(images_dir: Path, image_id: str) -> Path:
    direct = Path(image_id)
    if direct.is_file():
        return direct
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / f"{image_id}{suffix}"
        if candidate.is_file():
            return candidate
        candidate = images_dir / f"{image_id}{suffix.upper()}"
        if candidate.is_file():
            return candidate
    matches = [
        p for p in images_dir.glob(f"{image_id}.*")
        if p.suffix.lower() in IMAGE_SUFFIXES
    ]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Cannot uniquely resolve image '{image_id}' in {images_dir}")


def select_device(device_arg: str) -> torch.device:
    if str(device_arg).lower() == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")
    text = str(device_arg).replace("cuda:", "")
    return torch.device(f"cuda:{int(text)}")


def letterbox(image_bgr: np.ndarray, size: int):
    """Ultralytics-style centered letterbox and its geometric parameters."""
    h0, w0 = image_bgr.shape[:2]
    ratio = min(size / h0, size / w0)
    new_w = int(round(w0 * ratio))
    new_h = int(round(h0 * ratio))

    if (new_w, new_h) != (w0, h0):
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = image_bgr

    dw = (size - new_w) / 2
    dh = (size - new_h) / 2
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, ratio, left, top, new_w, new_h


def load_and_transform_labels(
    label_path: Path,
    original_w: int,
    original_h: int,
    ratio: float,
    pad_left: int,
    pad_top: int,
    image_size: int,
):
    """Return class ids and normalized xywh boxes after letterboxing."""
    classes = []
    original_boxes = []
    boxes = []
    if not label_path.is_file():
        print(f"Warning: label file not found: {label_path}")
        return classes, original_boxes, boxes

    for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
        fields = line.strip().split()
        if not fields:
            continue
        if len(fields) < 5:
            print(f"Warning: skipping malformed label line {line_number}: {line}")
            continue

        class_id, cx, cy, width, height = map(float, fields[:5])
        cx_px = cx * original_w * ratio + pad_left
        cy_px = cy * original_h * ratio + pad_top
        width_px = width * original_w * ratio
        height_px = height * original_h * ratio

        classes.append(int(class_id))
        original_boxes.append([cx, cy, width, height])
        boxes.append(
            [
                cx_px / image_size,
                cy_px / image_size,
                width_px / image_size,
                height_px / image_size,
            ]
        )
    return classes, original_boxes, boxes


def remove_letterbox_from_map(
    array: np.ndarray,
    model_size: int,
    pad_left: int,
    pad_top: int,
    content_w: int,
    content_h: int,
    original_w: int,
    original_h: int,
):
    """Map a feature-resolution array back onto the unpadded original image."""
    full = cv2.resize(array, (model_size, model_size), interpolation=cv2.INTER_CUBIC)
    x1 = max(0, pad_left)
    y1 = max(0, pad_top)
    x2 = min(model_size, x1 + content_w)
    y2 = min(model_size, y1 + content_h)
    cropped = full[y1:y2, x1:x2]
    if cropped.size == 0:
        raise RuntimeError("The valid image region became empty while removing letterbox padding.")
    return cv2.resize(cropped, (original_w, original_h), interpolation=cv2.INTER_CUBIC)


def robust_limits(first: np.ndarray, second: np.ndarray):
    values = np.concatenate([first.reshape(-1), second.reshape(-1)])
    low, high = np.percentile(values, [1.0, 99.0])
    if high <= low:
        low = float(values.min())
        high = float(values.max())
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def normalize_map(array: np.ndarray, low=None, high=None):
    array = np.asarray(array, dtype=np.float32)
    if low is None or high is None:
        low, high = np.percentile(array, [1.0, 99.0])
        if high <= low:
            low, high = float(array.min()), float(array.max())
        if high <= low:
            high = low + 1.0
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def colorize(normalized: np.ndarray, output_size: tuple[int, int]):
    resized = cv2.resize(
        normalized,
        output_size,
        interpolation=cv2.INTER_CUBIC,
    )
    values = np.clip(resized * 255.0, 0, 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(values, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB), np.clip(resized, 0.0, 1.0)


def overlay_heatmap(
    base_rgb: np.ndarray,
    normalized: np.ndarray,
    alpha: float,
    activation_weighted: bool,
):
    heat_rgb, resized = colorize(normalized, (base_rgb.shape[1], base_rgb.shape[0]))
    if activation_weighted:
        weight = alpha * np.clip(resized * 1.5, 0.0, 1.0)
    else:
        weight = np.full_like(resized, alpha, dtype=np.float32)
    weight = weight[..., None]
    blended = base_rgb.astype(np.float32) * (1.0 - weight) + heat_rgb.astype(np.float32) * weight
    return np.clip(blended, 0, 255).astype(np.uint8), heat_rgb


def draw_small_object_gt(
    image_rgb: np.ndarray,
    original_boxes: list[list[float]],
    small_flags: list[bool],
):
    image = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(image)
    image_h, image_w = image_rgb.shape[:2]
    line_width = max(2, min(image_w, image_h) // 180)
    small_count = 0

    # Only boxes used to build the DAGS target are shown.  Omitting text labels
    # keeps densely packed small-object boxes legible in a paper figure.
    for (cx, cy, width, height), is_small in zip(original_boxes, small_flags):
        if not is_small:
            continue
        small_count += 1
        color = (30, 210, 70)
        x1 = int(round((cx - width / 2) * image_w))
        y1 = int(round((cy - height / 2) * image_h))
        x2 = int(round((cx + width / 2) * image_w))
        y2 = int(round((cy + height / 2) * image_h))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_w - 1, x2), min(image_h - 1, y2)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
    return np.asarray(image), small_count


def save_rgb(path: Path, array: np.ndarray):
    Image.fromarray(array.astype(np.uint8)).save(path)


def build_composite(panels, output_path: Path):
    source_h, source_w = panels[0].shape[:2]
    cell_width = 640
    cell_height = min(640, max(320, int(round(cell_width * source_h / source_w))))
    header_height = 54
    column_gap = 10
    row_gap = 12
    outer_margin = 12
    titles = [
        "Input + small-object GT",
        "P3 response",
        "Enhanced P3'",
        "STGF difference |P3'-P3|",
        "Gaussian target (Gt)",
        "DAGS prediction (Gp)",
    ]

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 27
        )
    except Exception:
        font = ImageFont.load_default()

    canvas_w = outer_margin * 2 + cell_width * 3 + column_gap * 2
    canvas_h = outer_margin * 2 + (header_height + cell_height) * 2 + row_gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    for index, (title, panel_array) in enumerate(zip(titles, panels)):
        row = index // 3
        column = index % 3
        x = outer_margin + column * (cell_width + column_gap)
        y = outer_margin + row * (header_height + cell_height + row_gap)

        title_box = draw.textbbox((0, 0), title, font=font)
        title_w = title_box[2] - title_box[0]
        draw.text(
            (x + (cell_width - title_w) // 2, y + 10),
            title,
            fill=(20, 20, 20),
            font=font,
        )

        panel = Image.fromarray(panel_array.astype(np.uint8))
        panel.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
        panel_canvas = Image.new("RGB", (cell_width, cell_height), "white")
        panel_canvas.paste(
            panel,
            ((cell_width - panel.width) // 2, (cell_height - panel.height) // 2),
        )
        canvas.paste(panel_canvas, (x, y + header_height))

    canvas.save(output_path, dpi=(300, 300))


def main():
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1].")
    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    image_path = resolve_image(args.images_dir, args.image_id)
    label_path = args.labels_dir / f"{image_path.stem}.txt"
    output_dir = args.output_dir / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    original_bgr = cv2.imread(str(image_path))
    if original_bgr is None:
        raise RuntimeError(f"OpenCV could not read image: {image_path}")
    original_h, original_w = original_bgr.shape[:2]
    letterboxed_bgr, ratio, pad_left, pad_top, content_w, content_h = letterbox(
        original_bgr, args.imgsz
    )
    if letterboxed_bgr.shape[:2] != (args.imgsz, args.imgsz):
        raise RuntimeError(f"Unexpected letterbox shape: {letterboxed_bgr.shape}")

    classes, original_boxes, boxes = load_and_transform_labels(
        label_path,
        original_w,
        original_h,
        ratio,
        pad_left,
        pad_top,
        args.imgsz,
    )

    device = select_device(args.device)
    wrapper = RTDETR(str(args.weights))
    network = wrapper.model.to(device).float().eval()

    layers = network.model
    if args.p3_layer >= len(layers) or args.stgf_layer >= len(layers) or args.ggas_layer >= len(layers):
        raise IndexError(f"Model contains only {len(layers)} layers; requested layer index is invalid.")

    p3_module = layers[args.p3_layer]
    stgf_module = layers[args.stgf_layer]
    ggas_module = layers[args.ggas_layer]

    print(f"P3 layer {args.p3_layer}: {type(p3_module).__name__}")
    print(f"STGF layer {args.stgf_layer}: {type(stgf_module).__name__}")
    print(f"DAGS layer {args.ggas_layer}: {type(ggas_module).__name__}")

    if type(stgf_module).__name__ != "LiteCSAG":
        raise RuntimeError("The configured STGF layer is not LiteCSAG.")
    if type(ggas_module).__name__ != "GGASHead":
        raise RuntimeError("The configured DAGS layer is not GGASHead.")

    captured = {}

    def capture_p3(_module, _inputs, output):
        captured["p3"] = output.detach()

    def capture_p3_prime(_module, _inputs, output):
        captured["p3_prime"] = output.detach()

    hook_p3 = p3_module.register_forward_hook(capture_p3)
    hook_stgf = stgf_module.register_forward_hook(capture_p3_prime)

    model_input_rgb = cv2.cvtColor(letterboxed_bgr, cv2.COLOR_BGR2RGB)
    display_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(model_input_rgb.transpose(2, 0, 1)))
    tensor = tensor.unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0

    try:
        with torch.inference_mode():
            _ = network(tensor)
    finally:
        hook_p3.remove()
        hook_stgf.remove()

    if "p3" not in captured or "p3_prime" not in captured:
        raise RuntimeError("Forward hooks did not capture both P3 and enhanced P3.")

    p3 = captured["p3"]
    p3_prime = captured["p3_prime"]
    if p3.shape != p3_prime.shape:
        raise RuntimeError(f"P3 shape {p3.shape} != enhanced P3 shape {p3_prime.shape}")

    # Reproduce GGASHead forward activations without switching its BatchNorm to train mode.
    with torch.inference_mode():
        z = ggas_module.act(ggas_module.bn(ggas_module.depthwise(p3_prime)))
        z = ggas_module.act(ggas_module.pointwise(z))
        ggas_logits = ggas_module.out(z)
        ggas_prediction = ggas_logits.float().sigmoid()

    if boxes:
        bboxes_tensor = torch.tensor(boxes, dtype=torch.float32, device=device)
        batch_idx = torch.zeros((len(boxes),), dtype=torch.long, device=device)
    else:
        bboxes_tensor = torch.empty((0, 4), dtype=torch.float32, device=device)
        batch_idx = torch.empty((0,), dtype=torch.long, device=device)

    criterion = getattr(network, "ggas_criterion", None)
    if criterion is None:
        from ultralytics.nn.extra_modules.fpf_lite import GGASLoss

        criterion = GGASLoss(
            small_area=1024.0,
            sigma_ratio=1.0 / 6.0,
            loss_weight=0.10,
            truncate_sigma=3.0,
            background_weight=0.25,
        )

    with torch.inference_mode():
        target = criterion.build_targets(
            bboxes_tensor,
            batch_idx,
            batch_size=1,
            feature_size=ggas_prediction.shape[-2:],
            image_size=(args.imgsz, args.imgsz),
            device=device,
        )

    p3_response_raw = p3.abs().mean(dim=1)[0].float().cpu().numpy()
    p3_prime_response_raw = p3_prime.abs().mean(dim=1)[0].float().cpu().numpy()
    difference_response_raw = (p3_prime - p3).abs().mean(dim=1)[0].float().cpu().numpy()
    gaussian_target_raw = target[0, 0].float().cpu().numpy()
    dags_prediction_raw = ggas_prediction[0, 0].float().cpu().numpy()

    unletterbox_args = (
        args.imgsz,
        pad_left,
        pad_top,
        content_w,
        content_h,
        original_w,
        original_h,
    )
    p3_response = remove_letterbox_from_map(p3_response_raw, *unletterbox_args)
    p3_prime_response = remove_letterbox_from_map(p3_prime_response_raw, *unletterbox_args)
    difference_response = remove_letterbox_from_map(difference_response_raw, *unletterbox_args)
    gaussian_target = remove_letterbox_from_map(gaussian_target_raw, *unletterbox_args)
    dags_prediction = remove_letterbox_from_map(dags_prediction_raw, *unletterbox_args)

    common_low, common_high = robust_limits(p3_response, p3_prime_response)
    p3_norm = normalize_map(p3_response, common_low, common_high)
    p3_prime_norm = normalize_map(p3_prime_response, common_low, common_high)
    difference_norm = normalize_map(difference_response)
    target_norm = np.clip(gaussian_target, 0.0, 1.0)
    prediction_norm = np.clip(dags_prediction, 0.0, 1.0)

    small_area = float(getattr(criterion, "small_area", 1024.0))
    small_flags = [
        width * args.imgsz * height * args.imgsz < small_area
        for _cx, _cy, width, height in boxes
    ]
    input_with_gt, small_count = draw_small_object_gt(
        display_rgb,
        original_boxes,
        small_flags,
    )

    p3_overlay, p3_heat = overlay_heatmap(display_rgb, p3_norm, args.alpha, False)
    p3_prime_overlay, p3_prime_heat = overlay_heatmap(display_rgb, p3_prime_norm, args.alpha, False)
    difference_overlay, difference_heat = overlay_heatmap(display_rgb, difference_norm, args.alpha, True)
    target_overlay, target_heat = overlay_heatmap(display_rgb, target_norm, 0.70, True)
    prediction_overlay, prediction_heat = overlay_heatmap(display_rgb, prediction_norm, 0.62, True)

    foreground_mask = gaussian_target_raw > 0.10
    background_mask = gaussian_target_raw <= 1e-6
    foreground_mean = (
        float(dags_prediction_raw[foreground_mask].mean())
        if foreground_mask.any()
        else float("nan")
    )
    background_mean = (
        float(dags_prediction_raw[background_mask].mean())
        if background_mask.any()
        else float("nan")
    )
    contrast_ratio = (
        foreground_mean / max(background_mean, 1e-12)
        if np.isfinite(foreground_mean) and np.isfinite(background_mean)
        else float("nan")
    )
    if gaussian_target_raw.std() > 0 and dags_prediction_raw.std() > 0:
        spatial_correlation = float(
            np.corrcoef(gaussian_target_raw.reshape(-1), dags_prediction_raw.reshape(-1))[0, 1]
        )
    else:
        spatial_correlation = float("nan")

    save_rgb(output_dir / "01_input_small_gt.png", input_with_gt)
    save_rgb(output_dir / "02_p3_heatmap.png", p3_heat)
    save_rgb(output_dir / "02_p3_overlay.png", p3_overlay)
    save_rgb(output_dir / "03_p3_prime_heatmap.png", p3_prime_heat)
    save_rgb(output_dir / "03_p3_prime_overlay.png", p3_prime_overlay)
    save_rgb(output_dir / "04_stgf_difference_heatmap.png", difference_heat)
    save_rgb(output_dir / "04_stgf_difference_overlay.png", difference_overlay)
    save_rgb(output_dir / "05_gaussian_target_heatmap.png", target_heat)
    save_rgb(output_dir / "05_gaussian_target_overlay.png", target_overlay)
    save_rgb(output_dir / "06_dags_prediction_heatmap.png", prediction_heat)
    save_rgb(output_dir / "06_dags_prediction_overlay.png", prediction_overlay)

    np.savez_compressed(
        output_dir / "raw_heatmaps.npz",
        p3_feature_map=p3_response_raw,
        p3_prime_feature_map=p3_prime_response_raw,
        stgf_difference_feature_map=difference_response_raw,
        gaussian_target_feature_map=gaussian_target_raw,
        dags_prediction_feature_map=dags_prediction_raw,
        p3_display_map=p3_response,
        p3_prime_display_map=p3_prime_response,
        stgf_difference_display_map=difference_response,
        gaussian_target_display_map=gaussian_target,
        dags_prediction_display_map=dags_prediction,
    )

    composite_panels = [
        input_with_gt,
        p3_overlay,
        p3_prime_overlay,
        difference_overlay,
        target_overlay,
        prediction_overlay,
    ]
    build_composite(
        composite_panels,
        output_dir / "Fig_STGF_DAGS_heatmaps_2x3.png",
    )

    metadata = (
        f"image={image_path}\n"
        f"weights={args.weights}\n"
        f"input_size={args.imgsz}x{args.imgsz}\n"
        f"P3_layer={args.p3_layer}\n"
        f"STGF_layer={args.stgf_layer}\n"
        f"GGAS_layer={args.ggas_layer}\n"
        f"P3_shape={tuple(p3.shape)}\n"
        f"GGAS_shape={tuple(ggas_prediction.shape)}\n"
        f"GT_objects={len(boxes)}\n"
        f"DAGS_small_objects={small_count}\n"
        f"small_area_threshold={small_area}\n"
        f"P3_joint_display_range=[{common_low:.8g}, {common_high:.8g}]\n"
        f"DAGS_foreground_mean={foreground_mean:.8g}\n"
        f"DAGS_background_mean={background_mean:.8g}\n"
        f"DAGS_foreground_background_ratio={contrast_ratio:.8g}\n"
        f"DAGS_target_prediction_correlation={spatial_correlation:.8g}\n"
    )
    (output_dir / "metadata.txt").write_text(metadata, encoding="utf-8")

    print(f"Image: {image_path}")
    print(f"Labels: {label_path}")
    print(f"P3 shape: {tuple(p3.shape)}")
    print(f"DAGS map shape: {tuple(ggas_prediction.shape)}")
    print(f"GT objects: {len(boxes)}; DAGS small objects: {small_count}")
    print(f"DAGS foreground mean: {foreground_mean:.6f}")
    print(f"DAGS background mean: {background_mean:.6f}")
    print(f"DAGS foreground/background ratio: {contrast_ratio:.4f}")
    print(f"DAGS target/prediction correlation: {spatial_correlation:.4f}")
    print(f"All outputs saved to: {output_dir}")
    print(f"Composite: {output_dir / 'Fig_STGF_DAGS_heatmaps_2x3.png'}")


if __name__ == "__main__":
    main()