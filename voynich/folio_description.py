"""Pixel-only, deterministic descriptions. Shape proxies are not object labels."""
import cv2
import numpy as np

CONFIG = {
    "method": "pixel_layout_v1",
    "long_edge": 600,
    "border_fraction": 0.035,
    "green_a_delta": -7,
    "blue_b_delta": -14,
    "warm_a_delta": 8,
    "component_area_fraction": 0.0008,
    "circle_param1": 70,
    "circle_param2": 35,
    "circle_min_radius_fraction": 0.10,
    "circle_max_radius_fraction": 0.48,
    "circle_min_radial_coverage": 0.50,
    "circle_min_supported_sectors": 14,
}


def supported_circles(gray, config):
    """Reject accidental Hough circles in dense text using radial edge support."""
    gray = cv2.GaussianBlur(gray, (5, 5), 1)
    side = min(gray.shape)
    candidates = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,
                                  dp=1, minDist=20,
                                  param1=config["circle_param1"], param2=config["circle_param2"],
                                  minRadius=round(side * config["circle_min_radius_fraction"]),
                                  maxRadius=round(side * config["circle_max_radius_fraction"]))
    if candidates is None:
        return []
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    magnitude = np.hypot(dx, dy)
    theta = np.arange(360) * np.pi / 180
    cosine, sine = np.cos(theta), np.sin(theta)
    rings = []
    for x, y, radius in candidates[0]:
        if not (radius + 3 <= x <= gray.shape[1] - radius - 3 and
                radius + 3 <= y <= gray.shape[0] - radius - 3):
            continue
        support = []
        for offset in [-1, 0, 1]:
            xx = np.rint(x + (radius + offset) * cosine).astype(int)
            yy = np.rint(y + (radius + offset) * sine).astype(int)
            strength = magnitude[yy, xx]
            alignment = np.abs(dx[yy, xx] * cosine + dy[yy, xx] * sine) / (strength + 1e-6)
            support.append((strength > 25) & (alignment > .94))
        hits = np.any(support, axis=0)
        coverage = float(hits.mean())
        sectors = int(np.count_nonzero(hits.reshape(24, 15).mean(axis=1) > .5))
        if coverage < config["circle_min_radial_coverage"] or sectors < config["circle_min_supported_sectors"]:
            continue
        rings.append({"cx": round(float(x / gray.shape[1]), 4),
                      "cy": round(float(y / gray.shape[0]), 4),
                      "radius_fraction_of_short_edge": round(float(radius / side), 4),
                      "radial_edge_coverage": round(coverage, 4), "supported_sectors": sectors})
    kept = []
    for ring in sorted(rings, key=lambda r: -r["radial_edge_coverage"]):
        if all(np.hypot((ring["cx"] - other["cx"]) * gray.shape[1],
                        (ring["cy"] - other["cy"]) * gray.shape[0]) > side * .15 for other in kept):
            kept.append(ring)
    return sorted(kept, key=lambda r: (r["cy"], r["cx"]))


def describe(rgb, config=None):
    """Measure one cropped panel. Return JSON-compatible values and rule trace."""
    c = CONFIG if config is None else config
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    h, w = rgb.shape[:2]
    if h < 40 or w < 40:
        raise ValueError("Panel must be at least 40 pixels on each side")
    scale = min(1, c["long_edge"] / max(h, w))
    rgb = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    my, mx = [max(1, round(n * c["border_fraction"])) for n in rgb.shape[:2]]
    rgb = rgb[my:-my, mx:-mx]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    light = lab[:, :, 0]
    # Estimate parchment from the brighter half, suppressing dark ink/paint.
    paper = np.median(lab[light >= np.quantile(light, .5)], axis=0)
    da, db = lab[:, :, 1] - paper[1], lab[:, :, 2] - paper[2]
    valid = light > 45  # Ignore black mounting/background where possible.
    green = (da < c["green_a_delta"]) & (db >= c["blue_b_delta"]) & valid
    blue = (db < c["blue_b_delta"]) & valid
    warm = (da > c["warm_a_delta"]) & (db > 0) & valid
    chromatic = green | blue | warm
    cool = green | blue
    size = light.size
    background = cv2.GaussianBlur(light, (0, 0), 9)
    dark = (background - light > 16) & (background > 90)
    row_density = cv2.GaussianBlur(dark.mean(axis=1).astype(np.float32).reshape(-1, 1), (1, 5), 0).ravel()
    bands = row_density > max(.045, float(np.quantile(row_density, .65)))
    row_bands = int(np.count_nonzero(np.diff(np.r_[False, bands, False].astype(int)) == 1))

    # Components are pigment patches, not plants, people, stars, or jars.
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(cool.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    regions = []
    for i in range(1, count):
        x, y, rw, rh, area = stats[i]
        if area / size >= c["component_area_fraction"]:
            regions.append({"area_fraction": round(float(area / size), 5),
                            "cx": round(float(centers[i][0] / light.shape[1]), 4),
                            "cy": round(float(centers[i][1] / light.shape[0]), 4),
                            "width_fraction": round(float(rw / light.shape[1]), 4),
                            "height_fraction": round(float(rh / light.shape[0]), 4)})
    regions.sort(key=lambda r: (-r["area_fraction"], r["cx"], r["cy"]))
    grid = [round(float(cell.mean()), 5)
            for row in np.array_split(cool, 3, axis=0)
            for cell in np.array_split(row, 3, axis=1)]
    occupied = sum(v >= .02 for v in grid)
    ys, xs = np.where(cool)
    span_x = float((xs.max() - xs.min() + 1) / light.shape[1]) if len(xs) else 0
    span_y = float((ys.max() - ys.min() + 1) / light.shape[0]) if len(ys) else 0
    centroid = [float(xs.mean() / light.shape[1]), float(ys.mean() / light.shape[0])] if len(xs) else None
    # Reflective symmetry of coarse colour occupancy, not semantic symmetry.
    matrix = np.array(grid).reshape(3, 3)
    symmetry = float(1 - np.abs(matrix - matrix[:, ::-1]).sum() / (2 * matrix.sum())) if cool.mean() >= .008 else None
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ring_candidates = supported_circles(gray, c)
    fractions = {"green": float(green.mean()), "blue": float(blue.mean()), "warm": float(warm.mean())}
    trace = []
    if ring_candidates:
        layout = "circular_candidates"
        trace.append(f"R1: circle with radial edge coverage >= {c['circle_min_radial_coverage']} "
                     f"across >= {c['circle_min_supported_sectors']}/24 sectors")
    elif cool.mean() < .008 and row_bands >= 12:
        layout = "horizontal_marks_dominant"
        trace.append("R2: cool fraction < 0.008 and horizontal bands >= 12")
    elif len(regions) >= 6 and occupied >= 5:
        layout = "distributed_coloured_regions"
        trace.append("R3: cool components >= 6 and occupied grid cells >= 5")
    elif cool.mean() >= .008:
        layout = "localised_coloured_regions"
        trace.append("R4: cool fraction >= 0.008")
    else:
        layout = "sparse_or_unresolved"
        trace.append("R5: no preceding rule fired")
    palette = [name for name, value in fractions.items() if value >= .003]
    text = (f"{layout.replace('_', ' ').capitalize()}. "
            f"Green-like {fractions['green']:.1%}, blue-like {fractions['blue']:.1%}, "
            f"warm red/brown-like {fractions['warm']:.1%} of the analysed interior. "
            f"{len(regions)} substantial cool-colour patches across {occupied}/9 grid cells; "
            f"{len(ring_candidates)} large circle candidates; {row_bands} horizontal dark-mark bands.")
    if centroid:
        vertical = ["upper", "middle", "lower"][min(2, int(centroid[1] * 3))]
        horizontal = ["left", "centre", "right"][min(2, int(centroid[0] * 3))]
        text += f" Cool-colour centre of mass: {vertical} {horizontal}."
    return {"layout": layout, "domain": "unassigned", "description": text,
            "green_fraction": round(fractions["green"], 5),
            "blue_fraction": round(fractions["blue"], 5),
            "warm_fraction": round(fractions["warm"], 5),
            "dark_mark_fraction": round(float(dark.mean()), 5),
            "horizontal_band_count": row_bands, "cool_component_count": len(regions),
            "occupied_grid_cells": occupied, "cool_span_x": round(span_x, 4),
            "cool_span_y": round(span_y, 4), "coarse_colour_symmetry": round(symmetry, 4) if symmetry is not None else None,
            "circle_candidate_count": len(ring_candidates), "palette": palette,
            "cool_grid_3x3": grid, "cool_regions": regions, "circle_candidates": ring_candidates,
            "rules_fired": trace, "quality_flags": ["unvalidated_shape_proxies", "possible_show_through",
                                                       "domain_requires_visual_annotation"]}
