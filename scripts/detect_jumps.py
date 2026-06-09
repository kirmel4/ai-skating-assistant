"""Детекция прыжков в полном видео — эвристика по сигналу вращения.

Этап 1 сервисного пайплайна: найти ГДЕ прыжки во всём видео программы.
Не требует обучения — прыжок имеет чёткую кинематическую сигнатуру:
быстрое вращение корпуса (1-4 оборота за ~0.6 сек).

Алгоритм:
  1. YOLOv8-pose по кадрам видео (с прореживанием до sample_fps)
  2. Главный фигурист = человек с самым большим bbox в кадре
  3. Сигнал = угловая скорость линии плеч (вращение в плоскости кадра)
  4. Кандидат = сегмент высокой угловой скорости длительностью 0.3-1.5 сек
     (спины отсекаются — они длиннее)
  5. Для видео 9/10/11 — сверка с xlsx-разметкой → recall / precision

Использование:
    python scripts/detect_jumps.py --video 10
    python scripts/detect_jumps.py --video 9 --start 600 --end 1200
    python scripts/detect_jumps.py --video-path /path/to.mp4 --start 0 --end 300

Тестовый прогон на куске (--start/--end в секундах) — чтобы не ждать всё видео.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from scripts.train_verifier import (
    CHECKPOINT_DIR as VERIFIER_CKPT_DIR,
    DEFAULT_THRESHOLD as VERIFIER_THRESHOLD,
    FEATURE_DIM as VERIFIER_FEATURE_DIM,
    NUM_FRAMES as VERIFIER_NUM_FRAMES,
    SEEDS as VERIFIER_SEEDS,
    SMOOTH_SIGMA as VERIFIER_SMOOTH,
    TEMPORAL_HIDDEN_DIM as VERIFIER_HIDDEN,
    WINDOW_SEC as VERIFIER_WINDOW_SEC,
    VerifierModel,
    extract_features_f4,
)
from src.utils.xlsx_parser import parse_excel

BASE_DIR = _REPO_ROOT
VIDEOS_DIR = BASE_DIR / "data" / "videos"
TIMECODES_PATH = BASE_DIR / "data" / "Разметка прыжков.xlsx"
OUT_DIR = BASE_DIR / "data" / "jump_detection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# video_id → имя файла (videos 9-11 локальные)
VIDEO_FILES = {
    9: "Произвольная программа Девушки Омск.mp4",
    10: "Мужчины короткая программа Омск.mp4",
    11: "Произвольная программа Женщины Москва.mp4",
}

# COCO-17 keypoint индексы
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12

# Параметры детекции
SAMPLE_FPS = 15.0          # частота прореживания (>12 чтобы не алиасить вращение)
POSE_MODEL = "yolov8m-pose.pt"
POSE_IMGSZ = 1280
POSE_CONF = 0.15
SMOOTH_SIGMA = 1.5         # сглаживание сигнала угловой скорости (в кадрах)
SPEED_THRESH = 1.0         # порог об/сек — выше = "быстрое вращение"
MIN_JUMP_DUR = 0.30        # сек — короче не считаем прыжком
MAX_JUMP_DUR = 1.60        # сек — длиннее = спин, отсекаем
MERGE_GAP = 0.30           # сек — сегменты ближе склеиваем
MATCH_TOLERANCE = 1.5      # сек — допуск при сверке с разметкой
VERIFY_MERGE_GAP = 1.5     # сек — verified-окна ближе склеиваем в одну детекцию
                           # (детектор часто выдаёт 2 окна на один прыжок;
                           #  ВНИМАНИЕ: каскадные прыжки ближе этого зазора схлопнутся)

# Детекция по фазе полёта (баллистическая дуга) — дополняет rotation-эвристику
AIRBORNE_DETREND_SEC = 1.6   # окно медленного тренда — вычитает проводку/зум камеры
AIRBORNE_THRESH = 0.12       # подъём корпуса над трендом, в длинах торса (тюнится)


def _resolve_device() -> str:
    import torch
    if not torch.cuda.is_available():
        return "cpu"
    count = torch.cuda.device_count()
    if count == 1:
        # один видимый GPU (например при CUDA_VISIBLE_DEVICES=1) — он cuda:0
        return "cuda:0"
    idx = int(os.environ.get("POSE_CUDA_DEVICE", "1"))
    if idx < 0 or idx >= count:
        idx = 0
    return f"cuda:{idx}"


def _time_to_sec(t) -> float:
    """time object → секунды."""
    return t.hour * 3600 + t.minute * 60 + t.second


def _body_angle(kp: np.ndarray) -> float:
    """Угол поперечной оси корпуса по плечам и/или бёдрам.

    Раньше брались только плечи — при их пропаже (motion blur на быстром
    вращении, окклюзия) кадр терялся. Теперь направления линии плеч и линии
    бёдер усредняются как векторы: если плечи не видны, работает линия бёдер,
    если видно обе — оценка устойчивее.
    """
    vecs = []
    for a, b in ((L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP)):
        pa, pb = kp[a], kp[b]
        if pa[2] > 0.2 and pb[2] > 0.2:                       # confidence порог
            v = np.array([pb[0] - pa[0], pb[1] - pa[1]], dtype=np.float64)
            norm = float(np.hypot(v[0], v[1]))
            if norm > 1e-6:
                vecs.append(v / norm)
    if not vecs:
        return np.nan
    s = np.sum(vecs, axis=0)
    if float(np.hypot(s[0], s[1])) < 1e-6:
        return np.nan
    return float(np.arctan2(s[1], s[0]))


def extract_rotation_signal(video_path: str, start_sec: float, end_sec: float, device, model=None):
    """Прогоняет YOLOv8-pose по видео.

    Возвращает (times, shoulder_angle, keypoints) главного фигуриста:
    keypoints — (N, 17, 3) покадровые keypoints (нули, если фигурист не найден),
    переиспользуются verifier'ом без повторного прогона YOLO.

    model — заранее загруженная YOLO-модель (переиспользование / ONNX-вариант);
    если None — грузится yolov8m-pose.pt.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Не открыть видео: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps
    if end_sec <= 0 or end_sec > duration:
        end_sec = duration

    step = max(1, round(src_fps / SAMPLE_FPS))
    print(f"Video: {duration:.0f}s @ {src_fps:.1f}fps, обрабатываем [{start_sec:.0f}, {end_sec:.0f}]s, шаг {step} кадров")

    if model is None:
        from ultralytics import YOLO
        model = YOLO(POSE_MODEL)
        model.to(device)

    start_frame = int(start_sec * src_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    times: list[float] = []
    angles: list[float] = []   # угол линии плеч, NaN если не нашли
    keypoints: list[np.ndarray] = []   # (17,3) keypoints главного фигуриста / нули
    batch_imgs: list[np.ndarray] = []
    batch_times: list[float] = []

    def _flush(imgs, ts):
        results = model(imgs, verbose=False, conf=POSE_CONF, imgsz=POSE_IMGSZ, device=device)
        for t, res in zip(ts, results):
            angle = np.nan
            person_kp = np.zeros((17, 3), dtype=np.float32)
            if res.keypoints is not None and res.keypoints.data is not None and len(res.keypoints.data) > 0:
                kp = res.keypoints.data.cpu().numpy()  # (n, 17, 3)
                # главный фигурист = самый большой bbox
                if res.boxes is not None and len(res.boxes) > 0:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                    person = int(areas.argmax())
                else:
                    person = 0
                person_kp = kp[person].astype(np.float32)
                angle = _body_angle(person_kp)   # плечи + бёдра, устойчивее
            times.append(t)
            angles.append(angle)
            keypoints.append(person_kp)

    frame_idx = start_frame
    end_frame = int(end_sec * src_fps)
    total_kpts = max(1, (end_frame - start_frame + step - 1) // step)  # сколько кадров проанализируем
    log_every = 320          # кадров между строками прогресса
    next_log = log_every
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx - start_frame) % step == 0:
            batch_imgs.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            batch_times.append(frame_idx / src_fps)
            if len(batch_imgs) >= 32:
                _flush(batch_imgs, batch_times)
                batch_imgs, batch_times = [], []
                if len(times) >= next_log:
                    print(f"  проанализировано {len(times)}/{total_kpts} кадров", flush=True)
                    next_log += log_every
        frame_idx += 1
    if batch_imgs:
        _flush(batch_imgs, batch_times)
    cap.release()
    print(f"  проанализировано {len(times)}/{total_kpts} кадров — готово", flush=True)

    kpts_arr = np.stack(keypoints) if keypoints else np.zeros((0, 17, 3), dtype=np.float32)
    return np.array(times), np.array(angles), kpts_arr


def compute_angular_speed(times: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Угол плеч → угловая скорость в об/сек (сглаженная, abs)."""
    # интерполяция NaN
    valid = ~np.isnan(angles)
    if valid.sum() < 2:
        return np.zeros_like(times)
    angles = np.interp(np.arange(len(angles)), np.where(valid)[0], angles[valid])
    # unwrap чтобы убрать скачки ±π
    unwrapped = np.unwrap(angles)
    # угловая скорость: d(angle)/dt, переводим в обороты/сек
    dt = np.gradient(times)
    ang_vel = np.gradient(unwrapped) / np.maximum(dt, 1e-6) / (2 * np.pi)
    ang_speed = np.abs(gaussian_filter1d(ang_vel, sigma=SMOOTH_SIGMA))
    return ang_speed


def detect_candidates(times: np.ndarray, ang_speed: np.ndarray) -> list[tuple[float, float]]:
    """Сегменты высокой угловой скорости длительностью прыжка."""
    rotating = ang_speed > SPEED_THRESH
    raw = []
    i, n = 0, len(rotating)
    while i < n:
        if rotating[i]:
            j = i
            while j < n and rotating[j]:
                j += 1
            raw.append((times[i], times[j - 1]))
            i = j
        else:
            i += 1

    # склейка близких сегментов
    merged = []
    for seg in raw:
        if merged and seg[0] - merged[-1][1] <= MERGE_GAP:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(list(seg))

    # фильтр по длительности (отсекаем спины и слишком короткие)
    candidates = [(s, e) for s, e in merged if MIN_JUMP_DUR <= (e - s) <= MAX_JUMP_DUR]
    return candidates


def compute_vertical_signal(keypoints: np.ndarray) -> np.ndarray:
    """Покадровые keypoints → вертикальная позиция центра корпуса, нормированная
    длиной торса (масштаб фигуриста) — сигнал инвариантен к зуму камеры.
    NaN там, где торс (плечи+бёдра) не виден."""
    n = len(keypoints)
    y = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        kp = keypoints[i]
        sh = kp[[L_SHOULDER, R_SHOULDER]]
        hp = kp[[L_HIP, R_HIP]]
        if (sh[:, 2] > 0.2).all() and (hp[:, 2] > 0.2).all():
            sh_c = sh[:, :2].mean(axis=0)
            hp_c = hp[:, :2].mean(axis=0)
            torso = float(np.hypot(sh_c[0] - hp_c[0], sh_c[1] - hp_c[1]))
            if torso > 1e-3:
                y[i] = ((sh_c[1] + hp_c[1]) / 2.0) / torso
    return y


def detect_airborne_candidates(times: np.ndarray, keypoints: np.ndarray) -> list[tuple[float, float]]:
    """Кандидаты по фазе полёта: быстрый подъём корпуса над «обычным» уровнем.

    Прыжок — это отрыв ото льда, баллистическая дуга. Сигнал ортогонален
    вращению: ловит прыжки, где rotation-эвристика промахнулась.

    Медленный тренд (проводка/зум камеры) вычитается → остаётся быстрая дуга.
    Ось экрана: меньше y = выше, поэтому подъём корпуса = (slow - y) > 0.
    """
    if len(keypoints) == 0:
        return []
    y = compute_vertical_signal(keypoints)
    valid = ~np.isnan(y)
    if valid.sum() < 4:
        return []
    y = np.interp(np.arange(len(y)), np.where(valid)[0], y[valid])

    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0 / SAMPLE_FPS
    fps = 1.0 / max(dt, 1e-6)
    slow = gaussian_filter1d(y, sigma=max(fps * AIRBORNE_DETREND_SEC, 1.0))
    rise = gaussian_filter1d(slow - y, sigma=SMOOTH_SIGMA)   # >0 когда корпус выше тренда

    airborne = rise > AIRBORNE_THRESH
    raw = []
    i, n = 0, len(airborne)
    while i < n:
        if airborne[i]:
            j = i
            while j < n and airborne[j]:
                j += 1
            raw.append((times[i], times[j - 1]))
            i = j
        else:
            i += 1

    merged = []
    for seg in raw:
        if merged and seg[0] - merged[-1][1] <= MERGE_GAP:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(list(seg))
    return [(s, e) for s, e in merged if MIN_JUMP_DUR <= (e - s) <= MAX_JUMP_DUR]


def merge_candidate_lists(*lists, gap: float = MERGE_GAP) -> list[tuple[float, float]]:
    """Объединяет списки кандидатов (s,e) из разных эвристик (rotation +
    airborne), сливая пересекающиеся/близкие сегменты в один."""
    allc = sorted(seg for lst in lists for seg in lst)
    if not allc:
        return []
    merged = [list(allc[0])]
    for s, e in allc[1:]:
        if s - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def load_ground_truth(video_id: int, start_sec: float, end_sec: float) -> list[tuple[float, float]]:
    df = parse_excel(TIMECODES_PATH)
    sub = df[df["video_id"] == video_id]
    gt = []
    for _, row in sub.iterrows():
        ts = _time_to_sec(row["t_start_val"])
        te = _time_to_sec(row["t_end_val"])
        if te >= start_sec and ts <= end_sec:
            gt.append((ts, te))
    return sorted(gt)


def evaluate(candidates, gt_jumps):
    """Кандидат матчится с GT-прыжком если окна пересекаются (с допуском)."""
    matched = set()
    tp = 0
    for cs, ce in candidates:
        for gi, (gs, ge) in enumerate(gt_jumps):
            if gi in matched:
                continue
            if cs <= ge + MATCH_TOLERANCE and ce >= gs - MATCH_TOLERANCE:
                tp += 1
                matched.add(gi)
                break
    fp = len(candidates) - tp
    fn = len(gt_jumps) - len(matched)
    recall = tp / max(len(gt_jumps), 1)
    precision = tp / max(len(candidates), 1)
    return {"tp": tp, "fp": fp, "fn": fn, "recall": recall, "precision": precision}


def save_plot(times, ang_speed, candidates, rejected, gt_jumps, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(times, ang_speed, color="tab:blue", lw=0.8, label="angular speed (rev/s)")
    ax.axhline(SPEED_THRESH, color="gray", ls="--", lw=1, label=f"threshold {SPEED_THRESH}")
    for gs, ge in gt_jumps:
        ax.axvspan(gs, ge, color="tab:green", alpha=0.25)
    for cs, ce in rejected:
        ax.axvspan(cs, ce, color="gray", alpha=0.25)        # verifier отбраковал
    for cs, ce in candidates:
        ax.axvspan(cs, ce, color="tab:red", alpha=0.40)      # verifier оставил
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rev / sec")
    ax.set_title(f"{title}\nзелёный = GT, красный = детекция (после verifier), серый = отбраковано verifier")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ============================================================
# Verifier (шаг 2 пайплайна)
# ============================================================

def _extract_window_kpts(cap, center_sec, src_fps, total, yolo, device) -> torch.Tensor:
    """Окно кандидата → (VERIFIER_NUM_FRAMES, 17, 3) keypoints главного фигуриста."""
    t0 = center_sec - VERIFIER_WINDOW_SEC / 2
    start_frame = int(np.clip(t0 * src_fps, 0, max(total - 1, 0)))
    span = int(VERIFIER_WINDOW_SEC * src_fps) + 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    raw = []
    for _ in range(span):
        ret, fr = cap.read()
        if not ret:
            break
        raw.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))

    kpts = torch.zeros(VERIFIER_NUM_FRAMES, 17, 3, dtype=torch.float32)
    if raw:
        idx = np.linspace(0, len(raw) - 1, VERIFIER_NUM_FRAMES).round().astype(int)
        frames = [raw[i] for i in idx]
        results = yolo(frames, verbose=False, conf=POSE_CONF, imgsz=POSE_IMGSZ, device=device)
        for j, res in enumerate(results):
            if res.keypoints is None or res.keypoints.data is None or len(res.keypoints.data) == 0:
                continue
            kp = res.keypoints.data.cpu().numpy()
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                person = int(areas.argmax())
            else:
                person = 0
            kpts[j] = torch.from_numpy(kp[person].astype(np.float32))
    return kpts


@torch.no_grad()
def verify_candidates(video_path, candidates, device):
    """Re-extract окно каждого кандидата → verifier-ensemble → P(jump).
    Возвращает [(cs, ce, p_jump)] или None если чекпоинтов нет."""
    ckpt_paths = [VERIFIER_CKPT_DIR / f"seed_{s}" / "best.pt" for s in VERIFIER_SEEDS]
    ckpt_paths = [p for p in ckpt_paths if p.is_file()]
    if not ckpt_paths:
        print(f"Verifier чекпоинты не найдены в {VERIFIER_CKPT_DIR} — пропускаю верификацию.")
        print("Сначала: python scripts/train_verifier.py")
        return None

    models = []
    for p in ckpt_paths:
        m = VerifierModel(VERIFIER_FEATURE_DIM, VERIFIER_HIDDEN).to(device)
        m.load_state_dict(torch.load(p, map_location=device, weights_only=False)["model_state_dict"])
        m.eval()
        models.append(m)
    print(f"Verifier: ensemble из {len(models)} моделей, порог {VERIFIER_THRESHOLD}")

    from ultralytics import YOLO
    from tqdm import tqdm
    yolo = YOLO(POSE_MODEL)
    yolo.to(device)

    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    verified = []
    for cs, ce in tqdm(candidates, desc="verifier"):
        kpts = _extract_window_kpts(cap, (cs + ce) / 2.0, src_fps, total, yolo, device)
        feats = extract_features_f4(kpts, smooth_sigma=VERIFIER_SMOOTH).unsqueeze(0).to(device)
        probs = [torch.softmax(m(feats), dim=-1)[0, 1].item() for m in models]
        verified.append((cs, ce, float(np.mean(probs))))
    cap.release()
    return verified


def merge_verified(verified, gap: float):
    """Сливает пересекающиеся/близкие verified-окна в одну детекцию.

    Детектор вращения нередко выдаёт 2 окна на один прыжок — оба проходят
    verifier и без склейки раздувают FP. Здесь близкие по времени окна
    объединяются: интервал = объединение, p_jump = максимум по группе.

    verified: [(cs, ce, p)] — окна, прошедшие порог.
    Возвращает [(cs, ce, p)] — отсортированный список детекций.
    """
    if not verified:
        return []
    items = sorted(verified, key=lambda x: x[0])
    merged = [list(items[0])]
    for cs, ce, p in items[1:]:
        if cs - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], ce)
            merged[-1][2] = max(merged[-1][2], p)
        else:
            merged.append([cs, ce, p])
    return [(s, e, p) for s, e, p in merged]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str,
                        help="ID локального видео (9/10/11) ИЛИ путь к произвольному mp4")
    parser.add_argument("--video-path", type=str, help="путь к произвольному видео (синоним --video)")
    parser.add_argument("--start", type=float, default=0.0, help="начало обработки, сек")
    parser.add_argument("--end", type=float, default=0.0, help="конец, сек (0 = до конца)")
    parser.add_argument("--no-verify", action="store_true", help="пропустить verifier (только rotation-эвристика)")
    args = parser.parse_args()

    arg = args.video or args.video_path
    if not arg:
        parser.error("укажи --video (ID 9/10/11 или путь к mp4) или --video-path")

    # --video принимает и ID локального размеченного видео, и путь к файлу
    local_id = int(arg) if arg in ("9", "10", "11") else None
    if local_id is not None:
        args.video = local_id                              # дальше код сверяется с разметкой по ID
        video_path = str(VIDEOS_DIR / VIDEO_FILES[local_id])
        tag = f"video{local_id}"
    else:
        args.video = None
        video_path = arg
        tag = Path(video_path).stem

    device = _resolve_device()
    print(f"Device: {device}, Pose model: {POSE_MODEL}")

    times, angles, keypoints = extract_rotation_signal(video_path, args.start, args.end, device)
    if len(times) == 0:
        print("Не извлечено ни одного кадра")
        return

    ang_speed = compute_angular_speed(times, angles)
    rot_candidates = detect_candidates(times, ang_speed)
    air_candidates = detect_airborne_candidates(times, keypoints)
    candidates = merge_candidate_lists(rot_candidates, air_candidates)
    print(f"\nКандидатов: rotation={len(rot_candidates)}, airborne={len(air_candidates)}, "
          f"после объединения={len(candidates)}")

    # --- шаг 2: verifier ---
    verified = None if args.no_verify else verify_candidates(video_path, candidates, device)
    if verified is not None:
        kept = [(cs, ce, p) for cs, ce, p in verified if p >= VERIFIER_THRESHOLD]
        rejected = [(cs, ce) for cs, ce, p in verified if p < VERIFIER_THRESHOLD]
        merged = merge_verified(kept, VERIFY_MERGE_GAP)
        print(f"После verifier: {len(kept)}/{len(candidates)} прошли порог, "
              f"{len(merged)} детекций после склейки окон")
        final_candidates = [(cs, ce) for cs, ce, p in merged]
    else:
        rejected = []
        merged = []
        final_candidates = candidates

    result = {
        "video": tag,
        "candidates_raw": [[round(s, 1), round(e, 1)] for s, e in candidates],
    }
    if verified is not None:
        result["candidates_verified"] = [[round(s, 1), round(e, 1), round(p, 3)] for s, e, p in verified]
        result["candidates_final"] = [[round(s, 1), round(e, 1), round(p, 3)] for s, e, p in merged]
        result["verifier_threshold"] = VERIFIER_THRESHOLD

    # сверка с разметкой (только для видео 9/10/11)
    if args.video is not None:
        gt = load_ground_truth(args.video, times[0], times[-1])
        print(f"\nGT прыжков в диапазоне: {len(gt)}")

        m_raw = evaluate(candidates, gt)
        result["metrics_raw"] = m_raw
        print(f"  [rotation]  R={m_raw['recall']:.3f}  P={m_raw['precision']:.3f}  "
              f"(TP={m_raw['tp']} FP={m_raw['fp']} FN={m_raw['fn']})")

        if verified is not None:
            m_ver = evaluate(final_candidates, gt)
            result["metrics_verified"] = m_ver
            print(f"  [+verifier] R={m_ver['recall']:.3f}  P={m_ver['precision']:.3f}  "
                  f"(TP={m_ver['tp']} FP={m_ver['fp']} FN={m_ver['fn']})")
            print(f"  precision выросла в {m_ver['precision'] / max(m_raw['precision'], 1e-6):.1f}×")

        plot_path = OUT_DIR / f"{tag}_detection.png"
        save_plot(times, ang_speed, final_candidates, rejected, gt, plot_path, f"{tag}: jump detection")
        print(f"График: {plot_path}")
    else:
        save_plot(times, ang_speed, final_candidates, rejected, [],
                  OUT_DIR / f"{tag}_detection.png", f"{tag}: jump detection")

    out_json = OUT_DIR / f"{tag}_candidates.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"JSON: {out_json}")


if __name__ == "__main__":
    main()
