"""Refit command/optical correspondence from several captured sessions.

All input sessions become training data. Leave-one-recording-out diagnostics
are model-selection evidence, never a substitute for a fresh final test.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.optimize import lsq_linear
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision.runtime import result_to_detections
from vision.tracker import PixelToWorldMapper


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def basis(angles, knots):
    angles = np.asarray(angles)
    return np.column_stack([np.ones(len(angles)), *[
        np.clip(angles - left, 0, right - left)
        for left, right in zip(knots, knots[1:])
    ]])


def fit(rows, mode, knots, smoothness=10.):
    # Equal recording weight; robust Huber iterations limit isolated shifts.
    counts = Counter(row['session'] for row in rows)
    weights = np.array([1 / counts[row['session']] for row in rows])
    weights /= weights.mean()
    design = basis([row['command_deg'] for row in rows], knots)
    observed = np.array([np.median(row['optical'][mode]) for row in rows])
    current = weights.copy()
    penalty = np.zeros((len(knots) - 2, len(knots)))
    for index in range(len(penalty)):
        penalty[index, index + 1:index + 3] = (-1, 1)
    for _ in range(12):
        coeff = lsq_linear(np.vstack((design * np.sqrt(current)[:, None], np.sqrt(smoothness) * penalty)),
                           np.concatenate((observed * np.sqrt(current), np.zeros(len(penalty)))),
                           bounds=([-np.inf] + [0.10] * (len(knots) - 1),
                                   [np.inf] + [1.50] * (len(knots) - 1))).x
        residual = np.abs(observed - design @ coeff)
        current = weights * np.minimum(1., 0.4 / np.maximum(residual, 1e-9))
    optical = basis(knots, knots) @ coeff
    assert np.all(np.diff(optical) > 0)
    return optical


def inverse(values, optical, knots):
    values = np.asarray(values)
    predicted = np.interp(values, optical, knots)
    for mask, index in ((values < optical[0], 0), (values > optical[-1], -2)):
        predicted[mask] = knots[index] + (values[mask] - optical[index]) * (
            knots[index + 1] - knots[index]) / (optical[index + 1] - optical[index])
    return predicted


def metrics(rows, mode, optical, knots):
    encoder_errors, command_errors = [], []
    inside = total = 0
    for row in rows:
        values = np.asarray(row['optical'][mode])
        pred = inverse(values, optical, knots)
        encoder_errors.extend(pred - row['encoder_deg'])
        command_errors.extend(pred - row['command_deg'])
        inside += int(np.sum((values >= optical[0]) & (values <= optical[-1])))
        total += len(values)
    def summary(errors):
        a = np.asarray(errors); absolute = np.abs(a)
        return {'mae_deg': round(float(absolute.mean()), 4),
                'p95_deg': round(float(np.percentile(absolute, 95)), 4),
                'max_deg': round(float(absolute.max()), 4),
                'bias_deg': round(float(a.mean()), 4)}
    return {'holds': len(rows), 'frames': total, 'range_coverage': round(inside / total, 4),
            'encoder_error': summary(encoder_errors), 'command_error': summary(command_errors),
            'extrapolated_frames_included_for_diagnostics': total - inside}


def collect(reports, minimum, maximum, output):
    model_path = ROOT / 'models/foam_center_v9_gripper_axis_normal_bg03_20260816_candidate.pt'
    model = YOLO(str(model_path))
    mapper = PixelToWorldMapper(ROOT / 'configs/camera_2p1mm_640x480_fisheye.json')
    rows, excluded, sources = [], [], []
    for sid, report_path in enumerate(reports):
        report = json.loads(report_path.read_text(encoding='utf-8'))
        sources.append({'report': str(report_path.resolve()), 'report_sha256': file_hash(report_path), 'video': report['video'],
                        'video_sha256': file_hash(report['video'])})
        telemetry = [json.loads(line) for line in Path(report['rtt']).read_text(encoding='utf-8').splitlines() if line.strip()]
        capture = cv2.VideoCapture(report['video'])
        fps = capture.get(cv2.CAP_PROP_FPS)
        selected, states = [], {}
        for hold in report['holds']:
            step = hold['step']; command = hold['guide_angle_deg']
            if not minimum <= command <= maximum:
                excluded.append({'session': sid, 'step': step, 'reason': 'outside requested range'})
                continue
            snapshots = [r for r in telemetry if hold['start_elapsed'] <= r['elapsed_seconds'] <= hold['end_elapsed']]
            positions = [r['pos'] for r in snapshots if 'pos' in r]
            volts = [r['voltage'] for r in snapshots if 'voltage' in r]
            targets = [r['tgt'] for r in snapshots if 'tgt' in r]
            if (len(positions) < 2 or max(positions) - min(positions) > .25 or not volts
                    or min(volts) < 9000 or max(volts) > 12600 or not targets
                    or abs(np.median(targets) - command) > .1
                    or any(r.get('rd', 0) or r.get('stall', 0) for r in snapshots)):
                excluded.append({'session': sid, 'step': step, 'reason': 'invalid or unstable telemetry'})
                continue
            states[step] = {'session': sid, 'step': step, 'command_deg': command,
                            'encoder_deg': float(np.median(positions)), 'sampled': 0,
                            'low_confidence': 0, 'clipped': 0, 'center_sources': [],
                            'center_delta_px': [], 'optical': {'runtime_hybrid': [], 'box_center': []}}
            first = hold['start_frame'] + round(.8 * fps)
            last = hold['end_frame'] - round(.2 * fps)
            for frame_index in range(first, max(first, last) + 1, max(1, round(fps / 5))):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok: raise RuntimeError(f'Unreadable frame {sid}:{frame_index}')
                selected.append((step, frame))
                states[step]['sampled'] += 1
        capture.release()
        for start in range(0, len(selected), 16):
            batch = selected[start:start + 16]
            results = model.predict([frame for _, frame in batch], device='cpu', imgsz=640,
                                    conf=.5, max_det=5, verbose=False)
            for (step, frame), result in zip(batch, results):
                state = states[step]
                detections = result_to_detections(result, allowed_classes={0}, image_shape=frame.shape)
                if not detections:
                    state['low_confidence'] += 1
                    continue
                detection = max(detections, key=lambda d: d.score)
                x1, y1, x2, y2 = detection.bbox
                if x1 <= 2 or y1 <= 2 or x2 >= frame.shape[1] - 2 or y2 >= frame.shape[0] - 2:
                    state['clipped'] += 1
                    continue
                box = ((x1 + x2) / 2, (y1 + y2) / 2)
                center = detection.center
                state['center_sources'].append(detection.center_source)
                state['center_delta_px'].append(float(np.linalg.norm(np.array(center) - box)))
                for mode, pixel in (('runtime_hybrid', center), ('box_center', box)):
                    state['optical'][mode].append(float(mapper.camera_plane_angle_deg(mapper.to_world(pixel))))
        for state in states.values():
            valid = len(state['optical']['runtime_hybrid'])
            if valid / state['sampled'] < .90:
                excluded.append({**{k: state[k] for k in ('session', 'step', 'command_deg', 'sampled', 'low_confidence', 'clipped')},
                                 'valid_frames': valid, 'reason': 'fewer than 90% complete high-confidence targets'})
            else:
                rows.append(state)
        print(f'Recording {sid + 1}/{len(reports)} extracted; retained holds so far: {len(rows)}', flush=True)
    cache = {'sources': sources, 'model_sha256': file_hash(model_path), 'rows': rows, 'excluded': excluded}
    output.write_text(json.dumps(cache, indent=2), encoding='utf-8')
    return cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/merged_relation_20261006')
    parser.add_argument('--reuse-cache', action='store_true')
    parser.add_argument('--smoothness', type=float, default=10.)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.smoothness < 0: raise ValueError('smoothness must be nonnegative')
    cache_path = args.output_dir / 'observations.json'
    data = json.loads(cache_path.read_text(encoding='utf-8')) if args.reuse_cache else collect(args.reports, -77, -27, cache_path)
    if args.reuse_cache:
        if [str(p.resolve()) for p in args.reports] != [s['report'] for s in data['sources']]:
            raise ValueError('Cached recordings differ from requested reports')
        if any(s.get('report_sha256') != file_hash(s['report']) for s in data['sources']):
            raise ValueError('Cached input report changed; extract observations again')
        if any(s['video_sha256'] != file_hash(s['video']) for s in data['sources']):
            raise ValueError('Cached input recording changed; extract observations again')
        if data['model_sha256'] != file_hash(ROOT / 'models/foam_center_v9_gripper_axis_normal_bg03_20260816_candidate.pt'):
            raise ValueError('Detector weights changed; extract observations again')
    rows = data['rows']; knots = np.arange(-77., -26., 5.)
    if len(rows) < 20: raise RuntimeError('Insufficient usable stable holds')
    results = {}
    for mode in ('runtime_hybrid', 'box_center'):
        optical = fit(rows, mode, knots, args.smoothness)
        folds = []
        for session in sorted({row['session'] for row in rows}):
            train = [row for row in rows if row['session'] != session]
            held = [row for row in rows if row['session'] == session]
            folds.append({'held_recording': session, **metrics(held, mode, fit(train, mode, knots, args.smoothness), knots)})
        results[mode] = {'points': [{'servo_command_deg': float(g), 'optical_offset_deg': round(float(b), 6)} for g, b in zip(knots, optical)],
                         'training_diagnostics': metrics(rows, mode, optical, knots),
                         'leave_one_recording_out': folds}
    report = {'created_at': datetime.now().astimezone().isoformat(timespec='seconds'),
              'sources': data['sources'], 'retained_holds': len(rows), 'excluded': data['excluded'],
              'center_source_counts': dict(Counter(s for row in rows for s in row['center_sources'])),
              'center_delta_px_median': float(np.median([v for row in rows for v in row['center_delta_px']])),
              'fit_settings': {'knots': knots.tolist(), 'slope_bounds': [.10, 1.50],
                               'huber_optical_delta_deg': .4, 'slope_smoothness': args.smoothness,
                               'recordings_equal_weight': True},
              'results': results, 'final_independent_test': 'pending; all three recordings are now training data'}
    (args.output_dir / 'fit_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    chosen = results['runtime_hybrid']
    points = chosen['points']
    support = [min(row['command_deg'] for row in rows), max(row['command_deg'] for row in rows)]
    candidate = {
        'type': 'servo_optical_angle_lut', 'name': 'foam_center_servo_camera_lut_20261006_merged_v1',
        'version': 1, 'created_at': report['created_at'],
        'status': 'merged_training_complete_new_independent_test_pending',
        'definitions': {'servo_command_deg': 'Nominal g command from pooled recordings, not measured encoder position.',
                        'optical_offset_deg': 'Fisheye-corrected ray angle of the current runtime visible foam centre.'},
        'runtime_anchor': {'detector': 'foam_center_v9_gripper_axis_normal_bg03_20260816_candidate.pt',
                           'detector_sha256': data['model_sha256'], 'point': 'current runtime hybrid',
                           'inference_device': 'cpu', 'imgsz': 640, 'confidence_min': .5,
                           'reject_clipped_target_boxes': True},
        'model': {'kind': 'piecewise_linear_lut', 'interpolation': 'linear',
                  'inverse_interpolation': 'linear', 'points': points},
        'valid_range': {'servo_command_deg': [-77., -27.],
                        'optical_offset_deg': [points[0]['optical_offset_deg'], points[-1]['optical_offset_deg']]},
        'boundary_tolerance_deg': 0.,
        'fit': {'method': 'Recording-balanced Huber robust monotonic fit with slope smoothing.',
                'settings': report['fit_settings'], 'holds': len(rows),
                'frames': chosen['training_diagnostics']['frames'], 'observed_command_support_deg': support,
                'training_diagnostics': chosen['training_diagnostics'],
                'leave_one_recording_out': chosen['leave_one_recording_out'],
                'center_comparison': 'Fixed bbox centre had worse pooled MAE; current centre definition retained.',
                'quality_exclusions': data['excluded']},
        'source': {'training_recordings': data['sources'],
                   'fit_report': str((args.output_dir / 'fit_report.json').resolve()),
                   'camera_calibration': 'configs/camera_2p1mm_640x480_fisheye.json'},
        'independent_validation': None,
        'warning': 'Fourth independent test pending. Training P95 exceeds 2 degrees; '
                   'recordings show angle/direction drift and -29..-27 lacks complete-target fitting support. '
                   'Preview only; do not activate automatic HC13 output.',
    }
    (args.output_dir / 'candidate.json').write_text(json.dumps(candidate, indent=2), encoding='utf-8')
    print(json.dumps({k: {n: v for n, v in result.items() if n != 'points'} for k, result in results.items()}, indent=2))


if __name__ == '__main__':
    main()
