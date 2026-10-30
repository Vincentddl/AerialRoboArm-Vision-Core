"""Start a logged real-hardware follow/gripper test, armed by the operator."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time

from serial.tools import list_ports
from capture_ffmpeg_manual import RttTelemetryLogger

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / 'configs/servo_to_optical_angle_foam_center_lut_20261008_combined_v1.json'


def find_port(requested: str) -> str:
    ports = list(list_ports.comports())
    if requested != 'auto':
        if requested not in [p.device for p in ports]:
            raise RuntimeError(f'HC13 port {requested} is not connected.')
        return requested
    candidates = [p.device for p in ports if p.vid == 0x1A86 and p.pid == 0x7523]
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one HC13 CH340 adapter, found {candidates}; use --port COMx if needed.')
    return candidates[0]


def find_camera() -> int:
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('FFmpeg is unavailable.')
    result = subprocess.run([ffmpeg, '-hide_banner', '-list_devices', 'true', '-f', 'dshow', '-i', 'dummy'],
                            capture_output=True, timeout=10)
    names = re.findall(r'"([^"\r\n]+)" \(video\)',
                       (result.stdout + result.stderr).decode('utf-8', errors='replace'))
    if names.count('USB Video') != 1:
        raise RuntimeError(f'Analog USB Video input not found uniquely: {names}')
    return names.index('USB Video')


def check_mcu() -> str:
    with socket.create_connection(('127.0.0.1', 9090), timeout=2) as link:
        link.settimeout(.3)
        chunks = []
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                data = link.recv(65536)
            except socket.timeout:
                continue
            if not data:
                break
            chunks.append(data.decode('utf-8', errors='replace').replace('\x00', ''))
    text = ''.join(chunks)
    if 'SERVO1_DBG' in text or 'debug=SERVO1' in text:
        raise RuntimeError('SERVO1 debug is active; send s to exit before vision following.')
    lines = text.splitlines()
    values = {}
    home_stage = None
    mode = None
    for line in lines:
        for key, value in re.findall(r'\b(voltage|rc|stall|fb_age)=([0-9]+)', line):
            values[key] = int(value)
        if match := re.search(r'status=0x([0-9a-fA-F]+)', line):
            values['status'] = int(match.group(1), 16)
        if match := re.search(r'\[HOME\] stage=(\d+)', line):
            home_stage = int(match.group(1))
        if match := re.search(r'\bhome=(\d+)', line):
            home_stage = int(match.group(1))
        if match := re.search(r'\[DBG\]\s+(AUTO|MANUAL|ERROR|INIT)\b', line):
            mode = match.group(1)
    if not 9000 <= values.get('voltage', 0) <= 12600 or values.get('status', 255) & 0xFE:
        raise RuntimeError(f'HX8 supply/fault preflight failed: {values}')
    if values.get('fb_age', 999) >= 140 or values.get('stall', 1):
        raise RuntimeError(f'HX8 feedback preflight failed: {values}')
    if not values.get('rc'):
        raise RuntimeError('ELRS is offline; normal AUTO cannot run.')
    if home_stage != 2:
        raise RuntimeError('HX8 homing is not complete; wait for [HOME] stage=2 with the new firmware.')
    if mode not in ('MANUAL', 'AUTO'):
        raise RuntimeError('MCU is not in normal MANUAL/AUTO; clear the fault before testing.')
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='auto')
    parser.add_argument('--device', default='0')
    args = parser.parse_args()
    port = find_port(args.port)
    camera = find_camera()
    preflight = check_mcu()
    out = ROOT / 'outputs/real_foam_test' / ('live_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=True)
    (out / 'preflight.txt').write_text(preflight, encoding='utf-8')
    logger = RttTelemetryLogger('127.0.0.1', 9090, out / 'mcu_feedback.jsonl', time.monotonic())
    command = [sys.executable, '-u', str(ROOT / 'run_realtime.py'), '--source', str(camera),
               '--servo-angle-calibration', str(CANDIDATE), '--hc13-port', port,
               '--device', args.device, '--imgsz', '640', '--show-centers',
               '--new-track-confirm-frames', '1', '--min-age', '1', '--hc13-confirm-frames', '1',
               '--save-jsonl', str(out / 'visual_targets.jsonl'),
               '--window-title', f'Live foam follow - grip manual CH4 - {port} - A sends']
    logger.start()
    try:
        child = subprocess.Popen(command, cwd=str(ROOT))
        (out / 'session.json').write_text(json.dumps({
            'camera_name': 'USB Video', 'camera_index': camera, 'hc13_port': port,
            'preview_pid': child.pid, 'arm_confirmation_frames': 1,
            'gripper_control': 'manual_CH4', 'armed_on_start': False,
            'candidate': str(CANDIDATE), 'started_at': datetime.now().astimezone().isoformat(),
        }, indent=2), encoding='utf-8')
        print(f'LOGS: {out}', flush=True)
        print('Keep foam fully visible on the center axis. Clear hands before pressing A. Q ends and saves logs.', flush=True)
        return child.wait()
    finally:
        logger.stop()
        if logger.error:
            print(f'RTT logging error: {logger.error}', file=sys.stderr)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        print(f'PRECHECK FAILED: {exc}', file=sys.stderr)
        raise SystemExit(1)
