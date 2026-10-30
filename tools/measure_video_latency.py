"""Optical time-code loop test. No model, serial port, or actuator access.

The primary metric starts immediately before cv2.imshow and ends immediately
after VideoCapture.read returns a frame containing that code. Source display
latency is INCLUDED. Decoder time is reported separately, not subtracted from
an arbitrary wall-clock timestamp. See docs/video_latency_test.md.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import secrets
import sys
import threading
import time
import zlib

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_WINDOW = "SOURCE - point the FPV camera at this code only"
PREVIEW_WINDOW = "RECEIVED - SPACE start / Q stop"
BOUNDARY = (
    "Host imshow submission to return of the first decoded capture frame for "
    "each unique optical code. Includes source display, camera, transmitter, "
    "receiver, USB, driver and capture. Excludes QR decoder runtime from the "
    "primary timestamp difference. Not isolated RF latency or photon-to-host latency."
)


@dataclass
class SourceEvent:
    sequence: int
    created_ns: int
    submitted_ns: int
    payload: str
    imshow_returned_ns: int = 0
    gui_pump_returned_ns: int = 0


def make_payload(session: str, sequence: int, created_ns: int) -> str:
    base = f"VT1:{session}:{sequence}:{created_ns}"
    return f"{base}:{zlib.crc32(base.encode('ascii')) & 0xffffffff:08x}"


def parse_payload(payload: str) -> tuple[str, int, int] | None:
    if not isinstance(payload, str) or len(payload) > 160:
        return None
    parts = payload.split(":")
    if len(parts) != 5 or parts[0] != "VT1":
        return None
    if not re.fullmatch(r"[0-9a-f]{12}", parts[1]):
        return None
    if not re.fullmatch(r"[0-9]{1,10}", parts[2]) or not re.fullmatch(r"[0-9]{1,22}", parts[3]):
        return None
    base = ":".join(parts[:4])
    if parts[4] != f"{zlib.crc32(base.encode('ascii')) & 0xffffffff:08x}":
        return None
    sequence, created = int(parts[2]), int(parts[3])
    return (parts[1], sequence, created) if sequence > 0 and created > 0 else None


def make_source_panel(payload: str, sequence: int, created_ns: int, qr_size: int = 512) -> np.ndarray:
    qr = cv2.QRCodeEncoder_create().encode(payload)
    # An explicit quiet zone also survives moderate camera cropping/perspective.
    qr = cv2.copyMakeBorder(qr, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=255)
    qr = cv2.resize(qr, (qr_size, qr_size), interpolation=cv2.INTER_NEAREST)
    width, height = qr_size + 128, qr_size + 208
    panel = np.full((height, width, 3), 255, dtype=np.uint8)
    panel[96:96+qr_size, 64:64+qr_size] = cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)
    lines = [
        ("VIDEO LATENCY SOURCE", 32, 0.85),
        (f"Code {sequence:06d}  |  session {payload.split(':')[1]}", 66, 0.54),
        (f"Monotonic code timestamp: {created_ns / 1e6:.3f} ms", qr_size+130, 0.48),
        ("Camera must see this window, not the received preview.", qr_size+160, 0.46),
        ("SPACE: start recording    Q / ESC: save and exit", qr_size+187, 0.47),
    ]
    for label,y,scale in lines:
        cv2.putText(panel, label, (20,y), cv2.FONT_HERSHEY_SIMPLEX, scale, (15,15,15), 1, cv2.LINE_AA)
    return panel


def decode_frame(frame: np.ndarray, detector) -> tuple[str, np.ndarray | None, str]:
    """Retry camera/analog edge distortion without changing capture timestamps."""
    payload,points,_=detector.detectAndDecode(frame)
    if payload:
        return payload,points,"original"
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) if frame.ndim==3 else frame
    binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    payload,points,_=detector.detectAndDecode(binary)
    if payload:
        return payload,points,"otsu"
    if frame.shape[1]<=960:
        enlarged=cv2.resize(frame,None,fx=2,fy=2,interpolation=cv2.INTER_LINEAR)
        payload,points,_=detector.detectAndDecode(enlarged)
        if payload:
            return payload,points/2 if points is not None else None,"upscale_2x"
    return "",None,"not_decoded"


class Ledger:
    """Only first appearances of issued, session-matching codes are samples."""
    def __init__(self, session: str, max_latency_ms: float):
        self.session = session
        self.max_latency_ms = max_latency_ms
        self.issued: dict[int, SourceEvent] = {}
        self.seen: set[int] = set()
        self.samples: list[dict] = []
        self.counts: Counter = Counter()
        self.start_ns: int | None = None
        self.end_ns: int | None = None

    def observe(self, payload: str, frame_index: int, read_ns: int,
                decode_started_ns: int, decode_finished_ns: int) -> tuple[str, dict | None]:
        parsed = parse_payload(payload)
        if parsed is None:
            self.counts["empty_or_invalid_code"] += 1
            return "NO VALID CODE", None
        session, sequence, created_ns = parsed
        event = self.issued.get(sequence)
        if session != self.session or event is None or event.created_ns != created_ns or event.payload != payload:
            self.counts["foreign_or_unissued_code"] += 1
            return "FOREIGN CODE", None
        elapsed = (read_ns - event.submitted_ns) / 1e6
        if read_ns < event.submitted_ns or decode_started_ns < read_ns or decode_finished_ns < decode_started_ns:
            self.counts["invalid_timestamp_order"] += 1
            return "TIMESTAMP REJECTED", None
        if self.start_ns is None:
            return f"PREVIEW {elapsed:.1f} ms - SPACE to record", None
        if event.submitted_ns < self.start_ns or read_ns < self.start_ns:
            self.counts["pre_measurement_code"] += 1
            return "WARMUP", None
        if self.end_ns is not None and read_ns > self.end_ns:
            self.counts["after_measurement"] += 1
            return "FINISHED", None
        if elapsed > self.max_latency_ms:
            self.counts["above_max_latency"] += 1
            return "ABOVE LATENCY LIMIT", None
        if sequence in self.seen:
            self.counts["duplicate_code"] += 1
            return f"HOLD code {sequence} (not another sample)", None
        self.seen.add(sequence)
        row = dict(sequence=sequence, frame_index=frame_index,
                   source_created_ns=created_ns, source_submitted_ns=event.submitted_ns,
                   source_imshow_returned_ns=event.imshow_returned_ns,
                   capture_read_returned_ns=read_ns, decode_started_ns=decode_started_ns,
                   decode_finished_ns=decode_finished_ns,
                   submit_to_capture_ms=elapsed,
                   creation_to_capture_ms=(read_ns-created_ns)/1e6,
                   source_preparation_ms=(event.submitted_ns-created_ns)/1e6,
                   capture_to_decode_start_ms=(decode_started_ns-read_ns)/1e6,
                   qr_decode_ms=(decode_finished_ns-decode_started_ns)/1e6,
                   submit_to_decoder_finished_ms=(decode_finished_ns-event.submitted_ns)/1e6)
        self.samples.append(row)
        return f"NEW code {sequence}: {elapsed:.1f} ms", row


def statistics(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    values = np.asarray([r["submit_to_capture_ms"] for r in rows], dtype=np.float64)
    return dict(count=len(rows), mean_ms=float(values.mean()), median_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values,95)), minimum_ms=float(values.min()), maximum_ms=float(values.max()))


class CaptureWorker:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.latest = None
        self.frame_count = 0
        self.read_failures = 0
        self.error = ""
        self.properties = {}
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        if not self.ready.wait(8):
            raise RuntimeError("Camera opening timed out. Close other camera apps and check --source.")
        if self.error:
            raise RuntimeError(self.error)

    def _run(self):
        cap = None
        try:
            backend = {"dshow":cv2.CAP_DSHOW,"msmf":cv2.CAP_MSMF,"any":cv2.CAP_ANY}[self.args.backend]
            cap = cv2.VideoCapture(self.args.source, backend)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video source {self.args.source}. Close other camera apps or change --source.")
            requests = {"width":cap.set(cv2.CAP_PROP_FRAME_WIDTH,self.args.width),
                        "height":cap.set(cv2.CAP_PROP_FRAME_HEIGHT,self.args.height),
                        "buffer_size_one":cap.set(cv2.CAP_PROP_BUFFERSIZE,1)}
            if self.args.fps:
                requests["fps"] = cap.set(cv2.CAP_PROP_FPS,self.args.fps)
            if self.args.fourcc:
                requests["fourcc"] = cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*self.args.fourcc))
            code = int(cap.get(cv2.CAP_PROP_FOURCC))
            self.properties = dict(backend=cap.getBackendName(),requested_property_results=requests,
                reported_width=cap.get(cv2.CAP_PROP_FRAME_WIDTH),reported_height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                reported_fps=cap.get(cv2.CAP_PROP_FPS),reported_fourcc="".join(chr((code >> (8*i)) & 255) for i in range(4)),
                note="Requested properties and reported FPS do not prove actual sensor frame rate or zero buffering.")
            self.ready.set()
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                read_ns = time.perf_counter_ns()  # BEFORE QR decoding, drawing, logging or display.
                if not ok or frame is None:
                    self.read_failures += 1
                    time.sleep(.005)
                    continue
                with self.lock:
                    self.frame_count += 1
                    self.latest = (self.frame_count, read_ns, frame)
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.ready.set()
            if cap is not None:
                cap.release()

    def snapshot(self):
        with self.lock:
            return self.latest

    def close(self) -> bool:
        self.stop_event.set()
        self.thread.join(timeout=2)
        return not self.thread.is_alive()


def save_results(folder: Path, ledger: Ledger, args, capture, outcome: str, event_rows: list[dict], evidence: list) -> dict:
    stats = statistics(ledger.samples)
    flags = ["Source monitor presentation time is not calibrated; do not label results as isolated receiver/RF latency.",
             "QR decode misses and latest-frame overwrites can bias first-seen sampling; coverage is not RF packet loss."]
    if len(ledger.samples) < args.min_samples:
        flags.append(f"Insufficient samples: {len(ledger.samples)} < {args.min_samples}.")
    if ledger.counts["above_max_latency"]:
        flags.append("Samples above --max-latency-ms were excluded; accepted-sample quantiles are censored.")
    if outcome != "duration_complete":
        flags.append(f"Measurement ended with {outcome}; treat it as an incomplete/manual run.")
    source_rows = [asdict(e) for e in ledger.issued.values()]
    eligible = [e for e in ledger.issued.values() if ledger.start_ns is not None and ledger.start_ns <= e.submitted_ns <= (ledger.end_ns or 0)]
    intervals=np.diff([e.submitted_ns for e in eligible])/1e6
    metadata = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),session=ledger.session,
        metric="software_display_submission_to_first_decodable_capture_return_ms",boundary=BOUNDARY,
        measurement_method="Single-host monotonic clock and optical session/sequence/timestamp/CRC code; one sample per issued code.",
        source_display_latency_included=True,photon_timestamp_measured=False,actuator_commands_sent=False,
        outcome=outcome,run_completed=(outcome=="duration_complete"),
        sufficient_sample_count=len(ledger.samples)>=args.min_samples,
        start_ns=ledger.start_ns,end_ns=ledger.end_ns,statistics=stats,quality_flags=flags,
        emitted_codes_during_window=len(eligible),matched_unique_codes=len(ledger.samples),
        actual_source_interval_ms=(dict(median=float(np.median(intervals)),p95=float(np.percentile(intervals,95))) if len(intervals) else None),
        coverage_note="Includes source patterns near the end that may not have returned. This is not sensor FPS or wireless loss rate.",
        capture_read_count=capture.frame_count,capture_read_failures=capture.read_failures,
        capture_error=capture.error,capture_properties=capture.properties,counters=dict(ledger.counts),
        arguments=vars(args),python=sys.version,opencv=cv2.__version__,numpy=np.__version__,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (folder/"summary.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    for name,rows in [("samples.csv",ledger.samples),("source_events.csv",source_rows)]:
        with (folder/name).open("w",newline="",encoding="utf-8-sig") as handle:
            if rows:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (folder/"events.jsonl").open("w",encoding="utf-8") as handle:
        for row in event_rows:handle.write(json.dumps(row,ensure_ascii=False)+"\n")
    for index,(row,frame) in enumerate(evidence,1):
        cv2.imwrite(str(folder/f"received_{index:02d}_code_{row['sequence']}.jpg"),frame)
    return metadata


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=int,default=0,help="USB receiver camera index; verify device order after reconnecting; not a COM port")
    parser.add_argument("--backend",choices=("dshow","msmf","any"),default="dshow")
    parser.add_argument("--width",type=int,default=640)
    parser.add_argument("--height",type=int,default=480)
    parser.add_argument("--fps",type=float,default=0,help="Optional request, leave zero to keep driver default")
    parser.add_argument("--fourcc",default="",help="Optional four-letter format, e.g. MJPG")
    parser.add_argument("--pattern-hz",type=float,default=8)
    parser.add_argument("--seconds",type=float,default=45)
    parser.add_argument("--warmup",type=float,default=2)
    parser.add_argument("--min-samples",type=int,default=30)
    parser.add_argument("--max-latency-ms",type=float,default=5000)
    parser.add_argument("--qr-size",type=int,default=512)
    parser.add_argument("--output-dir",default=str(ROOT/"outputs/video_latency"))
    parser.add_argument("--note",default="",help="Optional monitor refresh rate, camera mode, receiver channel or setup notes")
    parser.add_argument("--auto-start",action="store_true",help="Explicitly begin after the receiver supplies a frame")
    parser.add_argument("--self-test",action="store_true",help="Run synthetic encode/decode checks without camera or windows")
    args=parser.parse_args(argv)
    if not 1<=args.pattern_hz<=30 or not 1<=args.seconds<=600 or not 0<=args.warmup<=30:
        parser.error("Use pattern-hz 1..30, seconds 1..600 and warmup 0..30")
    if args.width<=0 or args.height<=0 or not 192<=args.qr_size<=1200 or args.min_samples<1 or args.max_latency_ms<=0:
        parser.error("Invalid image size, sample count, or latency limit")
    if args.fourcc and (len(args.fourcc)!=4 or not args.fourcc.isascii()):
        parser.error("--fourcc must contain exactly four ASCII characters")
    return args


def self_test() -> None:
    session="012345abcdef";created=time.perf_counter_ns();payload=make_payload(session,1,created)
    panel=make_source_panel(payload,1,created)
    decoded,_,_=cv2.QRCodeDetector().detectAndDecode(panel)
    if decoded!=payload:raise RuntimeError("Synthetic QR decode failed")
    ledger=Ledger(session,5000);ledger.start_ns=created;ledger.end_ns=created+10**9
    ledger.issued[1]=SourceEvent(1,created,created+1_000_000,payload)
    status,row=ledger.observe(decoded,1,created+61_000_000,created+62_000_000,created+70_000_000)
    assert row is not None and abs(row['submit_to_capture_ms']-60)<1e-9
    assert row['qr_decode_ms']==8
    assert ledger.observe(decoded,2,created+81_000_000,created+82_000_000,created+90_000_000)[1] is None
    assert len(ledger.samples)==1
    print("PASS: optical QR roundtrip, timestamps, 60 ms injected delay, decoder separation and duplicate rejection.")


def main(argv=None) -> int:
    args=parse_args(argv)
    if not hasattr(cv2,"QRCodeEncoder_create"):
        raise RuntimeError("This OpenCV build has no QR encoder. Use the project's existing Python environment.")
    if args.self_test:
        self_test();return 0
    session=secrets.token_hex(6)
    folder=Path(args.output_dir)/f"{datetime.now():%Y%m%d_%H%M%S}_{session}"
    folder.mkdir(parents=True,exist_ok=False)
    ledger=Ledger(session,args.max_latency_ms);capture=CaptureWorker(args)
    events=[];evidence=[];outcome="preview_cancelled";measurement_started=False
    sequence=0;last_frame_index=0;processed=0;next_emit=0;status="Point the camera at SOURCE, then press SPACE"
    started_clock=time.perf_counter_ns();last_frame_clock=started_clock;last_console=0
    print("This test never opens a serial port or sends motor commands.")
    print(BOUNDARY)
    print("Close YOLO/OBS/other camera apps. Keep the RECEIVED window outside the camera's view.")
    print("Aim at the large SOURCE code; press SPACE to record. Q/ESC saves and exits.")
    print(f"Results: {folder}",flush=True)
    try:
        capture.start()
        print("Capture:",json.dumps(capture.properties),flush=True)
        cv2.namedWindow(SOURCE_WINDOW,cv2.WINDOW_NORMAL);cv2.resizeWindow(SOURCE_WINDOW,args.qr_size+128,args.qr_size+208)
        cv2.namedWindow(PREVIEW_WINDOW,cv2.WINDOW_NORMAL);cv2.resizeWindow(PREVIEW_WINDOW,640,480)
        cv2.moveWindow(SOURCE_WINDOW,10,10);cv2.moveWindow(PREVIEW_WINDOW,args.qr_size+155,30)
        detector=cv2.QRCodeDetector()
        while True:
            now=time.perf_counter_ns()
            first_key=255
            if now>=next_emit:
                sequence+=1;created=time.perf_counter_ns();payload=make_payload(session,sequence,created)
                panel=make_source_panel(payload,sequence,created,args.qr_size)
                event=SourceEvent(sequence,created,time.perf_counter_ns(),payload)
                ledger.issued[sequence]=event
                cv2.imshow(SOURCE_WINDOW,panel)
                event.imshow_returned_ns=time.perf_counter_ns()
                # Pump source presentation before potentially expensive decoding.
                # This remains a software timestamp, NOT a measured photon onset.
                first_key=cv2.waitKey(1)&255
                event.gui_pump_returned_ns=time.perf_counter_ns()
                next_emit=event.submitted_ns+int(1e9/args.pattern_hz)
            snap=capture.snapshot()
            if snap is not None and snap[0]!=last_frame_index:
                frame_index,read_ns,frame=snap;last_frame_index=frame_index;last_frame_clock=now;processed+=1
                if args.auto_start and not measurement_started:
                    measurement_started=True;ledger.start_ns=now+int(args.warmup*1e9)
                    ledger.end_ns=ledger.start_ns+int(args.seconds*1e9)
                    events.append(dict(event="record_requested",host_ns=now,automatic=True))
                decode_start=time.perf_counter_ns()
                payload,points,decode_method=decode_frame(frame,detector)
                decode_end=time.perf_counter_ns()
                status,row=ledger.observe(payload,frame_index,read_ns,decode_start,decode_end)
                if row:
                    row['decode_method']=decode_method
                    events.append(dict(event="sample",**row))
                    if len(evidence)<3:evidence.append((row,frame.copy()))
                view=frame.copy()
                if points is not None:
                    cv2.polylines(view,[np.rint(points).astype(np.int32).reshape(-1,2)],True,(0,200,0),2)
                cv2.rectangle(view,(0,0),(view.shape[1],83),(15,15,15),-1)
                phase="PREVIEW"
                if measurement_started:phase="WARMUP" if now<(ledger.start_ns or 0) else "RECORDING"
                cv2.putText(view,phase+f" samples={len(ledger.samples)}",(8,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
                cv2.putText(view,status[:85],(8,49),cv2.FONT_HERSHEY_SIMPLEX,.43,(100,255,255),1)
                stats=statistics(ledger.samples)
                if stats:
                    cv2.putText(view,f"Median {stats['median_ms']:.1f} ms  P95 {stats['p95_ms']:.1f} ms",(8,73),cv2.FONT_HERSHEY_SIMPLEX,.49,(255,255,255),1)
                cv2.imshow(PREVIEW_WINDOW,view)
            if capture.error:raise RuntimeError(capture.error)
            if now-last_frame_clock>10_000_000_000:
                raise RuntimeError("No new capture frames for 10 seconds")
            if measurement_started and now>=(ledger.end_ns or now+1):
                outcome="duration_complete";break
            if now-last_console>2_000_000_000:
                print(status,flush=True);last_console=now
            keys={first_key,cv2.waitKey(1)&255}
            if keys.intersection((27,ord('q'),ord('Q'))):
                outcome="manual_stop" if measurement_started else "preview_cancelled";break
            if 32 in keys and not measurement_started and snap is not None:
                measurement_started=True;pressed=time.perf_counter_ns()
                ledger.start_ns=pressed+int(args.warmup*1e9);ledger.end_ns=ledger.start_ns+int(args.seconds*1e9)
                events.append(dict(event="record_requested",host_ns=pressed,automatic=False))
                print(f"Starting after {args.warmup:g}s warmup, recording {args.seconds:g}s.",flush=True)
            if cv2.getWindowProperty(SOURCE_WINDOW,cv2.WND_PROP_VISIBLE)<1 or cv2.getWindowProperty(PREVIEW_WINDOW,cv2.WND_PROP_VISIBLE)<1:
                outcome="window_closed";break
    except KeyboardInterrupt:
        outcome="keyboard_interrupt"
    except Exception as exc:
        outcome="error";events.append(dict(event="error",message=str(exc),host_ns=time.perf_counter_ns()))
        print("ERROR:",exc,flush=True)
    finally:
        closed=capture.close() if capture.thread.ident is not None else True
        if not closed:events.append(dict(event="capture_shutdown_timeout"))
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        ledger.counts["processed_capture_frames"]=processed
        result=save_results(folder,ledger,args,capture,outcome,events,evidence)
        print("Saved:",folder,flush=True)
        print("Statistics:",json.dumps(result['statistics'],ensure_ascii=False),flush=True)
        for flag in result['quality_flags']:print("NOTE:",flag,flush=True)
    return 0 if outcome=="duration_complete" and result['sufficient_sample_count'] else 2


if __name__=="__main__":
    raise SystemExit(main())
