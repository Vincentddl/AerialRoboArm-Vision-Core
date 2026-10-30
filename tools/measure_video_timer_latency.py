"""Numeric optical timer: capture first, read digits only AFTER capture closes.

No model, OCR, serial port or actuator access during acquisition. Timestamp
boundary is imshow submission -> VideoCapture.read return, not photon onset.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets
import time

import cv2
import numpy as np

from measure_video_latency import CaptureWorker

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "TIMER SOURCE - camera sees ONLY the large numbers"
PREVIEW = "RECEIVED - SPACE record / Q save and exit"
PANEL_W, PANEL_H = 920, 340
BORDER = np.float32([[20,20],[900,20],[900,320],[20,320]])
# Seven segments a,b,c,d,e,f,g, in an 80 x 160 digit cell.
SEGMENTS = [(15,0,65,14),(66,15,80,73),(66,87,80,145),
            (15,146,65,160),(0,87,14,145),(0,15,14,73),(15,73,65,87)]
MASKS = {"0":"1111110","1":"0110000","2":"1101101","3":"1111001",
         "4":"0110011","5":"1011011","6":"1011111","7":"1110000",
         "8":"1111111","9":"1111011"}
REVERSE = {v:k for k,v in MASKS.items()}
BOUNDARY = ("Host numeric-source imshow submission to sampled capture read return. "
            "Includes source display, camera, analog transmitter/receiver, USB and "
            "capture buffering. Offline digit-reader runtime is excluded. "
            "Not isolated RF/receiver latency or measured photon-to-host latency.")


def digit_rects(index):
    ox,oy = 100+index*122,82
    return [(ox+round(x1*1.25),oy+round(y1*1.25),
             ox+round(x2*1.25),oy+round(y2*1.25)) for x1,y1,x2,y2 in SEGMENTS]


def make_panel(value):
    if not 0 <= value <= 999999:
        raise ValueError("Timer exceeds six digits; restart test")
    panel = np.full((PANEL_H,PANEL_W,3),255,np.uint8)
    cv2.rectangle(panel,(20,20),(900,320),(0,0,0),5)
    cv2.putText(panel,"SOURCE TIMER / milliseconds",(110,58),cv2.FONT_HERSHEY_SIMPLEX,.85,(0,0,0),2)
    for index,digit in enumerate(f"{value:06d}"):
        for bit,rect in zip(MASKS[digit],digit_rects(index)):
            if bit == '1':
                x1,y1,x2,y2=rect
                cv2.rectangle(panel,(x1,y1),(x2-1,y2-1),(0,0,0),-1)
    cv2.putText(panel,"SPACE: record 45 s    Q / ESC: save and exit",(110,308),cv2.FONT_HERSHEY_SIMPLEX,.58,(0,0,0),1)
    return panel


def order_quad(points):
    points=np.asarray(points,np.float32).reshape(4,2)
    sums=points.sum(axis=1);diff=points[:,1]-points[:,0]
    ordered=np.float32([points[np.argmin(sums)],points[np.argmin(diff)],
                       points[np.argmax(sums)],points[np.argmax(diff)]])
    return ordered if len(np.unique(ordered,axis=0))==4 else None


def decode_panel(warped):
    gray=cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY) if warped.ndim==3 else warped
    # Local contrast is normalized by Otsu; reject uncertain/mixed segments.
    _,ink=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    digits=[];confidence=1.0
    for index in range(6):
        bits=''
        for x1,y1,x2,y2 in digit_rects(index):
            patch=ink[y1+3:y2-3,x1+3:x2-3]
            score=float(np.mean(patch)/255)
            if score>=.70:bits+='1';confidence=min(confidence,score)
            elif score<=.25:bits+='0';confidence=min(confidence,1-score)
            else:return None
        if bits not in REVERSE:return None
        digits.append(REVERSE[bits])
    return int(''.join(digits)),confidence


_DIGIT_TEMPLATES = None


def decode_aligned_panel(warped):
    """Offline fallback: tolerate small character shifts after border warping.

    Decide digits from image similarity BEFORE consulting timestamps. Require a
    clear margin over the next digit; never choose a number to match a latency.
    """
    global _DIGIT_TEMPLATES
    if _DIGIT_TEMPLATES is None:
        _DIGIT_TEMPLATES=[]
        for digit in range(10):
            panel=make_panel(int(str(digit)*6))
            template=255-cv2.cvtColor(panel[82:282,100:200],cv2.COLOR_BGR2GRAY)
            _DIGIT_TEMPLATES.append(template)
    gray=cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
    _,ink=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    digits=[];confidence=1.;margin=1.
    for index in range(6):
        x=100+index*122
        region=ink[62:302,x-28:x+128]
        ranked=[]
        for digit,template in enumerate(_DIGIT_TEMPLATES):
            _,score,_,_=cv2.minMaxLoc(cv2.matchTemplate(region,template,cv2.TM_CCOEFF_NORMED))
            ranked.append((score,digit))
        ranked.sort(reverse=True)
        gap=ranked[0][0]-ranked[1][0]
        if ranked[0][0]<.72 or gap<.08:return None
        digits.append(str(ranked[0][1]))
        confidence=min(confidence,ranked[0][0]);margin=min(margin,gap)
    return int(''.join(digits)),confidence,margin


def candidate_readings(frame):
    """Offline only: locate the timer border, rectify, inspect digit segments."""
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    _,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours,_=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for contour in contours:
        area=abs(cv2.contourArea(contour))
        if not .02*gray.size <= area <= .97*gray.size:continue
        for epsilon in (.015,.025,.04):
            quad=cv2.approxPolyDP(contour,epsilon*cv2.arcLength(contour,True),True)
            if len(quad)!=4 or not cv2.isContourConvex(quad):continue
            ordered=order_quad(quad)
            if ordered is not None:candidates.append((area,ordered))
            break
    for area,quad in sorted(candidates,key=lambda x:-x[0]):
        warped=cv2.warpPerspective(frame,cv2.getPerspectiveTransform(quad,BORDER),(PANEL_W,PANEL_H),borderValue=(255,255,255))
        decoded=decode_panel(warped)
        if decoded is not None:
            yield dict(displayed_ms=decoded[0],segment_confidence=decoded[1],
                       reader_method='strict_segments',template_margin=None,
                       border_area_fraction=area/gray.size),warped
        else:
            aligned=decode_aligned_panel(warped)
            if aligned is not None:
                yield dict(displayed_ms=aligned[0],segment_confidence=aligned[1],
                           reader_method='aligned_digit_templates',template_margin=aligned[2],
                           border_area_fraction=area/gray.size),warped


def write_csv(path,rows,fields):
    with path.open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def analyze(folder):
    """No live devices opened. Invalid/ambiguous digits never become samples."""
    folder=Path(folder)
    metadata=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    with (folder/'source_events.csv').open(encoding='utf-8-sig') as handle:
        events={int(x['displayed_ms']):x for x in csv.DictReader(handle)}
    with (folder/'frames.csv').open(encoding='utf-8-sig') as handle:
        frames=list(csv.DictReader(handle))
    rows=[];rejected=[];evidence=0
    for row in frames:
        image=cv2.imread(str(folder/row['file']))
        if image is None:
            rejected.append(dict(file=row['file'],reason='image_missing'));continue
        accepted=None
        for reading,warped in candidate_readings(image):
            event=events.get(reading['displayed_ms'])
            if event is None:continue
            elapsed=(int(row['capture_read_returned_ns'])-int(event['submitted_ns']))/1e6
            if not 0<=elapsed<=metadata['arguments']['max_latency_ms']:continue
            accepted=dict(**row,**reading,source_submitted_ns=int(event['submitted_ns']),submit_to_capture_ms=elapsed)
            if evidence<6:
                cv2.imwrite(str(folder/f"audit_{evidence+1:02d}_rectified.png"),warped)
                evidence+=1
            break
        if accepted:rows.append(accepted)
        else:rejected.append(dict(file=row['file'],reason='no_unambiguous_issued_timer_within_latency_limit'))
    values=np.array([x['submit_to_capture_ms'] for x in rows],dtype=float)
    stats=(dict(count=len(rows),mean_ms=float(values.mean()),median_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values,95)),minimum_ms=float(values.min()),maximum_ms=float(values.max())) if len(values) else None)
    flags=["Source display latency is included and uncalibrated.",
           "Offline reader may reject blurred/mixed digits; accepted samples can be biased.",
           "Automatic readings require visual checks of raw and rectified images.",
           "Sampling estimates frame age, unlike the QR first-decodable-appearance sampling.",
           "Latency-limit filtering may censor results; rejected count is not RF packet loss."]
    if len(rows)<30:flags.append('Fewer than 30 valid readings; insufficient samples.')
    if not metadata['run_completed']:flags.append('Acquisition did not complete its planned duration.')
    result=dict(metric='software_numeric_display_submission_to_sampled_capture_return_ms',
                boundary=BOUNDARY,statistics=stats,recorded_frames=len(frames),accepted_frames=len(rows),
                rejected_frames=len(rejected),run_completed=metadata['run_completed'],
                sufficient_sample_count=len(rows)>=30,visual_audit_status='pending',quality_flags=flags,
                reader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    fields=list(rows[0]) if rows else ['file','submit_to_capture_ms']
    write_csv(folder/'samples.csv',rows,fields)
    write_csv(folder/'rejected.csv',rejected,['file','reason'])
    (folder/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    return result


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=int,default=0)
    parser.add_argument('--backend',choices=['dshow','msmf','any'],default='dshow')
    parser.add_argument('--width',type=int,default=640)
    parser.add_argument('--height',type=int,default=480)
    parser.add_argument('--fps',type=float,default=0)
    parser.add_argument('--fourcc',default='')
    parser.add_argument('--seconds',type=float,default=45)
    parser.add_argument('--warmup',type=float,default=2)
    parser.add_argument('--sample-hz',type=float,default=5)
    parser.add_argument('--display-hz',type=float,default=60)
    parser.add_argument('--max-latency-ms',type=float,default=5000)
    parser.add_argument('--output-dir',default=str(ROOT/'outputs/video_timer_latency'))
    parser.add_argument('--note',default='USB Video; verify receiver preview before pressing SPACE')
    parser.add_argument('--analyze',help='Analyze a saved run without opening camera/windows')
    args=parser.parse_args(argv)
    if not 1<=args.seconds<=90 or not 0<=args.warmup<=10 or not 1<=args.sample_hz<=10 or not 10<=args.display_hz<=120:
        parser.error('Use seconds 1..90, warmup 0..10, sample-hz 1..10, display-hz 10..120')
    if args.width<=0 or args.height<=0 or args.width*args.height>640*480 or args.seconds*args.sample_hz>450:
        parser.error('RAM capture limited to 640x480 and 450 frames')
    if args.max_latency_ms<=0 or args.fps<0 or (args.fourcc and len(args.fourcc)!=4):
        parser.error('Invalid latency/FPS/FourCC')
    return args


def main(argv=None):
    args=parse_args(argv)
    if args.analyze:
        analyze(args.analyze);return 0
    session=secrets.token_hex(6)
    folder=Path(args.output_dir)/f"{datetime.now():%Y%m%d_%H%M%S}_{session}"
    folder.mkdir(parents=True,exist_ok=False)
    capture=CaptureWorker(args);events=[];frames=[];errors=[]
    outcome='preview_cancelled';start=None;end=None;base=time.perf_counter_ns()
    next_display=0;last_frame=0;last_saved_ns=None;last_live=base
    print('No QR/OCR/model/serial during capture. Aim ONLY at the numeric SOURCE.',flush=True)
    print(f'Press SPACE to record. Output: {folder}',flush=True)
    try:
        capture.start()
        cv2.namedWindow(SOURCE,cv2.WINDOW_NORMAL);cv2.resizeWindow(SOURCE,PANEL_W,PANEL_H)
        cv2.moveWindow(SOURCE,10,30)
        cv2.namedWindow(PREVIEW,cv2.WINDOW_NORMAL);cv2.resizeWindow(PREVIEW,640,480)
        cv2.moveWindow(PREVIEW,940,30)
        while True:
            now=time.perf_counter_ns();keys=[]
            if now>=next_display:
                created=time.perf_counter_ns();value=(created-base)//1_000_000
                panel=make_panel(value)
                event=dict(displayed_ms=value,created_ns=created,submitted_ns=time.perf_counter_ns())
                cv2.imshow(SOURCE,panel)
                keys.append(cv2.waitKey(1)&255)
                event['gui_pump_returned_ns']=time.perf_counter_ns();events.append(event)
                next_display=event['submitted_ns']+int(1e9/args.display_hz)
            snap=capture.snapshot()
            if snap is not None and snap[0]!=last_frame:
                index,read_ns,image=snap;last_frame=index;last_live=time.perf_counter_ns()
                if start is not None and start<=read_ns<=end and (last_saved_ns is None or read_ns-last_saved_ns>=int(1e9/args.sample_hz)):
                    if len(frames)>=450:raise RuntimeError('RAM sample limit reached')
                    frames.append((dict(file=f'frames/frame_{len(frames)+1:04d}.png',frame_index=index,capture_read_returned_ns=read_ns),image.copy()))
                    last_saved_ns=read_ns
                view=image.copy()
                phase='PREVIEW - aim at the numbers; SPACE starts'
                if start is not None:
                    phase='WARMUP' if now<start else f'RECORDING {max(0,(end-now)/1e9):.1f}s left / saved {len(frames)}'
                cv2.rectangle(view,(0,0),(view.shape[1],55),(20,20,20),-1)
                cv2.putText(view,phase,(8,23),cv2.FONT_HERSHEY_SIMPLEX,.47,(255,255,255),1)
                cv2.putText(view,'No digit recognition during recording',(8,45),cv2.FONT_HERSHEY_SIMPLEX,.44,(100,255,255),1)
                cv2.imshow(PREVIEW,view)
            if capture.error:raise RuntimeError(capture.error)
            if now-last_live>10_000_000_000:raise RuntimeError('No capture frames for 10 seconds')
            if end is not None and time.perf_counter_ns()>=end:
                outcome='duration_complete';break
            keys.append(cv2.waitKey(1)&255)
            if any(x in (27,ord('q'),ord('Q')) for x in keys):
                outcome='manual_stop' if start is not None else 'preview_cancelled';break
            if 32 in keys and start is None and snap is not None:
                start=time.perf_counter_ns()+int(args.warmup*1e9);end=start+int(args.seconds*1e9)
            if cv2.getWindowProperty(SOURCE,cv2.WND_PROP_VISIBLE)<1 or cv2.getWindowProperty(PREVIEW,cv2.WND_PROP_VISIBLE)<1:
                outcome='window_closed';break
    except KeyboardInterrupt:
        outcome='keyboard_interrupt'
    except Exception as exc:
        outcome='error';errors.append(str(exc));print(f'ERROR: {exc}',flush=True)
    finally:
        if capture.thread.ident is not None and not capture.close():errors.append('capture_shutdown_timeout')
        try:cv2.destroyAllWindows()
        except cv2.error:pass
        # All compression and disk writes happen after camera acquisition stops.
        (folder/'frames').mkdir(exist_ok=True)
        frame_rows=[]
        for row,image in frames:
            if cv2.imwrite(str(folder/row['file']),image):frame_rows.append(row)
            else:errors.append(f"image_write_failed:{row['file']}")
        write_csv(folder/'frames.csv',frame_rows,['file','frame_index','capture_read_returned_ns'])
        write_csv(folder/'source_events.csv',events,['displayed_ms','created_ns','submitted_ns','gui_pump_returned_ns'])
        eligible=[x['submitted_ns'] for x in events if start is not None and start<=x['submitted_ns']<=(end or 0)]
        intervals=np.diff(eligible)/1e6
        metadata=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),session=session,boundary=BOUNDARY,
                      source_display_latency_included=True,photon_timestamp_measured=False,
                      recognition_during_capture=False,actuator_commands_sent=False,
                      outcome=outcome,run_completed=outcome=='duration_complete' and not errors,
                      recorded_frames=len(frame_rows),base_ns=base,start_ns=start,end_ns=end,
                      source_interval_ms=(dict(median=float(np.median(intervals)),p95=float(np.percentile(intervals,95))) if len(intervals) else None),
                      capture_properties=capture.properties,capture_read_count=capture.frame_count,
                      capture_read_failures=capture.read_failures,capture_error=capture.error,errors=errors,
                      arguments=vars(args),opencv=cv2.__version__,
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      capture_helper_sha256=hashlib.sha256((Path(__file__).parent/'measure_video_latency.py').read_bytes()).hexdigest())
        (folder/'summary.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        print(f'Saved: {folder}',flush=True)
    if frames:
        print('Camera closed. Starting OFFLINE digit reading.',flush=True)
        analyze(folder)
    return 0 if outcome=='duration_complete' and not errors else 2


if __name__=='__main__':
    raise SystemExit(main())
