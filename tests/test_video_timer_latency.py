import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from measure_video_timer_latency import make_panel,decode_panel,decode_aligned_panel,candidate_readings,analyze,write_csv


class TimerTests(unittest.TestCase):
    def test_perspective_camera_view_preserves_readable_time(self):
        panel=make_panel(123456)
        src=np.float32([[0,0],[919,0],[919,339],[0,339]])
        dst=np.float32([[35,130],[606,102],[582,352],[51,370]])
        frame=cv2.warpPerspective(panel,cv2.getPerspectiveTransform(src,dst),(640,480),borderValue=(150,150,150))
        ok,data=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,80]);self.assertTrue(ok)
        frame=cv2.imdecode(data,cv2.IMREAD_COLOR)
        self.assertTrue(any(x[0]['displayed_ms']==123456 for x in candidate_readings(frame)))

    def test_blank_and_uncertain_digits_do_not_become_zero(self):
        self.assertIsNone(decode_panel(np.full((340,920,3),255,np.uint8)))
        self.assertIsNone(decode_aligned_panel(np.full((340,920,3),255,np.uint8)))
        panel=make_panel(123456)
        # Mix an on-segment with white; ambiguous pixels must be rejected.
        panel[100:137,182:200]=255
        self.assertIsNone(decode_panel(panel))

    def test_offline_reader_tolerates_character_shift_without_timestamps(self):
        original=make_panel(123456);shifted=original.copy()
        shifted[70:290,80:850]=255
        for index,dx in enumerate([2,7,12,16,18,17]):
            x=100+index*122
            shifted[77:277,x+dx:x+dx+100]=original[82:282,x:x+100]
        decoded=decode_aligned_panel(shifted)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded[0],123456)

    def test_offline_known_latency_and_empty_run(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);(folder/'frames').mkdir()
            summary=dict(arguments=dict(max_latency_ms=5000),run_completed=True)
            (folder/'summary.json').write_text(json.dumps(summary))
            write_csv(folder/'source_events.csv',[dict(displayed_ms=123456,submitted_ns=1_000_000_000)],['displayed_ms','submitted_ns'])
            cv2.imwrite(str(folder/'frames/f.png'),make_panel(123456))
            write_csv(folder/'frames.csv',[dict(file='frames/f.png',frame_index=5,capture_read_returned_ns=1_160_000_000)],['file','frame_index','capture_read_returned_ns'])
            result=analyze(folder)
            self.assertEqual(result['statistics']['median_ms'],160)
            self.assertEqual(result['visual_audit_status'],'pending')
            self.assertFalse(result['sufficient_sample_count'])
            write_csv(folder/'frames.csv',[],['file','frame_index','capture_read_returned_ns'])
            result=analyze(folder)
            self.assertIsNone(result['statistics'])


if __name__=='__main__':unittest.main()
