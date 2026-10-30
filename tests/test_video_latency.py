from __future__ import annotations

import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import cv2
import numpy as np
from tools.measure_video_latency import (Ledger, SourceEvent, make_payload,
    parse_payload, make_source_panel, statistics, parse_args, decode_frame, save_results)


class LatencyTests(unittest.TestCase):
    def setUp(self):
        self.session="012345abcdef"
        self.ledger=Ledger(self.session,5000)
        self.ledger.start_ns=1_000_000_000
        self.ledger.end_ns=20_000_000_000
        self.payload=make_payload(self.session,1,1_010_000_000)
        self.ledger.issued[1]=SourceEvent(1,1_010_000_000,1_020_000_000,self.payload,1_021_000_000)

    def observe(self,payload=None,read=1_080_000_000):
        return self.ledger.observe(payload or self.payload,1,read,read+2_000_000,read+12_000_000)

    def test_exact_injected_latency_excludes_decoder(self):
        _,row=self.observe()
        self.assertEqual(row['submit_to_capture_ms'],60)
        self.assertEqual(row['source_preparation_ms'],10)
        self.assertEqual(row['creation_to_capture_ms'],70)
        self.assertEqual(row['qr_decode_ms'],10)
        self.assertEqual(row['submit_to_decoder_finished_ms'],72)

    def test_duplicate_does_not_inflate_delay_or_sample_count(self):
        self.observe();self.assertIsNone(self.observe(read=1_500_000_000)[1])
        self.assertEqual(len(self.ledger.samples),1)
        self.assertEqual(statistics(self.ledger.samples)['median_ms'],60)

    def test_crc_session_and_unknown_sequence_rejected(self):
        self.assertIsNone(self.observe(self.payload[:-1]+'x')[1])
        self.assertIsNone(self.observe(make_payload('fedcba987654',1,1_010_000_000))[1])
        self.assertIsNone(self.observe(make_payload(self.session,2,1_010_000_000))[1])
        self.assertIsNone(self.observe(make_payload(self.session,1,1_011_000_000))[1])

    def test_old_capture_and_too_late_capture_rejected(self):
        self.assertIsNone(self.observe(read=1_019_000_000)[1])
        self.assertIsNone(self.observe(read=7_000_000_000)[1])
        self.assertEqual(self.ledger.counts['above_max_latency'],1)

    def test_before_start_after_end_and_preview_rejected(self):
        self.ledger.start_ns=1_100_000_000
        self.assertIsNone(self.observe()[1])
        self.ledger.start_ns=None
        self.assertTrue(self.observe()[0].startswith('PREVIEW'))
        self.ledger.start_ns=1_000_000_000;self.ledger.end_ns=1_050_000_000
        self.assertIsNone(self.observe()[1])

    def test_empty_results_have_no_fake_zero_latency(self):
        self.assertIsNone(statistics([]))
        for invalid in ['', 'VT1:x:1:2:00000000','VT1:012345abcdef:-1:2:00000000','anything']:
            self.assertIsNone(parse_payload(invalid))

    def test_qr_survives_realistic_resize_jpeg_and_perspective(self):
        panel=make_source_panel(self.payload,1,1_010_000_000)
        h,w=panel.shape[:2]
        src=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
        dst=np.float32([[100,50],[530,65],[510,447],[83,432]])
        warped=cv2.warpPerspective(panel,cv2.getPerspectiveTransform(src,dst),(640,480),borderValue=(180,180,180))
        ok,data=cv2.imencode('.jpg',warped,[cv2.IMWRITE_JPEG_QUALITY,82]);self.assertTrue(ok)
        frame=cv2.imdecode(data,cv2.IMREAD_COLOR)
        decoded,_,_=decode_frame(frame,cv2.QRCodeDetector())
        self.assertEqual(decoded,self.payload)

    def test_no_actuator_options_and_default_camera_parameters(self):
        args=parse_args([])
        self.assertEqual(args.source,0)
        self.assertFalse(args.auto_start)
        self.assertEqual(args.fourcc,'')
        self.assertEqual(args.fps,0)

    def test_manual_stop_preserves_samples_and_marks_incomplete(self):
        self.observe()
        capture=SimpleNamespace(frame_count=10,read_failures=0,error='',properties={})
        with tempfile.TemporaryDirectory() as temp:
            report=save_results(Path(temp),self.ledger,parse_args([]),capture,'manual_stop',[],[])
            saved=json.loads((Path(temp)/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['statistics']['median_ms'],60)
            self.assertFalse(saved['run_completed'])
            self.assertFalse(saved['sufficient_sample_count'])
            self.assertIn('60.0',(Path(temp)/'samples.csv').read_text(encoding='utf-8-sig'))

    def test_no_decode_never_produces_a_successful_zero_ms_result(self):
        capture=SimpleNamespace(frame_count=10,read_failures=0,error='',properties={})
        with tempfile.TemporaryDirectory() as temp:
            report=save_results(Path(temp),self.ledger,parse_args([]),capture,'duration_complete',[],[])
            self.assertIsNone(report['statistics'])
            self.assertFalse(report['sufficient_sample_count'])
            self.assertEqual(report['matched_unique_codes'],0)


if __name__=='__main__':
    unittest.main()
