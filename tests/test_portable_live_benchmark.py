"""Read-only live benchmark math and exact stream/frame cohort checks."""
import json
import struct

import pytest

from tools.analysis.benchmark_live_session import (clock_ticks, cohort, complete_rows,
    interval_stats, prediction_summary, process_stat, summarize, total_cpu, viewer_progress)


def test_linux_process_stat_with_space_in_comm():
    fields = ['S'] + ['0']*20
    fields[11], fields[12], fields[19] = '120', '30', '999'
    assert process_stat('42 (python worker) ' + ' '.join(fields)) == {'cpu_ticks':150,'start_ticks':999}


def test_auxv_reads_actual_clock_rate():
    assert clock_ticks(struct.pack('<QQQQQQ',6,4096,17,250,0,0)) == 250


@pytest.mark.parametrize('data',[b'bad',struct.pack('<QQ',0,0)])
def test_auxv_never_guesses_missing_clock(data):
    with pytest.raises(ValueError): clock_ticks(data)


def test_aggregate_cpu_excludes_guest_double_count():
    assert total_cpu('cpu 100 20 30 400 10 5 5 2 30 10\ncpu0 0\ncpu1 0\n') == {
        'total':572,'idle':410,'logical_cpus':2}


def test_complete_rows_drops_only_partial_edges():
    data=b'partial}\n'+json.dumps({'id':1}).encode()+b'\n{"id":2'
    assert complete_rows(data,True)==[{'id':1}]
    with pytest.raises(json.JSONDecodeError): complete_rows(b'{bad}\n')


def record(i, stream='decoder', session='session'):
    return {'frame_id':i,'stream_id':stream,'session':session}


def test_encoder_cohort_excludes_detector_tail_and_other_streams():
    first={'session':'session','stream_id':'decoder','last_frame_id':10,'frames':11}
    last={'last_frame_id':12,'frames':13}
    fs,ds=cohort([record(i) for i in range(9,14)],
                 [record(i) for i in range(9,16)]+[record(11,'other')],first,last)
    assert [r['frame_id'] for r in fs]==[11,12]
    assert [r['frame_id'] for r in ds]==[11,12]


@pytest.mark.parametrize('mode',['missing_frame','missing_detection','duplicate_detection','duplicate_frame'])
def test_incomplete_or_duplicate_cohort_refused(mode):
    first={'session':'session','stream_id':'decoder','last_frame_id':10,'frames':11}
    last={'last_frame_id':12,'frames':13}
    fs,ds=[record(11),record(12)],[record(11),record(12)]
    if mode=='missing_frame':fs.pop()
    elif mode=='missing_detection':ds.pop()
    elif mode=='duplicate_detection':ds.append(record(12))
    else:fs.append(record(12))
    with pytest.raises(ValueError):cohort(fs,ds,first,last)


def test_raw_class_presence_keeps_weak_fail_and_duplicates():
    row={'detections':[{'label':'IC_Pass','score':.8},{'label':'Terminal_pass','score':.6},
                       {'label':'Terminal_pass','score':.4},{'label':'IC_Fail','score':.0487}]}
    r=prediction_summary([row,{'detections':[]}],['IC_Pass','Terminal_pass'])
    assert r['both_or_all_expected']==1 and r['any_unexpected']==1 and r['empty']==1
    assert r['classes']['Terminal_pass']['frames_with_multiple_candidates']==1
    assert r['classes']['IC_Fail']['score_max']==.0487


def test_percentile_gap_units():
    r=interval_stats([10.,20.,30.,40.])
    assert r['p50_ms']==25 and r['p95_ms']==pytest.approx(38.5) and r['max_ms']==40
    assert interval_stats([])['p99_ms'] is None


@pytest.mark.parametrize('leading_gap',[0,2])
def test_summary_units_cpu_memory_and_boundary_frame_gap(leading_gap):
    first={'board_uptime':100.,'health':{'frames':0,'last_frame_id':-1,'valid_timed_frames':0,
           'rtp_frame_markers':0,'errors':[],'media_ready':True,
           'timestamp_cache':{'pending':10,'misses':0,'evicted':0}},
           'process':{'cpu_ticks':100,'start_ticks':999},'cpu':{'total':1000,'idle':900,'logical_cpus':16},
           'rss_kib':1000,'mem_available_kib':10000,
           'viewer':{'reported_drops':2,'clock_seconds':20.}}
    last={'board_uptime':110.,'health':dict(first['health'],frames=5,last_frame_id=4+leading_gap,
           valid_timed_frames=5,rtp_frame_markers=5),
           'process':{'cpu_ticks':150,'start_ticks':999},'cpu':{'total':2000,'idle':1650,'logical_cpus':16},
           'rss_kib':1256,'mem_available_kib':9900,
           'viewer':{'reported_drops':6,'clock_seconds':30.}}
    frames=[{'frame_id':i+leading_gap,'monotonic_seconds':101.+i*2,'pts':int((1+i*2)*1e9)} for i in range(5)]
    result=summarize([first,last],frames,[{'detections':[]} for _ in frames],['IC_Pass'],100)
    assert result['encoder_fps_from_health_window']==.5
    assert result['encoder_fps_from_delivery_span']==.5
    assert result['encoder_delivery_intervals']['p50_ms']==2000
    assert result['worker_cpu_percent_one_core']['mean']==5
    assert result['board_cpu_percent_all_cores']['mean']==25
    assert result['worker_rss_kib']['end_minus_start']==256
    assert result['encoder_frame_id_gaps']==leading_gap
    assert result['viewer']['reported_drop_delta']==4


def test_viewer_tail_is_not_reported_as_decoded_fps(tmp_path):
    p=tmp_path/'viewer.log';p.write_text(' 1.25 M-V: -0.015 fd= 3 aq=0\r 2.50 M-V: 0.010 fd= 9 aq=0\r')
    assert viewer_progress(p)=={'clock_seconds':2.5,'master_minus_video_seconds':.01,'reported_drops':9}
    assert viewer_progress(tmp_path/'missing') is None
