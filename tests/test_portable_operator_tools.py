"""Host-side operator tools: no board access, native SDK or display required."""
import argparse
import json
import shlex

import pytest

from tools import portable_live_view as live
from tools.analysis.portable_labelled_eval import LABELS, box_iou, load_labels, match, metrics, source_predictions


def det(cls=0, box=(0, 0, 10, 10), score=.9):
    return {'class_id': cls, 'xywh': list(box), 'score': score}


@pytest.mark.parametrize('seconds', [0, -1, float('nan'), float('inf')])
def test_invalid_duration(seconds):
    with pytest.raises(ValueError):
        live.remote_command('/release', '/output', seconds)


@pytest.mark.parametrize('release,output', [('relative', '/output'), ('/release', '/release'), ('/release', '/release/output'), ('/r/../release', '/output')])
def test_unsafe_paths(release, output):
    with pytest.raises(ValueError):
        live.remote_command(release, output, 10)


def test_shell_paths_quoted_and_pinned():
    command = live.remote_command('/release', '/media/a b; echo bad', 10)
    tokens = shlex.split(command.split('exec ', 1)[1])
    assert tokens == ['python3', '-u', '/release/tools/launch.py', 'run', '/release', '--output', '/media/a b; echo bad', '--seconds', '10']
    assert '--diagnostic' not in tokens


def test_receiver_sdp_and_command():
    assert 'm=video 5010 RTP/AVP 96' in live.sdp('172.21.30.13', 5010)
    command = live.receiver_command('ffmpeg', 'input.sdp', True, 300, 'progress.txt')
    assert command[command.index('-frames:v')+1] == '300'
    assert command[command.index('-progress')+1] == 'progress.txt'
    assert '-frames:v' not in live.receiver_command('ffplay', 'input.sdp', False, 0, 'progress.txt')
    with pytest.raises(ValueError):
        live.receiver_command('ffmpeg', 'input.sdp', True, 0, 'progress.txt')
    with pytest.raises(ValueError):
        live.sdp('172.21.30.13', 65535)


def test_stop_checks_exact_process_identity(monkeypatch):
    captured = []
    monkeypatch.setattr(live, 'remote_python', lambda client, code: captured.append(code))
    live.stop_launcher(None, 12345, '/release', '/output')
    assert "'/release/tools/launch.py' not in a" in captured[0]
    assert "'/output' not in a" in captured[0]
    assert 'signal.SIGTERM' in captured[0]
    assert 'pkill' not in captured[0]
    compile(captured[0], '<stop>', 'exec')


@pytest.mark.parametrize('occupied', [True, False])
def test_preflight_never_starts_receiver_on_busy_board_or_existing_output(tmp_path, monkeypatch, occupied):
    import paramiko
    class FakeSFTP:
        def stat(self, _path):
            return object()  # an existing board output
    class FakeClient:
        def load_system_host_keys(self): pass
        def set_missing_host_key_policy(self, _policy): pass
        def connect(self, *_args, **_kwargs): pass
        def open_sftp(self): return FakeSFTP()
        def close(self): pass
    monkeypatch.setattr(paramiko, 'SSHClient', FakeClient)
    monkeypatch.setattr(live.shutil, 'which', lambda _name: 'ffplay')
    monkeypatch.setattr(live, 'busy_processes', lambda _client: [{'pid': 123}] if occupied else [])
    def forbidden(*_args, **_kwargs):
        pytest.fail('receiver must not start during failed preflight')
    monkeypatch.setattr(live.subprocess, 'Popen', forbidden)
    credentials = tmp_path / 'credentials.json'
    credentials.write_text(json.dumps({'board': {'host': 'example', 'user': 'test', 'password': 'never-log-this'}}))
    args = argparse.Namespace(output=tmp_path / 'output', release='/release', seconds=10., headless=False,
                              frames=300, credentials=credentials)
    assert live.run(args) == 1
    text = (args.output / 'result.json').read_text()
    assert ('Board occupied' if occupied else 'Board output already exists') in text
    assert 'never-log-this' not in text


def test_iou_and_duplicate_matching():
    gt = [det()]
    pred = [det(score=.8), det(score=.9)]
    pairs, fp, fn = match(gt, pred)
    assert pairs == [(0, 1, 1.)]
    assert fp == [0] and fn == []
    assert box_iou([0,0,0,0], [0,0,0,0]) == 0
    assert box_iou([0,0,10,10], [20,20,10,10]) == 0


def test_wrong_class_is_both_false_positive_and_false_negative():
    rows = [{'ground_truth': [det(0)], 'compiled': [det(1)]}]
    report = metrics(rows, 'compiled')
    assert report['overall']['tp'] == 0
    assert report['overall']['fp'] == report['overall']['fn'] == 1
    assert report['class_agnostic_confusion']['gt_rows_prediction_columns'][0][1] == 1
    assert report['classes'][LABELS[0]]['recall'] == 0


def test_empty_and_missing_predictions():
    report = metrics([{'ground_truth': [det(2)], 'compiled': []}], 'compiled')
    assert report['overall']['precision'] is None
    assert report['overall']['recall'] == 0
    assert report['class_agnostic_confusion']['gt_rows_prediction_columns'][2][4] == 1


def test_yolo_labels_and_invalid_class(tmp_path):
    labels = tmp_path / 'label.txt'
    labels.write_text('3 0.5 0.5 0.2 0.4\n')
    assert load_labels(labels, 100, 200) == [{'class_id': 3, 'xywh': [40., 60., 20., 80.]}]
    labels.write_text('4 0.5 0.5 0.2 0.4\n')
    with pytest.raises(ValueError):
        load_labels(labels, 100, 200)


def test_model_to_source_coordinates_clips_padding():
    result = source_predictions([{'class_id': 2, 'score': .5, 'model_xywh': [10, 270, 20, 40]}],
                                {'scale': 1, 'padding_left_top': [0, 280], 'original_size': [1280,720]})
    assert result[0]['xywh'] == [10., 0., 20., 30.]
