"""Architecture guards: independent video, bounded inference, explicit stale UI."""
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime/launcher'))
from latest_frame_graph import graph


def test_video_passthrough_separate_from_decoder():
    g=graph('172.21.30.230',5010)
    video,ai=g.split('compressed. ! ')[1:]
    assert 'rtph264pay' in video and 'udpsink' in video
    assert 'decoder' not in video and 'encoder' not in video
    assert 'simaaidecoder' in ai
    assert g.count('simaaidecoder')==1


def test_latest_input_has_no_inference_backlog():
    assert 'appsink name=input sync=false max-buffers=1 drop=true' in graph('172.21.30.230',5010)


def test_no_stale_live_overlay():
    g=graph('172.21.30.230',5010)
    assert 'overlay' not in g and 'appsrc' not in g


def test_unapproved_target_refused():
    with pytest.raises(ValueError):graph('8.8.8.8',5010)


def test_result_panel_is_explicitly_historical_and_stales():
    text=(ROOT/'tools/analysis/latest_frame_review.html').read_text(encoding='utf-8')
    assert 'STALE' in text and 'DISCONNECTED' in text
    assert 'not a live overlay' in text
    assert 'textContent' in text and 'innerHTML' not in text
