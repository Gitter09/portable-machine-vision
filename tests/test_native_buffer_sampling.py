"""Periodic native capture selection is deterministic and prediction-independent."""
import pytest
from tools.analysis.inspect_native_buffers import STAGES, missing_frames, selected_frames


def test_legacy_capture_defaults():
    assert selected_frames(20,1,2)==(20,21)


def test_periodic_rotation_samples():
    frames=selected_frames(40,29,40)
    assert len(frames)==40 and frames[0]==40 and frames[-1]==1171
    assert len(set(frames))==40


@pytest.mark.parametrize('values',[(-1,1,2),(0,0,2),(0,1,0),(0,1,65),(True,1,2),(0,True,2),(0.,1,2),(0,1,2.)])
def test_invalid_capture_bounds(values):
    with pytest.raises(ValueError):selected_frames(*values)


def test_missing_stages_checked_by_exact_frame_ids():
    rows=[{'label':name,'frame_id':f} for name in STAGES for f in (20,21)]
    assert not any(missing_frames(rows,(20,21)).values())
    rows=[r for r in rows if not (r['label']=='decoder' and r['frame_id']==21)]
    rows.append({'label':'decoder','frame_id':20})
    assert missing_frames(rows,(20,21))['decoder']==[21]
