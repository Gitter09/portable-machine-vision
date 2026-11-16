"""Holdout membership and deterministic validation preprocessing guards."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/analysis'))
import prepare_cropped_validation_review as review


def record(split,source,model):return {'split':split,'source_sha256':source,'model_sha256':model}


def test_only_reserved_validation_selected():
    rows=[record('train','a','b'),record('val','c','d'),record('test','e','f')]
    assert review.reserved(rows)==[rows[1]]


@pytest.mark.parametrize('field',['source_sha256','model_sha256'])
def test_duplicate_holdout_refused(field):
    a=record('train','a','b');b=record('val','c','d');b[field]=a[field]
    with pytest.raises(ValueError,match='overlaps'):review.reserved([a,b])


def test_missing_validation_refused():
    with pytest.raises(ValueError,match='no reserved'):review.reserved([record('test','a','b')])


def test_validation_letterbox_is_deterministic_pad114():
    image=np.full((100,200,3),80,np.uint8)
    a,m=review.augmentation.letterbox(image);b,n=review.augmentation.letterbox(image)
    assert np.array_equal(a,b) and np.array_equal(m,n)
    assert a.shape==(1280,1280,3)
    assert (a[:320]==114).all() and (a[320:960]==80).all()
