"""Offline geometry helpers; these are not native/accuracy qualification."""
import importlib.util
from pathlib import Path
import cv2
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('site_experiment',Path(__file__).parents[1]/'tools/analysis/portable_site_experiment.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def synthetic():
    im=np.full((400,600,3),200,np.uint8)
    # Saturated blue substrate, neutral dark header at its left end.
    cv2.rectangle(im,(170,155),(410,245),(120,40,20),-1)
    cv2.rectangle(im,(155,165),(185,235),(30,30,30),-1)
    return im


def test_blank_is_unknown():
    assert m.locate(np.full((400,600,3),200,np.uint8))['status']=='unknown'


def test_two_boards_are_ambiguous():
    im=np.full((720,1280,3),200,np.uint8)
    im[0:400,0:600]=synthetic();im[300:700,650:1250]=synthetic()
    assert m.locate(im)['status']=='unknown'


def test_substrate_without_header_is_unknown():
    im=synthetic();cv2.rectangle(im,(150,155),(420,245),(120,40,20),-1)
    assert m.locate(im)['status']=='unknown'


@pytest.mark.parametrize('rotation',[0,1,2,3])
def test_header_orientation_under_rotation(rotation):
    im=np.ascontiguousarray(np.rot90(synthetic(),rotation))
    pose=m.locate(im);assert pose['status']=='located'
    canonical=cv2.warpPerspective(im,np.float64(pose['source_to_canonical']),m.SIZE)
    assert canonical[:,0:40].mean()<canonical[:,-40:].mean()


def test_site_roundtrip():
    pose=m.locate(synthetic());matrix=np.float64(pose['source_to_canonical'])
    for name,bounds in m.SITES.items():
        out=cv2.perspectiveTransform(np.float32([pose['sites'][name]]),matrix)[0]
        np.testing.assert_allclose(out,m.site_quad(bounds),atol=1e-3)


def test_gate_is_class_site_specific_and_does_not_create_verdicts():
    pose={'status':'located','sites':{'IC':[[0,0],[10,0],[10,10],[0,10]],'terminal':[[20,0],[30,0],[30,10],[20,10]]}}
    ds=[{'class_id':2,'xywh':[2,2,2,2],'score':.1},{'class_id':2,'xywh':[22,2,2,2],'score':.9},{'class_id':3,'xywh':[22,2,2,2],'score':.2}]
    kept,rejected=m.gate(ds,pose)
    assert kept==[ds[0],ds[2]] and rejected==[ds[1]]
    assert all('verdict' not in d for d in kept)


def test_unknown_keeps_nothing_but_preserves_evidence():
    ds=[{'class_id':0,'xywh':[2,2,2,2],'score':.8}]
    kept,rejected=m.gate(ds,{'status':'unknown'})
    assert kept==[] and rejected==ds


def test_reference_does_not_accept_blank():
    with pytest.raises(ValueError,match='reference pose unknown'):
        m.ReferenceLocator(np.full((400,600,3),200,np.uint8))


def test_reference_features_do_not_include_distant_background():
    im=synthetic()
    rng=np.random.default_rng(42);im[0:90]=rng.integers(0,255,(90,600,3),dtype=np.uint8)
    reference=m.ReferenceLocator(im)
    for k in reference.kp:
        x,y=np.array(k.pt)/2
        assert 120<x<470 and 100<y<300
    assert reference(np.full_like(im,200))['status']=='unknown'
