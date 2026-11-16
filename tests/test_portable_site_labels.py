"""Draft annotation evidence and safety, not model accuracy."""
import importlib.util
import json
from argparse import Namespace
from pathlib import Path
import cv2
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('site_labels',Path(__file__).parents[1]/'tools/analysis/prepare_portable_site_labels.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
QUAD=[[100,100],[300,100],[300,200],[100,200]]


def item(state='both_missing'):
    return {'id':'example','state':state,'path':'source.png'}


def annotation(state='both_missing'):
    return m.make_annotation(item(state),{'status':'located','board_quad':QUAD},None,(400,600,3),'hash')


@pytest.mark.parametrize('bad',[ [[1,2]], [[0,0],[20,20],[0,20],[20,0]], [[0,0],[0,20],[20,20],[20,0]], [[-1,0],[20,0],[20,20],[0,20]], [[0,0],[600,0],[600,20],[0,20]], [[0,0],[20,0],[20,float('nan')],[0,20]] ])
def test_invalid_quads_refused(bad):
    with pytest.raises(ValueError):m.validate_quad(bad,600,400)


def test_pending_never_training_ground_truth():
    row=annotation()
    assert row['review_status']=='pending' and row['reviewer'] is None
    assert not row['training_eligible'] and not row['geometry_independently_reviewed']
    assert row['split']=='development_only_not_held_out'
    assert [row['sites'][s]['legacy_class_id'] for s in ['IC','terminal']]==[0,3]


def test_present_contract_and_unknown_state():
    row=annotation('both_present')
    assert [row['sites'][s]['legacy_class_id'] for s in ['IC','terminal']]==[2,1]
    with pytest.raises(ValueError,match='state'):annotation('unconfirmed')


def test_unknown_pose_requires_explicit_manual_proposal():
    with pytest.raises(ValueError,match='missing manual proposal'):m.make_annotation(item(),{'status':'unknown'},None,(400,600,3),'hash')
    override={'board_quad':QUAD,'notes':'manual, inferred'}
    row=m.make_annotation(item(),{'status':'unknown'},override,(400,600,3),'hash')
    assert row['review_priority']=='high' and row['review_status']=='pending'


def test_projection_and_orientation():
    r=annotation()
    np.testing.assert_allclose(r['sites']['IC']['polygon'][0],[192,126])
    assert r['orientation']['header_to_terminal_degrees_clockwise_from_image_x']==0
    assert r['orientation']['header_center']==[100,150]


def test_labelme_retains_geometry_and_draft_warning():
    r=annotation();record=m.labelme_record(r)
    assert record['flags']['unreviewed']
    assert record['shapes'][0]['points']==QUAD
    assert record['imagePath']=='../images/example.png'
    assert len(record['shapes'])==4


def inputs(tmp_path):
    source=tmp_path/'source.png';cv2.imwrite(str(source),np.full((400,600,3),100,np.uint8))
    i={**item(),'path':str(source),'sha256':m.digest(source)}
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'images':[i]}))
    proposals=tmp_path/'proposals.json';proposals.write_text(json.dumps({'frames':[{'id':'example','pose':{'status':'located','board_quad':QUAD}}]}))
    overrides=tmp_path/'overrides.json';overrides.write_text('{}')
    return Namespace(manifest=manifest,proposals=proposals,overrides=overrides,output=tmp_path/'pack')


def test_pack_copies_exact_source_and_exposes_offline_review(tmp_path):
    args=inputs(tmp_path);m.build(args)
    data=json.loads((args.output/'annotations.draft.json').read_text())
    assert data['coverage']=={'both_missing':1,'both_present':0,'ic_missing_terminal_present':0,'approved_geometry':0}
    assert m.digest(args.output/'images/example.png')==data['annotations'][0]['source_image_sha256']
    html=(args.output/'review.html').read_text(encoding='utf-8')
    assert '/*ANNOTATION_DATA*/null' not in html and 'Export review JSON' in html
    assert 'fetch(' not in html and '<script src=' not in html
    assert (args.output/'review-sheets/page-01.jpg').is_file()
    with pytest.raises(FileExistsError):m.build(args)


def test_changed_source_refused_before_output(tmp_path):
    args=inputs(tmp_path);(tmp_path/'source.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'):m.build(args)
    assert not args.output.exists()


@pytest.mark.parametrize('problem',['duplicate','unsafe','unknown_override'])
def test_bad_manifest_refused(tmp_path,problem):
    args=inputs(tmp_path);data=json.loads(args.manifest.read_text())
    if problem=='duplicate':data['images']*=2
    if problem=='unsafe':data['images'][0]['id']='../escape'
    if problem=='unknown_override':args.overrides.write_text('{"other":{}}')
    args.manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):m.build(args)
    assert not args.output.exists()
