"""Fail-closed split/config checks. Real-data sampler acceptance is a separate CPU check."""
from copy import deepcopy
from dataclasses import asdict
import json,runpy,shutil
from pathlib import Path
import numpy as np
import pytest
from holosoma.config_values.wbt.g1.hdmi_goal043 import make_config,make_eval_config
from holosoma.utils import m15_motion_sets as ms


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    selection=deepcopy(ms.load_multi40_selection())
    root=tmp_path/'source';(root/'motions').mkdir(parents=True)
    (root/'suitcase.urdf').write_text('<robot name="fixture"/>')
    from holosoma.config_values.wbt.g1.hdmi_goal import make_legacy_m15_config
    robot=make_legacy_m15_config().robot; t=3;b=len(robot.body_names)
    data={'fps':np.array(50.),'joint_names':np.array(robot.dof_names),'body_names':np.array(robot.body_names),
          'joint_pos':np.zeros((t,36)),'joint_vel':np.zeros((t,35)),
          'body_pos_w':np.zeros((t,b,3)),'body_quat_w':np.tile([1.,0,0,0],(t,b,1)),
          'body_lin_vel_w':np.zeros((t,b,3)),'body_ang_vel_w':np.zeros((t,b,3)),
          'object_pos_w':np.zeros((t,3)),'object_quat_w':np.tile([1.,0,0,0],(t,1)),
          'object_lin_vel_w':np.zeros((t,3)),'object_ang_vel_w':np.zeros((t,3))}
    for i,record in enumerate(selection['train']+selection['eval']):
        data['joint_pos'][:,-1]=i
        f=root/'motions'/record['motion'];np.savez(f,**data)
        record.update(frames=t,fps=50.,sha256=ms.sha256(f))
    selection['geometry']={'suitcase.urdf':ms.sha256(root/'suitcase.urdf')}
    frozen=tmp_path/'selection.json';frozen.write_text(json.dumps(selection))
    monkeypatch.setattr(ms,'MULTI40_SELECTION_PATH',frozen)
    monkeypatch.setattr(ms,'MULTI40_SELECTION_SHA256',ms.sha256(frozen))
    helper=runpy.run_path(str(Path(__file__).parents[2]/'scripts/prepare_m15_multi40.py'))
    destination=tmp_path/'prepared'
    helper['prepare'](root,root,destination)
    return destination,root,helper


@pytest.mark.parametrize('extra_dr',[False,True])
def test_config_only_changes_motion_paths_and_name(prepared,extra_dr):
    root,_,_=prepared
    a=asdict(make_config(root,extra_dr=extra_dr))
    for cfg in (make_config(root,extra_dr=extra_dr,motion_set='multi40'),make_eval_config(root,extra_dr=extra_dr)):
        c=asdict(cfg);c['training']['name']=a['training']['name']
        cm=c['command']['setup_terms']['motion_command']['params']['motion_config']
        am=a['command']['setup_terms']['motion_command']['params']['motion_config']
        cm['motion_dir']=am['motion_dir'];cm['motion_file']=am['motion_file']
        assert c==a
    evaluation=make_eval_config(root).command.setup_terms['motion_command'].params['motion_config']
    assert Path(evaluation.motion_file).parent.name=='motions_eval'


@pytest.mark.parametrize('damage',['missing_train','missing_eval','eval_leak','reserve_extra','bytes','selection','geometry','manifest','nan'])
def test_rejects_invalid_split_before_native_loader(prepared,damage):
    root,_,_=prepared;s=ms.load_multi40_selection();p=root/'motions_train'/s['train'][0]['motion']
    if damage=='missing_train':p.unlink()
    elif damage=='missing_eval':(root/'motions_eval'/s['eval'][0]['motion']).unlink()
    elif damage=='eval_leak':shutil.copy2(root/'motions_eval'/s['eval'][0]['motion'],root/'motions_train'/s['eval'][0]['motion'])
    elif damage=='reserve_extra':shutil.copy2(p,root/'motions_train'/s['reserve'][0]['motion'])
    elif damage=='bytes':p.write_bytes(p.read_bytes()+b'changed')
    elif damage=='selection':(root/'multi40_selection.json').write_text('{}')
    elif damage=='geometry':(root/'suitcase.urdf').write_text('changed')
    elif damage=='manifest':
        f=root/'multi40_manifest.json';x=json.loads(f.read_text());x['motions']['train'].pop(p.name);f.write_text(json.dumps(x))
    elif damage=='nan':
        with np.load(p) as z:d=dict(z)
        d['joint_pos'][0,0]=np.nan;np.savez(p,**d)
    with pytest.raises((ValueError,FileNotFoundError)):make_config(root,motion_set='multi40')


def test_preparation_does_not_overwrite_and_uses_independent_files(prepared):
    root,source,helper=prepared
    with pytest.raises(FileExistsError):helper['prepare'](source,source,root)
    f=ms.load_multi40_selection()['train'][0]['motion']
    assert (root/'motions_train'/f).stat().st_ino!=(source/'motions'/f).stat().st_ino


def test_missing_source_does_not_publish(prepared):
    root,source,helper=prepared
    (source/'motions'/ms.load_multi40_selection()['eval'][0]['motion']).unlink()
    destination=root.parent/'must_not_publish'
    with pytest.raises(FileNotFoundError):helper['prepare'](source,source,destination)
    assert not destination.exists() and not destination.with_name(destination.name+'.preparing').exists()


@pytest.mark.parametrize('motion_set,split',[('single043','eval'),('multi50','eval'),('multi40','typo')])
def test_invalid_split_selector_fails(tmp_path,motion_set,split):
    with pytest.raises(ValueError,match='motion_split'):make_config(tmp_path,motion_set=motion_set,motion_split=split)
