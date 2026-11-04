"""CPU-only real C40 loader/sampler acceptance; no simulator, GPU or policy."""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
from types import SimpleNamespace,MethodType
import numpy as np
import torch
from holosoma.config_values.wbt.g1.m15 import make_config,make_eval_config
from holosoma.managers.command.terms.wbt import MotionCommand,AdaptiveTimestepsSampler,FAKE_BODY_NAME_ALIASES
from holosoma.utils.m15_motion_sets import load_multi40_selection,validate_multi40


def check(root):
    torch.set_num_threads(1)
    selection=load_multi40_selection(); reports={}
    train_names={x['motion'] for x in selection['train']}
    eval_names={x['motion'] for x in selection['eval']}
    assert not train_names & eval_names
    for split,cfg in [('train',make_config(root,motion_set='multi40')),('eval',make_eval_config(root))]:
        mc=cfg.command.setup_terms['motion_command'].params['motion_config']
        names=[FAKE_BODY_NAME_ALIASES.get(x,x) for x in cfg.robot.body_names]
        command=SimpleNamespace(device='cpu',num_envs=8192,motion_cfg=mc,object_groups=False,_env=SimpleNamespace(dt=.02))
        command._ensure_index_tensor=MethodType(MotionCommand._ensure_index_tensor,command)
        command._sample_motion_ids_for_reset=MethodType(MotionCommand._sample_motion_ids_for_reset,command)
        loader=MotionCommand._load_motion_data(command,names,cfg.robot.dof_names)
        command.motion=loader
        paths=sorted(Path(mc.motion_dir).glob('*.npz'))
        assert len(paths)==loader.num_motions==len(selection[split])
        assert {p.name for p in paths}=={x['motion'] for x in selection[split]}
        starts=loader.motion_start_idx.clone();ends=loader.motion_end_idx.clone()
        raw_frames=int(loader.time_step_total)
        assert starts[0]==0 and torch.equal(starts[1:],ends[:-1]) and int(ends[-1])==raw_frames
        # Independent NPZ oracle: compare every loaded tensor/frame, including
        # name reordering and wxyz->xyzw conversion, not only loader counts.
        for i,p in enumerate(paths):
            a,b=int(starts[i]),int(ends[i])
            with np.load(p,allow_pickle=False) as data:
                assert b-a==len(data['joint_pos'])
                ji=[data['joint_names'].tolist().index(x) for x in cfg.robot.dof_names]
                bi=[data['body_names'].tolist().index(x) for x in names]
                expected={'joint_pos':data['joint_pos'][:,7:][:,ji],'joint_vel':data['joint_vel'][:,6:][:,ji]}
                for key in ['body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']:
                    value=data[key][:,bi]
                    if key=='body_quat_w':value=value[:,:,[1,2,3,0]]
                    expected[key]=value
                for key in ['object_pos_w','object_quat_w','object_lin_vel_w','object_ang_vel_w']:
                    value=data[key]
                    if key=='object_quat_w':value=value[:,[1,2,3,0]]
                    expected[key]=value
                for key,value in expected.items():
                    assert torch.equal(getattr(loader,key)[a:b],torch.as_tensor(value,dtype=torch.float32)),(p.name,key)
        assert not mc.enable_default_pose_prepend and not mc.enable_default_pose_append
        hold=round(mc.terminal_pose_hold_duration_s/.02) if mc.enable_terminal_pose_hold else 0
        terminals={key:getattr(loader,key)[ends-1].clone() for key in ['joint_pos','body_pos_w','body_quat_w','object_pos_w','object_quat_w']}
        MotionCommand._maybe_add_terminal_pose_hold(command)
        loader=command.motion
        assert loader.time_step_total==raw_frames+hold*len(paths)
        assert torch.equal(loader.motion_end_idx-loader.motion_start_idx,ends-starts+hold)
        for i in range(len(paths)):
            end=int(loader.motion_end_idx[i])
            for key,value in terminals.items():
                assert torch.equal(getattr(loader,key)[end-hold:end],value[i].expand_as(getattr(loader,key)[end-hold:end]))
            for key in ['joint_vel','body_lin_vel_w','body_ang_vel_w','object_lin_vel_w','object_ang_vel_w']:
                assert getattr(loader,key)[end-hold:end].count_nonzero()==0
        command.motion_ids=torch.zeros(command.num_envs,dtype=torch.long)
        ids=torch.arange(command.num_envs)
        def within(mid,frames):
            assert ((mid>=0)&(mid<len(paths))).all()
            assert ((frames>=loader.motion_start_idx[mid])&(frames<loader.motion_end_idx[mid])).all()
        torch.manual_seed(42)
        mid,frames=MotionCommand._sample_uniform_reset_states(command,ids);within(mid,frames)
        uniform_counts=torch.bincount(mid,minlength=len(paths));assert (uniform_counts>0).all()
        before=command.motion_ids.clone(); subset=torch.tensor([1,17,51])
        partial,steps=MotionCommand._sample_uniform_reset_states(command,subset);within(partial,steps)
        mask=torch.ones(command.num_envs,dtype=torch.bool);mask[subset]=False
        assert torch.equal(before[mask],command.motion_ids[mask])
        state=torch.get_rng_state().clone()
        empty=MotionCommand._sample_uniform_reset_states(command,torch.tensor([],dtype=torch.long))
        assert all(x.numel()==0 for x in empty) and torch.equal(state,torch.get_rng_state())
        adaptive_cfg=asdict(mc.adaptive_timestep_sampler)
        args={k:adaptive_cfg[k] for k in ['bin_width_s','ema_alpha','bin_uniform_mix','motion_uniform_mix','score_power','min_segment_steps']}
        sampler=AdaptiveTimestepsSampler(loader.motion_start_idx,loader.motion_end_idx,'cpu',.02,['track','gen'],**args)
        pool=torch.arange(len(paths)); probabilities=sampler.motion_probabilities(0,pool)
        assert torch.allclose(probabilities,torch.ones(len(paths))/len(paths))
        torch.manual_seed(42)
        mid,bins,steps=sampler.sample_from_motion_pool(0,pool,32768);within(mid,steps)
        assert sampler.valid_bin_mask[mid,bins].all()
        counts=torch.bincount(mid,minlength=len(paths));assert (counts>0).all()
        # Changed difficulty must change probabilities without losing boundaries.
        sampler.bin_fail_ema[0,0,sampler.valid_bin_mask[0]]=5.
        biased=sampler.motion_probabilities(0,pool);assert biased[0]>biased[1:].max()
        mid,bins,steps=sampler.sample_from_motion_pool(0,pool,8192);within(mid,steps)
        assert sampler.valid_bin_mask[mid,bins].all()
        eval_fixed=[]
        if split=='eval':
            for i,p in enumerate(paths):
                command.motion_cfg=replace(mc,eval_fixed_reset=replace(mc.eval_fixed_reset,enabled=True,motion_id=i,local_frame=0))
                mid,steps=MotionCommand._sample_eval_fixed_reset_states(command,torch.arange(32));within(mid,steps)
                assert (mid==i).all() and (steps==loader.motion_start_idx[i]).all()
                eval_fixed.append({'motion':p.name,'cpu_reset_states':32,'start_frame':0})
        reports[split]={'motion_count':len(paths),'motion_files':[p.name for p in paths],
            'raw_frames':raw_frames,'loaded_frames_after_hold':loader.time_step_total,'hold_frames_per_motion':hold,
            'all_frame_tensor_oracle_passed':True,'no_skipped_files':True,'boundaries_passed':True,
            'uniform_samples':8192,'uniform_counts':uniform_counts.tolist(),
            'adaptive_samples':32768,'adaptive_counts':counts.tolist(),'adaptive_biased_samples':8192,
            'adaptive_difficulty_response_passed':True,'partial_and_empty_reset_passed':True,
            'eval_fixed_reset_checks':eval_fixed,'start_at_timestep_zero_prob':mc.start_at_timestep_zero_prob}
    return {'passed':True,'selection_id':selection['selection_id'],'splits':reports,
        'train_eval_disjoint':True,'reserve_and_excluded_not_loaded':True,
        'device':'cpu','torch':torch.__version__,'numpy':np.__version__,
        'scope':'Actual MultiMotionLoader, MotionCommand sampling methods, AdaptiveTimestepsSampler and terminal hold; no simulator/reset writeback, PPO, physics or policy evaluation.',
        'gpu_smoke_passed':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--asset-root',required=True);p.add_argument('--report',required=True);a=p.parse_args()
    result=check(a.asset_root);Path(a.report).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'counts':{k:v['motion_count'] for k,v in result['splits'].items()},'device':'cpu'}))
