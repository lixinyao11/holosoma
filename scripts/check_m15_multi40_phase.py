"""CPU phase acceptance over all real train40/eval2 frames and sampled resets."""
import argparse,json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from holosoma.config_values.wbt.g1.m15 import make_config
from holosoma.managers.command.terms.wbt import MotionCommand,AdaptiveTimestepsSampler,FAKE_BODY_NAME_ALIASES
from holosoma.managers.observation.terms.track_phase import track_phase


def check(root):
    torch.set_num_threads(1);reports={}
    for split,n in [('train',40),('eval',2)]:
        c=make_config(root,motion_set='multi40',motion_split=split)
        d=make_config(root,motion_set='multi40',motion_split=split,track_phase=True)
        before,after=asdict(c),asdict(d)
        term=after['observation']['groups']['actor_obs']['terms'].pop('track_phase')
        after['training']['name']=before['training']['name'];assert before==after
        mc=d.command.setup_terms['motion_command'].params['motion_config']
        cmd=NS(device='cpu',motion_cfg=mc,_env=NS(dt=.02))
        names=[FAKE_BODY_NAME_ALIASES.get(x,x) for x in d.robot.body_names]
        cmd.motion=MotionCommand._load_motion_data(cmd,names,d.robot.dof_names)
        assert cmd.motion.num_motions==n
        original_lengths=(cmd.motion.motion_end_idx-cmd.motion.motion_start_idx).clone()
        MotionCommand._maybe_add_terminal_pose_hold(cmd)
        hold=round(mc.terminal_pose_hold_duration_s/.02) if mc.enable_terminal_pose_hold else 0
        lengths=cmd.motion.motion_end_idx-cmd.motion.motion_start_idx
        assert torch.equal(lengths,original_lengths+hold)
        cmd.motion_ids=torch.repeat_interleave(torch.arange(n),lengths)
        cmd.time_steps=torch.arange(cmd.motion.time_step_total)
        cmd.track_env_mask=torch.ones_like(cmd.time_steps,dtype=torch.bool)
        env=NS(dt=.02,command_manager=NS(get_state=lambda _:cmd),episode_length_buf=torch.full_like(cmd.time_steps,99999))
        expected=torch.cat([torch.cat([torch.linspace(0,1,int(k)),torch.ones(hold)]) for k in original_lengths])
        phase=track_phase(env).squeeze(-1)
        assert torch.allclose(phase,expected,atol=1e-7,rtol=1e-7)
        assert torch.isfinite(phase).all() and ((phase>=0)&(phase<=1)).all()
        assert (phase[cmd.motion.motion_start_idx]==0).all()
        assert (phase[cmd.motion.motion_end_idx-1]==1).all()
        # Explicitly move a single reset into a different motion at its beginning.
        saved=phase.clone();cmd.motion_ids[1]=n-1;cmd.time_steps[1]=cmd.motion.motion_start_idx[-1]
        changed=track_phase(env).squeeze(-1);saved[1]=0
        assert torch.equal(changed,saved)
        cmd.track_env_mask[::7]=False
        assert (track_phase(env)[::7]==0).all()
        options=asdict(mc.adaptive_timestep_sampler)
        kwargs={k:options[k] for k in ['bin_width_s','ema_alpha','bin_uniform_mix','motion_uniform_mix','score_power','min_segment_steps']}
        sampler=AdaptiveTimestepsSampler(cmd.motion.motion_start_idx,cmd.motion.motion_end_idx,'cpu',.02,['track','gen'],**kwargs)
        torch.manual_seed(42)
        ids,_,steps=sampler.sample_from_motion_pool(0,torch.arange(n),32768)
        cmd.motion_ids=ids;cmd.time_steps=steps;cmd.track_env_mask=torch.ones_like(ids,dtype=torch.bool)
        sampled_phase=track_phase(env).squeeze(-1)
        assert torch.allclose(sampled_phase,expected[steps],atol=1e-7,rtol=1e-7)
        assert len(ids.unique())==n
        reports[split]={'motion_count':n,'raw_frames':int(original_lengths.sum()),
            'phase_frames_checked':len(expected),'hold_frames_per_motion':hold,'adaptive_reset_samples':32768,
            'all_clip_lengths_offsets_boundaries_hold_passed':True,'partial_reset_passed':True,'gen_phase_zero_passed':True,
            'motion_dir':mc.motion_dir,'phase_term':term,'D_equals_C_except_actor_phase_and_name':True}
    return {'passed':True,'device':'cpu','splits':reports,'phase_meaning':'current local reference-frame progress, terminal hold=1, gen=0',
        'expected_actor_dim':572,'expected_critic_dim':776,'dimension_delta':{'actor':1,'critic':0},
        'full_native_GPU_observation_dimensions_verified':False,'GPU_smoke':'not_run','policy_eval':'not_run'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--asset-root',required=True);p.add_argument('--report',required=True)
    a=p.parse_args();result=check(a.asset_root);Path(a.report).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'splits':[v['motion_count'] for v in result['splits'].values()],'device':'cpu'}))
