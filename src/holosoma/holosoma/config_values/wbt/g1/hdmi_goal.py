"""Opt-in HDMI-informed state teacher. Original hybrid presets stay unchanged."""
from dataclasses import replace
from math import pi
from holosoma.config_values.wbt.g1.robot_mode15 import with_mode15
from holosoma.utils.collision_exclusions import HDMI_G1_SELF_COLLISION_EXCLUSIONS
from holosoma.config_types.observation import ObsHistoryAugCfg, ObsTermCfg
from holosoma.config_types.reward import RewardTermCfg
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_hybrid_object

base = g1_29dof_wbt_hybrid_object
proprio_offsets = (0, 1, 2, 3, 4, 8, 16)
actor = base.observation.groups['actor_obs']
terms = dict(actor.terms)
for name in ('base_ang_vel', 'dof_pos', 'dof_vel'):
    terms[name] = replace(terms[name], history_offsets=proprio_offsets)
terms['projected_gravity'] = ObsTermCfg(
    func='holosoma.managers.observation.terms.wbt:projected_gravity',
    noise=0.01, history_offsets=proprio_offsets,
)
terms['actions'] = replace(terms['actions'], history_offsets=(0, 1, 2))
terms['obj_ang_vel_b'] = ObsTermCfg(func='holosoma.managers.observation.terms.hdmi_goal:object_ang_vel_b')
terms['object_to_hands_b'] = ObsTermCfg(func='holosoma.managers.observation.terms.hdmi_goal:object_to_hands_b')
groups = dict(base.observation.groups)
groups['actor_obs'] = replace(actor, terms=terms, history_augmentation=ObsHistoryAugCfg())
# Critic retains privileged reference terms, and gets the same history to value a history-conditioned actor.
critic_terms = dict(groups['critic_obs'].terms)
for name in ('base_ang_vel', 'dof_pos', 'dof_vel', 'projected_gravity', 'actions', 'obj_ang_vel_b', 'object_to_hands_b'):
    critic_terms[name] = replace(terms[name], noise=0.0)
groups['critic_obs'] = replace(groups['critic_obs'], terms=critic_terms, history_augmentation=ObsHistoryAugCfg())

reward_terms = dict(base.reward.terms)
for name, term in list(reward_terms.items()):
    function_name = term.func.split(':')[-1]
    if function_name.startswith('hybrid_'):
        reward_terms[name] = replace(term, func='holosoma.managers.reward.terms.hdmi_goal:track_only',
                                    params={**term.params, 'reference_term': function_name.removeprefix('hybrid_')})
reward_terms.update({
    'joint_pos_tracking': RewardTermCfg(func='holosoma.managers.reward.terms.hdmi_goal:joint_tracking',
                                      params={'sigma':0.25, 'velocity':False}, weight=0.5),
    'joint_vel_tracking': RewardTermCfg(func='holosoma.managers.reward.terms.hdmi_goal:joint_tracking',
                                      params={'sigma':2.5, 'velocity':True}, weight=0.5),
    'joint_speed_l2': RewardTermCfg(func='holosoma.managers.reward.terms.hdmi_goal:joint_speed_penalty', weight=-1e-4),
    'goal_progress': RewardTermCfg(func='holosoma.managers.reward.terms.hdmi_goal:GoalProgress',
                                 params={'pos_scale':1.0, 'ori_scale':pi}, weight=1.0),
    'goal_object_reached_once': RewardTermCfg(func='holosoma.managers.reward.terms.hdmi_goal:GoalSettledOnce',
        params={'bonus':3.0, 'xy_threshold':0.2, 'max_bottom_height':0.05, 'ori_threshold':pi/6,
                'lin_vel_threshold':0.05, 'ang_vel_threshold':0.2, 'hold_seconds':0.5}, weight=1.0),
})

# Goal-conditioned rollouts may choose their own route and phase. Keep the
# original track checks and independent gen root-height safety condition.
termination_terms = dict(base.termination.terms)
bad_tracking = termination_terms['bad_tracking']
termination_terms['bad_tracking'] = replace(bad_tracking, params={
    **bad_tracking.params,
    'gen_bad_ref_pos_z_threshold': None,
    'gen_bad_ref_pos_xyz_threshold': None,
    'gen_bad_object_pos_z_threshold': None,
    'gen_bad_object_ori_threshold': None,
})

g1_29dof_wbt_hybrid_object_hdmi = replace(
    base,
    training=replace(base.training, name='g1_29dof_wbt_hybrid_object_hdmi'),
    robot=with_mode15(replace(base.robot, asset=replace(base.robot.asset,
        collision_exclusions=HDMI_G1_SELF_COLLISION_EXCLUSIONS))),
    observation=replace(base.observation, groups=groups, history_augmentation=False),
    reward=replace(base.reward, terms=reward_terms),
    termination=replace(base.termination, terms=termination_terms),
)
