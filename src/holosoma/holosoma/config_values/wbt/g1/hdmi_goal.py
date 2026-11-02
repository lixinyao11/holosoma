"""Legacy HDMI teacher: M15 with the three independently switchable enhancements."""
from dataclasses import replace
from math import pi
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

_hdmi_original = replace(
    base,
    training=replace(base.training, name='g1_29dof_wbt_hybrid_object_hdmi'),
    observation=replace(base.observation, groups=groups, history_augmentation=False),
    reward=replace(base.reward, terms=reward_terms),
    termination=replace(base.termination, terms=termination_terms),
)


def make_legacy_m15_config(*, collision_exclusions=True, motor_dr=True, delay_filter=True):
    """Build from the original teacher so switches can be independently disabled."""
    from holosoma.config_values.wbt.g1.legacy_m15 import with_legacy_m15
    return with_legacy_m15(_hdmi_original, collision_exclusions=collision_exclusions,
                          motor_dr=motor_dr, delay_filter=delay_filter)


# The established HDMI/goal043 entry points now use the approved legacy profile.
g1_29dof_wbt_hybrid_object_hdmi = make_legacy_m15_config()
# Explicit M15 reference: original full DR, all three new switches disabled.
g1_29dof_wbt_hybrid_object_hdmi_m15_reference = make_legacy_m15_config(
    collision_exclusions=False, motor_dr=False, delay_filter=False)
