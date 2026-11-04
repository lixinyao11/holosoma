> Historical experiment specification: the sections below describe the frozen
> pre-adoption A/B/C/D runs. The candidate branch now uses upstream joint rewards
> and hand observations; see [upstream adoption](upstream_hands_profile.md).
> Old dimensions and no-reference-actor claims below do not describe the new default.

# Legacy M15默认配置

2026-11-02用户选择：M15资产/名义电机参数 + 四对自碰撞排除 + 电机DR + 物理子步延迟/低通滤波。
代码维护位置：desktop `/home/pummpkin/projects/holosoma-legacy`，分支 `pummpkin/legacy`。

默认入口仍为 `exp:g1-29dof-wbt-hybrid-object-hdmi`；`hdmi_goal043.make_config`继承同一默认配置。
其他机器人/通用WBT预设保持原定义。保留M15原全DR参考入口：
`exp:g1-29dof-wbt-hybrid-object-hdmi-m15-reference`，三个新增开关均关闭。

## 独立开关

```python
from holosoma.config_values.wbt.g1.hdmi_goal import make_legacy_m15_config

cfg = make_legacy_m15_config(
    collision_exclusions=True,
    motor_dr=True,
    delay_filter=True,
)
```

每次从原教师配置构造，不在已有增强配置上反复叠加。CLI高级字段覆盖仍可用；开关组合统一通过上述工厂构造。

- 碰撞：左右 `elbow_link ↔ wrist_pitch_link`，`pelvis ↔ left/right_hip_roll_link`；其他自碰撞及外部接触保留。
- 电机DR：手臂Kp/Kd倍率0.9–1.1，腰腿0.8–1.2，均使用已验证的log-uniform采样；startup armature倍率0.75–1.25，不在reset累乘。
- 动作：物理子步延迟1–3；200Hz物理时对应5–15ms。逐子步低通alpha0.8–1.0，不与控制步延迟叠加。
- PPO、奖励、终止、观测、原材质/质量/COM/inertia/push/restitution/startup bias配置保持不变。观测契约仍继承原565/770配置。

## 验证边界

三个独立消融各完成5000轮，最终checkpoint各做32次小样本配对eval。联合扰动测试并不等于三项联合训练已验证。
本次合入完成CPU配置/开关/数值语义检查，不启动训练、修改RP冻结目录、升级环境或自动commit/push。
新默认完整组合仍需独立fresh smoke验收后才能声称GPU运行通过；Isaac启动、驱动、资源及原物理门禁仍按实验计划执行。

本次源代码留档：`operations/rp-workflow/version_snapshots/legacy-m15-combined-20261102-v1/`。
评测依据：`experiments/isaac-m15-eval32-v1/`；最终数据在RP48的 `holosoma-isaac-m15-eval32-v1/rp8_multigpu_archive/`。
