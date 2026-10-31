"""Holosoma state interface over the desktop MJLab scene/entity runtime.

The Scene/Simulation construction follows active-adaptation's MJLab backend.
Policy, rewards, motion sampling and PPO stay in Holosoma. This backend does not
claim PhysX-equivalent contacts. Approximate materials require explicit opt-in.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import os
import torch
import warp as wp
import mujoco_warp as mjwarp

from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.shared.object_registry import ObjectType
from holosoma.simulator.mjlab.assets import load_urdf_spec, source_path
from holosoma.simulator.mjlab.randomization import randomize
from mjlab.utils.lab_api.math import quat_apply
from holosoma.simulator.mjlab.physics_compat import NormalContacts, object_damping_wrench


def to_xyzw(state):
    out = state.clone()
    out[..., 3:7] = state[..., [4, 5, 6, 3]]
    return out


def to_wxyz(state):
    out = state.clone()
    out[..., 3:7] = state[..., [6, 3, 4, 5]]
    return out


class MuJoCo(BaseSimulator):
    """MJLab runtime; class name follows Holosoma's MuJoCo family registry."""
    backend_name = "mjlab"
    supports_object_states = True

    def __init__(self, tyro_config, terrain_manager, device):
        super().__init__(tyro_config, terrain_manager, device)
        self.device = device
        self.num_envs = self.training_config.num_envs
        self.audit_dir = Path(os.environ["HOLOSOMA_MJLAB_AUDIT_DIR"])
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.randomization_samples = {}
        self.tyro_config = tyro_config
        self._sim_step_counter = 0
        self._dirty = False
        self.viewer = None
        self.torso_name = self.robot_config.torso_name

    def setup(self):
        self.sim_dt = 1 / self.simulator_config.sim.fps

    def setup_terrain(self):
        if self.terrain_manager.get_state("locomotion_terrain").mesh_type != "plane":
            raise NotImplementedError("Initial MJLab port supports the frozen 043 plane task only")

    def load_assets(self):
        from mjlab.entity import EntityCfg
        from mjlab.scene import Scene, SceneCfg
        from mjlab.sim import Simulation, SimulationCfg, MujocoCfg
        from mjlab.sensor import ContactSensorCfg, ContactMatch
        from mjlab.terrains import TerrainEntityCfg

        if self.video_config.enabled:
            raise NotImplementedError("This initial backend comparison is explicitly non-rendering")
        cfg = self.robot_config
        robot_path = source_path(cfg.asset.asset_root) / cfg.asset.urdf_file
        object_path = Path(cfg.object.object_urdf_path)
        if self.simulator_config.scene.rigid_objects or self.simulator_config.scene.scene_files:
            raise NotImplementedError("Use the original single-object URDF configuration")
        init = cfg.init_state
        q = init.rot
        robot_init = EntityCfg.InitialStateCfg(pos=tuple(init.pos), rot=(q[3], *q[:3]),
            lin_vel=tuple(init.lin_vel), ang_vel=tuple(init.ang_vel),
            joint_pos=dict(init.default_joint_angles))
        entities = {
            "robot": EntityCfg(spec_fn=lambda: load_urdf_spec(robot_path, cfg, self.audit_dir), init_state=robot_init),
            "object": EntityCfg(spec_fn=lambda: load_urdf_spec(object_path, None, self.audit_dir),
                init_state=EntityCfg.InitialStateCfg(pos=(0., 0., 1.))),
        }
        sensor = ContactSensorCfg(name="robot_contacts",
            primary=ContactMatch(mode="body", pattern=tuple(cfg.body_names), entity="robot"),
            fields=("force", "found"), reduce="netforce", global_frame=True)
        self._scene = Scene(SceneCfg(num_envs=self.num_envs,
            env_spacing=self.simulator_config.scene.env_spacing, entities=entities,
            sensors=(sensor,), terrain=TerrainEntityCfg(terrain_type="plane")), device=self.device)
        # Same desktop MJLab defaults: only dt is fixed to the original task's 200 Hz.
        self._sim = Simulation(num_envs=self.num_envs,
            cfg=SimulationCfg(nconmax=512, njmax=2048, contact_sensor_maxmatch=80,
                mujoco=MujocoCfg(timestep=self.sim_dt, iterations=10, ls_iterations=20)),
            model=self._scene.compile(), device=self.device)
        self._scene.initialize(self._sim.mj_model, self._sim.model, self._sim.data)
        if self._scene.sensor_context is not None:
            self._sim.set_sensor_context(self._scene.sensor_context)
        self._sim.expand_model_fields(("body_mass", "body_inertia", "body_iquat", "body_ipos", "geom_friction", "geom_solref"))
        self.entities = self._scene.entities
        self._robot = self.entities["robot"]
        self.scene = SimpleNamespace(env_origins=self._scene.env_origins,
            articulations={"robot": self._robot}, rigid_objects={"object": self.entities["object"]},
            entities=self.entities, write_data_to_sim=self.write_state_updates)
        self.dof_names = list(cfg.dof_names)
        self._body_list = list(cfg.body_names)
        self.num_dof = len(self.dof_names)
        self.num_dofs = self.num_dof
        self.body_names = self._body_list
        self.num_bodies = len(self._body_list)
        self.dof_ids = torch.tensor([self._robot.joint_names.index(n) for n in self.dof_names], device=self.device)
        self.body_ids = torch.tensor([self._robot.body_names.index(n) for n in self._body_list], device=self.device)
        self._qfrc_ids = self._robot.indexing.joint_v_adr[self.dof_ids]
        self.contact_sensor = self._scene.sensors["robot_contacts"]
        contact_names = [n.split("/")[-1] for n in self.contact_sensor.primary_names]
        self._contact_ids = torch.tensor([contact_names.index(n) for n in self._body_list], device=self.device)
        self.contact_forces_history = torch.zeros(self.num_envs, self.simulator_config.contact_sensor_history_length,
            self.num_bodies, 3, device=self.device)
        self._normal_contacts = NormalContacts(self)
        self._torques = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self._scene.reset()
        for name in ("robot", "object"):
            entity = self.entities[name]
            assert not entity.is_fixed_base, name
            entity.write_root_state_to_sim(entity.data.default_root_state)
            if entity.is_articulated:
                entity.write_joint_state_to_sim(entity.data.default_joint_pos, entity.data.default_joint_vel)
        self._sim.forward()
        self._scene.update(0.)
        self.all_root_states = torch.zeros(self.num_envs * 2, 13, device=self.device)
        self.robot_root_states = self.all_root_states.view(self.num_envs, 2, 13)[:, 0]
        for pos, name in enumerate(("robot", "object")):
            ent = self.entities[name]
            state = to_xyzw(torch.cat((ent.data.root_link_pose_w, ent.data.root_com_vel_w), -1))
            self.object_registry.register_object(name, ObjectType.ROBOT if pos == 0 else ObjectType.INDIVIDUAL,
                0, state[:, :7].clone())
        self.object_registry.setup_ranges(self.num_envs, 1, 0, 1)
        self.object_registry.finalize_registration()
        self._initial_poses = {n: to_xyzw(e.data.default_root_state)[:, :7].clone() for n,e in self.entities.items()}
        self.refresh_sim_tensors()
        (self.audit_dir / "backend.json").write_text(json.dumps({
            "backend":"mjlab", "num_envs":self.num_envs, "dt":self.sim_dt,
            "control_decimation":self.simulator_config.sim.control_decimation,
            "joint_names":self.dof_names,"body_names":self._body_list,
            "frame":"independent MuJoCo worlds with zero env origins",
            "contact":"net normal-only world-frame force per body, physical-step history",
            "object_damping":{"linear_rate":0.01,"angular_rate":0.01,"method":"mass/inertia-scaled world COM drag"},
            "materials":"approximation only when explicitly enabled; not PhysX equivalent"},indent=2))

    def create_envs(self, num_envs, env_origins, base_init_state, *args):
        assert num_envs == self.num_envs

    def get_dof_limits_properties(self):
        self.hard_dof_pos_limits = torch.zeros(
            self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False
        )
        self.dof_pos_limits = torch.zeros(
            self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False
        )
        self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        for i in range(self.num_dof):
            self.hard_dof_pos_limits[i, 0] = self.robot_config.dof_pos_lower_limit_list[i]
            self.hard_dof_pos_limits[i, 1] = self.robot_config.dof_pos_upper_limit_list[i]
            self.dof_pos_limits[i, 0] = self.robot_config.dof_pos_lower_limit_list[i]
            self.dof_pos_limits[i, 1] = self.robot_config.dof_pos_upper_limit_list[i]
            self.dof_vel_limits[i] = self.robot_config.dof_vel_limit_list[i]
            self.torque_limits[i] = self.robot_config.dof_effort_limit_list[i]
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = m - 0.5 * r * self.robot_config.soft_dof_pos_limit
            self.dof_pos_limits[i, 1] = m + 0.5 * r * self.robot_config.soft_dof_pos_limit
        return self.dof_pos_limits, self.dof_vel_limits, self.torque_limits

    def find_rigid_body_indice(self, name):
        return self._body_list.index(name)

    def prepare_sim(self):
        self.refresh_sim_tensors()
        if os.environ.get("HOLOSOMA_MJLAB_STATE_GATE") == "1":
            from .contracts import audit
            audit(self)

    def prepare_manager_fields(self, **managers):
        # Fields were expanded before graph capture. No shared fields are silently randomized.
        pass

    def refresh_sim_tensors(self):
        self.write_state_updates()
        for pos, name in enumerate(("robot", "object")):
            d = self.entities[name].data
            self.all_root_states.view(self.num_envs, 2, 13)[:, pos] = to_xyzw(torch.cat((d.root_link_pose_w, d.root_com_vel_w), -1))
        d = self._robot.data
        self.dof_pos = d.joint_pos[:, self.dof_ids].clone()
        self.dof_vel = d.joint_vel[:, self.dof_ids].clone()
        self.base_quat = self.robot_root_states[:, 3:7]
        self._rigid_body_pos = d.body_link_pos_w[:, self.body_ids]
        self._rigid_body_rot = d.body_link_quat_w[:, self.body_ids][..., [1,2,3,0]]
        # IsaacLab body_lin_vel_w uses COM velocity, while poses use link frames.
        self._rigid_body_vel = d.body_com_lin_vel_w[:, self.body_ids]
        self._rigid_body_ang_vel = d.body_com_ang_vel_w[:, self.body_ids]
        self.contact_forces = self._normal_contacts.read().clone()

    @property
    def dof_state(self):
        return torch.stack((self.dof_pos, self.dof_vel), -1)

    def apply_torques_at_dof(self, torques):
        self._torques.copy_(torques)
        self._sim.data.qfrc_applied[:, self._qfrc_ids] = torques

    def simulate_at_each_physics_step(self):
        body = self.entities["object"].indexing.root_body_id
        damping = object_damping_wrench(self)
        external = self._sim.data.xfrc_applied[:, body].clone()
        self._sim.data.xfrc_applied[:, body] = external + damping
        self._sim.step()
        self._refresh_derived_state()
        self._sim.data.xfrc_applied[:, body] = external
        self._sim_step_counter += 1
        self._scene.update(self.sim_dt)
        # The unchanged Holosoma PD controller reads these before every physics
        # substep, just as with IsaacSim; control-rate refresh alone is stale.
        self.dof_pos = self._robot.data.joint_pos[:, self.dof_ids]
        self.dof_vel = self._robot.data.joint_vel[:, self.dof_ids]
        f = self._normal_contacts.read()
        self.contact_forces_history[:, 1:] = self.contact_forces_history[:, :-1].clone()
        self.contact_forces_history[:, 0] = f

    def clear_contact_forces_history(self, env_id):
        self.contact_forces_history[env_id] = 0
        self.contact_sensor.reset(env_id)

    def get_actor_indices(self, names, env_ids=None):
        return self.object_registry.get_object_indices(names, env_ids)

    def get_actor_initial_poses(self, names, env_ids=None):
        return self.object_registry.get_initial_poses_batch(names, env_ids)

    def get_actor_states_by_index(self, indices):
        return self.all_root_states[indices].clone()

    def set_actor_states_by_index(self, indices, states, write_updates=True):
        self.all_root_states[indices] = states
        for slot, name in enumerate(("robot", "object")):
            mask = indices.remainder(2) == slot
            ids = torch.div(indices[mask], 2, rounding_mode="floor")
            if ids.numel():
                self._write_root(name, states[mask], ids)
        self._dirty = True
        if write_updates:
            self.write_state_updates()

    def _write_root(self, name, state, ids):
        entity = self.entities[name]
        converted = to_wxyz(state)
        com_offset = self._sim.model.body_ipos[ids, entity.indexing.root_body_id]
        offset_w = quat_apply(converted[:, 3:7], com_offset)
        converted[:, 7:10] -= torch.linalg.cross(converted[:, 10:13], offset_w)
        entity.data.write_root_state(converted, ids)

    def get_actor_states_by_names(self, names, env_ids=None):
        return self.get_actor_states_by_index(self.get_actor_indices(names, env_ids))

    def set_actor_states_by_names(self, names, env_ids, states, write_updates=True):
        self.set_actor_states_by_index(self.get_actor_indices(names, env_ids), states, write_updates)

    def set_actor_root_state_tensor_robots(self, env_ids=None, root_states=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        src = self.robot_root_states if root_states is None or root_states is self.all_root_states else root_states
        self._write_root("robot", src[ids], ids)
        self._robot.clear_state(ids)
        self._sim.data.qacc_warmstart[ids] = 0
        self._sim.data.qfrc_applied[ids] = 0
        self._sim.data.xfrc_applied[ids] = 0
        self._dirty = True

    def set_dof_state_tensor_robots(self, env_ids=None, dof_states=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        s = self.dof_state if dof_states is None else dof_states
        self._robot.write_joint_state_to_sim(s[ids,:,0], s[ids,:,1], joint_ids=self.dof_ids, env_ids=ids)
        self._dirty = True

    def set_actor_root_state_tensor(self, set_env_ids=None, root_states=None):
        self.set_actor_root_state_tensor_robots(set_env_ids, root_states)

    def set_dof_state_tensor(self, set_env_ids=None, dof_states=None):
        self.set_dof_state_tensor_robots(set_env_ids, dof_states)

    def _refresh_derived_state(self):
        """Expose post-integration body state without re-solving contacts.

        MuJoCo step leaves xpos/cvel at the pre-integration state while qpos/
        qvel have advanced. Refresh only kinematics and COM velocities. Keep
        the just-completed step's contact forces and solver warm start intact.
        """
        with wp.ScopedDevice(self._sim.wp_device):
            mjwarp.kinematics(self._sim.wp_model, self._sim.wp_data)
            mjwarp.com_pos(self._sim.wp_model, self._sim.wp_data)
            mjwarp.com_vel(self._sim.wp_model, self._sim.wp_data)
        # EntityData reads these Warp arrays through PyTorch immediately below.
        if self._sim.data.qpos.is_cuda:
            torch.cuda.current_stream(self._sim.data.qpos.device).wait_stream(
                torch.cuda.ExternalStream(wp.get_stream(self._sim.wp_device).cuda_stream))

    def write_state_updates(self):
        if self._dirty:
            self._sim.forward()
            self._scene.update(0.)
            self._dirty = False

    def get_actor_scale_factors(self, name):
        assert name in self.entities
        return torch.ones(self.num_envs, 3, device=self.device)

    def get_actor_local_collision_bounds(self, name):
        from holosoma.envs.wbt.hybrid_size_randomization import load_urdf_collision_bounds
        if name != "object":
            raise KeyError(name)
        return torch.tensor(load_urdf_collision_bounds(self.robot_config.object.object_urdf_path), device=self.device)

    def get_dof_forces(self, env_id=0):
        return self._torques[env_id]

    def randomize(self, kind, params):
        return randomize(self, kind, params)

    def setup_viewer(self):
        raise NotImplementedError("Use headless mode for backend timing")

    def render(self, *args, **kwargs):
        if not self.headless or self.video_config.enabled:
            raise RuntimeError("Rendering must be disabled in the matched timing runs")

    def on_episode_start(self, env_id=0):
        pass

    def on_episode_end(self, env_id=0):
        pass
