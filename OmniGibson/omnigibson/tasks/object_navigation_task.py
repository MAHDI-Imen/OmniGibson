import math
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.object_states import Pose
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.reward_functions.collision_reward import CollisionReward
from omnigibson.reward_functions.object_goal_reward import ObjectGoalReward
from omnigibson.reward_functions.potential_reward import PotentialReward
from omnigibson.scenes.traversable_scene import TraversableScene
from omnigibson.tasks.task_base import BaseTask
from omnigibson.termination_conditions.falling import Falling
from omnigibson.termination_conditions.max_collision import MaxCollision
from omnigibson.termination_conditions.object_goal import ObjectGoal
from omnigibson.termination_conditions.timeout import Timeout
from omnigibson.utils.python_utils import assert_valid_key, classproperty
from omnigibson.utils.sim_utils import land_object, test_valid_pose
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)


class ObjectNavigationTask(BaseTask):
    def __init__(
        self,
        goal_object_category,
        robot_idn=0,
        floor=0,
        initial_pos=None,
        initial_quat=None,
        goal_tolerance=0.5,
        goal_in_polar=False,
        path_range=None,
        termination_config=None,
        reward_config=None,
    ):
        # Store inputs
        self._robot_idn = robot_idn
        self._floor = floor
        self._initial_pos = (
            initial_pos if initial_pos is None else th.tensor(initial_pos)
        )
        self._initial_quat = (
            initial_quat if initial_quat is None else th.tensor(initial_quat)
        )
        self._randomize_initial_pos = initial_pos is None
        self._randomize_initial_quat = initial_quat is None
        self._goal_object_category = goal_object_category
        self._goal_tolerance = goal_tolerance
        self._goal_in_polar = goal_in_polar
        self._path_range = path_range

        # Create other attributes that will be filled in at runtime
        self._path_length = None
        self._current_robot_pos = None
        self._l2_dist = None

        # Run super
        super().__init__(
            termination_config=termination_config, reward_config=reward_config
        )

    def _create_termination_conditions(self):
        # Initialize termination conditions dict and fill in with MaxCollision, Timeout, Falling, and PointGoal
        terminations = dict()
        terminations["max_collision"] = MaxCollision(
            max_collisions=self._termination_config["max_collisions"]
        )
        terminations["timeout"] = Timeout(
            max_steps=self._termination_config["max_steps"]
        )
        terminations["falling"] = Falling(
            robot_idn=self._robot_idn,
            fall_height=self._termination_config["fall_height"],
        )
        terminations["objectgoal"] = ObjectGoal(
            robot_idn=self._robot_idn,
            distance_tol=self._goal_tolerance,
            distance_axes="xy",
        )

        return terminations

    def _create_reward_functions(self):
        # Initialize reward functions dict and fill in with Potential, Collision, and PointGoal rewards
        rewards = dict()

        rewards["potential"] = PotentialReward(
            potential_fcn=self.get_potential,
            r_potential=self._reward_config["r_potential"],
        )
        rewards["collision"] = CollisionReward(
            r_collision=self._reward_config["r_collision"]
        )
        rewards["objectgoal"] = ObjectGoalReward(
            objectgoal=self._termination_conditions["objectgoal"],
            r_objectgoal=self._reward_config["r_objectgoal"],
        )

        return rewards

    def _load(self, env):
        self.goal_objects = [
            o
            for o in env.scene.object_registry.objects
            if o.category == self._goal_object_category
        ]
        assert len(self.goal_objects), "The scene does not contain the goal category"
        # Auto-initialize all markers
        og.sim.play()
        self._reset_agent(env=env)
        env.scene.update_initial_file()
        og.sim.stop()

    # TODO: add load_visualization_markers

    def _sample_initial_pose(self, env, max_trials=100):
        """
        Potentially sample the robot initial pos / ori, based on whether we're using randomized
        initial state. If not randomzied, then this value will return the corresponding values inputted
        during this task initialization.

        Args:
            env (Environment): Environment instance
            max_trials (int): Number of trials to attempt to sample valid poses and positions

        Returns:
            3-tuple:
                - 3-array: (x,y,z) global sampled initial position
                - 4-array: (x,y,z,w) global sampled initial orientation in quaternion form
        """
        # Possibly sample initial pos
        if self._randomize_initial_pos:
            in_range_dist = False
            for _ in range(max_trials):
                _, initial_pos = env.scene.get_random_point(
                    floor=self._floor, robot=env.robots[self._robot_idn]
                )
                # make sure the objects are within the path range.
                dist = self._get_l2_potential(env)
                if dist is not None and (
                    (
                        self._path_range is None
                        or (self._path_range[0] < dist < self._path_range[1])
                    )
                ):
                    in_range_dist = True
                    break

            # Notify if we weren't able to get a valid start / end point sampled in the requested range
            if not in_range_dist:
                log.warning(
                    "Failed to sample initial and target positions within requested path range"
                )

        else:
            initial_pos = self._initial_pos

        # Possibly sample initial ori
        quat_lo, quat_hi = 0, math.pi * 2
        initial_quat = (
            T.euler2quat(
                th.tensor([0, 0, (th.rand(1) * (quat_hi - quat_lo) + quat_lo).item()])
            )
            if self._randomize_initial_quat
            else self._initial_quat
        )

        # Add additional logging info
        log.info("Sampled initial pose: {}, {}".format(initial_pos, initial_quat))
        return initial_pos, initial_quat

    def _get_l2_potential(self, env):
        """
        Get potential based on L2 distance

        Args:
            env: environment instance

        Returns:
            float: L2 distance to the closest target position
        """
        min_dist = float("inf")
        closest_goal = ""
        for goal_pos in self.get_goal_pos(env):
            dist = T.l2_distance(
                env.robots[self._robot_idn].states[Pose].get_value()[0][:2],
                goal_pos[:2],
            )
            min_dist = min(min_dist, dist)
        return min_dist

    def get_potential(self, env):
        """
        Compute task-specific potential: distance to the goal

        Args:
            env (Environment): Environment instance

        Returns:
            float: Computed potential
        """
        return self._get_l2_potential(env)

    def _reset_agent(self, env):
        # Reset agent
        env.robots[self._robot_idn].reset()

        # We attempt to sample valid initial poses and goal positions
        success, max_trials = False, 100

        initial_pos, initial_quat, goal_pos = None, None, None
        for i in range(max_trials):
            initial_pos, initial_quat = self._sample_initial_pose(env)
            goal_pos = self.get_goal_pos(env)
            # Make sure the sampled robot start pose and goal position are both collision-free
            success = test_valid_pose(
                env.robots[self._robot_idn],
                initial_pos,
                initial_quat,
                env.initial_pos_z_offset,
            )

            # Don't need to continue iterating if we succeeded
            if success:
                break

        # Notify user if we failed to reset a collision-free sampled pose
        if not success:
            log.warning("Failed to reset robot without collision")

        # Land the robot
        land_object(
            env.robots[self._robot_idn],
            initial_pos,
            initial_quat,
            env.initial_pos_z_offset,
        )

        # Store the sampled values internally
        self._initial_pos = initial_pos
        self._initial_quat = initial_quat
        self._goal_pos = goal_pos

    def _reset_variables(self, env):
        # Run super first
        super()._reset_variables(env=env)

        # Reset internal variables
        self._path_length = 0.0
        self._current_robot_pos = self._initial_pos
        self._l2_dist = self._get_l2_potential(env)

    def _step_termination(self, env, action, info=None):
        # Run super first
        done, info = super()._step_termination(env=env, action=action, info=info)

        # Add additional info
        info["path_length"] = self._path_length
        info["spl"] = (
            float(info["success"]) * min(1.0, self._l2_dist / self._path_length)
            if done and self._path_length != 0.0
            else 0.0
        )

        return done, info

    def get_shortest_path_to_goal(self, env, start_xy_pos=None, entire_path=False):
        """
        Get the shortest path and geodesic distance from @start_pos to the closest target position

        Args:
            env (TraversableEnv): Environment instance
            start_xy_pos (None or 2-array): If specified, should be the global (x,y) start position from which
                to calculate the shortest path to the goal position. If None (default), the robot's current xy position
                will be used
            entire_path (bool): Whether to return the entire shortest path

        Returns:
            3-tuple:
                - list of 2-array: List of (x,y) waypoints representing the path # TODO: is this true?
                - float: geodesic distance of the path to the goal position
                - 3d tensor (x,y,z) : closest goal_pos
        """
        start_xy_pos = (
            env.robots[self._robot_idn].states[Pose].get_value()[0][:2]
            if start_xy_pos is None
            else start_xy_pos
        )
        min_path, min_geodesic_dist, closest_goal_pos = None, float("inf"), None

        return min_path, min_geodesic_dist, closest_goal_pos

    def _global_pos_to_robot_frame(self, env, pos):
        """
        Convert a 3D point in global frame to agent's local frame

        Args:
            env (TraversableEnv): Environment instance
            pos (th.Tensor): global (x,y,z) position

        Returns:
            th.Tensor: (x,y,z) position in self._robot_idn agent's local frame
        """
        delta_pos_global = pos - env.robots[self._robot_idn].states[Pose].get_value()[0]
        return (
            T.quat2mat(env.robots[self._robot_idn].states[Pose].get_value()[1]).T
            @ delta_pos_global
        )

    def _get_obs(self, env):
        # linear velocity and angular velocity
        ori_t = T.quat2mat(env.robots[self._robot_idn].states[Pose].get_value()[1]).T
        lin_vel = ori_t @ env.robots[self._robot_idn].get_linear_velocity()
        ang_vel = ori_t @ env.robots[self._robot_idn].get_angular_velocity()

        # Compose observation dict
        low_dim_obs = dict(
            robot_lin_vel=lin_vel,
            robot_ang_vel=ang_vel,
        )

        # We have no non-low-dim obs, so return empty dict for those
        return low_dim_obs, dict()

    def _load_non_low_dim_observation_space(self):
        # No non-low dim observations so we return an empty dict
        return dict()

    def get_goal_pos(self, env):
        """
        Returns:
            N,3-array: (N, x,y,z) global current goal position for each object of goal category
        """
        goal_pos = th.stack(
            [o.get_position_orientation()[0] for o in self.goal_objects]
        )
        log.info("Category goal positions: {}".format(goal_pos))

        return goal_pos

    def get_current_pos(self, env):
        """
        Returns:
            3-array: (x,y,z) global current position representing the robot
        """
        return env.robots[self._robot_idn].states[Pose].get_value()[0]

    def step(self, env, action):
        # Run super method first
        reward, done, info = super().step(env=env, action=action)

        # Update other internal variables
        new_robot_pos = env.robots[self._robot_idn].states[Pose].get_value()[0]
        self._path_length += T.l2_distance(
            self._current_robot_pos[:2], new_robot_pos[:2]
        )
        self._current_robot_pos = new_robot_pos

        return reward, done, info

    @classproperty
    def valid_scene_types(cls):
        # Must be a traversable scene
        return {TraversableScene}

    @classproperty
    def default_termination_config(cls):
        return {
            "max_collisions": 500,
            "max_steps": 500,
            "fall_height": 0.03,
        }

    @classproperty
    def default_reward_config(cls):
        return {
            "r_potential": 1.0,
            "r_collision": 0.1,
            "r_objectgoal": 10.0,
        }
