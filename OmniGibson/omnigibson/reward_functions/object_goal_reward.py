from omnigibson.reward_functions.reward_function_base import BaseRewardFunction


class ObjectGoalReward(BaseRewardFunction):
    """
    Object goal reward
    Success reward for reaching the object goal with the robot's base

    Args:
        objectgoal (ObjectGoal): Termination condition for checking whether an object goal is reached
        r_objecttgoal (float): Reward for reaching the object goal
    """

    def __init__(self, objectgoal, r_objectgoal=10.0):
        # Store internal vars
        self._objectgoal = objectgoal
        self._r_objectgoal = r_objectgoal

        # Run super
        super().__init__()

    def _step(self, task, env, action):
        # Reward received the objectgoal success condition is met
        reward = self._r_objectgoal if self._objectgoal.success else 0.0

        return reward, {}
