import gymnasium as gym

from ..logging_utils import logger

class FailAwareWrapper(gym.Wrapper):
    """
    Uniformly catch all exception crashes (e.g. IK Fail) at the outermost layer. Convert thrown code Errors to status code info = {"status": "error"} and terminate the episode, ensuring all peripheral execution scripts do not need to write try-except manually.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self._last_obs = None
    
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        try:
            obs, reward, terminated, truncated, info = super().step(action)
            self._last_obs = obs
            return obs, reward, terminated, truncated, info
        except Exception as e:
            # Record exceptions for traceability and debugging
            logger.error(f"Environment execution interrupted due to exception: {str(e)}")
            
            # Directly trigger the terminated exit mechanism and inject an error flag into info
            return (
                None,            # no obs
                0.0,            # Punitive reward or maintain 0
                True,           # terminated: True, cut off loop
                False,          # truncated
                {
                    "status": "error", 
                    "error_message": f"FailAwareWrapper caught specific exception: {str(e)}",
                    "exception_type": type(e).__name__
                }
            )
