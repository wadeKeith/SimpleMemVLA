# Optional joint-delta limit override for the joint-space dataset variant.
#
# ManiSkill's Panda pd_joint_delta_pos caps every arm joint step at +/-0.1 rad
# (2 rad/s at 20 Hz). Under the official RC episode timeouts (25 steps, minus
# 2-10 frozen cue/empty frames) that speed cap makes the RememberColor tasks
# borderline-infeasible in joint-delta space (the EE oracle's executed joint
# steps peak at ~0.36 rad). Setting MIKASA_JOINT_DELTA_LIMIT (e.g. 0.25)
# widens the COMMAND bound; actual motion stays governed by the controller's
# stiffness/damping/force limits. Applies identically to collection and
# closed-loop eval (both import this package), preserving train/eval parity.
# Unset -> stock ManiSkill behaviour.
import os as _os

_joint_delta_limit = _os.environ.get("MIKASA_JOINT_DELTA_LIMIT")
if _joint_delta_limit:
    _lim = float(_joint_delta_limit)

    from mani_skill.agents.robots.panda.panda import Panda as _Panda

    _orig_controller_configs = _Panda._controller_configs.fget

    def _patched_controller_configs(self):
        configs = _orig_controller_configs(self)
        for _mode in ("pd_joint_delta_pos", "pd_joint_target_delta_pos"):
            _cfg = configs.get(_mode)
            _arm = _cfg.get("arm") if isinstance(_cfg, dict) else getattr(_cfg, "arm", None)
            if _arm is not None:
                _arm.lower = -_lim
                _arm.upper = _lim
        return configs

    _Panda._controller_configs = property(_patched_controller_configs)
