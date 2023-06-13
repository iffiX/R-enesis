from experiments.multiple.interpolator import Interpolator
from launch.run_experiments import run_multiple_experiments


run_multiple_experiments(
    "experiments/vec_patch_anneal_std_voxcraft/optimize_rl.py",
    [
        {
            "config": {
                "model": {
                    "custom_model_config": {
                        "anneal_func": Interpolator(100, 0, 3000, 1, 0)
                    }
                }
            }
        },
        {
            "config": {
                "model": {
                    "custom_model_config": {
                        "anneal_func": Interpolator(100, 0, 3000, 1, 0.25)
                    }
                }
            }
        },
        {
            "config": {
                "model": {
                    "custom_model_config": {
                        "anneal_func": Interpolator(100, 0, 3000, 1, 0.5)
                    }
                }
            }
        },
        {
            "config": {
                "model": {
                    "custom_model_config": {
                        "anneal_func": Interpolator(100, 0, 3000, 1, 1),
                    }
                }
            }
        },
        {
            "config": {
                "env_config": {"reward_type": "distance_traveled_com"},
                "model": {
                    "custom_model_config": {
                        "anneal_func": Interpolator(100, 0, 3000, 1, 0),
                    }
                },
            }
        },
        {
            "config": {
                "env_config": {"reward_type": "distance_traveled_com"},
                "model": {
                    "custom_model_config": {
                        "anneal_func": Interpolator(100, 0, 3000, 1, 0.25),
                    },
                },
            }
        },
        {
            "config": {
                "env_config": {"reward_type": "distance_traveled_com"},
                "model": {
                    "custom_model_config": {
                        "anneal_func": Interpolator(100, 0, 3000, 1, 0.5),
                    },
                },
            }
        },
        {
            "config": {
                "env_config": {"reward_type": "distance_traveled_com"},
                "model": {
                    "custom_model_config": {
                        "anneal_func": Interpolator(100, 0, 3000, 1, 1),
                    },
                },
            }
        },
    ],
    [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ],
    [
        "linear std multipler 1->0, seed=317276",
        "linear std multipler 1->0.25, seed=317276",
        "linear std multipler 1->0.5, seed=317276",
        "linear std multipler 1->1, seed=317276",
        "linear std multipler 1->0, com, seed=317276",
        "linear std multipler 1->0.25, com, seed=317276",
        "linear std multipler 1->0.5, com, seed=317276",
        "linear std multipler 1->1, com, seed=317276",
    ],
)
