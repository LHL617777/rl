# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import torch
import swanlab
from omegaconf import DictConfig

from torchrl.record.loggers.common import Logger

__all__ = ["SwanLabLogger"]


class SwanLabLogger(Logger):
    """SwanLab logger for tracking experiments.
    
    Args:
        log_dir (str): Directory where logs will be stored.
        exp_name (str): Name of the experiment.
        **kwargs: Additional arguments passed to swanlab.init.
    """
    def __init__(
        self,
        exp_name: str,
        save_dir: str | None = None,
        id: str | None = None,
        project: str | None = None,
        *,
        video_fps: int = 32,
        **kwargs,
    ) -> None:
        log_dir = kwargs.pop("log_dir", None)
        if save_dir and log_dir:
            raise ValueError(
                "log_dir and save_dir point to the same value in "
                "SwanLabLogger. Both cannot be specified."
            )
        save_dir = save_dir if save_dir and not log_dir else log_dir
        self.save_dir = save_dir
        self.id = id
        self.project = project
        self.video_fps = video_fps
        self._swanlab_kwargs = {
            "experiment_name": exp_name,
            "logdir": save_dir,
            "id": id,
            "project": project,
            **kwargs,
        }

        super().__init__(exp_name=exp_name, log_dir=save_dir)
    
    def _create_experiment(self) -> Any:
        """Creates a swanlab experiment.
        
        Returns:
            The swanlab experiment object.
        """
        # SwanLab uses project name and run name, so we'll use exp_name as run name
        experiment = swanlab.init(**self._swanlab_kwargs)
        return experiment
    
    def log_scalar(self, name: str, value: float, step: int | None = None) -> None:
        """Logs a scalar value.
        
        Args:
            name: Name of the scalar.
            value: Value of the scalar.
            step: Step at which the scalar is logged.
        """
        if step is not None:
            swanlab.log({name: value}, step=step)
        else:
            swanlab.log({name: value})
    def log_video(self, name: str, video: torch.Tensor, **kwargs) -> None:
        """Log videos inputs to wandb.

        Args:
            name (str): The name of the video.
            video (Tensor): The video to be logged.
            **kwargs: Other keyword arguments. By construction, log_video
                supports 'step' (integer indicating the step index), 'format'
                (default is 'mp4') and 'fps' (defaults to ``self.video_fps``). Other kwargs are
                passed as-is to the :obj:`experiment.log` method.
        """
        import wandb
        fps = kwargs.pop("fps", self.video_fps)
        wandb_video = wandb.Video(video, fps=fps, format='gif')
        log_data = {name: swanlab.Video(wandb_video._path)}
        self.experiment.log(
            {name: log_data},
            **kwargs,
        )
    
    def log_hparams(self, cfg: DictConfig | dict) -> None:
        """Logs hyperparameters.
        
        Args:
            cfg: Hyperparameters to log.
        """
        # SwanLab automatically logs config during init
        pass
    
    def log_histogram(self, name: str, data: Sequence, **kwargs) -> None:
        """Logs a histogram.
        
        Args:
            name: Name of the histogram.
            data: Data to create the histogram from.
            **kwargs: Additional arguments passed to swanlab.log.
        """
        swanlab.log({name: swanlab.Histogram(data, **kwargs)})
    
    def __repr__(self) -> str:
        return f"SwanLabLogger(exp_name={self.exp_name}, log_dir={self.log_dir})"
    
    def finish(self) -> None:
        """Finish the swanlab experiment."""
        if hasattr(swanlab, "finish"):
            swanlab.finish()