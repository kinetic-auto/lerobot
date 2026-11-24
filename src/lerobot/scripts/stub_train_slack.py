#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Post a stub training thread to Slack without a dataset or a GPU.

The parent message is the run config. One thread reply is edited at each
eval and test checkpoint, including a single loss plot.

Requires ``SLACK_BOT_TOKEN`` and ``SLACK_CHANNEL``. The bot needs ``chat:write``
and ``files:write``, and it must already be in the channel.
Matplotlib (``lerobot[matplotlib-dep]``) draws the plot; without it the reply
text still updates.

Example:

```shell
SLACK_BOT_TOKEN=xoxb-... SLACK_CHANNEL=C... python -m lerobot.scripts.stub_train_slack
```
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from pathlib import Path

from lerobot.common.train_monitor import SLACK_CHANNEL_ENV, SLACK_TOKEN_ENV, TrainMonitor
from lerobot.configs import NormalizationMode
from lerobot.configs.accelerator import AcceleratorConfig
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.transforms import ImageTransformsConfig

logger = logging.getLogger(__name__)

# Same flags as the single-GPU command in scripts/setup_lambda_train.sh.
# split_ratio is [train, val, test]. Val weight 0 leaves no validation set.
# The 0.1 test set is scored on save steps, which is when lerobot-train
# runs the test dataloader.
_DATASET_ROOT = "/data/datasets/dataset"
_JOB_NAME = "act-chunk100-aug"
_STEPS = 100_000
_LOG_FREQ = 200
_SAVE_FREQ = 10_000
_SPLIT_RATIO = [0.9, 0.0, 0.1]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    token = os.environ.get(SLACK_TOKEN_ENV, "").strip()
    channel = os.environ.get(SLACK_CHANNEL_ENV, "").strip()
    if not token or not channel:
        logger.error("Set %s and %s.", SLACK_TOKEN_ENV, SLACK_CHANNEL_ENV)
        raise SystemExit(1)

    with tempfile.TemporaryDirectory(prefix="stub-train-") as tmp:
        cfg = TrainPipelineConfig(
            dataset=DatasetConfig(
                root=_DATASET_ROOT,
                image_transforms=ImageTransformsConfig(enable=True, max_num_transforms=3),
                split_ratio=_SPLIT_RATIO,
            ),
            policy=ACTConfig(
                device="cpu",
                chunk_size=100,
                n_action_steps=100,
                kl_weight=10.0,
                normalization_mapping={
                    "ACTION": NormalizationMode.MEAN_STD,
                    "STATE": NormalizationMode.MEAN_STD,
                    "VISUAL": NormalizationMode.IDENTITY,
                },
            ),
            job_name=_JOB_NAME,
            backbone="resnet18",
            batch_size=8,
            steps=_STEPS,
            save_freq=_SAVE_FREQ,
            log_freq=_LOG_FREQ,
            accelerator=AcceleratorConfig(mixed_precision="bf16"),
            output_dir=Path(tmp) / "model_zoo" / "dataset" / _JOB_NAME,
        )
        cfg.validate()
        cfg.wandb.enable = False
        cfg.monitor.enable = True
        cfg.monitor.slack_channel = channel
        monitor = TrainMonitor(cfg)
        logger.info("Posting the parent message to %s", channel)
        monitor.start(0)
        for step in range(_LOG_FREQ, _STEPS + 1, _LOG_FREQ):
            train_loss = 1.0 / (step / _LOG_FREQ)
            monitor.log_scalars({"loss": train_loss}, step)
            if step % _SAVE_FREQ != 0:
                continue
            test_loss = train_loss + 0.1
            monitor.log_scalars({"loss": test_loss}, step, mode="test")
            logger.info("step %s train_loss %.4f test_loss %.4f", step, train_loss, test_loss)
            time.sleep(2)
        monitor.finish(_STEPS)
        logger.info("Updated the progress reply through step %s", _STEPS)


if __name__ == "__main__":
    sys.exit(main() or 0)
