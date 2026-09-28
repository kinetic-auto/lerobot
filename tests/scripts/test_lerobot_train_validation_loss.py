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

from contextlib import nullcontext

import pytest
import torch

from lerobot.scripts.lerobot_train import _compute_validation_loss


class _Accelerator:
    device = torch.device("cpu")

    def unwrap_model(self, policy):
        return policy

    def autocast(self):
        return nullcontext()

    def gather_for_metrics(self, value):
        return value


class _Policy(torch.nn.Module):
    validation_loss_in_train_mode = False

    def __init__(self):
        super().__init__()
        self.modes = []

    def forward(self, batch):
        self.modes.append(self.training)
        return batch["loss"].float().mean(), {}


def test_compute_validation_loss_means_batches_and_restores_train_mode():
    policy = _Policy()
    policy.train()
    loss = _compute_validation_loss(
        policy,
        [{"loss": torch.tensor([1.0, 3.0])}, {"loss": torch.tensor([5.0])}],
        _Accelerator(),
        lambda batch: batch,
    )
    assert loss == 3.0
    assert policy.modes == [False, False]
    assert policy.training


def test_compute_validation_loss_supports_train_mode_policy():
    policy = _Policy()
    policy.validation_loss_in_train_mode = True
    _compute_validation_loss(
        policy,
        [{"loss": torch.tensor([1.0])}],
        _Accelerator(),
        lambda batch: batch,
    )
    assert policy.modes == [True]
    assert policy.training


def test_compute_validation_loss_rejects_empty_loader():
    with pytest.raises(ValueError, match="empty"):
        _compute_validation_loss(_Policy(), [], _Accelerator(), lambda batch: batch)
