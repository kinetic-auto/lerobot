# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""KineticACT policy.

Extends ACT (https://huggingface.co/papers/2304.13705) with depth cameras and per-timestep object distance,
stage and object mask heads.
"""

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.act.modeling_act import ACT, ACTPolicy, ACTTemporalEnsembler
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from .configuration_kinetic_act import KineticACTConfig


class KineticACTPolicy(ACTPolicy):
    config_class = KineticACTConfig
    name = "kinetic_act"

    def __init__(self, config: KineticACTConfig, **kwargs):
        # ACTPolicy.__init__ hard-codes self.model to `ACT`.
        # So skip super().__init__ and call PreTrainedPolicy.__init__ directly to build `KineticACT`.
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config

        self.model = KineticACT(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        batch = self._add_camera_image_list(batch)

        predicted_actions, (latent_mean, latent_log_variance), auxiliary_outputs = self.model(batch)

        action_l1_loss = self._compute_action_l1_loss(batch, predicted_actions)
        loss = action_l1_loss
        loss_dict = {"l1_loss": action_l1_loss.item()}

        if self.config.use_vae and latent_log_variance is not None:
            latent_kl_divergence_loss = self._compute_latent_kl_divergence_loss(
                latent_mean, latent_log_variance
            )
            loss = loss + latent_kl_divergence_loss * self.config.kl_weight
            loss_dict["kld_loss"] = latent_kl_divergence_loss.item()

        if self.config.object_distance_label_key is not None:
            object_distance_l1_loss = self._compute_object_distance_l1_loss(batch, auxiliary_outputs)
            loss = loss + object_distance_l1_loss * self.config.object_distance_loss_weight
            loss_dict["object_distance_l1_loss"] = object_distance_l1_loss.item()

        if self.config.stage_label_key is not None:
            stage_cross_entropy_loss = self._compute_stage_cross_entropy_loss(batch, auxiliary_outputs)
            loss = loss + stage_cross_entropy_loss * self.config.stage_loss_weight
            loss_dict["stage_cross_entropy_loss"] = stage_cross_entropy_loss.item()
            loss_dict["stage_accuracy"] = self._compute_stage_accuracy(batch, auxiliary_outputs)

        if self.config.object_mask_label_key_per_camera:
            object_mask_binary_cross_entropy_loss = self._compute_object_mask_binary_cross_entropy_loss(
                batch, auxiliary_outputs
            )
            loss = loss + object_mask_binary_cross_entropy_loss * self.config.object_mask_loss_weight
            loss_dict["object_mask_binary_cross_entropy_loss"] = object_mask_binary_cross_entropy_loss.item()

        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk_with_auxiliary_outputs(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        self.eval()

        batch = self._add_camera_image_list(batch)

        actions, _, auxiliary_outputs = self.model(batch)
        return actions, auxiliary_outputs

    def _add_camera_image_list(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Add the camera images as a list to the batch.

        Args:
            batch (dict[str, Tensor]): Training batch.

        Returns:
            dict[str, Tensor]: Batch with the camera image list.
        """
        if not self.config.image_features:
            return batch

        batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
        batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        return batch

    def _compute_action_l1_loss(self, batch: dict[str, Tensor], predicted_actions: Tensor) -> Tensor:
        """Compute the action L1 loss.

        Args:
            batch (dict[str, Tensor]): Training batch.
            predicted_actions (Tensor): Predicted actions.

        Returns:
            Tensor: L1 loss.
        """
        absolute_error = F.l1_loss(batch[ACTION], predicted_actions, reduction="none")
        valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
        valid_element_count = valid_mask.sum() * absolute_error.shape[-1]
        return (absolute_error * valid_mask).sum() / valid_element_count.clamp_min(1)

    def _compute_latent_kl_divergence_loss(self, latent_mean: Tensor, latent_log_variance: Tensor) -> Tensor:
        """Compute the latent KL divergence loss.

        Args:
            latent_mean (Tensor): Latent mean.
            latent_log_variance (Tensor): Latent log variance.

        Returns:
            Tensor: KL divergence loss.
        """
        # Calculate Dₖₗ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
        # each dimension independently, we sum over the latent dimension to get the total
        # KL-divergence per batch element, then take the mean over the batch.
        # (See App. B of https://huggingface.co/papers/1312.6114 for more details).
        return (
            (-0.5 * (1 + latent_log_variance - latent_mean.pow(2) - latent_log_variance.exp())).sum(-1).mean()
        )

    def _compute_object_distance_l1_loss(
        self, batch: dict[str, Tensor], auxiliary_outputs: dict[str, Tensor]
    ) -> Tensor:
        """Compute the object distance L1 loss.

        Args:
            batch (dict[str, Tensor]): Training batch.
            auxiliary_outputs (dict[str, Tensor]): Auxiliary outputs.

        Returns:
            Tensor: L1 loss.
        """
        object_distance_label_key = self.config.object_distance_label_key
        predicted_distance = auxiliary_outputs["object_distance"]
        target_distance = (
            batch[object_distance_label_key].to(predicted_distance.dtype).reshape(predicted_distance.shape)
        )
        absolute_error = F.l1_loss(input=predicted_distance, target=target_distance, reduction="none")

        return self._average_over_unpadded_steps(absolute_error, batch[f"{object_distance_label_key}_is_pad"])

    def _compute_stage_cross_entropy_loss(
        self, batch: dict[str, Tensor], auxiliary_outputs: dict[str, Tensor]
    ) -> Tensor:
        """Compute the stage cross-entropy loss.

        Args:
            batch (dict[str, Tensor]): Training batch.
            auxiliary_outputs (dict[str, Tensor]): Auxiliary outputs.

        Returns:
            Tensor: Cross-entropy loss.
        """
        stage_label_key = self.config.stage_label_key
        stage_logits = auxiliary_outputs["stage_logits"]
        target_stage = self._get_stage_labels(batch, stage_logits)
        per_step_cross_entropy = F.cross_entropy(
            stage_logits.flatten(0, 1), target_stage.flatten(), reduction="none"
        ).reshape(target_stage.shape)

        return self._average_over_unpadded_steps(per_step_cross_entropy, batch[f"{stage_label_key}_is_pad"])

    @torch.no_grad()
    def _compute_stage_accuracy(
        self, batch: dict[str, Tensor], auxiliary_outputs: dict[str, Tensor]
    ) -> float:
        """Compute the stage accuracy.

        Args:
            batch (dict[str, Tensor]): Training batch.
            auxiliary_outputs (dict[str, Tensor]): Auxiliary outputs.

        Returns:
            float: Accuracy.
        """
        stage_label_key = self.config.stage_label_key
        stage_logits = auxiliary_outputs["stage_logits"]
        target_stage = self._get_stage_labels(batch, stage_logits)
        is_prediction_correct = (stage_logits.argmax(dim=-1) == target_stage).float()

        return self._average_over_unpadded_steps(
            is_prediction_correct, batch[f"{stage_label_key}_is_pad"]
        ).item()

    def _get_stage_labels(self, batch: dict[str, Tensor], stage_logits: Tensor) -> Tensor:
        """Get the stage labels as class indices.

        Args:
            batch (dict[str, Tensor]): Training batch.
            stage_logits (Tensor): Stage logits.

        Returns:
            Tensor: Stage labels.
        """
        return batch[self.config.stage_label_key].reshape(stage_logits.shape[:2]).long()

    def _compute_object_mask_binary_cross_entropy_loss(
        self, batch: dict[str, Tensor], auxiliary_outputs: dict[str, Tensor]
    ) -> Tensor:
        """Compute the object mask binary cross-entropy loss.

        Args:
            batch (dict[str, Tensor]): Training batch.
            auxiliary_outputs (dict[str, Tensor]): Auxiliary outputs.

        Returns:
            Tensor: Binary cross-entropy loss.
        """
        per_camera_losses = []
        for mask_label_key in self.config.object_mask_label_key_per_camera.values():
            mask_logits = auxiliary_outputs[mask_label_key].float()
            positive_class_weight = mask_logits.new_tensor(self.config.object_mask_positive_class_weight)
            target_mask = F.adaptive_avg_pool2d(
                batch[mask_label_key].float().unsqueeze(1), mask_logits.shape[-2:]
            ).squeeze(1)
            per_camera_losses.append(
                F.binary_cross_entropy_with_logits(mask_logits, target_mask, pos_weight=positive_class_weight)
            )

        return torch.stack(per_camera_losses).mean()

    @staticmethod
    def _average_over_unpadded_steps(values: Tensor, is_pad: Tensor) -> Tensor:
        """Average the values over the unpadded steps.

        Args:
            values (Tensor): Per-step values.
            is_pad (Tensor): Padding mask.

        Returns:
            Tensor: Mean value.
        """
        valid_mask = ~is_pad
        return (values * valid_mask).sum() / valid_mask.sum().clamp_min(1)


class KineticACT(ACT):
    def __init__(self, config: KineticACTConfig):
        super().__init__(config)

        if config.object_distance_label_key is not None:
            self.object_distance_head = nn.Linear(config.dim_model, 1)
        if config.stage_label_key is not None:
            self.stage_head = nn.Linear(config.dim_model, config.stage_class_count)
        if config.object_mask_label_key_per_camera:
            # Shared across the supervised cameras.
            self.object_mask_head = nn.Linear(config.dim_model, 1)

        self.camera_keys = list(config.image_features)
        self.single_channel_camera_keys = {
            camera_key for camera_key, feature in config.image_features.items() if feature.shape[0] == 1
        }
        if config.use_camera_identity_embedding:
            self.encoder_camera_identity_embedding = nn.Embedding(len(self.camera_keys), config.dim_model)

    def forward(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None], dict[str, Tensor]]:
        """Run the network on a batch.

        Args:
            batch (dict[str, Tensor]): Batch with the camera image list.

        Returns:
            tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None], dict[str, Tensor]]: Actions, latent mean and log variance, and auxiliary outputs.
        """
        # Encode the latent style variable.
        latent_sample, latent_mean, latent_log_variance = self._encode_latent(batch)

        # Build the encoder inputs.
        (
            encoder_tokens,
            encoder_position_embedding,
            token_slice_per_camera,
            token_grid_shape_per_camera,
        ) = self._build_encoder_inputs(batch, latent_sample)

        # Run the ACT transformer.
        encoder_output, decoder_output = self._run_transformer(encoder_tokens, encoder_position_embedding)

        # Predict the actions.
        actions = self.action_head(decoder_output)

        # Predict the auxiliary outputs.
        auxiliary_outputs = self._predict_auxiliary_outputs(
            encoder_output, decoder_output, token_slice_per_camera, token_grid_shape_per_camera
        )

        return actions, (latent_mean, latent_log_variance), auxiliary_outputs

    def _encode_latent(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Encode the latent style variable.

        Args:
            batch (dict[str, Tensor]): Batch of observations.

        Returns:
            tuple[Tensor, Tensor | None, Tensor | None]: Latent sample, latent mean and latent log variance.
        """
        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]
        device = (batch[OBS_STATE] if OBS_STATE in batch else batch[OBS_IMAGES][0]).device

        if not (self.config.use_vae and ACTION in batch and self.training):
            # When not using the VAE encoder, we set the latent to be all zeros.
            latent_sample = torch.zeros(
                [batch_size, self.config.latent_dim], dtype=torch.float32, device=device
            )
            return latent_sample, None, None

        # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
        class_token_embedding = einops.repeat(
            self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
        )  # (B, 1, D)
        if self.config.robot_state_feature:
            robot_state_embedding = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
            robot_state_embedding = robot_state_embedding.unsqueeze(1)  # (B, 1, D)
        action_embedding = self.vae_encoder_action_input_proj(batch[ACTION])  # (B, S, D)

        if self.config.robot_state_feature:
            vae_encoder_input = [
                class_token_embedding,
                robot_state_embedding,
                action_embedding,
            ]  # (B, S+2, D)
        else:
            vae_encoder_input = [class_token_embedding, action_embedding]
        vae_encoder_input = torch.cat(vae_encoder_input, dim=1)

        position_embedding = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

        # Key padding mask: the cls and robot state tokens are never padding (False).
        leading_token_is_pad = torch.full(
            (batch_size, 2 if self.config.robot_state_feature else 1),
            False,
            device=device,
        )
        key_padding_mask = torch.cat([leading_token_is_pad, batch["action_is_pad"]], dim=1)

        class_token_output = self.vae_encoder(
            vae_encoder_input.permute(1, 0, 2),
            pos_embed=position_embedding.permute(1, 0, 2),
            key_padding_mask=key_padding_mask,
        )[0]  # select the class token, with shape (B, D)
        latent_distribution_parameters = self.vae_encoder_latent_output_proj(class_token_output)
        latent_mean = latent_distribution_parameters[:, : self.config.latent_dim]
        # This is 2log(sigma). Done this way to match the original implementation.
        latent_log_variance = latent_distribution_parameters[:, self.config.latent_dim :]

        # Sample the latent with the reparameterization trick.
        latent_sample = latent_mean + latent_log_variance.div(2).exp() * torch.randn_like(latent_mean)
        return latent_sample, latent_mean, latent_log_variance

    def _build_encoder_inputs(
        self, batch: dict[str, Tensor], latent_sample: Tensor
    ) -> tuple[Tensor, Tensor, dict[str, slice], dict[str, tuple[int, int]]]:
        """Build the encoder tokens and their position embedding.

        Args:
            batch (dict[str, Tensor]): Batch with the camera image list.
            latent_sample (Tensor): Latent sample.

        Returns:
            tuple[Tensor, Tensor, dict[str, slice], dict[str, tuple[int, int]]]: Encoder tokens, position embedding, token slice per camera and token grid shape per camera.
        """
        # Build the encoder tokens.
        encoder_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_position_embedding = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        # Add the robot state.
        if self.config.robot_state_feature:
            encoder_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))

        # Add the environment state.
        if self.config.env_state_feature:
            encoder_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        # Add the camera images.
        token_slice_per_camera: dict[str, slice] = {}
        token_grid_shape_per_camera: dict[str, tuple[int, int]] = {}
        if self.config.image_features:
            for camera_index, (camera_key, image) in enumerate(
                zip(self.camera_keys, batch[OBS_IMAGES], strict=True)
            ):
                # Expand single-channel cameras to three channels.
                if camera_key in self.single_channel_camera_keys:
                    image = image.expand(-1, 3, -1, -1)

                camera_features = self.backbone(image)["feature_map"]
                camera_position_embedding = self.encoder_cam_feat_pos_embed(camera_features).to(
                    dtype=camera_features.dtype
                )

                # Add the camera identity embedding.
                if self.config.use_camera_identity_embedding:
                    camera_identity = self.encoder_camera_identity_embedding.weight[camera_index]
                    camera_position_embedding = camera_position_embedding + camera_identity.view(1, -1, 1, 1)

                # Project the camera features.
                camera_features = self.encoder_img_feat_input_proj(camera_features)

                grid_height, grid_width = camera_features.shape[-2:]
                first_token_index = len(encoder_tokens)
                token_slice_per_camera[camera_key] = slice(
                    first_token_index, first_token_index + grid_height * grid_width
                )
                token_grid_shape_per_camera[camera_key] = (grid_height, grid_width)

                # Rearrange features to (sequence, batch, dim).
                camera_features = einops.rearrange(camera_features, "b c h w -> (h w) b c")
                camera_position_embedding = einops.rearrange(
                    camera_position_embedding, "b c h w -> (h w) b c"
                )

                # Add the camera features
                encoder_tokens.extend(list(camera_features))
                encoder_position_embedding.extend(list(camera_position_embedding))

        return (
            torch.stack(encoder_tokens, dim=0),
            torch.stack(encoder_position_embedding, dim=0),
            token_slice_per_camera,
            token_grid_shape_per_camera,
        )

    def _run_transformer(
        self, encoder_tokens: Tensor, encoder_position_embedding: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Run the transformer encoder and decoder.

        Args:
            encoder_tokens (Tensor): Encoder tokens.
            encoder_position_embedding (Tensor): Encoder position embedding.

        Returns:
            tuple[Tensor, Tensor]: Encoder output and decoder output.
        """
        encoder_output = self.encoder(encoder_tokens, pos_embed=encoder_position_embedding)
        decoder_input = torch.zeros(
            (self.config.chunk_size, encoder_tokens.shape[1], self.config.dim_model),
            dtype=encoder_position_embedding.dtype,
            device=encoder_position_embedding.device,
        )
        decoder_output = self.decoder(
            decoder_input,
            encoder_output,
            encoder_pos_embed=encoder_position_embedding,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # Move back to (B, S, C).
        return encoder_output, decoder_output.transpose(0, 1)

    def _predict_auxiliary_outputs(
        self,
        encoder_output: Tensor,
        decoder_output: Tensor,
        token_slice_per_camera: dict[str, slice],
        token_grid_shape_per_camera: dict[str, tuple[int, int]],
    ) -> dict[str, Tensor]:
        """Predict the object distance, stage logits and object mask logits.

        Args:
            encoder_output (Tensor): Encoder output.
            decoder_output (Tensor): Decoder output.
            token_slice_per_camera (dict[str, slice]): Token slice per camera.
            token_grid_shape_per_camera (dict[str, tuple[int, int]]): Token grid shape per camera.

        Returns:
            dict[str, Tensor]: Auxiliary outputs.
        """
        auxiliary_outputs: dict[str, Tensor] = {}

        # Predict the object distance.
        if self.config.object_distance_label_key is not None:
            auxiliary_outputs["object_distance"] = self.object_distance_head(decoder_output).squeeze(-1)

        # Predict the stage logits.
        if self.config.stage_label_key is not None:
            auxiliary_outputs["stage_logits"] = self.stage_head(decoder_output)

        # Predict the object masks.
        for camera_key, mask_label_key in self.config.object_mask_label_key_per_camera.items():
            grid_height, grid_width = token_grid_shape_per_camera[camera_key]
            mask_logits = self.object_mask_head(encoder_output[token_slice_per_camera[camera_key]]).squeeze(
                -1
            )
            auxiliary_outputs[mask_label_key] = einops.rearrange(
                mask_logits, "(h w) b -> b h w", h=grid_height, w=grid_width
            )

        return auxiliary_outputs
