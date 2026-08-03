from __future__ import annotations

from typing import Any

import ray
import torch.distributed as dist

from vime.ray.train_actor import TrainRayActor

from .draft_trainer import ExternalDraftTrainer


def _resolve_feature_ref(value):
    if isinstance(value, ray.ObjectRef):
        return ray.get(value)
    return value


class ExternalDraftRayActor(TrainRayActor):
    def init(self, args, role, with_ref=False, with_opd_teacher=False):
        del with_ref, with_opd_teacher
        super().init(args, role)
        self.trainer = ExternalDraftTrainer(args)
        self.rollout_manager = None
        return self.trainer.last_trained_rollout + 1

    def set_rollout_manager(self, rollout_manager):
        self.rollout_manager = rollout_manager

    def collect_features(self, feature_refs: list[Any], target_lm_head, target_version: str) -> dict[str, int]:
        if target_lm_head is not None:
            self.trainer.sync_target_lm_head(target_lm_head, target_version)
        payloads = []
        for value in feature_refs:
            resolved = _resolve_feature_ref(value)
            if resolved:
                payloads.extend(resolved)
        # Every Draft rank receives the manifests, then owns a deterministic shard.
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        if payloads and len(payloads) < world_size:
            # DDP requires every rank to enter the same backward collectives.
            # Replicate only in this low-sample corner case so no rank is empty.
            owned = [payloads[rank % len(payloads)]]
        else:
            owned = [payload for index, payload in enumerate(payloads) if index % world_size == rank]
        accepted = self.trainer.collect(owned, expected_version=target_version)
        return {
            "accepted": accepted,
            "received": len(owned),
            "queued": self.trainer.queue.count(target_version),
            "rejected_version_mismatch": self.trainer.queue.rejected_version_mismatch,
        }

    def train_draft(self, rollout_id: int):
        return self.trainer.train(rollout_id)

    def prepare_publish_snapshot(self):
        snapshot = self.trainer.prepare_publish_snapshot()
        if snapshot is None:
            return None
        return ray.put(snapshot)

    def save_model(self, rollout_id: int, force_sync: bool = False):
        del force_sync
        dist.barrier()
        result = self.trainer.save_checkpoint(rollout_id)
        dist.barrier()
        return result

    def train(self, rollout_id, rollout_data_ref, external_data=None):
        del rollout_data_ref, external_data
        return self.train_draft(rollout_id)

    def update_weights(self):
        return None

    def sleep(self, tags=None):
        del tags
        return None

    def wake_up(self, tags=None):
        del tags
        return None

    def _get_parallel_config(self):
        return {"world_size": dist.get_world_size(), "rank": dist.get_rank()}
