from __future__ import annotations

import copy

import ray

from vime.ray.actor_group import RayTrainGroup

from .draft_actor import ExternalDraftRayActor


class ExternalDraftTrainGroup(RayTrainGroup):
    def __init__(self, args, pg) -> None:
        draft_args = copy.deepcopy(args)
        draft_args.num_gpus_per_node = int(args.draft_num_gpus_per_node)
        draft_args.offload_train = False
        super().__init__(
            args=draft_args,
            num_nodes=int(args.draft_num_nodes),
            num_gpus_per_node=int(args.draft_num_gpus_per_node),
            pg=pg,
            num_gpus_per_actor=1,
            role="draft",
            actor_cls=ExternalDraftRayActor,
        )
        self.base_args = args
        self.last_train_result = None
        self.last_published_draft_version = -1

    def collect_actor_results(self, actor_results) -> dict[str, int | str]:
        results = [value for value in actor_results if isinstance(value, dict)]
        if not results:
            return {"accepted": 0, "received": 0, "reason": "no_actor_feature_manifests"}
        versions = {str(value["target_weight_version"]) for value in results}
        if len(versions) != 1:
            raise RuntimeError(f"Actor Draft feature versions diverged: {sorted(versions)}")
        target_version = versions.pop()
        feature_refs = [value["draft_features_ref"] for value in results if value.get("draft_features_ref") is not None]
        head_refs = [value["draft_target_lm_head_ref"] for value in results if value.get("draft_target_lm_head_ref") is not None]
        if not head_refs:
            raise RuntimeError("No Actor rank exported the Target LM Head for Draft supervision")
        # Passing the ObjectRef as a top-level Ray argument dereferences it in each Draft worker.
        worker_results = ray.get(
            [
                actor.collect_features.remote(feature_refs, head_refs[0], target_version)
                for actor in self._actor_handlers
            ]
        )
        return {
            "accepted": sum(int(value.get("accepted", 0)) for value in worker_results),
            "received": sum(int(value.get("received", 0)) for value in worker_results),
            "queued": sum(int(value.get("queued", 0)) for value in worker_results),
            "rejected_version_mismatch": sum(
                int(value.get("rejected_version_mismatch", 0)) for value in worker_results
            ),
            "target_weight_version": target_version,
        }

    def train_draft(self, rollout_id: int):
        results = ray.get([actor.train_draft.remote(rollout_id) for actor in self._actor_handlers])
        trained_flags = {int(value.get("trained", 0)) for value in results if isinstance(value, dict)}
        if len(trained_flags) > 1:
            raise RuntimeError(f"Draft ranks disagreed on whether an optimizer step completed: {results}")
        trained = [value for value in results if isinstance(value, dict) and int(value.get("trained", 0))]
        if trained:
            versions = {
                (str(value.get("draft_version")), str(value.get("target_weight_version"))) for value in trained
            }
            if len(versions) != 1:
                raise RuntimeError(f"Draft ranks diverged in version state: {sorted(versions)}")
        self.last_train_result = trained[0] if trained else (results[0] if results else None)
        return self.last_train_result

    def prepare_publish_snapshot(self):
        if not isinstance(self.last_train_result, dict) or not int(self.last_train_result.get("trained", 0)):
            return None
        candidate_version = int(self.last_train_result["draft_version"])
        if candidate_version <= self.last_published_draft_version:
            return None
        results = ray.get([actor.prepare_publish_snapshot.remote() for actor in self._actor_handlers])
        snapshots = [value for value in results if value is not None]
        if len(snapshots) > 1:
            raise RuntimeError("More than one Draft rank produced a global publish snapshot")
        if not snapshots:
            return None
        snapshot_ref = snapshots[0] if isinstance(snapshots[0], ray.ObjectRef) else ray.put(snapshots[0])
        return snapshot_ref, str(candidate_version)

    def mark_published(self, draft_version: str) -> None:
        draft_version = int(draft_version)
        if draft_version < self.last_published_draft_version:
            raise RuntimeError(
                f"Draft publish version regressed: {draft_version} < {self.last_published_draft_version}"
            )
        self.last_published_draft_version = draft_version

    def save_draft(self, rollout_id: int, force_sync: bool = False):
        return ray.get(
            [actor.save_model.remote(rollout_id, force_sync=force_sync) for actor in self._actor_handlers]
        )
