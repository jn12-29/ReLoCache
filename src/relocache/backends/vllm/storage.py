"""Read tier residency through native lookup and lookup-batch lifecycle calls."""

from vllm.v1.kv_offload.base import ReqContext, ScheduleEndContext
from vllm.v1.kv_offload.tiering.manager import TieringOffloadingManager


def storage_tiers(manager):
    if isinstance(manager, TieringOffloadingManager):
        return {
            "cpu": manager.primary_tier,
            **{
                f"{tier.tier_type}:{index}": tier
                for index, tier in enumerate(manager.secondary_tiers)
            },
        }
    return {"cpu": manager}


def inspect_tiers(manager, keys, request_id):
    context = ReqContext(request_id)
    result = {}
    for name, tier in storage_tiers(manager).items():
        result[name] = [tier.lookup(key, context).name for key in keys]
        # Flush deferred existence checks even while generation is idle.
        tier.on_schedule_end(ScheduleEndContext((), ()))
    return result


def finish_inspection(manager, request_id):
    for tier in storage_tiers(manager).values():
        tier.on_request_finished(ReqContext(request_id))
