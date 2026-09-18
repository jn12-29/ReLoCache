# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression coverage for ReLoCache's native offload execution changes."""

import mmap
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import torch
from vllm.v1.kv_offload.base import LookupResult, ReqContext, ScheduleEndContext, make_offload_key
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.tiering.example.manager import ExampleSecondaryTierManager
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)


def test_scheduler_unlink_preserves_existing_worker_mapping():
    options = {
        "engine_id": f"relocache-test-{uuid4().hex}",
        "num_blocks": 4,
        "rank": None,
        "kv_bytes_per_block": mmap.PAGESIZE,
        "cpu_page_size": mmap.PAGESIZE,
    }
    worker = SharedOffloadRegion(**options)
    try:
        scheduler = SharedOffloadRegion(**options)
        try:
            scheduler.unlink()
            assert not Path(worker.mmap_path).exists()
            worker.mmap_obj[0] = 17
            assert scheduler.mmap_obj[0] == 17
        finally:
            scheduler.cleanup()
    finally:
        worker.cleanup()


def test_required_promotion_retries_when_primary_is_full():
    """Required secondary hits survive temporary primary write pressure."""
    region = MagicMock()
    view = memoryview(torch.zeros((5, 16), dtype=torch.int8).numpy())
    region.create_kv_memoryview.return_value = view
    primary = CPUPrimaryTierOffloadingManager(num_blocks=5, mmap_region=region)
    tiers = [
        ExampleSecondaryTierManager(
            offloading_spec=MagicMock(), primary_kv_view=view, tier_type="example"
        )
        for _ in range(2)
    ]
    manager = TieringOffloadingManager(primary_tier=primary, secondary_tiers=tiers)
    optional = ReqContext(req_id="optional")
    required = ReqContext(req_id="required", require_load=True)
    manager.on_new_request(optional)
    manager.on_new_request(required)
    occupied = [make_offload_key(str(i).encode(), 0) for i in range(5)]
    assert manager.prepare_store(occupied, optional) is not None
    key, missing = [make_offload_key(str(i).encode(), 0) for i in (10, 11)]
    tiers[0].blocks[key] = True

    assert manager.lookup(key, required) is LookupResult.RETRY
    assert manager.lookup(key, optional) is LookupResult.MISS
    assert manager.lookup(missing, required) is LookupResult.MISS

    manager.complete_store(occupied, optional, success=True)
    step = ScheduleEndContext((), ())
    manager.on_schedule_end(step)
    manager.on_schedule_end(step)
    assert manager.lookup(key, required) is LookupResult.HIT_PENDING
    manager.on_schedule_end(step)
    manager.on_schedule_end(step)
    assert manager.lookup(key, required) is LookupResult.HIT
