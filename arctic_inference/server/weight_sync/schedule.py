"""TransferSchedule — static topology computation for weight transfer.

Schedules at **TP-worker granularity**: each sender GPU is assigned a set
of (replica_id, tp_rank) targets.  When enough training GPUs are
available (T >= R * TP), every sender handles exactly one TP worker,
maximising parallelism.

All groups operate in parallel, enabling multi-NIC bandwidth scaling.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TransferGroup:
    """One sender GPU and the TP workers it feeds.

    Each (replica_id, tp_rank) target gets its own independent
    ``world_size=2`` NCCL connection with the sender.
    """

    WORLD_SIZE = 2

    group_id: int
    sender_train_rank: int
    targets: list[tuple[int, int]]  # (replica_id, tp_rank) pairs

    @property
    def world_size(self) -> int:
        return self.WORLD_SIZE

    @property
    def replica_ids(self) -> list[int]:
        """Unique replica IDs served by this group."""
        return sorted(set(r for r, _ in self.targets))


@dataclass
class TransferSchedule:
    """Static transfer plan computed once at initialization.

    Port scheme (global):
        TP worker ``(r, t)`` uses port ``base_port + r * tp + t``.
    """

    training_sharding: str
    training_gpus: int
    inference_replicas: int
    inference_tp: int
    groups: list[TransferGroup] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        training_sharding: str,
        training_gpus: int,
        inference_replicas: int,
        inference_tp: int,
    ) -> TransferSchedule:
        """Assign TP workers to sender GPUs.

        Enumerates all ``inference_replicas * inference_tp`` TP workers
        and round-robin assigns them to
        ``min(training_gpus, total_tp_workers)`` senders.

        For **DP** each training GPU already holds the full model.
        For **FSDP** every rank calls ``full_tensor()`` (collective
        all-gather) per parameter, so after the gather each rank holds
        the full parameter and can independently send to its assigned
        inference targets.  For **dp** and **fsdp**, the number of
        active senders is ``min(training_gpus, total_tp_workers)``.

        For **zero3** every training rank must participate in the
        DeepSpeed ``GatheredParameters`` all-gather collective, so
        ``n_senders = training_gpus`` even when ``T > R*TP``.  Extra
        ranks get an empty ``targets`` list but still iterate
        parameters in lockstep to keep the collective in sync.  To
        keep a single target per sender (so the weights iterable is
        consumed at most once and never replayed while a gather is
        resharded), ``training_gpus >= R*TP`` is required.

        Example — T=8, R=4, TP=2 (8 senders, 8 TP workers)::

            Group 0: GPU 0 → (R0,T0)
            Group 1: GPU 1 → (R0,T1)
            Group 2: GPU 2 → (R1,T0)
            ...
            Group 7: GPU 7 → (R3,T1)

        Example — T=4, R=4, TP=2 (4 senders, 8 TP workers)::

            Group 0: GPU 0 → (R0,T0), (R2,T0)
            Group 1: GPU 1 → (R0,T1), (R2,T1)
            Group 2: GPU 2 → (R1,T0), (R3,T0)
            Group 3: GPU 3 → (R1,T1), (R3,T1)
        """
        if training_sharding not in ("dp", "fsdp", "zero3"):
            raise ValueError(
                f"Unknown training_sharding={training_sharding!r}. "
                f"Use 'dp', 'fsdp', or 'zero3'."
            )

        all_targets = [
            (r, t)
            for r in range(inference_replicas)
            for t in range(inference_tp)
        ]
        n_targets = len(all_targets)

        if training_sharding == "zero3":
            # Every training rank must participate in the ZeRO-3
            # GatheredParameters collective, so each rank is a "sender"
            # even if it has no inference targets to send to.
            if training_gpus < n_targets:
                raise ValueError(
                    f"zero3 requires training_gpus ({training_gpus}) >= "
                    f"inference_replicas * inference_tp ({n_targets}) so "
                    f"each sender has at most one target and the weights "
                    f"iterable is never replayed during a gather."
                )
            n_senders = training_gpus
        else:
            n_senders = min(training_gpus, n_targets)

        assignments: list[list[tuple[int, int]]] = [[] for _ in range(n_senders)]
        for i, target in enumerate(all_targets):
            assignments[i % n_senders].append(target)

        groups = [
            TransferGroup(
                group_id=i,
                sender_train_rank=i,
                targets=targets,
            )
            for i, targets in enumerate(assignments)
        ]

        return cls(
            training_sharding=training_sharding,
            training_gpus=training_gpus,
            inference_replicas=inference_replicas,
            inference_tp=inference_tp,
            groups=groups,
        )

    def port_for_target(self, replica_id: int, tp_rank: int, base_port: int) -> int:
        """Compute the NCCL port for a specific TP worker."""
        return base_port + replica_id * self.inference_tp + tp_rank

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    @property
    def active_sender_ranks(self) -> list[int]:
        """Training GPU ranks that are actively sending."""
        return [g.sender_train_rank for g in self.groups]
