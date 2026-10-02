"""
Distributed Execution Module for GNN

Provides Ray and Dask-based parallel dispatching for script execution and parameter sweeps.
Includes robust retry semantics for node failure in external cloud instances.
"""

import logging
import math
import os
import time
from typing import Any, Callable, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

DEFAULT_WAIT_TIMEOUT_SECONDS = 7200
WAIT_TIMEOUT_ENV = "GNN_DISTRIBUTED_WAIT_TIMEOUT"
TRANSFER_POLL_SECONDS = 0.1


def _resolve_wait_timeout() -> int:
    """Resolve the bounded distributed-wait timeout (seconds) from the env."""
    raw = os.environ.get(WAIT_TIMEOUT_ENV)
    if raw is None:
        return DEFAULT_WAIT_TIMEOUT_SECONDS
    try:
        wait_timeout = int(raw)
        if wait_timeout <= 0:
            raise ValueError(raw)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; using default of %ss.",
            WAIT_TIMEOUT_ENV,
            raw,
            DEFAULT_WAIT_TIMEOUT_SECONDS,
        )
        return DEFAULT_WAIT_TIMEOUT_SECONDS
    return wait_timeout


def _wait_timeout_failures(count: int, wait_timeout: float) -> List[Dict[str, Any]]:
    """Failure records for executions that outlived the distributed wait window."""
    error = (
        "Execution did not finish within the distributed wait timeout "
        f"({wait_timeout:.3f}s effective budget); review {WAIT_TIMEOUT_ENV} and the invocation deadline if a longer window is needed"
    )
    return [
        {
            "success": False,
            "error": error,
            "error_type": "DistributedWaitTimeout",
            "collection_timeout_seconds": wait_timeout,
        }
        for _ in range(count)
    ]


class Dispatcher:
    """
    Dispatcher for distributed parameter sweeps and script execution.
    Supports both Ray and Dask backends.
    """

    def __init__(
        self,
        backend: Literal["ray", "dask"] = "ray",
        address: Optional[str] = None,
        num_cpus: Optional[int] = None,
        max_retries: int = 3,
        deadline_monotonic: float | None = None,
        client: Any = None,
    ) -> None:
        """Initialize connection to distributed cluster."""
        self.backend: str = backend
        self.address = address
        self.num_cpus = num_cpus
        self.max_retries = max_retries
        if deadline_monotonic is not None and (
            isinstance(deadline_monotonic, bool)
            or not math.isfinite(deadline_monotonic)
        ):
            raise ValueError("deadline_monotonic must be finite")
        self.deadline_monotonic = deadline_monotonic
        self._initialized = False
        self.client: Any = client
        self._owns_client = False
        self._owned_cluster: Any = None
        self._owns_ray = False

    def connect_to_cluster(self, *, deadline_monotonic: float | None = None) -> bool:
        """Adopt a ready cluster or start one inside the supervised pipeline.

        Direct callers must provide a ready Dask client or initialize Ray in
        their own process boundary. Backend startup has no portable interrupt
        API; only pipeline child-process supervision supplies a hard deadline.
        """
        from gnn.pipeline.run_context import current_run_context

        deadline, budget = self._collection_deadline(deadline_monotonic)
        if budget <= 0:
            return False
        if self.backend == "dask" and self.client is not None:
            self._initialized = getattr(self.client, "status", None) == "running"
            return self._initialized
        context = current_run_context()
        if self.backend == "ray":
            try:
                import ray
            except ImportError:
                return False
            if time.monotonic() >= deadline:
                return False
            if ray.is_initialized():
                self._initialized = True
                return True
        if context is None or context.remaining_seconds() == 0:
            logger.error(
                "Distributed startup requires a ready caller-owned cluster or supervised pipeline context"
            )
            return False
        if self.backend == "ray":
            try:
                import ray

                if not ray.is_initialized():
                    if time.monotonic() >= deadline:
                        return False
                    ray.init(
                        address=self.address,
                        num_cpus=self.num_cpus,
                        ignore_reinit_error=True,
                    )
                    self._owns_ray = True
                self._initialized = True
                if time.monotonic() >= deadline:
                    self.shutdown(deadline_monotonic=deadline)
                    return False
                logger.info(
                    f"Successfully connected to Ray cluster (Active Nodes: {len(ray.nodes())})"
                )
                return True
            except ImportError:
                logger.warning("Ray is not installed. Run: pip install ray")
                return False
            except Exception as e:
                logger.error(f"Failed to initialize Ray: {e}")
                return False
        elif self.backend == "dask":
            try:
                from dask.distributed import Client, LocalCluster

                if time.monotonic() >= deadline:
                    return False
                if self.address:
                    self.client = Client(self.address)
                    self._owns_client = True
                else:
                    cluster = LocalCluster(
                        n_workers=self.num_cpus if self.num_cpus else 4
                    )
                    self._owned_cluster = cluster
                    if time.monotonic() >= deadline:
                        self.shutdown(deadline_monotonic=deadline)
                        return False
                    self.client = Client(cluster)
                    self._owns_client = True
                self._initialized = True
                if time.monotonic() >= deadline:
                    self.shutdown(deadline_monotonic=deadline)
                    return False
                logger.info(f"Successfully connected to Dask cluster: {self.client}")
                return True
            except ImportError:
                logger.warning(
                    "Dask is not installed. Run: pip install dask distributed"
                )
                return False
            except Exception as e:
                logger.error(f"Failed to initialize Dask: {e}")
                self.shutdown(deadline_monotonic=deadline)
                return False
        return False

    def shutdown(self, *, deadline_monotonic: float | None = None) -> Any:
        """Close owned resources, bounding Dask cleanup by the invocation budget.

        An externally initialized Ray runtime is caller-owned. Dask client and
        local cluster are distinct resources; a client constructed from a cluster
        does not close that cluster itself. An exhausted deadline queues public
        asynchronous close requests, never another synchronous scheduler wait.
        """
        if deadline_monotonic is not None:
            self._collection_deadline(deadline_monotonic)
        if self.backend == "dask":
            from gnn.pipeline.run_context import current_run_context

            deadline = time.monotonic() + 5.0
            context = current_run_context()
            for limit in (
                self.deadline_monotonic,
                deadline_monotonic,
                context.deadline_monotonic if context else None,
            ):
                if limit is not None:
                    deadline = min(deadline, limit)
            for resource in (
                self.client if self._owns_client else None,
                self._owned_cluster,
            ):
                if resource is None:
                    continue
                remaining = deadline - time.monotonic()
                try:
                    if remaining <= 0:
                        resource.loop.add_callback(resource.close)
                    else:
                        resource.close(timeout=remaining)
                except (TimeoutError, RuntimeError, OSError) as exc:
                    logger.warning("Dask resource cleanup did not settle: %s", exc)
                    try:
                        resource.loop.add_callback(resource.close)
                    except (RuntimeError, OSError) as queue_exc:
                        logger.warning("Dask cleanup request failed: %s", queue_exc)
            self._initialized = False
            return None
        if self._initialized and self._owns_ray:
            try:
                if self.backend == "ray":
                    import ray

                    from gnn.pipeline.run_context import current_run_context

                    context = current_run_context()
                    if (
                        context is not None
                        and context.remaining_seconds() != 0
                        and self._collection_deadline(deadline_monotonic)[1] > 0
                    ):
                        ray.shutdown()
                    else:
                        logger.warning(
                            "Owned Ray cleanup requires pipeline process supervision after deadline expiry"
                        )
                self._initialized = False
                self._owns_ray = False
            except Exception as e:  # noqa: BLE001
                logger.warning("Distributed backend shutdown failed: %s", e)
        elif self.backend == "ray":
            self._initialized = False

    def _collection_deadline(
        self, deadline_monotonic: float | None = None
    ) -> tuple[float, float]:
        """Intersect backend collection and invocation budgets once."""
        from gnn.pipeline.run_context import current_run_context

        if deadline_monotonic is not None and (
            isinstance(deadline_monotonic, bool)
            or not math.isfinite(deadline_monotonic)
        ):
            raise ValueError("deadline_monotonic must be finite")
        timeout = _resolve_wait_timeout()
        started = time.monotonic()
        limits = [started + timeout]
        if self.deadline_monotonic is not None:
            limits.append(self.deadline_monotonic)
        if deadline_monotonic is not None:
            limits.append(deadline_monotonic)
        context = current_run_context()
        if context is not None and context.deadline_monotonic is not None:
            limits.append(context.deadline_monotonic)
        deadline = min(limits)
        return deadline, max(0.0, deadline - started)

    @staticmethod
    def _task_failure(kind: str, exc: BaseException | None = None) -> Dict[str, Any]:
        """A bare per-task receipt, bound to script identity by the processor."""
        return {
            "success": False,
            "error_type": kind,
            "error": str(exc) if exc is not None else "Remote task was cancelled",
            "exception_type": type(exc).__name__ if exc is not None else kind,
        }

    def _initialization_failures(self, count: int) -> List[Dict[str, Any]]:
        return [
            self._task_failure(
                "DistributedInitializationError",
                RuntimeError(
                    f"{self.backend} is unavailable or not ready; direct callers must supply a caller-owned ready cluster"
                ),
            )
            for _ in range(count)
        ]

    def _ray_get_bounded(
        self, futures: List[Any], *, deadline_monotonic: float | None = None
    ) -> List[Any]:
        """Collect each Ray result independently under one monotonic deadline.

        Wait readiness does not grant an unbounded data transfer. A sibling's
        exception or cancellation does not discard successful task receipts.
        """
        if not futures:
            return []
        deadline, timeout = self._collection_deadline(deadline_monotonic)
        import ray

        results: dict[int, Any] = {}
        pending = list(range(len(futures)))
        while pending and (remaining := deadline - time.monotonic()) > 0:
            try:
                ready, _ = ray.wait(
                    [futures[i] for i in pending],
                    num_returns=len(pending),
                    timeout=0,
                    fetch_local=False,
                )
                if not ready:
                    ready, _ = ray.wait(
                        [futures[i] for i in pending],
                        num_returns=1,
                        timeout=min(remaining, TRANSFER_POLL_SECONDS),
                        fetch_local=False,
                    )
            except ray.exceptions.RayError as exc:
                for i in pending:
                    results[i] = self._task_failure("DistributedTransportError", exc)
                    try:
                        ray.cancel(futures[i], recursive=True)
                    except ray.exceptions.RayError:
                        pass
                pending.clear()
                break
            if not ready:
                continue
            quantum = min(TRANSFER_POLL_SECONDS, remaining / len(pending))
            for i in tuple(pending):
                if futures[i] not in ready:
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    results[i] = ray.get(futures[i], timeout=min(remaining, quantum))
                except ray.exceptions.GetTimeoutError:
                    # Fetch can time out after the scheduler reported readiness.
                    continue
                except ray.exceptions.TaskCancelledError as exc:
                    results[i] = self._task_failure("DistributedTaskCancelled", exc)
                except ray.exceptions.RayError as exc:
                    results[i] = self._task_failure("DistributedTaskError", exc)
                pending.remove(i)
        for i in pending:
            try:
                # A request, never a new wait window or an assertion of worker death.
                ray.cancel(futures[i], recursive=True)
            except ray.exceptions.RayError as exc:
                logger.warning("Ray cancellation request failed: %s", exc)
            results[i] = _wait_timeout_failures(1, timeout)[0]
        return [results[i] for i in range(len(futures))]

    def _cancel_dask(self, futures: List[Any]) -> None:
        """Queue cancellation without blocking beyond the exhausted deadline."""
        if not futures:
            return
        # Client.cancel normally performs a synchronous scheduler round trip.
        # Tornado runs the public asynchronous call on the client's own loop.
        try:
            self.client.loop.add_callback(
                self.client.cancel,
                futures,
                asynchronous=True,
                reason="GNN distributed collection deadline",
            )
        except (RuntimeError, OSError) as exc:
            logger.warning("Dask cancellation request failed: %s", exc)

    def _dask_gather_bounded(
        self, futures: List[Any], *, deadline_monotonic: float | None = None
    ) -> List[Any]:
        """Preserve ordered sibling results across errors, cancellation and loss.

        ``lost`` is recoverable scheduler state, so it remains pending until
        recovery or the shared deadline. Status snapshots classify each future
        once per iteration; every finished/error result transfer is also bounded.
        There is no unbounded gather fallback when distributed is unavailable.
        """
        if not futures:
            return []
        deadline, timeout = self._collection_deadline(deadline_monotonic)
        try:
            from dask.distributed import wait as dask_wait
            from distributed.client import (
                FutureCancelledError,
                FuturesCancelledError,
            )
        except ImportError as exc:
            return [
                self._task_failure("DistributedBackendUnavailable", exc)
                for _ in futures
            ]

        results: dict[int, Any] = {}
        pending = list(range(len(futures)))
        while pending and (remaining := deadline - time.monotonic()) > 0:
            states = {i: futures[i].status for i in pending}
            quantum = min(TRANSFER_POLL_SECONDS, remaining / len(pending))
            for i, status in states.items():
                if status == "cancelled":
                    results[i] = self._task_failure("DistributedTaskCancelled")
                elif status in {"finished", "error"}:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        results[i] = futures[i].result(timeout=min(remaining, quantum))
                    except TimeoutError as exc:
                        # A finished result can be lost during transfer and retry.
                        if futures[i].status == "error":
                            results[i] = self._task_failure("DistributedTaskError", exc)
                        else:
                            continue
                    except FutureCancelledError as exc:
                        results[i] = self._task_failure("DistributedTaskCancelled", exc)
                    except Exception as exc:  # noqa: BLE001 - remote Future.result boundary
                        results[i] = self._task_failure("DistributedTaskError", exc)
                else:
                    continue
                pending.remove(i)
            if not pending or (remaining := deadline - time.monotonic()) <= 0:
                break
            try:
                with self.client.as_current():
                    dask_wait(
                        [futures[i] for i in pending],
                        timeout=min(remaining, TRANSFER_POLL_SECONDS),
                        return_when="FIRST_COMPLETED",
                    )
            except (TimeoutError, FuturesCancelledError):
                # A cancelled sibling makes wait raise even with successful siblings.
                # Reclassify, retrieve independent results, and keep the same deadline.
                continue
            except OSError as exc:
                self._cancel_dask([futures[i] for i in pending])
                for i in pending:
                    results[i] = self._task_failure("DistributedTransportError", exc)
                pending.clear()
                break
        self._cancel_dask([futures[i] for i in pending])
        for i in pending:
            results[i] = _wait_timeout_failures(1, timeout)[0]
        return [results[i] for i in range(len(futures))]

    def _submit_dask(
        self,
        fn: Callable,
        calls: List[tuple[tuple[Any, ...], Dict[str, Any]]],
        *,
        deadline_monotonic: float,
        timeout_seconds: float,
    ) -> List[Any]:
        """Bound every submission and retain independent, ordered receipts."""
        receipts: dict[int, Any] = {}
        positions: list[int] = []
        futures = []
        for index, (args, kwargs) in enumerate(calls):
            if time.monotonic() >= deadline_monotonic:
                for position in range(index, len(calls)):
                    receipts[position] = _wait_timeout_failures(1, timeout_seconds)[0]
                    receipts[position]["dispatch_phase"] = "submission"
                break
            try:
                future = self.client.submit(
                    fn, *args, retries=self.max_retries, **kwargs
                )
            except OSError as exc:
                receipts[index] = self._task_failure("DistributedTransportError", exc)
                continue
            except RuntimeError as exc:
                # Dask uses RuntimeError for submit on a closed client. Other
                # runtime setup/serialization errors remain caller failures.
                if getattr(self.client, "status", None) != "closed":
                    raise
                receipts[index] = self._task_failure("DistributedTransportError", exc)
                continue
            positions.append(index)
            futures.append(future)
        for index, result in zip(
            positions,
            self._dask_gather_bounded(futures, deadline_monotonic=deadline_monotonic),
        ):
            receipts[index] = result
        return [receipts[index] for index in range(len(calls))]

    def run_scripts_parallel(
        self, script_infos: List[Dict[str, Any]], execute_fn: Callable, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple scripts in parallel across workers with robust retries.
        """
        if not script_infos:
            return []
        deadline, timeout = self._collection_deadline()
        if timeout <= 0:
            return _wait_timeout_failures(len(script_infos), timeout)
        if not self._initialized and not self.connect_to_cluster(
            deadline_monotonic=deadline
        ):
            if time.monotonic() >= deadline:
                return _wait_timeout_failures(len(script_infos), timeout)
            return self._initialization_failures(len(script_infos))
        if time.monotonic() >= deadline:
            return _wait_timeout_failures(len(script_infos), timeout)

        logger.info(
            f"Dispatching {len(script_infos)} scripts to {self.backend.capitalize()} cluster..."
        )

        if self.backend == "ray":
            import ray

            # Context switch to a remote function with robust retries
            @ray.remote(max_retries=self.max_retries, retry_exceptions=True)
            def _remote_execute(script_info: Any, kwargs_dict: Any) -> Any:
                """Handle remote execute for internal callers."""
                return execute_fn(script_info, **kwargs_dict)

            receipts: dict[int, Any] = {}
            submitted = []
            positions = []
            for index, info in enumerate(script_infos):
                if time.monotonic() >= deadline:
                    for position in range(index, len(script_infos)):
                        receipts[position] = _wait_timeout_failures(1, timeout)[0]
                        receipts[position]["dispatch_phase"] = "submission"
                    break
                try:
                    submitted.append(_remote_execute.remote(info, kwargs))
                    positions.append(index)
                except ray.exceptions.RayError as exc:
                    receipts[index] = self._task_failure(
                        "DistributedTransportError", exc
                    )
            for index, result in zip(
                positions,
                self._ray_get_bounded(submitted, deadline_monotonic=deadline),
            ):
                receipts[index] = result
            return [receipts[index] for index in range(len(script_infos))]

        elif self.backend == "dask":
            return self._submit_dask(
                execute_fn,
                [((info,), kwargs) for info in script_infos],
                deadline_monotonic=deadline,
                timeout_seconds=timeout,
            )

        return []

    def parameter_sweep(
        self, model_fn: Callable, param_grid: List[Dict[str, Any]]
    ) -> List[Any]:
        """
        Execute a parameter sweep with built-in retry semantics.
        """
        if not param_grid:
            return []
        deadline, timeout = self._collection_deadline()
        if timeout <= 0:
            return _wait_timeout_failures(len(param_grid), timeout)
        if not self._initialized and not self.connect_to_cluster(
            deadline_monotonic=deadline
        ):
            if time.monotonic() >= deadline:
                return _wait_timeout_failures(len(param_grid), timeout)
            return self._initialization_failures(len(param_grid))
        if time.monotonic() >= deadline:
            return _wait_timeout_failures(len(param_grid), timeout)

        logger.info(
            f"Dispatching {len(param_grid)} parameter combinations for sweep using {self.backend.capitalize()}..."
        )

        if self.backend == "ray":
            import ray

            @ray.remote(max_retries=self.max_retries, retry_exceptions=True)
            def _remote_eval(params: Any) -> Any:
                """Handle remote eval for internal callers."""
                return model_fn(**params)

            receipts: dict[int, Any] = {}
            submitted = []
            positions = []
            for index, params in enumerate(param_grid):
                if time.monotonic() >= deadline:
                    for position in range(index, len(param_grid)):
                        receipts[position] = _wait_timeout_failures(1, timeout)[0]
                        receipts[position]["dispatch_phase"] = "submission"
                    break
                try:
                    submitted.append(_remote_eval.remote(params))
                    positions.append(index)
                except ray.exceptions.RayError as exc:
                    receipts[index] = self._task_failure(
                        "DistributedTransportError", exc
                    )
            for index, result in zip(
                positions,
                self._ray_get_bounded(submitted, deadline_monotonic=deadline),
            ):
                receipts[index] = result
            return [receipts[index] for index in range(len(param_grid))]

        elif self.backend == "dask":
            return self._submit_dask(
                model_fn,
                [((), params) for params in param_grid],
                deadline_monotonic=deadline,
                timeout_seconds=timeout,
            )

        return []
