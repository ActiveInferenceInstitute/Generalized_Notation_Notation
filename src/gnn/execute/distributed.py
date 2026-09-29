"""
Distributed Execution Module for GNN

Provides Ray and Dask-based parallel dispatching for script execution and parameter sweeps.
Includes robust retry semantics for node failure in external cloud instances.
"""

import logging
import os
from typing import Any, Callable, Dict, List, Literal, Optional, cast

logger = logging.getLogger(__name__)

DEFAULT_WAIT_TIMEOUT_SECONDS = 7200
WAIT_TIMEOUT_ENV = "GNN_DISTRIBUTED_WAIT_TIMEOUT"


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


def _wait_timeout_failures(count: int, wait_timeout: int) -> List[Dict[str, Any]]:
    """Failure records for executions that outlived the distributed wait window."""
    error = (
        "Execution did not finish within the distributed wait timeout "
        f"({wait_timeout}s); raise {WAIT_TIMEOUT_ENV} if a longer window is needed"
    )
    return [
        {"success": False, "error": error, "error_type": "DistributedWaitTimeout"}
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
    ) -> None:
        """Initialize connection to distributed cluster."""
        self.backend: str = backend
        self.address = address
        self.num_cpus = num_cpus
        self.max_retries = max_retries
        self._initialized = False
        self.client: Any = None

    def connect_to_cluster(self) -> bool:
        """Connect to distributed cluster."""
        if self.backend == "ray":
            try:
                import ray

                if not ray.is_initialized():
                    ray.init(
                        address=self.address,
                        num_cpus=self.num_cpus,
                        ignore_reinit_error=True,
                    )
                self._initialized = True
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

                if self.address:
                    self.client = Client(self.address)
                else:
                    cluster = LocalCluster(
                        n_workers=self.num_cpus if self.num_cpus else 4
                    )
                    self.client = Client(cluster)
                self._initialized = True
                logger.info(f"Successfully connected to Dask cluster: {self.client}")
                return True
            except ImportError:
                logger.warning(
                    "Dask is not installed. Run: pip install dask distributed"
                )
                return False
            except Exception as e:
                logger.error(f"Failed to initialize Dask: {e}")
                return False
        return False

    def shutdown(self) -> Any:
        """Shutdown connection."""
        if self._initialized:
            try:
                if self.backend == "ray":
                    import ray

                    ray.shutdown()
                elif self.backend == "dask" and self.client:
                    self.client.close()
                self._initialized = False
            except Exception as e:  # noqa: BLE001
                logger.warning("Distributed backend shutdown failed: %s", e)

    def _ray_get_bounded(self, futures: List[Any]) -> List[Any]:
        """Collect Ray futures under a bounded wait.

        On wait-timeout the outstanding remote tasks are cancelled
        (recursive — they carry max_retries and would otherwise keep
        executing) and each reported as a failed execution.
        """
        import ray

        wait_timeout = _resolve_wait_timeout()
        try:
            return list(ray.get(futures, timeout=wait_timeout))
        except ray.exceptions.GetTimeoutError:
            logger.warning(
                "Ray wait exceeded %ss; cancelling %d outstanding executions.",
                wait_timeout,
                len(futures),
            )
            for future in futures:
                try:
                    ray.cancel(future, recursive=True)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Ray cancel failed: %s", e)
            return list(_wait_timeout_failures(len(futures), wait_timeout))

    def _dask_gather_bounded(self, futures: List[Any]) -> List[Dict[str, Any]]:
        """Gather Dask futures under a bounded wait.

        On wait-timeout the outstanding futures are cancelled and reported
        as failed executions; completed results keep their submission order.
        When dask.distributed is not importable (only possible for
        non-dask clients — a real dask client has dask installed by
        definition) the pre-change unbounded gather applies.
        """
        try:
            from dask.distributed import wait as dask_wait
        except ImportError:
            dask_wait = None
        if dask_wait is None:
            return cast("list[dict[str, Any]]", self.client.gather(futures))
        wait_timeout = _resolve_wait_timeout()
        # dask.distributed.wait resolves the process-global default client, so
        # scope it to this dispatcher's client; on expiry it raises rather than
        # returning the pending set, so derive done/not_done from future status.
        try:
            with self.client.as_current():
                dask_wait(futures, timeout=wait_timeout)
        except TimeoutError:
            pass
        # Read each future's status exactly once: a future that finishes
        # mid-classification must land in exactly one bucket, never in neither
        # (which would drop its result without cancelling it).
        done: List[Any] = []
        not_done: List[Any] = []
        for fut in futures:
            (not_done if fut.status == "pending" else done).append(fut)
        if not not_done:
            return cast("list[dict[str, Any]]", self.client.gather(futures))
        logger.warning(
            "Dask wait exceeded %ss; cancelling %d outstanding executions.",
            wait_timeout,
            len(not_done),
        )
        for future in not_done:
            try:
                future.cancel()
            except Exception as e:  # noqa: BLE001
                logger.warning("Dask cancel failed: %s", e)
        gathered = dict(zip(done, self.client.gather(list(done))))
        return [
            gathered.get(fut, _wait_timeout_failures(1, wait_timeout)[0])
            for fut in futures
        ]

    def run_scripts_parallel(
        self, script_infos: List[Dict[str, Any]], execute_fn: Callable, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple scripts in parallel across workers with robust retries.
        """
        if not self._initialized and not self.connect_to_cluster():
            logger.warning(
                "Falling back to sequential execution due to initialization failure."
            )
            return [execute_fn(info, **kwargs) for info in script_infos]

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

            futures = [_remote_execute.remote(info, kwargs) for info in script_infos]
            return cast("list[dict[str, Any]]", self._ray_get_bounded(futures))

        elif self.backend == "dask":
            futures = [
                self.client.submit(
                    execute_fn,
                    info,
                    retries=self.max_retries,
                    **kwargs,
                )
                for info in script_infos
            ]
            return self._dask_gather_bounded(futures)

        return []

    def parameter_sweep(
        self, model_fn: Callable, param_grid: List[Dict[str, Any]]
    ) -> List[Any]:
        """
        Execute a parameter sweep with built-in retry semantics.
        """
        if not self._initialized and not self.connect_to_cluster():
            logger.warning("Falling back to sequential parameter sweep.")
            return [model_fn(**params) for params in param_grid]

        logger.info(
            f"Dispatching {len(param_grid)} parameter combinations for sweep using {self.backend.capitalize()}..."
        )

        if self.backend == "ray":
            import ray

            @ray.remote(max_retries=self.max_retries, retry_exceptions=True)
            def _remote_eval(params: Any) -> Any:
                """Handle remote eval for internal callers."""
                return model_fn(**params)

            futures = [_remote_eval.remote(p) for p in param_grid]
            return self._ray_get_bounded(futures)

        elif self.backend == "dask":
            futures = [
                self.client.submit(model_fn, retries=self.max_retries, **p)
                for p in param_grid
            ]
            return cast("list[Any]", self._dask_gather_bounded(futures))

        return []
