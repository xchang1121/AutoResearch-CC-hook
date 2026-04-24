from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict, Union


class WorkerInterface(ABC):
    """Abstract base class for AutoResearch Workers."""

    @abstractmethod
    async def verify(self, package_data: Union[bytes, str], task_id: str,
                     op_name: str, timeout: int = 300
                     ) -> Tuple[bool, str, Dict[str, Any]]:
        """Correctness verification.

        Returns (success, log_output, artifacts). `artifacts` is
        {relative_path: json_content} for every .json/.jsonl file produced
        under the extract dir during execution.
        """
        pass

    @abstractmethod
    async def profile(self, package_data: bytes, task_id: str, op_name: str,
                      profile_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Performance profile.

        Returns {gen_time, base_time, speedup, roofline_time,
        roofline_speedup, roofline, artifacts, ...}.
        """
        pass
