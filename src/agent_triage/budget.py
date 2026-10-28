"""Bounded run budget shared by the model loop and tool executor."""

from __future__ import annotations

import time
from typing import Callable

from .contracts import BudgetLimits
from .store import BudgetState


class Budget:
    """Counters are cumulative across ``resume``; elapsed time is preserved."""

    def __init__(
        self,
        limits: BudgetLimits,
        *,
        state: BudgetState | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limits = limits
        self.state = state.model_copy(deep=True) if state else BudgetState()
        self.clock = clock
        self._started = clock()
        self._base_elapsed_ms = self.state.elapsed_ms

    # -- time ---------------------------------------------------------------
    def elapsed_ms(self) -> int:
        delta = max(0.0, self.clock() - self._started)
        return self._base_elapsed_ms + int(delta * 1000)

    def remaining_seconds(self) -> float:
        return max(0.0, self.limits.total_seconds - self.elapsed_ms() / 1000.0)

    def snapshot(self) -> BudgetState:
        state = self.state.model_copy(deep=True)
        state.elapsed_ms = self.elapsed_ms()
        return state

    # -- counters -----------------------------------------------------------
    def model_requests_remaining(self) -> int:
        return max(0, self.limits.max_model_requests - self.state.model_requests_used)

    def tool_attempts_remaining(self) -> int:
        return max(0, self.limits.max_tool_attempts - self.state.tool_attempts_used)

    def can_start_model_request(self) -> tuple[bool, str | None]:
        if self.remaining_seconds() <= 0:
            return False, "total_time_exhausted"
        if self.model_requests_remaining() <= 0:
            return False, "model_request_limit"
        return True, None

    def consume_model_request(self) -> None:
        self.state.model_requests_used += 1

    def can_attempt_tool(self) -> tuple[bool, str | None]:
        if self.remaining_seconds() <= 0:
            return False, "total_time_exhausted"
        if self.tool_attempts_remaining() <= 0:
            return False, "tool_attempt_limit"
        return True, None

    def consume_tool_attempt(self) -> None:
        self.state.tool_attempts_used += 1

    def corrections_remaining(self) -> int:
        return max(0, self.limits.max_corrections - self.state.corrections_used)

    def consume_correction(self) -> None:
        self.state.corrections_used += 1

    def per_model_timeout(self) -> float:
        return max(0.1, min(float(self.limits.per_model_seconds), self.remaining_seconds()))

    def per_tool_timeout(self) -> float:
        return max(0.1, min(float(self.limits.per_tool_seconds), self.remaining_seconds()))