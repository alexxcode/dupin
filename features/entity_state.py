"""Estado histórico por entidad — acumulación causal estricta.

`DestState` mantiene los agregados necesarios para las features de comportamiento
del receptor, usando exclusivamente transacciones ya vistas. `FeatureState` es el
almacén de estados que comparten el batch (offline) y el serving (online): esa
compartición es lo que hace imposible el train/serve skew.

Regla causal: una transacción se incorpora al estado SOLO DESPUÉS de haber emitido
su vector de features. Nunca antes. El llamador es responsable de ese orden
(ver build_features.build_features).
"""
from __future__ import annotations

import math
from collections import deque

from features.config import FeatureConfig


class DestState:
    """Estado acumulado de una cuenta receptora (`nameDest`)."""

    __slots__ = ("count", "sum_amt", "sumsq_amt", "last_step", "_events")

    def __init__(self) -> None:
        self.count: int = 0
        self.sum_amt: float = 0.0
        self.sumsq_amt: float = 0.0
        self.last_step: int = -1
        # Eventos (step, amount) dentro de la ventana máxima, en orden ascendente
        # de step. Se evictan los más viejos para acotar memoria.
        self._events: deque[tuple[int, float]] = deque()

    # ── Consultas (solo lectura; reflejan el estado PREVIO a la tx actual) ──
    def mean(self) -> float:
        return self.sum_amt / self.count if self.count else 0.0

    def std(self) -> float:
        """Desviación estándar poblacional. 0.0 si <2 observaciones."""
        if self.count < 2:
            return 0.0
        mean = self.sum_amt / self.count
        var = self.sumsq_amt / self.count - mean * mean
        return math.sqrt(var) if var > 0.0 else 0.0

    def recency(self, step: int) -> int:
        return step - self.last_step

    def window_count(self, step: int, window: int) -> int:
        """Nº de eventos previos con step > (step - window)."""
        threshold = step - window
        n = 0
        for ev_step, _ in reversed(self._events):
            if ev_step > threshold:
                n += 1
            else:
                break
        return n

    def window_sum(self, step: int, window: int) -> float:
        """Suma de montos de eventos previos con step > (step - window)."""
        threshold = step - window
        total = 0.0
        for ev_step, ev_amt in reversed(self._events):
            if ev_step > threshold:
                total += ev_amt
            else:
                break
        return total

    # ── Actualización (incorpora la tx; llamar DESPUÉS de emitir features) ──
    def update(self, step: int, amount: float, max_window: int) -> None:
        self.count += 1
        self.sum_amt += amount
        self.sumsq_amt += amount * amount
        self.last_step = step
        self._events.append((step, amount))
        # Evicción: descarta eventos fuera de la ventana máxima.
        cutoff = step - max_window
        events = self._events
        while events and events[0][0] <= cutoff:
            events.popleft()


class FeatureState:
    """Almacén de estados por entidad. Compartido por batch y serving."""

    __slots__ = ("config", "_dest", "_orig_count")

    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self._dest: dict[str, DestState] = {}
        self._orig_count: dict[str, int] = {}

    def dest_state(self, dest_id: str) -> DestState | None:
        """Estado actual del receptor, o None si nunca se ha visto."""
        return self._dest.get(dest_id)

    def orig_prior_count(self, orig_id: str) -> int:
        return self._orig_count.get(orig_id, 0)

    def update(self, orig_id: str, dest_id: str, step: int, amount: float) -> None:
        """Incorpora la transacción al estado de ambas entidades."""
        ds = self._dest.get(dest_id)
        if ds is None:
            ds = DestState()
            self._dest[dest_id] = ds
        ds.update(step, amount, self.config.max_window)
        self._orig_count[orig_id] = self._orig_count.get(orig_id, 0) + 1
