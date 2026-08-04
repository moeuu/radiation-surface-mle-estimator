"""Streaming linear operators for memory-bounded surface-MLE responses."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class ResponseBlock:
    """Store one observations-by-source response block and global indices."""

    observation_indices: NDArray[np.int64]
    source_indices: NDArray[np.int64]
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate an immutable finite non-negative block."""
        rows = np.asarray(self.observation_indices, dtype=np.int64).reshape(-1)
        columns = np.asarray(self.source_indices, dtype=np.int64).reshape(-1)
        values = np.asarray(self.values, dtype=np.float64)
        if values.shape != (rows.size, columns.size):
            raise ValueError(
                "ResponseBlock values must align with row and column indices."
            )
        if np.any(rows < 0) or np.any(columns < 0):
            raise ValueError("ResponseBlock indices must be non-negative.")
        if np.any(~np.isfinite(values)) or np.any(values < -1.0e-12):
            raise ValueError("ResponseBlock values must be finite and non-negative.")
        rows = np.array(rows, dtype=np.int64, copy=True)
        columns = np.array(columns, dtype=np.int64, copy=True)
        values = np.maximum(np.array(values, dtype=np.float64, copy=True), 0.0)
        rows.setflags(write=False)
        columns.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "observation_indices", rows)
        object.__setattr__(self, "source_indices", columns)
        object.__setattr__(self, "values", values)


@runtime_checkable
class ResponseOperator(Protocol):
    """Define the matrix-free response operations consumed by the solver."""

    observation_shape: tuple[int, ...]
    patch_count: int
    isotope_count: int

    @property
    def observation_count(self) -> int:
        """Return the flattened observation count."""

    @property
    def source_count(self) -> int:
        """Return the flattened patch-isotope count."""

    def iter_blocks(self) -> Iterator[ResponseBlock]:
        """Yield deterministic non-overlapping response blocks."""

    def matvec(self, values: ArrayLike) -> NDArray[np.float64]:
        """Return ``A @ values`` without materializing ``A``."""

    def rmatvec(self, values: ArrayLike) -> NDArray[np.float64]:
        """Return ``A.T @ values`` without materializing ``A``."""

    def row_sums(self) -> NDArray[np.float64]:
        """Return non-negative absolute row sums."""

    def column_sums(self) -> NDArray[np.float64]:
        """Return non-negative absolute column sums."""

    def select_measurements(self, indices: Sequence[int]) -> "ResponseOperator":
        """Return a view containing complete selected measurement rows."""

    def masked_sources(self, mask: ArrayLike) -> "ResponseOperator":
        """Return an operator with excluded source columns set to zero."""


class BlockResponseOperator:
    """Implement a response operator from a restartable block iterator."""

    def __init__(
        self,
        observation_shape: Sequence[int],
        patch_count: int,
        isotope_count: int,
        block_factory: Callable[[], Iterator[ResponseBlock]],
        *,
        diagnostics: dict[str, object] | None = None,
    ) -> None:
        """Store validated dimensions and a deterministic block factory."""
        shape = tuple(int(value) for value in observation_shape)
        if not shape or any(value < 1 for value in shape):
            raise ValueError("observation_shape must contain positive dimensions.")
        if int(patch_count) < 1 or int(isotope_count) < 1:
            raise ValueError("patch_count and isotope_count must be positive.")
        if not callable(block_factory):
            raise TypeError("block_factory must be callable.")
        self.observation_shape = shape
        self.patch_count = int(patch_count)
        self.isotope_count = int(isotope_count)
        self._block_factory = block_factory
        self.diagnostics = {} if diagnostics is None else dict(diagnostics)
        self._row_sums: NDArray[np.float64] | None = None
        self._column_sums: NDArray[np.float64] | None = None

    @property
    def observation_count(self) -> int:
        """Return the flattened observation count."""
        return int(np.prod(self.observation_shape, dtype=np.int64))

    @property
    def source_count(self) -> int:
        """Return the flattened patch-isotope count."""
        return self.patch_count * self.isotope_count

    def iter_blocks(self) -> Iterator[ResponseBlock]:
        """Yield validated blocks inside the declared global dimensions."""
        for block in self._block_factory():
            if not isinstance(block, ResponseBlock):
                raise TypeError("block_factory must yield ResponseBlock instances.")
            if np.any(block.observation_indices >= self.observation_count) or np.any(
                block.source_indices >= self.source_count
            ):
                raise ValueError("ResponseBlock index exceeds operator dimensions.")
            yield block

    def matvec(self, values: ArrayLike) -> NDArray[np.float64]:
        """Return a streamed forward product."""
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.shape != (self.source_count,) or np.any(~np.isfinite(vector)):
            raise ValueError("matvec values must be one finite value per source.")
        result = np.zeros(self.observation_count, dtype=np.float64)
        for block in self.iter_blocks():
            result[block.observation_indices] += (
                block.values @ vector[block.source_indices]
            )
        return result

    def rmatvec(self, values: ArrayLike) -> NDArray[np.float64]:
        """Return a streamed transpose product."""
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.shape != (self.observation_count,) or np.any(~np.isfinite(vector)):
            raise ValueError("rmatvec values must be one finite value per observation.")
        result = np.zeros(self.source_count, dtype=np.float64)
        for block in self.iter_blocks():
            result[block.source_indices] += (
                block.values.T @ vector[block.observation_indices]
            )
        return result

    def row_sums(self) -> NDArray[np.float64]:
        """Return cached streamed absolute row sums."""
        if self._row_sums is None:
            values = np.zeros(self.observation_count, dtype=np.float64)
            for block in self.iter_blocks():
                values[block.observation_indices] += np.sum(block.values, axis=1)
            values.setflags(write=False)
            self._row_sums = values
        return self._row_sums

    def column_sums(self) -> NDArray[np.float64]:
        """Return cached streamed absolute column sums."""
        if self._column_sums is None:
            values = np.zeros(self.source_count, dtype=np.float64)
            for block in self.iter_blocks():
                values[block.source_indices] += np.sum(block.values, axis=0)
            values.setflags(write=False)
            self._column_sums = values
        return self._column_sums

    def materialize(self, *, maximum_bytes: int | None = None) -> NDArray[np.float64]:
        """Materialize the operator for tests and bounded diagnostics only."""
        required = self.observation_count * self.source_count * 8
        if maximum_bytes is not None and required > int(maximum_bytes):
            raise MemoryError(
                f"Materialized response requires {required} bytes, above the limit."
            )
        matrix = np.zeros(
            (self.observation_count, self.source_count),
            dtype=np.float64,
        )
        for block in self.iter_blocks():
            matrix[np.ix_(block.observation_indices, block.source_indices)] += (
                block.values
            )
        return matrix.reshape(
            *self.observation_shape, self.patch_count, self.isotope_count
        )

    def select_measurements(self, indices: Sequence[int]) -> "BlockResponseOperator":
        """Return a compact operator over complete selected measurement rows."""
        if len(self.observation_shape) < 2:
            raise ValueError(
                "Measurement selection requires measurement-first responses."
            )
        selected = np.asarray(tuple(indices), dtype=np.int64).reshape(-1)
        measurement_count = self.observation_shape[0]
        if (
            selected.size == 0
            or np.any(selected < 0)
            or np.any(selected >= measurement_count)
            or np.unique(selected).size != selected.size
        ):
            raise ValueError("Measurement indices must be unique and in range.")
        trailing = int(np.prod(self.observation_shape[1:], dtype=np.int64))
        global_rows = np.concatenate(
            [np.arange(index * trailing, (index + 1) * trailing) for index in selected]
        ).astype(np.int64, copy=False)
        inverse = np.full(self.observation_count, -1, dtype=np.int64)
        inverse[global_rows] = np.arange(global_rows.size, dtype=np.int64)

        def factory() -> Iterator[ResponseBlock]:
            """Yield only selected observation rows with compact indices."""
            for block in self.iter_blocks():
                keep = inverse[block.observation_indices] >= 0
                if not np.any(keep):
                    continue
                yield ResponseBlock(
                    observation_indices=inverse[block.observation_indices[keep]],
                    source_indices=block.source_indices,
                    values=block.values[keep],
                )

        return BlockResponseOperator(
            (selected.size, *self.observation_shape[1:]),
            self.patch_count,
            self.isotope_count,
            factory,
            diagnostics={
                **self.diagnostics,
                "selected_measurements": selected.tolist(),
            },
        )

    def masked_sources(self, mask: ArrayLike) -> "BlockResponseOperator":
        """Return an operator whose excluded columns are exactly zero."""
        values = np.asarray(mask, dtype=bool)
        if values.shape == (self.patch_count, self.isotope_count):
            vector = values.reshape(-1)
        elif values.shape == (self.source_count,):
            vector = values
        else:
            raise ValueError("Source mask must match patches by isotopes.")

        def factory() -> Iterator[ResponseBlock]:
            """Yield source-masked response blocks."""
            for block in self.iter_blocks():
                selected = vector[block.source_indices]
                if not np.any(selected):
                    continue
                yield ResponseBlock(
                    observation_indices=block.observation_indices,
                    source_indices=block.source_indices[selected],
                    values=block.values[:, selected],
                )

        return BlockResponseOperator(
            self.observation_shape,
            self.patch_count,
            self.isotope_count,
            factory,
            diagnostics={**self.diagnostics, "source_masked": True},
        )


def atomic_save_npy(path: str | Path, values: ArrayLike) -> None:
    """Atomically publish one NumPy cache block without replacing valid data."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values, dtype=np.float64), allow_pickle=False)
    try:
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "BlockResponseOperator",
    "ResponseBlock",
    "ResponseOperator",
    "atomic_save_npy",
]
