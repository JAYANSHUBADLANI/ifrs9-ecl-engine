"""Streaming access to Release 47 Freddie Mac quarterly archives."""

from __future__ import annotations

import io
import os
import re
import zipfile
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .schemas import ORIGINATION_SCHEMA, PERFORMANCE_SCHEMA, RawFileSchema


PathLike = str | os.PathLike[str]
_VINTAGE_PATTERN = re.compile(r"^(?P<year>\d{4})Q(?P<quarter>[1-4])$")
_ORIGINATION_MEMBER_PATTERN = re.compile(r"^orig_(?P<vintage>\d{4}Q[1-4])\.txt$")
_PERFORMANCE_MEMBER_PATTERN = re.compile(r"^perf_(?P<vintage>\d{4}Q[1-4])\.txt$")


class ArchiveIngestionError(ValueError):
    """Base class for invalid source archive content."""


class ArchiveLayoutError(ArchiveIngestionError):
    """Raised when a ZIP does not have the expected Release 47 layout."""


class RowWidthError(ArchiveIngestionError):
    """Raised when a headerless row does not match its exact schema width."""

    def __init__(
        self,
        *,
        archive_path: Path,
        member_name: str,
        row_number: int,
        expected_width: int,
        actual_width: int,
    ) -> None:
        self.archive_path = archive_path
        self.member_name = member_name
        self.row_number = row_number
        self.expected_width = expected_width
        self.actual_width = actual_width
        super().__init__(
            f"{archive_path}:{member_name}:{row_number} has {actual_width} fields; "
            f"expected {expected_width}"
        )


class RawValueError(ArchiveIngestionError):
    """Raised when a required raw value is blank or otherwise unusable."""


class LoanOrderingError(ArchiveIngestionError):
    """Raised when performance loan IDs are not grouped in ascending order."""


@dataclass(frozen=True, slots=True)
class ArchiveMembers:
    """The two data members in one quarterly Release 47 archive."""

    vintage: str
    origination: str
    performance: str


@dataclass(frozen=True, slots=True)
class RawRow:
    """A width-validated raw row with name-based and positional access."""

    schema: RawFileSchema
    values: tuple[str, ...]
    row_number: int

    def __getitem__(self, key: str | int | slice) -> str | tuple[str, ...]:
        if isinstance(key, str):
            return self.values[self.schema.index(key)]
        return self.values[key]

    def get(self, column: str, default: Any = None) -> str | Any:
        """Return a named raw value, or a default for an unknown column."""

        try:
            return self[column]
        except KeyError:
            return default

    def as_dict(self) -> dict[str, str]:
        """Materialize the row as a dictionary when a downstream API needs one."""

        return dict(zip(self.schema.columns, self.values, strict=True))


@dataclass(frozen=True, slots=True)
class OriginationSample:
    """A deterministic file-order prefix of origination rows."""

    rows: tuple[RawRow, ...]
    loan_ids: tuple[str, ...]

    @property
    def loan_id_set(self) -> frozenset[str]:
        return frozenset(self.loan_ids)


@dataclass(slots=True)
class ScanDiagnostics:
    """Live counters for a performance stream."""

    requested_loans: int | None
    loan_order_check_enabled: bool
    physical_rows_scanned: int = 0
    selected_rows_yielded: int = 0
    selected_loans_found: int = 0
    ordering_checks: int = 0
    loan_order_valid: bool | None = None
    early_stop_occurred: bool = False
    completed_archive_scan: bool = False
    selection_complete: bool = False


def locate_vintage_archive(source_root: PathLike, vintage: str) -> Path:
    """Resolve a quarterly archive below the configured dataset root."""

    match = _VINTAGE_PATTERN.fullmatch(vintage)
    if match is None:
        raise ValueError(f"Invalid vintage {vintage!r}; expected YYYYQ1 through YYYYQ4")

    root = Path(source_root).expanduser()
    year_directory = f"historical_data_{match.group('year')}"
    archive_name = f"historical_data_{vintage}.zip"
    candidates = (
        root / year_directory / archive_name,
        root / archive_name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not locate archive for {vintage}; searched: {searched}")


def resolve_archive_members(archive_path: PathLike) -> ArchiveMembers:
    """Validate and return the two flat data member names in a quarterly ZIP."""

    path = Path(archive_path).expanduser()
    try:
        with zipfile.ZipFile(path) as archive:
            regular_members = [info.filename for info in archive.infolist() if not info.is_dir()]
    except zipfile.BadZipFile as exc:
        raise ArchiveLayoutError(f"Not a valid ZIP archive: {path}") from exc

    if len(regular_members) != 2:
        raise ArchiveLayoutError(
            f"{path} contains {len(regular_members)} regular members; expected exactly 2"
        )
    if any(PurePosixPath(name).name != name for name in regular_members):
        raise ArchiveLayoutError(f"{path} must contain two flat members with no directories")

    origination_matches = [
        (name, _ORIGINATION_MEMBER_PATTERN.fullmatch(name)) for name in regular_members
    ]
    performance_matches = [
        (name, _PERFORMANCE_MEMBER_PATTERN.fullmatch(name)) for name in regular_members
    ]
    origination = [(name, match) for name, match in origination_matches if match is not None]
    performance = [(name, match) for name, match in performance_matches if match is not None]

    if len(origination) != 1 or len(performance) != 1:
        raise ArchiveLayoutError(
            f"{path} must contain exactly one orig_YYYYQn.txt and one perf_YYYYQn.txt member"
        )

    origination_name, origination_match = origination[0]
    performance_name, performance_match = performance[0]
    assert origination_match is not None
    assert performance_match is not None
    origination_vintage = origination_match.group("vintage")
    performance_vintage = performance_match.group("vintage")
    if origination_vintage != performance_vintage:
        raise ArchiveLayoutError(
            f"{path} member vintages differ: {origination_vintage} and {performance_vintage}"
        )

    return ArchiveMembers(
        vintage=origination_vintage,
        origination=origination_name,
        performance=performance_name,
    )


def iter_member_rows(
    archive_path: PathLike,
    member_name: str,
    schema: RawFileSchema,
) -> Iterator[RawRow]:
    """Yield decoded rows from one ZIP member without extracting it to disk."""

    path = Path(archive_path).expanduser()

    def rows() -> Iterator[RawRow]:
        try:
            with zipfile.ZipFile(path) as archive:
                try:
                    member = archive.getinfo(member_name)
                except KeyError as exc:
                    raise ArchiveLayoutError(
                        f"{path} does not contain expected member {member_name!r}"
                    ) from exc

                with archive.open(member, mode="r") as binary_stream:
                    with io.TextIOWrapper(
                        binary_stream,
                        encoding="utf-8",
                        errors="strict",
                        newline="",
                    ) as text_stream:
                        for row_number, line in enumerate(text_stream, start=1):
                            values = tuple(line.rstrip("\r\n").split("|"))
                            if len(values) != schema.width:
                                raise RowWidthError(
                                    archive_path=path,
                                    member_name=member_name,
                                    row_number=row_number,
                                    expected_width=schema.width,
                                    actual_width=len(values),
                                )
                            yield RawRow(
                                schema=schema,
                                values=values,
                                row_number=row_number,
                            )
        except zipfile.BadZipFile as exc:
            raise ArchiveLayoutError(f"Not a valid ZIP archive: {path}") from exc

    return rows()


def iter_origination_rows(archive_path: PathLike) -> Iterator[RawRow]:
    """Yield all origination rows in source-file order."""

    members = resolve_archive_members(archive_path)
    return iter_member_rows(archive_path, members.origination, ORIGINATION_SCHEMA)


def read_origination_sample(archive_path: PathLike, limit: int) -> OriginationSample:
    """Read the first N origination rows deterministically and keep their order."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    if limit == 0:
        return OriginationSample(rows=(), loan_ids=())

    selected_rows: list[RawRow] = []
    loan_ids: list[str] = []
    seen_loan_ids: set[str] = set()
    row_iterator = iter_origination_rows(archive_path)
    try:
        for row in row_iterator:
            loan_id = row["loan_id"]
            assert isinstance(loan_id, str)
            if not loan_id:
                raise RawValueError(
                    f"Origination row {row.row_number} has a blank loan_id"
                )
            if loan_id in seen_loan_ids:
                raise RawValueError(
                    f"Origination row {row.row_number} repeats loan_id {loan_id!r}"
                )
            seen_loan_ids.add(loan_id)
            selected_rows.append(row)
            loan_ids.append(loan_id)
            if len(selected_rows) == limit:
                break
    finally:
        close = getattr(row_iterator, "close", None)
        if close is not None:
            close()

    return OriginationSample(rows=tuple(selected_rows), loan_ids=tuple(loan_ids))


class PerformanceRowStream(Iterator[RawRow]):
    """A single-use performance iterator with live scan diagnostics."""

    def __init__(
        self,
        *,
        archive_path: PathLike,
        member_name: str,
        selected_loan_ids: frozenset[str] | None,
        stop_after_selected: bool,
        validate_loan_order: bool,
    ) -> None:
        self._archive_path = Path(archive_path).expanduser()
        self._member_name = member_name
        self._selected_loan_ids = selected_loan_ids
        self._stop_after_selected = stop_after_selected
        self._validate_loan_order = validate_loan_order
        self.diagnostics = ScanDiagnostics(
            requested_loans=(
                None if selected_loan_ids is None else len(selected_loan_ids)
            ),
            loan_order_check_enabled=validate_loan_order,
            loan_order_valid=True if validate_loan_order else None,
        )
        self._iterator = self._scan()

    def __iter__(self) -> PerformanceRowStream:
        return self

    def __next__(self) -> RawRow:
        return next(self._iterator)

    def __enter__(self) -> PerformanceRowStream:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying ZIP stream if iteration ended early by the caller."""

        self._iterator.close()

    def _scan(self) -> Iterator[RawRow]:
        selected = self._selected_loan_ids
        diagnostics = self.diagnostics
        if selected is not None and not selected:
            diagnostics.selection_complete = True
            return

        selected_seen: set[str] = set()
        previous_loan_id: str | None = None
        maximum_selected_id = max(selected) if selected else None
        row_iterator = iter_member_rows(
            self._archive_path,
            self._member_name,
            PERFORMANCE_SCHEMA,
        )
        try:
            for row in row_iterator:
                diagnostics.physical_rows_scanned += 1
                loan_id = row["loan_id"]
                assert isinstance(loan_id, str)
                if not loan_id:
                    raise RawValueError(
                        f"Performance row {row.row_number} has a blank loan_id"
                    )

                if self._validate_loan_order and previous_loan_id is not None:
                    diagnostics.ordering_checks += 1
                    if loan_id < previous_loan_id:
                        diagnostics.loan_order_valid = False
                        raise LoanOrderingError(
                            f"Performance loan IDs are not nondecreasing at row "
                            f"{row.row_number}: {loan_id!r} follows {previous_loan_id!r}"
                        )
                previous_loan_id = loan_id

                if selected is None or loan_id in selected:
                    if selected is not None and loan_id not in selected_seen:
                        selected_seen.add(loan_id)
                        diagnostics.selected_loans_found = len(selected_seen)
                    diagnostics.selected_rows_yielded += 1
                    yield row
                    continue

                can_stop = (
                    self._stop_after_selected
                    and self._validate_loan_order
                    and maximum_selected_id is not None
                    and len(selected_seen) == len(selected)
                    and loan_id > maximum_selected_id
                )
                if can_stop:
                    diagnostics.selection_complete = True
                    diagnostics.early_stop_occurred = True
                    return

            diagnostics.completed_archive_scan = True
            diagnostics.selection_complete = selected is None or len(selected_seen) == len(selected)
        finally:
            close = getattr(row_iterator, "close", None)
            if close is not None:
                close()


def iter_performance_rows(
    archive_path: PathLike,
    selected_loan_ids: Collection[str] | None = None,
    *,
    stop_after_selected: bool = False,
    validate_loan_order: bool = True,
) -> PerformanceRowStream:
    """Stream selected performance rows and expose physical-scan diagnostics.

    When ``stop_after_selected`` is true, an early stop is used only while the
    observed loan IDs remain nondecreasing and every requested loan has been
    seen. If order validation is disabled, the iterator safely scans the full
    member instead of applying the optimization.
    """

    if isinstance(selected_loan_ids, (str, bytes)):
        raise TypeError("selected_loan_ids must be a collection of complete loan IDs")
    normalized_ids: frozenset[str] | None
    if selected_loan_ids is None:
        normalized_ids = None
    else:
        normalized_ids = frozenset(selected_loan_ids)
        if any(not isinstance(loan_id, str) or not loan_id for loan_id in normalized_ids):
            raise ValueError("selected_loan_ids must contain non-empty strings")

    members = resolve_archive_members(archive_path)
    return PerformanceRowStream(
        archive_path=archive_path,
        member_name=members.performance,
        selected_loan_ids=normalized_ids,
        stop_after_selected=stop_after_selected,
        validate_loan_order=validate_loan_order,
    )

