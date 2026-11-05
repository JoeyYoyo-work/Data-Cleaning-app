"""Clean tabular data from CSV/TSV/TXT/JSON/JSONL/XLSX files,
report data-quality issues, and generate visual summaries."""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)


SUPPORTED_FORMATS = {".csv", ".tsv", ".txt", ".json", ".jsonl", ".ndjson", ".xlsx"}


class ProcessLog:
    """Print and retain a readable log of the cleaning steps."""

    def __init__(self) -> None:
        self.entries: list[str] = []

    def add(self, message: str) -> None:
        self.entries.append(message)
        print(f"[LOG] {message}")


def normalize_column_names(columns: list[Any]) -> tuple[list[str], int]:
    """Normalize column names and make any resulting duplicates unique."""
    normalized: list[str] = []
    used_names: set[str] = set()
    changed = 0

    for position, original in enumerate(columns, start=1):
        name = str(original).strip().lower()
        name = re.sub(r"[^\w]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")

        if not name:
            name = f"column_{position}"

        base_name = name
        suffix = 2
        while name in used_names:
            name = f"{base_name}_{suffix}"
            suffix += 1

        used_names.add(name)
        normalized.append(name)

        if name != str(original):
            changed += 1

    return normalized, changed


def count_outliers(frame: pd.DataFrame, excluded: set[str] | None = None) -> int:
    """Count numeric outlier cells using the 1.5 × IQR rule."""
    excluded = excluded or set()
    total = 0

    for column in frame.columns:
        if column in excluded:
            continue

        series = frame[column]
        if not is_numeric_dtype(series) or is_bool_dtype(series):
            continue

        values = series.dropna()
        if len(values) < 4:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        total += int(((series < lower_bound) | (series > upper_bound)).sum())

    return total


def detect_outlier_rows(
    frame: pd.DataFrame,
    row_id_column: str,
) -> tuple[set[int], int]:
    """Return original row IDs and outlier-cell count for numeric columns."""
    row_ids: set[int] = set()
    outlier_cells = 0

    for column in frame.columns:
        if column == row_id_column:
            continue

        series = frame[column]
        if not is_numeric_dtype(series) or is_bool_dtype(series):
            continue

        values = series.dropna()
        if len(values) < 4:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        mask = ((series < lower_bound) | (series > upper_bound)).fillna(False)

        outlier_cells += int(mask.sum())
        row_ids.update(frame.loc[mask, row_id_column].astype(int).tolist())

    return row_ids, outlier_cells


def load_table(path: Path) -> tuple[pd.DataFrame, str | None]:
    """Load a supported tabular file. Returns the data and any detected text delimiter."""
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig"), ","

    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", encoding="utf-8-sig"), "\t"

    if suffix == ".txt":
        sample = path.read_text(encoding="utf-8-sig")[:8192]
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
        except csv.Error:
            delimiter = None

        if delimiter:
            return pd.read_csv(path, sep=delimiter, encoding="utf-8-sig"), delimiter

        return (
            pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig"),
            None,
        )

    if suffix == ".json":
        return pd.read_json(path), None

    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True), None

    if suffix == ".xlsx":
        return pd.read_excel(path, engine="openpyxl"), None

    raise ValueError(
        f"Unsupported file format: {suffix or '(no extension)'}. "
        f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
    )


def save_table(frame: pd.DataFrame, path: Path, delimiter: str | None = None) -> None:
    """Save cleaned data using the input file's format."""
    suffix = path.suffix.lower()

    if suffix == ".csv":
        frame.to_csv(path, index=False)
    elif suffix == ".tsv":
        frame.to_csv(path, index=False, sep="\t")
    elif suffix == ".txt":
        frame.to_csv(path, index=False, sep=delimiter or ",")
    elif suffix == ".json":
        frame.to_json(path, orient="records", indent=2, date_format="iso")
    elif suffix in {".jsonl", ".ndjson"}:
        frame.to_json(path, orient="records", lines=True, date_format="iso")
    elif suffix == ".xlsx":
        frame.to_excel(path, index=False, engine="openpyxl")
    else:
        raise ValueError(f"Cannot save unsupported file format: {suffix}")


def make_charts(
    issue_counts: dict[str, int],
    problematic_rows: int,
    total_rows: int,
    before_metrics: dict[str, int],
    after_metrics: dict[str, int],
    reports_dir: Path,
) -> None:
    """Save and display the three requested data-quality charts."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Issue counts
    labels = list(issue_counts)
    values = [issue_counts[label] for label in labels]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(labels, values, color="#4472C4")
    ax.set_title("Data Issues Found")
    ax.set_ylabel("Affected items (rows, cells, or columns)")
    ax.tick_params(axis="x", labelrotation=25)
    ax.bar_label(bars, padding=3)
    fig.tight_layout()
    fig.savefig(reports_dir / "issue_counts.png", dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # 2. Pie chart: clean vs problematic rows
    clean_rows = max(total_rows - problematic_rows, 0)
    pie_values = [clean_rows, problematic_rows]

    fig, ax = plt.subplots(figsize=(7, 7))
    if total_rows:
        ax.pie(
            pie_values,
            labels=["No detected issues", "Had one or more issues"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#70AD47", "#ED7D31"],
        )
    else:
        ax.text(0.5, 0.5, "No data rows", ha="center", va="center")
    ax.set_title("Rows: Clean vs. Problematic")
    fig.tight_layout()
    fig.savefig(reports_dir / "clean_vs_problematic.png", dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # 3. Before/after quality comparison
    metric_names = ["Missing cells", "Duplicate rows", "Outlier cells"]
    before_values = [before_metrics[name] for name in metric_names]
    after_values = [after_metrics[name] for name in metric_names]

    x = np.arange(len(metric_names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 6))
    before_bars = ax.bar(x - width / 2, before_values, width, label="Before", color="#A5A5A5")
    after_bars = ax.bar(x + width / 2, after_values, width, label="After", color="#5B9BD5")
    ax.set_title("Data Quality: Before and After Cleaning")
    ax.set_ylabel("Count")
    ax.set_xticks(x, metric_names)
    ax.legend()
    ax.bar_label(before_bars, padding=3)
    ax.bar_label(after_bars, padding=3)
    fig.tight_layout()
    fig.savefig(reports_dir / "before_after_quality.png", dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def clean_file(input_path: Path) -> None:
    """Clean one supported tabular file and save output plus visual reports."""
    log = ProcessLog()

    if input_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported file format: {input_path.suffix or '(no extension)'}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    if not input_path.is_file():
        raise ValueError(f"The provided path is not a file: {input_path}")

    if input_path.stat().st_size == 0:
        raise ValueError("The file is empty.")

    frame, text_delimiter = load_table(input_path)

    if frame.empty:
        raise ValueError("The file contains no data rows.")

    rows_before = len(frame)
    raw_frame = frame.copy(deep=True)
    before_metrics = {
        "Missing cells": int(raw_frame.isna().sum().sum()),
        "Duplicate rows": int(raw_frame.duplicated().sum()),
        "Outlier cells": count_outliers(raw_frame),
    }

    print("\n--- Cleaning process ---")
    print(f"Input file: {input_path}")
    print(f"Rows before cleaning: {rows_before}")

    frame.columns, renamed_columns = normalize_column_names(list(frame.columns))
    log.add(
        f"Column names: normalized {renamed_columns} column(s) "
        "to lowercase snake_case; duplicate normalized names were made unique."
    )

    row_id_column = "__cleaner_original_row_id__"
    while row_id_column in frame.columns:
        row_id_column = f"_{row_id_column}"

    frame.insert(0, row_id_column, np.arange(rows_before))
    row_ids_by_issue: dict[str, set[int]] = defaultdict(set)
    issue_counts: dict[str, int] = {}

    def mark_rows(issue: str, mask: pd.Series) -> set[int]:
        ids = set(frame.loc[mask.fillna(False), row_id_column].astype(int).tolist())
        row_ids_by_issue[issue].update(ids)
        return ids

    data_columns = [column for column in frame.columns if column != row_id_column]

    # Trim whitespace and convert blank strings to missing values.
    whitespace_cells = 0
    whitespace_mask = pd.Series(False, index=frame.index)

    for column in data_columns:
        if not (
            pd.api.types.is_object_dtype(frame[column])
            or pd.api.types.is_string_dtype(frame[column])
        ):
            continue

        original = frame[column]
        trimmed = original.map(lambda value: value.strip() if isinstance(value, str) else value)
        changed = (original.astype("string") != trimmed.astype("string")).fillna(False)

        whitespace_cells += int(changed.sum())
        whitespace_mask |= changed
        frame[column] = trimmed
        frame[column] = frame[column].map(
            lambda value: pd.NA if isinstance(value, str) and value == "" else value
        )

    mark_rows("whitespace", whitespace_mask)
    issue_counts["Whitespace cells"] = whitespace_cells
    log.add(
        f"Whitespace: trimmed {whitespace_cells} cell(s) across "
        f"{int(whitespace_mask.sum())} row(s)."
    )

    # Missing values detection.
    missing_mask = frame[data_columns].isna().any(axis=1)
    missing_rows = mark_rows("missing", missing_mask)
    missing_cells = int(frame[data_columns].isna().sum().sum())
    issue_counts["Missing cells"] = missing_cells
    log.add(
        f"Missing values: found {missing_cells} missing cell(s) in "
        f"{len(missing_rows)} row(s); values will be imputed where possible."
    )

    # Remove rows with no usable values.
    empty_row_mask = frame[data_columns].isna().all(axis=1)
    empty_row_ids = mark_rows("unusable", empty_row_mask)
    unusable_rows_removed = len(empty_row_ids)
    frame = frame.loc[~empty_row_mask].copy()
    issue_counts["Unusable rows removed"] = unusable_rows_removed
    log.add(f"Unusable rows: removed {unusable_rows_removed} completely empty row(s).")

    # Remove duplicates.
    data_columns = [column for column in frame.columns if column != row_id_column]
    duplicate_mask = frame.duplicated(subset=data_columns, keep="first")
    duplicate_ids = mark_rows("duplicates", duplicate_mask)
    duplicate_rows_removed = len(duplicate_ids)
    frame = frame.loc[~duplicate_mask].copy()
    issue_counts["Duplicate rows"] = duplicate_rows_removed
    log.add(f"Duplicates: removed {duplicate_rows_removed} duplicate row(s).")

    # Type conversions.
    malformed_cells = 0
    converted_columns = 0

    for column in [c for c in frame.columns if c != row_id_column]:
        series = frame[column]
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            continue

        non_missing = series.notna()
        if not non_missing.any():
            continue

        text = series.astype("string").str.strip()
        numeric_text = text.str.replace(",", "", regex=False).str.replace("$", "", regex=False)
        numeric_values = pd.to_numeric(numeric_text, errors="coerce")
        numeric_rate = float(numeric_values[non_missing].notna().mean())

        if numeric_rate >= 0.80 and numeric_values[non_missing].notna().any():
            malformed_mask = non_missing & numeric_values.isna()
            malformed_ids = mark_rows("malformed", malformed_mask)
            affected_cells = int(malformed_mask.sum())
            malformed_cells += affected_cells

            frame[column] = numeric_values
            converted_columns += 1
            log.add(
                f"Data types: converted '{column}' to numeric; "
                f"{affected_cells} malformed value(s) in {len(malformed_ids)} row(s) became missing "
                "and will be imputed."
            )
            continue

        looks_like_date = (
            "date" in column
            or "time" in column
            or text.dropna().astype(str).str.contains(
                r"[-/]\d{1,2}[-/]\d{2,4}|T\d{2}:\d{2}",
                regex=True,
            ).any()
        )

        if looks_like_date:
            try:
                date_values = pd.to_datetime(text, errors="coerce", format="mixed")
                date_rate = float(date_values[non_missing].notna().mean())

                if date_rate >= 0.80 and date_values[non_missing].notna().any():
                    malformed_mask = non_missing & date_values.isna()
                    malformed_ids = mark_rows("malformed", malformed_mask)
                    affected_cells = int(malformed_mask.sum())
                    malformed_cells += affected_cells

                    frame[column] = date_values
                    converted_columns += 1
                    log.add(
                        f"Data types: converted '{column}' to datetime; "
                        f"{affected_cells} malformed value(s) in {len(malformed_ids)} row(s) became missing "
                        "and will be imputed."
                    )
            except (TypeError, ValueError, OverflowError) as error:
                log.add(f"Data types: could not safely convert '{column}' to datetime ({error}).")

    issue_counts["Malformed values"] = malformed_cells
    issue_counts["Converted columns"] = converted_columns
    if converted_columns == 0:
        log.add("Data types: no safe type conversions were needed.")
    log.add(
        f"Malformed values: found {malformed_cells} unparseable value(s) "
        "in columns that could otherwise be converted."
    )

    # Drop columns with no usable values.
    data_columns = [column for column in frame.columns if column != row_id_column]
    all_missing_columns = [column for column in data_columns if frame[column].isna().all()]

    if all_missing_columns:
        frame.drop(columns=all_missing_columns, inplace=True)
        log.add(
            f"Missing values: dropped {len(all_missing_columns)} entirely empty "
            f"column(s): {', '.join(all_missing_columns)}."
        )

    # Fill remaining missing values.
    imputed_cells = 0
    for column in [c for c in frame.columns if c != row_id_column]:
        missing_count = int(frame[column].isna().sum())
        if missing_count == 0:
            continue

        series = frame[column]
        if is_numeric_dtype(series) and not is_bool_dtype(series):
            fill_value = series.median()
        elif is_datetime64_any_dtype(series):
            modes = series.mode(dropna=True)
            fill_value = modes.iloc[0] if not modes.empty else pd.NaT
        else:
            modes = series.mode(dropna=True)
            fill_value = modes.iloc[0] if not modes.empty else pd.NA

        if pd.isna(fill_value):
            log.add(
                f"Missing values: could not find a fill value for '{column}'; "
                "its remaining missing values will be retained."
            )
            continue

        frame[column] = series.fillna(fill_value)
        imputed_cells += missing_count
        log.add(
            f"Missing values: filled {missing_count} missing cell(s) in "
            f"'{column}' using {fill_value!r}."
        )

    if imputed_cells == 0:
        log.add("Missing values: no cells needed imputation.")

    # Outlier detection.
    outlier_row_ids, outlier_cells = detect_outlier_rows(frame, row_id_column)
    row_ids_by_issue["outliers"].update(outlier_row_ids)
    issue_counts["Outlier cells"] = outlier_cells

    flag_column = "outlier_flag"
    suffix = 2
    while flag_column in frame.columns:
        flag_column = f"outlier_flag_{suffix}"
        suffix += 1

    frame[flag_column] = frame[row_id_column].astype(int).isin(outlier_row_ids)
    log.add(
        f"Outliers: flagged {outlier_cells} outlier cell(s) across "
        f"{len(outlier_row_ids)} row(s) using the IQR rule; values were preserved "
        f"in '{flag_column}'."
    )

    # Summaries.
    problematic_row_ids: set[int] = set()
    for ids in row_ids_by_issue.values():
        problematic_row_ids.update(ids)

    rows_with_issues = len(problematic_row_ids)
    rows_after = len(frame)

    final_columns = [column for column in frame.columns if column not in {row_id_column, flag_column}]
    after_metrics = {
        "Missing cells": int(frame[final_columns].isna().sum().sum()),
        "Duplicate rows": int(frame.duplicated(subset=final_columns).sum()),
        "Outlier cells": outlier_cells,
    }

    output_path = input_path.with_name(f"cleaned_{input_path.name}")
    output_frame = frame.drop(columns=[row_id_column])
    save_table(output_frame, output_path, delimiter=text_delimiter)

    reports_dir = Path.cwd() / "reports"
    make_charts(
        issue_counts=issue_counts,
        problematic_rows=rows_with_issues,
        total_rows=rows_before,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        reports_dir=reports_dir,
    )

    print("\n--- Cleaning summary ---")
    print(f"Total rows before cleaning: {rows_before}")
    print(f"Rows that had one or more issues: {rows_with_issues}")
    print(f"Unusable rows removed: {unusable_rows_removed}")
    print(f"Duplicate rows removed: {duplicate_rows_removed}")
    print(f"Total rows after cleaning: {rows_after}")
    print(f"Cleaned file: {output_path}")
    print(f"Reports folder: {reports_dir.resolve()}")


def main() -> None:
    filename = input(
        "Enter the filename (e.g., test.csv, data.json, data.xlsx): "
    ).strip().strip('"')

    if not filename:
        print("Error: please enter a filename.")
        sys.exit(1)

    try:
        clean_file(Path(filename).expanduser())
    except FileNotFoundError as error:
        print(f"Error: {error}")
        sys.exit(1)
    except PermissionError:
        print("Error: permission denied while reading, writing, or creating reports.")
        sys.exit(1)
    except ValueError as error:
        print(f"Error: {error}")
        sys.exit(1)
    except pd.errors.EmptyDataError as error:
        print(f"Error: the file is empty or not valid tabular data: {error}")
        sys.exit(1)
    except pd.errors.ParserError as error:
        print(f"Error: the file could not be parsed as a valid tabular file: {error}")
        sys.exit(1)
    except UnicodeDecodeError as error:
        print(f"Error: the file is not valid UTF-8 text: {error}")
        sys.exit(1)
    except OSError as error:
        print(f"Error: an operating-system error occurred: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()