# Smart Data Cleaner

A Python project for cleaning messy tabular data, detecting quality issues, and generating visual reports for data validation.

## Overview

This project helps users clean structured datasets by automatically handling:
- missing values
- duplicate rows
- inconsistent column names
- extra whitespace
- malformed values
- outliers
- type conversion issues

It saves the cleaned output as a new file and creates visual quality reports for portfolio presentation.

## Features

- Prompts the user for the filename to clean
- Supports CSV, TSV, TXT, JSON, JSONL, and XLSX files
- Normalizes column names to lowercase snake_case
- Removes duplicate records
- Trims extra whitespace from string entries
- Converts numeric and date-like values where possible
- Flags outliers instead of silently deleting data
- Saves a cleaned file named:
  - cleaned_[original_filename]
- Produces a detailed processing log
- Generates 3 charts in the reports folder:
  - issue counts
  - clean vs. problematic rows
  - before/after quality comparison

## Project Structure

```text
smart-data-cleaner/
├── clean_data.py
├── requirements.txt
├── README.md
├── sample_data.csv
├── cleaned_sample_data.csv
└── reports/
    ├── issue_counts.png
    ├── clean_vs_problematic.png
    └── before_after_quality.png