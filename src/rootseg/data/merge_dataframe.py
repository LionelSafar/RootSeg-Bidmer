"""
Prepare and clean root-segmentation dataframe data.

Usage:
    PYTHONPATH=src python -m rootseg.data.prepare \
        --workspace /path/to/workspace \
        --rv-file RV_output/EA25.csv \
        --area-file Area_metrics/metrics_EA25.csv \
        --type EA

For Bidmer:
    PYTHONPATH=src python -m rootseg.data.prepare \
        --workspace /path/to/workspace \
        --rv-file RV_output/Bidmer25.csv \
        --area-file Area_metrics/metrics_Bidmer25.csv \
        --type Bidmer
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------

def get_date(filename: str, ea: bool = False) -> datetime:
    parts = filename.split("_")

    # EA filenames:
    # ... _ ... _ ... _ YYYY.MM.DD ...
    # Bidmer filenames:
    # ... _ ... _ ... _ ... _ ... _ YYYY.MM.DD ...
    date_str = parts[3] if ea else parts[5]

    return datetime.strptime(date_str, "%Y.%m.%d")


def get_tube_name(filename: str, ea: bool = False) -> str:
    parts = filename.split("_")

    if ea:
        return parts[1]

    return parts[1] + parts[0].replace("Bidmer", "")


def get_depth_level(filename: str) -> str:
    parts = filename.split("_")
    depth = parts[2] + "_" + parts[3]

    if depth == "0_10cm":
        return "0_10cm"

    if depth == "10_Xcm":
        return "10_Xcm"

    print(
        f"ERROR IN {filename}: {depth} is neither "
        "'0_10cm' nor '10_Xcm'"
    )
    return np.nan


# ---------------------------------------------------------------------
# Area conversion
# ---------------------------------------------------------------------

def pix_to_cm2(num_pixels: float, dpi: int = 1200) -> float:
    pixel_size_cm = 2.54 / dpi
    pixel_area_cm2 = pixel_size_cm ** 2

    return num_pixels * pixel_area_cm2


# ---------------------------------------------------------------------
# Treatment metadata
# ---------------------------------------------------------------------

def append_treatment_ea(
    df: pd.DataFrame,
    treatment_file: Path,
) -> pd.DataFrame:

    df["Tube"] = df["filename"].apply(
        lambda x: get_tube_name(x, ea=True)
    )

    df["Date"] = df["filename"].apply(
        lambda x: get_date(x, ea=True)
    )

    meta_df = pd.read_excel(treatment_file)

    meta_df["Tube"] = "T" + meta_df["Tube"].astype(str)

    df = pd.merge(
        df,
        meta_df,
        how="left",
        on="Tube",
    )

    treatment_mapping = {
        "C": "control",
        "D": "drought",
    }

    df["Treatment"] = df["Treatment"].replace(treatment_mapping)

    return df


def append_treatment_bidmer(
    df: pd.DataFrame,
    treatment_file: Path,
) -> pd.DataFrame:

    df["Tube"] = df["filename"].apply(get_tube_name)

    df["Depth"] = df["filename"].apply(get_depth_level)

    df["Date"] = df["filename"].apply(
        lambda x: get_date(x, ea=False)
    )

    meta_df = pd.read_excel(treatment_file)

    # Create north and south versions of every plot
    north = meta_df.copy()
    north["Tube"] = "T" + north["plot"].astype(str) + "N"

    south = meta_df.copy()
    south["Tube"] = "T" + south["plot"].astype(str) + "S"

    expanded = pd.concat(
        [north, south],
        ignore_index=True,
    )

    expanded = expanded.rename(
        columns={
            "block$": "block",
            "snow$": "snow",
            "drought$": "drought",
        }
    )

    expanded.drop(
        columns=["plot"],
        inplace=True,
    )

    df = pd.merge(
        df,
        expanded,
        how="left",
        on="Tube",
    )

    drought_mapping = {
        "C": "control",
        "L": "10 weeks",
        "S": "5 weeks",
    }

    snow_mapping = {
        "L": "early",
        "C": "control",
        "H": "late",
    }

    df["drought"] = df["drought"].replace(drought_mapping)
    df["snow"] = df["snow"].replace(snow_mapping)

    return df


# ---------------------------------------------------------------------
# Area metrics
# ---------------------------------------------------------------------

def add_metrics(
    df: pd.DataFrame,
    metrics_csv: Path,
    ea: bool = False,
) -> pd.DataFrame:

    df_met = pd.read_csv(metrics_csv)

    # Make sure filenames have the same format in both dataframes
    df["filename"] = (
        df["filename"]
        .str.replace("_segmented", "", regex=False)
        .str.replace(".png", ".tiff", regex=False)
    )

    df = pd.merge(
        df,
        df_met,
        on="filename",
        how="left",
    )

    # Effective image area
    effective_area = (
        df["image_area_px2"]
        - df["excluded_area_px2"]
    )

    effective_area_cm2 = pix_to_cm2(effective_area)

    # Root length density
    df["rootlength"] = (
        df["Total.Root.Length.mm"]
        / effective_area_cm2
    )

    # Diameter distributions
    diam_cols = [
        col
        for col in df.columns
        if "Root.Length.Diameter.Range" in col
    ]

    for col in diam_cols:
        df[col] /= effective_area_cm2

        df[col + "_normed"] = (
            df[col] / df["rootlength"]
        )

    # Sort before calculating temporal differences
    df.sort_values(
        by="Date",
        inplace=True,
    )

    if ea:
        group_cols = ["Tube"]
    else:
        group_cols = ["Depth", "Tube"]

    df["rootlength_dif"] = (
        df.groupby(group_cols)["rootlength"]
        .diff()
        .fillna(0)
    )

    time_diff_days = (
        df.groupby(group_cols)["Date"]
        .diff()
        .dt.days
    )

    df["rootlength_dif_norm"] = (
        df["rootlength_dif"]
        / time_diff_days
        * 7
    )

    df["rootlength_dif_norm"] = (
        df["rootlength_dif_norm"]
        .fillna(0)
    )

    df["date_disp"] = df["Date"].dt.strftime("%d %b")

    return df


# ---------------------------------------------------------------------
# EA cleaning
# ---------------------------------------------------------------------

EA_SPECIES = [
    "Anthoxantum",
    "Helictotrichon",
    "Geum",
    "Potentilla",
    "Leontodon",
    "Carex",
]


def clean_ea(df: pd.DataFrame) -> pd.DataFrame:

    # Remove whitespace in species names
    df["Species"] = (
        df["Species"]
        .astype(str)
        .str.strip()
    )

    # Standardise spelling
    df["Species"] = df["Species"].replace(
        {
            "Anthox": "Anthoxantum",
        }
    )

    # Standardise functional-group names
    df["Class"] = df["Class"].replace(
        {
            "Graminoid": "Graminoids",
            "Herb": "Forbs",
        }
    )

    # Replication flag
    #
    # This reproduces the notebook logic:
    # species belonging to the six experimental species
    # receive replication = 1.
    df["replication"] = (
        df["Species"]
        .isin(EA_SPECIES)
        .astype(int)
    )

    # Year
    df["year"] = (
        df["Date"]
        .dt.year
        .astype(str)
    )

    return df


# ---------------------------------------------------------------------
# Bidmer cleaning
# ---------------------------------------------------------------------

def clean_bidmer(df: pd.DataFrame) -> pd.DataFrame:

    return df


# ---------------------------------------------------------------------
# Main preparation function
# ---------------------------------------------------------------------

def prepare_dataframe(
    rv_file: Path,
    area_file: Path,
    treatment_file: Path,
    dataset_type: str,
) -> pd.DataFrame:

    print(f"Reading RV file:      {rv_file}")
    print(f"Reading area file:    {area_file}")
    print(f"Reading treatment:    {treatment_file}")
    print(f"Dataset type:         {dataset_type}")

    df = pd.read_csv(rv_file)

    # Internally use one consistent filename column
    if "File.Name" in df.columns:
        df = df.rename(
            columns={"File.Name": "filename"}
        )

    if "filename" not in df.columns:
        raise ValueError(
            "Could not find 'File.Name' or 'filename' "
            "in the RV file."
        )

    dataset_type = dataset_type.lower()

    if dataset_type == "ea":

        df = append_treatment_ea(
            df,
            treatment_file,
        )

        df = clean_ea(df)

        df = add_metrics(
            df,
            area_file,
            ea=True,
        )

    elif dataset_type == "bidmer":

        df = append_treatment_bidmer(
            df,
            treatment_file,
        )

        df = clean_bidmer(df)

        df = add_metrics(
            df,
            area_file,
            ea=False,
        )

    else:
        raise ValueError(
            f"Unknown dataset type: {dataset_type}"
        )

    return df


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Prepare and clean root-segmentation data "
            "from an RV output file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("."),
        help=(
            "Root directory containing the treatment "
            "metadata and data directories."
        ),
    )

    parser.add_argument(
        "--rv-file",
        type=Path,
        required=True,
        help=(
            "RV output CSV file. Relative paths are "
            "interpreted relative to --workspace."
        ),
    )

    parser.add_argument(
        "--area-file",
        type=Path,
        required=True,
        help=(
            "Area-metrics CSV file. Relative paths are "
            "interpreted relative to --workspace."
        ),
    )

    parser.add_argument(
        "--type",
        choices=["EA", "Bidmer"],
        required=True,
        help="Dataset type.",
    )

    parser.add_argument(
        "--treatment-file",
        type=Path,
        default=None,
        help=(
            "Treatment Excel file. If omitted, "
            "<workspace>/EA_treatments.xlsx is used."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output Excel file. If omitted, the file is "
            "saved as <workspace>/cleaned/"
            "<rv_filename>_cleaned.xlsx."
        ),
    )

    return parser


def main() -> None:

    parser = build_parser()
    args = parser.parse_args()

    workspace = args.workspace.resolve()

    # -------------------------------------------------------------
    # Resolve input paths
    # -------------------------------------------------------------

    rv_file = (
        args.rv_file
        if args.rv_file.is_absolute()
        else workspace / args.rv_file
    )

    area_file = (
        args.area_file
        if args.area_file.is_absolute()
        else workspace / args.area_file
    )

    if args.treatment_file is None:
        treatment_file = (
            workspace / "EA_treatments.xlsx"
        )
    else:
        treatment_file = (
            args.treatment_file
            if args.treatment_file.is_absolute()
            else workspace / args.treatment_file
        )

    # -------------------------------------------------------------
    # Check input files
    # -------------------------------------------------------------

    for path in [
        rv_file,
        area_file,
        treatment_file,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"File does not exist: {path}"
            )

    # -------------------------------------------------------------
    # Determine output path
    # -------------------------------------------------------------

    if args.output is None:

        output_dir = workspace / "cleaned"
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = (
            output_dir
            / f"{rv_file.stem}_cleaned.xlsx"
        )

    else:

        output_file = (
            args.output
            if args.output.is_absolute()
            else workspace / args.output
        )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    # -------------------------------------------------------------
    # Prepare dataframe
    # -------------------------------------------------------------

    df = prepare_dataframe(
        rv_file=rv_file,
        area_file=area_file,
        treatment_file=treatment_file,
        dataset_type=args.type,
    )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    print()
    print(f"Rows:    {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print()
    print(f"Saving cleaned dataframe to:")
    print(output_file)

    df.to_excel(
        output_file,
        index=False,
    )

    print("Done.")


if __name__ == "__main__":
    main()