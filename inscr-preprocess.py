import argparse
import json
import pandas as pd
import os

# Configuration
INPUT_FILE = 'data/EDCS_text_cleaned_2022-09-12.json'
FILTERED_FILE = 'data/edcs_filtered_inscriptions.csv'

DATE_FROM = 399
DATE_TO = 1199

DEFAULT_EXCLUDED_PROVINCES = [
    "Achaia", "Aegyptus", "Arabia", "Armenia", "Asia", "Cappadocia",
    "Cilicia", "Creta et Cyrenaica", "Cyprus", "Dacia", "Dalmatia",
    "Galatia", "Lycia et Pamphylia", "Macedonia", "Mesopotamia",
    "Moesia inferior", "Moesia superior", "Palaestina", "Pannonia inferior",
    "Pannonia superior", "Pontus et Bithynia", "Regnum Bospori", "Syria", "Thracia"]


def get_excluded_provinces() -> list:
    print(f"\n{'='*80}")
    print("PROVINCE CONFIGURATION")
    print(f"{'='*80}")
    print(f"\nDefault excluded provinces ({len(DEFAULT_EXCLUDED_PROVINCES)}):")
    print(f"  {', '.join(DEFAULT_EXCLUDED_PROVINCES)}")
    print()

    choice = input("Use default excluded provinces? [Y/n]: ").strip().lower()

    if choice in ('', 'y', 'yes'):
        print(f"✓ Using default excluded provinces ({len(DEFAULT_EXCLUDED_PROVINCES)} provinces).")
        return DEFAULT_EXCLUDED_PROVINCES
    else:
        print("\nEnter provinces to exclude, separated by commas.")
        print("Example: Achaia, Aegyptus, Syria")
        raw = input("Excluded provinces: ").strip()

        if not raw:
            print("⚠ No input provided. Falling back to default excluded provinces.")
            return DEFAULT_EXCLUDED_PROVINCES

        custom_provinces = [p.strip() for p in raw.split(',') if p.strip()]
        print(f"✓ Using custom excluded provinces ({len(custom_provinces)} provinces):")
        print(f"  {', '.join(custom_provinces)}")
        return custom_provinces


def is_in_date_range(row) -> bool:
    try:
        date_from = pd.to_numeric(row.get('date_not_before', None), errors='coerce')
        date_to = pd.to_numeric(row.get('date_not_after', None), errors='coerce')

        if pd.isna(date_from) or pd.isna(date_to):
            return False

        return not (date_to < DATE_FROM or date_from > DATE_TO)
    except Exception:
        return False


def load_and_filter_data(filepath: str, excluded_provinces: list, output_path: str) -> pd.DataFrame:
    print(f"\n{'='*80}")
    print("STEP 1: LOADING AND FILTERING DATA")
    print(f"{'='*80}")
    print(f"Loading data from {filepath}...")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    print(f"✓ Total inscriptions loaded: {len(df):,}")

    print(f"\nFiltering by date range: {DATE_FROM}-{DATE_TO}")
    df['in_date_range'] = df.apply(is_in_date_range, axis=1)
    before_date_filter = len(df)
    df = df[df['in_date_range']].copy()
    removed_by_date = before_date_filter - len(df)
    print(f"  ✓ After date filtering: {len(df):,} inscriptions")
    print(f"    (removed {removed_by_date:,} inscriptions outside date range)")

    print(f"\nFiltering by province (excluding {len(excluded_provinces)} provinces)")
    print(f"  Excluded provinces: {', '.join(excluded_provinces[:5])}{'...' if len(excluded_provinces) > 5 else ''}")
    before_province_filter = len(df)
    df = df[~df['province'].isin(excluded_provinces)].copy()
    removed_by_province = before_province_filter - len(df)
    print(f"  ✓ After province filtering: {len(df):,} inscriptions")
    print(f"    (removed {removed_by_province:,} inscriptions from excluded provinces)")

    dirpath = os.path.dirname(output_path) or '.'
    os.makedirs(dirpath, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✓ Filtered data saved to: {output_path}")

    df = df.reset_index(drop=True)
    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess EDCS inscriptions by date and province, then save filtered results.'
    )
    parser.add_argument('-i', '--input', default=INPUT_FILE,
                        help='Input JSON file path (default: %(default)s)')
    parser.add_argument('-o', '--output', default=FILTERED_FILE,
                        help='Output CSV file path for filtered data (default: %(default)s)')
    parser.add_argument('--exclude-provinces', default=None,
                        help='Comma-separated provinces to exclude; if omitted, prompt interactively.')
    parser.add_argument('--no-prompt', action='store_true',
                        help='Use the default excluded provinces without prompting.')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.exclude_provinces is not None:
        excluded_provinces = [p.strip() for p in args.exclude_provinces.split(',') if p.strip()]
        print(f"✓ Using custom excluded provinces ({len(excluded_provinces)} provinces).")
    elif args.no_prompt:
        excluded_provinces = DEFAULT_EXCLUDED_PROVINCES
        print(f"✓ Using default excluded provinces ({len(DEFAULT_EXCLUDED_PROVINCES)} provinces).")
    else:
        excluded_provinces = get_excluded_provinces()

    df = load_and_filter_data(args.input, excluded_provinces, args.output)

    if len(df) == 0:
        print("\nERROR: No inscriptions match the filtering criteria!")
        print("Check your date range and province exclusions.")
        return

    print(f"\nPreprocessing complete. {len(df):,} filtered inscriptions saved to {args.output}.")


if __name__ == '__main__':
    main()
