import argparse
import os
import pandas as pd
import numpy as np

# Configuration
LEMMATIZED_FILE = 'data/edcs_lemmatized_inscriptions.csv'
OUTPUT_FILE = 'output/edcs_architectural_scores.csv'
OUTPUT_HIGH_FILE = 'output/edcs_architectural_scores_gt50.csv'
AUTO_TERMS_FILE = 'data/dicts/auto_terms.csv'
ASSO_TERMS_FILE = 'data/dicts/asso_terms.csv'
MAT_TERMS_FILE = 'data/dicts/mat_terms.csv'

WEIGHTS = {
    'term_count': 0.45,
    'cooccurrence': 0.25,
    'proximity': 0.20,
    'density': 0.10
}


def load_terms(filepath: str, label: str) -> list:
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
        if 'term' not in df.columns:
            raise ValueError(f"Column 'term' not found in {filepath}. Available columns: {list(df.columns)}")
        terms = df['term'].dropna().str.strip().tolist()
        print(f"  ✓ {label}: {len(terms)} terms loaded from {filepath}")
        return terms
    except FileNotFoundError:
        print(f"  ✗ ERROR: File not found — {filepath}")
        raise
    except Exception as e:
        print(f"  ✗ ERROR loading {filepath}: {e}")
        raise


def load_term_dictionaries() -> tuple:
    print(f"\n{'='*80}")
    print("LOADING TERM DICTIONARIES")
    print(f"{'='*80}\n")
    autonomous_terms = load_terms(AUTO_TERMS_FILE, "Autonomous terms")
    associative_terms = load_terms(ASSO_TERMS_FILE, "Associative terms")
    material_terms = load_terms(MAT_TERMS_FILE, "Material terms")
    return autonomous_terms, associative_terms, material_terms


def calculate_score(lemmatized_text: str,
                    auto_terms: set,
                    assoc_terms: set,
                    mate_terms: set) -> dict:
    if pd.isna(lemmatized_text) or not lemmatized_text:
        return {
            'score': 0.0,
            'autonomous': [],
            'associative': [],
            'material': [],
            'n_autonomous': 0,
            'n_associative': 0,
            'n_material': 0
        }

    lemmas = str(lemmatized_text).split()
    autonomous_found = []
    associative_found = []
    material_found = []
    auto_assoc_positions = []
    material_positions = []

    for idx, lemma in enumerate(lemmas):
        if lemma in auto_terms:
            autonomous_found.append(lemma)
            auto_assoc_positions.append(idx)
        elif lemma in assoc_terms:
            associative_found.append(lemma)
            auto_assoc_positions.append(idx)
        elif lemma in mate_terms:
            material_found.append(lemma)
            material_positions.append(idx)

    n_auto = len(autonomous_found)
    n_assoc = len(associative_found)
    n_mate = len(material_found)
    total_terms = n_auto + n_assoc
    include_material = (n_auto + n_assoc) > 0
    positions = list(auto_assoc_positions)
    if include_material:
        positions.extend(material_positions)

    total_terms = n_auto + n_assoc + (n_mate if include_material else 0)
    if total_terms == 0:
        return {
            'score': 0.0,
            'autonomous': [],
            'associative': [],
            'material': [],
            'n_autonomous': 0,
            'n_associative': 0,
            'n_material': 0
        }

    score_count = min(np.log1p(total_terms) / np.log1p(5), 1.0)
    if n_auto == 0:
        score_cooc = 0.0
    elif n_auto > 0 and n_assoc > 0:
        score_cooc = 1.0
    elif n_auto > 1:
        score_cooc = 0.6
    else:
        score_cooc = 0.3

    if len(positions) > 1:
        positions.sort()
        distances = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        avg_distance = np.mean(distances)
        score_prox = np.exp(-0.1 * avg_distance)
    else:
        score_prox = 0.0

    density = total_terms / len(lemmas)
    score_density = min(density / 0.15, 1.0)

    final_score = (
        WEIGHTS['term_count'] * score_count +
        WEIGHTS['cooccurrence'] * score_cooc +
        WEIGHTS['proximity'] * score_prox +
        WEIGHTS['density'] * score_density
    ) * 100

    return {
        'score': round(final_score, 2),
        'autonomous': autonomous_found,
        'associative': associative_found,
        'material': material_found,
        'n_autonomous': n_auto,
        'n_associative': n_assoc,
        'n_material': n_mate
    }


def process_corpus(df: pd.DataFrame, autonomous_terms: list, associative_terms: list, material_terms: list) -> pd.DataFrame:
    print(f"\n{'='*80}")
    print("STEP 3: CALCULATING ARCHITECTURAL SCORES")
    print(f"{'='*80}")

    auto_set = set(autonomous_terms)
    assoc_set = set(associative_terms)
    mate_set = set(material_terms)

    print(f"\nArchitectural vocabulary:")
    print(f"  - Autonomous terms: {len(autonomous_terms)} ({', '.join(autonomous_terms[:5])}{'...' if len(autonomous_terms) > 5 else ''})")
    print(f"  - Associative terms: {len(associative_terms)} ({', '.join(associative_terms[:5])}{'...' if len(associative_terms) > 5 else ''})")
    print(f"  - Material terms: {len(material_terms)} ({', '.join(material_terms[:5])}{'...' if len(material_terms) > 5 else ''})")

    print(f"\nScore calculation weights:")
    for component, weight in WEIGHTS.items():
        print(f"  - {component}: {weight} ({weight*100:.0f}%)")

    print(f"\nProcessing {len(df):,} inscriptions...")

    scores = []
    autonomous_lists = []
    associative_lists = []
    material_lists = []
    inscriptions_with_terms = 0
    total_auto_terms = 0
    total_assoc_terms = 0
    total_mate_terms = 0

    for idx in range(len(df)):
        if idx % 500 == 0:
            print(f"  Progress: {idx:,}/{len(df):,} ({(idx/len(df)*100):.1f}%)")

        row = df.iloc[idx]
        result = calculate_score(row['lemmatized_text'], auto_set, assoc_set, mate_set)

        scores.append(result['score'])
        autonomous_lists.append('|'.join(result['autonomous']) if result['autonomous'] else '')
        associative_lists.append('|'.join(result['associative']) if result['associative'] else '')
        material_lists.append('|'.join(result['material']) if result['material'] else '')

        if result['score'] > 0:
            inscriptions_with_terms += 1
            total_auto_terms += result['n_autonomous']
            total_assoc_terms += result['n_associative']
            total_mate_terms += result['n_material']

    df['arch_score'] = scores
    df['autonomous_terms'] = autonomous_lists
    df['associative_terms'] = associative_lists
    df['material_terms'] = material_lists

    print(f"\n✓ Processing complete!\n")
    print(f"Results summary:")
    print(f"  - Total inscriptions: {len(df):,}")
    print(f"  - With architectural terms: {inscriptions_with_terms:,} ({(inscriptions_with_terms/len(df)*100):.1f}%)")
    print(f"  - Without terms: {len(df) - inscriptions_with_terms:,} ({((len(df)-inscriptions_with_terms)/len(df)*100):.1f}%)")
    print(f"\nTerm frequency:")
    print(f"  - Total autonomous terms found: {total_auto_terms:,}")
    print(f"  - Total associative terms found: {total_assoc_terms:,}")
    print(f"  - Average autonomous per inscription (with terms): {(total_auto_terms/inscriptions_with_terms if inscriptions_with_terms > 0 else 0):.2f}")
    print(f"  - Average associative per inscription (with terms): {(total_assoc_terms/inscriptions_with_terms if inscriptions_with_terms > 0 else 0):.2f}")
    print(f"\nScore statistics:")
    print(f"  - Mean score: {df['arch_score'].mean():.2f}")
    print(f"  - Median score: {df['arch_score'].median():.2f}")
    print(f"  - Max score: {df['arch_score'].max():.2f}")
    print(f"  - Score > 30: {(df['arch_score'] > 30).sum():,} inscriptions")
    print(f"  - Score > 40: {(df['arch_score'] > 40).sum():,} inscriptions")
    print(f"  - Score > 50: {(df['arch_score'] > 50).sum():,} inscriptions")
    print(f"  - Score > 60: {(df['arch_score'] > 60).sum():,} inscriptions")
    print(f"  - Score > 70: {(df['arch_score'] > 70).sum():,} inscriptions")

    return df


def show_top_inscriptions(df: pd.DataFrame, n: int = 10) -> None:
    print(f"\n{'='*80}")
    print(f"Top {n} inscriptions by architectural score:")
    print(f"{'='*80}\n")

    top = df.nlargest(n, 'arch_score')
    for idx, (_, row) in enumerate(top.iterrows(), 1):
        print(f"{idx}. Score: {row['arch_score']:.2f}")
        print(f"   ID: {row['EDCS-ID']}")
        print(f"   Province: {row.get('province', 'N/A')}")
        print(f"   Place: {row.get('place', 'N/A')}")
        print(f"   Date: {row.get('date_not_before', '?')} to {row.get('date_not_after', '?')}")
        print(f"   Autonomous: {row['autonomous_terms']}")
        print(f"   Associative: {row['associative_terms']}")
        print(f"   Text: {row.get('clean_text_interpretive_word', 'N/A')[:100]}...")
        print()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Calculate architectural scores for lemmatized inscription texts.'
    )
    parser.add_argument('-i', '--input', default=LEMMATIZED_FILE,
                        help='Input CSV file with lemmatized texts (default: %(default)s)')
    parser.add_argument('-o', '--output', default=OUTPUT_FILE,
                        help='Output CSV file path for scored data (default: %(default)s)')
    parser.add_argument('--high-output', default=OUTPUT_HIGH_FILE,
                        help='Output CSV file path for high-score subset (default: %(default)s)')
    parser.add_argument('--no-subsets', action='store_true',
                        help='Skip creating per-bin subset CSV files.')
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Lemmatized file not found: {args.input}")
        print("Run lemmatize.py first to generate the lemmatized text file.")
        return

    df = pd.read_csv(args.input, encoding='utf-8')
    autonomous_terms, associative_terms, material_terms = load_term_dictionaries()
    df_scored = process_corpus(df, autonomous_terms, associative_terms, material_terms)

    show_top_inscriptions(df_scored, n=10)

    out_dir = os.path.dirname(args.output) or '.'
    os.makedirs(out_dir, exist_ok=True)
    df_scored.to_csv(args.output, index=False, encoding='utf-8')
    print(f"\n✓ File saved successfully: {args.output}")

    if not args.no_subsets:
        df_scored['score_bin'] = df_scored['arch_score'].apply(lambda x: int(x // 10) * 10)
        for score_bin in sorted(df_scored['score_bin'].unique()):
            subset = df_scored[df_scored['score_bin'] == score_bin].copy()
            subset_file = f"{out_dir}/edcs_architectural_scores_{score_bin}_{score_bin+9}.csv"
            subset.to_csv(subset_file, index=False, encoding='utf-8')
            print(f"Subset file saved: {subset_file} ({len(subset):,} rows)")

    high_df = df_scored[df_scored['arch_score'] > 50].copy()
    if len(high_df) > 0:
        high_out_dir = os.path.dirname(args.high_output) or '.'
        os.makedirs(high_out_dir, exist_ok=True)
        high_df.to_csv(args.high_output, index=False, encoding='utf-8')
        print(f"High-score file saved: {args.high_output} ({len(high_df):,} rows)")
    else:
        print("No inscriptions with arch_score > 50 found; no high-score CSV created.")


if __name__ == '__main__':
    main()
