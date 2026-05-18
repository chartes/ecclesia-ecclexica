import argparse
import os
import pandas as pd
import spacy

# Configuration
FILTERED_FILE = 'data/edcs_filtered_inscriptions.csv'
LEMMATIZED_FILE = 'data/edcs_lemmatized_inscriptions.csv'


def lemmatize_texts(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*80}")
    print("STEP 2: LEMMATIZING TEXTS")
    print(f"{'='*80}")
    print("\nLoading Latin spaCy model (la_core_web_md)...")
    try:
        nlp = spacy.load("la_core_web_md")
        print("✓ Model loaded successfully")
    except OSError:
        print("⚠ Latin model not found. Installing...")
        os.system("python -m spacy download la_core_web_md")
        nlp = spacy.load("la_core_web_md")
        print("✓ Model installed and loaded")

    print(f"\nProcessing {len(df):,} inscriptions...")
    print("Using field: 'clean_text_interpretive_word'")

    lemmatized_texts = []
    empty_texts = 0

    for i in range(len(df)):
        if i % 100 == 0 and i > 0:
            print(f"  Progress: {i:,}/{len(df):,} ({(i/len(df)*100):.1f}%)")

        row = df.iloc[i]
        text = row.get('clean_text_interpretive_word', '')

        if pd.isna(text) or not text:
            lemmatized_texts.append('')
            empty_texts += 1
            continue

        doc = nlp(str(text))
        lemmas = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha
        ]
        lemmatized_texts.append(" ".join(lemmas))

        if i == 0 and lemmas:
            print(f"\n--- Example: First inscription ---")
            print(f"  EDCS-ID: {row.get('EDCS-ID', 'N/A')}")
            print(f"  Original text: {text}")
            print(f"  Lemmatized: {' '.join(lemmas)}")
            print(f"  Word count: {len(text.split())} → {len(lemmas)} lemmas")
            print()

    df['lemmatized_text'] = lemmatized_texts
    print(f"\n✓ Lemmatization complete!")
    print(f"  - Processed: {len(df):,} inscriptions")
    print(f"  - Empty/missing texts: {empty_texts:,}")
    print(f"  - With content: {len(df) - empty_texts:,}")

    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description='Lemmatize filtered inscriptions and save the results.'
    )
    parser.add_argument('-i', '--input', default=FILTERED_FILE,
                        help='Input filtered CSV file (default: %(default)s)')
    parser.add_argument('-o', '--output', default=LEMMATIZED_FILE,
                        help='Output CSV file path for lemmatized data (default: %(default)s)')
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Filtered file not found: {args.input}")
        print("Run preprocess.py first to generate the filtered data.")
        return

    df = pd.read_csv(args.input, encoding='utf-8')
    df = lemmatize_texts(df)

    dirpath = os.path.dirname(args.output) or '.'
    os.makedirs(dirpath, exist_ok=True)
    df.to_csv(args.output, index=False, encoding='utf-8')
    print(f"\nLemmatized data saved to: {args.output}")


if __name__ == '__main__':
    main()
