import pandas as pd
from pathlib import Path
import emoji
import argparse
from sklearn.model_selection import train_test_split


def remove_emojis(series: pd.Series) -> pd.Series:
    """
    Convert emojis in a text Series to their textual representation.
    """
    return series.astype(str).map(emoji.demojize)

def load_and_clean_data(path: Path) -> pd.DataFrame:
    """
    Load raw data, remove emojis, and filter removed contents.
    """
    data = pd.read_csv(path)
    
    comments = remove_emojis(data["Comment"])
    mask = comments != "[removed]"
    
    return pd.DataFrame({
        "Comment": comments[mask],
        "Topic": data.loc[mask, "Topic"]
    })

def main():
    parser = argparse.ArgumentParser(description="Clean and split text data for modeling")
    parser.add_argument("--input", type=Path, default=Path("data/raw_data/train.csv"), help="Path to raw csv file")
    parser.add_argument("--outdir", type=Path, default=Path("data/processed"), help="Output directory for processed files")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of the dataset to include in the test split")
    args = parser.parse_args()

    print(f"Loading and cleaning data from {args.input}...")
    df_clean = load_and_clean_data(args.input)

    print(f"Splitting data into train and test sets (Test size: {args.test_size})...")
    df_train, df_test = train_test_split(df_clean, test_size=args.test_size, random_state=42)

    args.outdir.mkdir(parents=True, exist_ok=True)
    
    train_path = args.outdir / "train.csv"
    test_path = args.outdir / "test.csv"
    
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    print(f"Data saved successfully:")
    print(f" - Train set: {train_path} ({len(df_train)} samples)")
    print(f" - Test set:  {test_path} ({len(df_test)} samples)")

if __name__ == "__main__":
    main()