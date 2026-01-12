import pandas as pd
from pathlib import Path
import emoji


def remove_emojis(series: pd.Series) -> pd.Series:
    """
    Convert emojis in a text Series to their textual representation.
    
    """
    return series.astype(str).map(emoji.demojize)

def load_and_process_data(path: Path) -> pd.DataFrame:
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

def  save_processed_data(data: pd.DataFrame, output_path: Path) -> None:
    data.to_csv(output_path, index=False)
    
    
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    input_file = project_root / "data" / "raw_data" / "train.csv"
    output_file = project_root / "data" / "processed" / "data_processed.csv"

    processed_data = load_and_process_data(input_file)
    save_processed_data(processed_data, output_file)
    

    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    