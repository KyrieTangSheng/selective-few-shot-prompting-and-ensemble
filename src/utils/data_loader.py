import pandas as pd
from sklearn.model_selection import train_test_split
import re

class IMDBDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        
    def clean_text(self, text):
        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convert to lowercase
        text = text.lower().strip()
        return text
        
    def load_data(self, test_size=0.2, random_state=42):
        # Read CSV
        df = pd.read_csv(self.data_path)
        
        # Clean text
        df['review'] = df['review'].apply(self.clean_text)
        
        # Convert sentiment to numeric
        df['sentiment'] = (df['sentiment'] == 'positive').astype(int)
        
        # Split data
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size,
            random_state=random_state,
            stratify=df['sentiment']
        )
        
        return train_df, test_df