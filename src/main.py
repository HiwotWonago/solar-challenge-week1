# main.py

import pandas as pd
from feature_engineering import TransactionTimeFeatures
from pipelines import build_transaction_pipeline

def main():
    df = pd.read_csv('data/transactions.csv')

    # Build pipeline
    pipeline = build_transaction_pipeline()

    # Fit and transform
    processed_data = pipeline.fit_transform(df)

    # Convert to DataFrame
    output_df = pd.DataFrame(
        processed_data,
        columns=pipeline.get_feature_names_out()
    )

    # Save or display
    print(output_df.head())
    output_df.to_csv('data/processed_output.csv', index=False)

if __name__ == "__main__":
    main()
