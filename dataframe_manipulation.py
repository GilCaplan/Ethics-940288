import pandas as pd


def concatenate_csvs_with_labels(csv1_path, csv2_path, output_path):
    """
    Concatenates two CSV files with the same columns and adds
    'moral_label' and 'liability_level' columns for manual evaluation

    Args:
        csv1_path (str): Path to first CSV file
        csv2_path (str): Path to second CSV file
        output_path (str): Path for the combined output CSV
    """

    try:
        # Read both CSV files
        df1 = pd.read_csv(csv1_path)
        df2 = pd.read_csv(csv2_path)

        print(f"CSV 1 shape: {df1.shape}")
        print(f"CSV 2 shape: {df2.shape}")

        # Check if columns match
        if list(df1.columns) != list(df2.columns):
            print("Warning: Column names don't match exactly")
            print(f"CSV 1 columns: {list(df1.columns)}")
            print(f"CSV 2 columns: {list(df2.columns)}")

        # Concatenate the dataframes
        combined_df = pd.concat([df1, df2], ignore_index=True)

        # Add the new evaluation columns
        combined_df['moral_label'] = ''  # Empty string for manual filling
        combined_df['liability_level'] = ''  # Empty string for manual filling

        # Save to new CSV
        combined_df.to_csv(output_path, index=False)

        print(f"Successfully combined CSVs!")
        print(f"Combined shape: {combined_df.shape}")
        print(f"Output saved to: {output_path}")
        print(f"Columns in output: {list(combined_df.columns)}")

        return combined_df

    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def add_evaluation_columns_to_existing_csv(input_path, output_path):
    """
    Alternative function: Just add evaluation columns to an existing CSV

    Args:
        input_path (str): Path to existing CSV file
        output_path (str): Path for output CSV with new columns
    """

    try:
        # Read the CSV
        df = pd.read_csv(input_path)

        # Add the evaluation columns
        df['moral_label'] = ''
        df['liability_level'] = ''

        # Save to new file
        df.to_csv(output_path, index=False)

        print(f"Added evaluation columns to existing CSV")
        print(f"Shape: {df.shape}")
        print(f"Output saved to: {output_path}")

        return df

    except Exception as e:
        print(f"Error: {str(e)}")
        return None


# Usage examples
if __name__ == "__main__":
    # Example 1: Concatenate two CSVs and add evaluation columns
    concatenate_csvs_with_labels('generated_datasets/llm_morality_dataset_20250529_104113.csv',
                                 'generated_datasets/llm_morality_dataset_220250529_105518.csv', 'combined_for_evaluation.csv')

    # Example 2: Just add evaluation columns to existing CSV
    # add_evaluation_columns_to_existing_csv('all_responses.csv', 'responses_ready_for_evaluation.csv')

    # Print instructions
    print("Script ready to use!")
    print("\nTo concatenate two CSVs:")
    print("concatenate_csvs_with_labels('file1.csv', 'file2.csv', 'output.csv')")
    print("\nTo add columns to existing CSV:")
    print("add_evaluation_columns_to_existing_csv('input.csv', 'output.csv')")
    print("\nThe output CSV will have these additional columns:")
    print("- moral_label (empty, for you to fill: MORAL/NEUTRAL/IMMORAL)")
    print("- liability_level (empty, for you to fill: LOW/MEDIUM/HIGH)")