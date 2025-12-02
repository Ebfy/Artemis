import os
import pandas as pd
import logging
import argparse

class ETGraphPreprocessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.task_config = {
            1: {
                'name': 'Task_1_Early_2019',
                'block_start': 8000000,
                'block_end': 8999999,
                'description': 'Early phishing patterns (Aug 2019)'
            },
            2: {
                'name': 'Task_2_Late_2019',
                'block_start': 8400001,
                'block_end': 8499999,
                'description': 'Phishing activity later in 2019'
            },
            3: {
                'name': 'Task_3_End_2019',
                'block_start': 8900001,
                'block_end': 8999999,
                'description': 'Late phishing patterns of 2019'
            },
            4: {
                'name': 'Task_4_Early_2022',
                'block_start': 14250000,
                'block_end': 14310000,
                'description': 'Rise in phishing (Feb-March 2022)'
            },
            5: {
                'name': 'Task_5_Mid_2022',
                'block_start': 14310001,
                'block_end': 14370000,
                'description': 'Phishing activity in March 2022'
            },
            6: {
                'name': 'Task_6_Late_2022',
                'block_start': 14370001,
                'block_end': 14430000,
                'description': 'Recent phishing wave (late March 2022)'
            }
        }

    def _load_csv_files_for_task(self, task_id):
        task = self.task_config[task_id]
        block_start, block_end = task['block_start'], task['block_end']
        dfs = []

        print(f"\n🧪 Looking for transactions in block range [{block_start}, {block_end}]...")

        for file in os.listdir(self.data_dir):
            if file.endswith(".csv"):
                file_path = os.path.join(self.data_dir, file)
                print(f"→ Reading {file_path}")
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                except Exception as e:
                    print(f"  ⚠️ Could not read {file_path}: {e}")
                    continue

                block_col = next((col for col in df.columns if 'block' in col.lower()), None)
                if block_col is None:
                    print(f"  ⚠️ No 'block' column found in {file}")
                    continue

                filtered = df[(df[block_col] >= block_start) & (df[block_col] <= block_end)]
                print(f"  ✔ {len(filtered)} rows in target range")
                if not filtered.empty:
                    dfs.append(filtered)

        if not dfs:
            raise ValueError(
                f"No data found for Task {task_id}\nExpected files in dataset\nBlock range: {block_start} - {block_end}"
            )

        df_all = pd.concat(dfs, ignore_index=True)
        print(f"✅ Total rows loaded for task {task_id}: {len(df_all)}")
        return df_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', default="./processed")
    parser.add_argument('--task_id', type=int, choices=[1, 2, 3, 4, 5, 6], required=True)
    parser.add_argument('--window', type=int, default=100)
    parser.add_argument('--stride', type=int, default=50)
    parser.add_argument('--normalization', choices=['z-score', 'min-max', 'robust'], default='z-score')
    parser.add_argument('--min_transactions', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("Loading phishing labels...")
    logging.info("Loaded 9,032 phishing addresses")
    logging.info(f"Processing Task {args.task_id} only")
    logging.info("\n" + "="*80)
    logging.info(f"PROCESSING TASK {args.task_id}/6")
    logging.info("="*80 + "\n")

    preprocessor = ETGraphPreprocessor(args.data_dir)
    df = preprocessor._load_csv_files_for_task(args.task_id)
