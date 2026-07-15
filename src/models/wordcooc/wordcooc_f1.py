import glob
import pandas as pd
import os

def collect_f1_scores(reports_dir='model_output/reports/wordcooc_adjusted', experiment_name='learning-curve', output_file='f1_summary.txt'):
    report_pattern = f'{reports_dir}/{experiment_name}/*.csv'
    report_files = glob.glob(report_pattern)
    
    if not report_files:
        print(f"No report files found in {report_pattern}")
        return
    
    results = []
    
    for report_path in report_files:
        try:
            # Each line is separated by ##### (custom delimiter)
            df = pd.read_csv(report_path, sep='#####', engine='python')
            df.columns = [col.strip() for col in df.columns]  # Clean up column names
            for _, row in df.iterrows():
                results.append({
                    'report_file': report_path,
                    'model': row['model'],
                    'train_set': row['train_set'],
                    'test_set': row['test_set'],
                    'f1_test': float(row['f1_test'])
                })
        except Exception as e:
            print(f"⚠️ Skipping {report_path} due to error: {e}")
    
    # Convert all results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Print results sorted by F1-score
    results_df = results_df.sort_values(by='f1_test', ascending=False)
    
    # Output path
    os.makedirs('model_output/summary', exist_ok=True)
    output_path = os.path.join('model_output/summary', output_file.replace('.txt', '.csv'))

    # Save the results DataFrame as CSV
    results_df.to_csv(output_path, index=False, columns=["model", "train_set", "test_set", "f1_test"])

    return results_df


if __name__ == '__main__':
    collect_f1_scores()
