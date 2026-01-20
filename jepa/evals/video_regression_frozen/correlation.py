import pandas as pd
import matplotlib.pyplot as plt 

def correlation(csv_path):
    # Read the CSV file without a header
    df = pd.read_csv(csv_path, header=None)
    # Assign column names
    df.columns = ['original_label', 'predicted_label', 'file_path']
    
    original_label = df['original_label'].to_list()
    predicted_label = df['predicted_label'].to_list()
    correlation = pd.Series(original_label).corr(pd.Series(predicted_label))
    
    output_png = csv_path.replace('.csv', '_correlation_plot.png')
    plt.figure(figsize=(8, 6))
    plt.scatter(original_label, predicted_label, alpha=0.5)
    plt.title(f'Scatter Plot of Original vs Predicted Labels\nCorrelation: {correlation:.4f}')
    plt.xlabel('Original Labels')
    plt.ylabel('Predicted Labels')
    plt.savefig(output_png)
    print(f"Saved correlation plot to {output_png}\nCorrelation: {correlation:.4f}")
    plt.close()
    return correlation