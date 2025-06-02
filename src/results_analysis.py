import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu

# === Utilities ===

def load_data(path):
    df = pd.read_csv(path)
    return df

def compute_neighbor_stats(df):
    # Compute count of same-class neighbors
    if 'Nearest Neighbors Classes' in df.columns and 'Original Class' in df.columns:
        def count_same_neighbors(row):
            neighbors = [int(x) for x in str(row['Nearest Neighbors Classes']).replace(',', ' ').split()]
            return sum(1 for n in neighbors if n == int(row['Original Class']))
        df['SameClassNeighborCount'] = df.apply(count_same_neighbors, axis=1)

        k = df['Nearest Neighbors Classes'].iloc[0]
        if isinstance(k, str):
            k = len(k.replace(',', ' ').split())
        else:
            k = 13  # default if ambiguous
        df['FractionSame'] = df['SameClassNeighborCount'] / k

        df['MajoritySame'] = df['SameClassNeighborCount'] >= (k // 2 + 1)
    else:
        raise ValueError("Expected columns not found.")

def compute_outcome_columns(df):
    df['OutcomeBinary'] = (df['Classification Outcome'].str.lower() == 'correct').astype(int)

def plot_boxplot(df, y_col, title, ylabel):
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Classification Outcome', y=y_col, data=df)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def run_mannwhitney(df, col):
    corr = df[df['OutcomeBinary'] == 1][col]
    inc  = df[df['OutcomeBinary'] == 0][col]
    stat, p = mannwhitneyu(corr, inc, alternative='two-sided')
    print(f"Mann-Whitney U on '{col}': U={stat:.1f}, p={p:.3g}")
    return p

def analyze_correct_neighbors(df):
    print("\n[Average Correct Neighbors by Classification Outcome]")
    grouped = df.groupby('Classification Outcome')['SameClassNeighborCount'].mean()
    k = df['SameClassNeighborCount'].max()
    for outcome, avg in grouped.items():
        print(f"  {outcome:15s}: {avg:.2f} ({(avg/k)*100:.2f}% of {k})")

def plot_neighbor_heatmap(df):
    print("\n[Neighbor-Class Frequency Heatmap]")
    classes = sorted([int(x) for x in df['Original Class'].unique()])
    k = len(str(df.iloc[0]['Nearest Neighbors Classes']).replace(',', ' ').split())
    mat = np.zeros((len(classes), len(classes)))
    for _, row in df.iterrows():
        true = int(row['Original Class'])
        neigh = [int(x) for x in str(row['Nearest Neighbors Classes']).replace(',', ' ').split()]
        for n in neigh:
            mat[true, n] += 1
    mat = mat / mat.sum(axis=1, keepdims=True)
    plt.figure(figsize=(8,6))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Neighbor Class')
    plt.ylabel('True Class')
    plt.title('Fraction of Nearest Neighbors per True Class')
    plt.tight_layout()
    plt.show()

def run_all_stats(df):
    print("\n========= Statistical Test Results =========")
    # 1. Same-Class Neighbor Count
    p1 = run_mannwhitney(df, "SameClassNeighborCount")
    if p1 < 0.05:
        print("   ➤ Statistically significant difference: Correct predictions have more same-class neighbors.")
    else:
        print("   ➤ No statistically significant difference found.")

    # 2. Neighborhood Entropy
    if 'Entropy Value' in df.columns:
        p2 = run_mannwhitney(df, "Entropy Value")
        if p2 < 0.05:
            print("   ➤ Statistically significant: Misclassified cases have higher entropy (mixed neighbors).")
        else:
            print("   ➤ Not significant: No difference in entropy.")

    # 3. Mean Distance (All Neighbors)
    if 'Mean Distance' in df.columns:
        p3 = run_mannwhitney(df, 'Mean Distance')
        if p3 < 0.05:
            print("   ➤ Statistically significant: Distance to all neighbors differs.")
        else:
            print("   ➤ Not significant: No difference for all neighbors.")

    # 4. Mean Distance Same-Class
    if 'Mean Distance (Correct Neighbors)' in df.columns:
        p4 = run_mannwhitney(df, 'Mean Distance (Correct Neighbors)')
        if p4 < 0.05:
            print("   ➤ Statistically significant: Distance to same-class neighbors differs.")
        else:
            print("   ➤ Not significant: No difference for same-class neighbors.")

    # 5. Mean Distance Different-Class
    if 'Mean Distance (Incorrect Neighbors)' in df.columns:
        p5 = run_mannwhitney(df, 'Mean Distance (Incorrect Neighbors)')
        if p5 < 0.05:
            print("   ➤ Statistically significant: Distance to different-class neighbors differs.")
        else:
            print("   ➤ Not significant: No difference for different-class neighbors.")

    # 6. Point-Biserial Correlation
    r_pb, r_pb_p = stats.pointbiserialr(df['OutcomeBinary'], df['SameClassNeighborCount'])
    print(f"\nPoint-Biserial Correlation: r = {r_pb:.3f}, p-value = {r_pb_p:.4g}")
    if r_pb_p < 0.05:
        print("   ➤ Significant positive correlation: More same-class neighbors means higher accuracy.")
    else:
        print("   ➤ No significant correlation found.")

    # 7. Chi-Square Test
    contingency = pd.crosstab(df['MajoritySame'], df['Classification Outcome'])
    chi2, p_chi, dof, expected = chi2_contingency(contingency)
    print(f"\nMajority Same-Class Neighbors (Chi-square test): p-value = {p_chi:.4g}")
    if p_chi < 0.05:
        print("   ➤ Statistically significant: Having a majority of same-class neighbors predicts correct classification.")
    else:
        print("   ➤ Not significant: Majority of same-class neighbors does not predict outcome.")

    print("\n========= End of Statistical Results =========\n")

# === MAIN PIPELINE ===

def main():
    df = load_data('../outputs/analysis_data.csv')
    compute_neighbor_stats(df)
    compute_outcome_columns(df)

    plot_boxplot(df, 'FractionSame', 'Fraction of Same-Class Neighbors by Outcome', 'FractionSame')
    run_mannwhitney(df, 'FractionSame')

    if 'Mean Distance (Correct Neighbors)' in df.columns:
        plot_boxplot(df, 'Mean Distance (Correct Neighbors)', 'Mean Distance to Correct-Class Neighbors', 'Mean Distance')
        run_mannwhitney(df, 'Mean Distance (Correct Neighbors)')

    if 'Entropy Value' in df.columns:
        plot_boxplot(df, 'Entropy Value', 'Neighbor Class Entropy by Outcome', 'Entropy')
        run_mannwhitney(df, 'Entropy Value')

    run_all_stats(df)

    analyze_correct_neighbors(df)

    plot_neighbor_heatmap(df)

if __name__ == '__main__':
    main()
