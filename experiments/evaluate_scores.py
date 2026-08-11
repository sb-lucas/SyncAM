import numpy as np
import pandas as pd
import os
from scipy.stats import spearmanr, linregress
from dotenv import load_dotenv

EPSILON = 1e-8  # small constant to avoid division by zero in stability

#mean and std of each scoring method
def get_baseline_summary(df, score_cols):
    """
    Computes baseline scores (mean ± std) for uncorrupted audio-MIDI pairs (corruption_amount == 0)
    pooled across all data, with an optional runtime column.
    
    Parameters:
      - df: DataFrame containing evaluation scores
                 
    Returns:
      - summary_df: DataFrame indexed by metric name
    """

    # Filter for uncorrupted baseline pairs (S == 0)
    baseline_df = df[np.isclose(df['corruption_amount'], 0)].copy()

    # Deduplicate so $S=0$ tracks aren't counted multiple times across different corruption types
    dedup_cols = [c for c in ['dataset', 'audio'] if c in baseline_df.columns]
    if dedup_cols:
        baseline_df = baseline_df.drop_duplicates(subset=dedup_cols)

    # Compute pooled mean and std across all uncorrupted pairs
    means = baseline_df[score_cols].mean()
    stds = baseline_df[score_cols].std()

    rows = []
    for m in score_cols:
        mean_val = means[m]
        std_val = stds[m]
        
        row = {
            'Metric': m,
            'Baseline Mean': round(mean_val, 4),
            'Baseline Std': round(std_val, 4),
        }
        rows.append(row)

    summary_df = pd.DataFrame(rows).set_index('Metric')
    return summary_df

def _compute_monotonicity(df, score_cols):
    def _spearman_per_group(g):
        x = g['corruption_amount'].values
        res = {}
        for col in score_cols:
            y = g[col].values
            if np.all(y == y[0]):  # Handle constant outputs
                res[col] = 0.0
            else:
                res[col] = (-1) * spearmanr(x, y).statistic
        return pd.Series(res)

    # Compute per-song Spearman, then average over songs per corruption type
    per_song = df.groupby(['audio', 'corruption_type'], group_keys=True).apply(_spearman_per_group)
    return per_song.groupby(level='corruption_type')[score_cols].mean()

def _compute_stability(df, metric_cols):
    # Group by corruption type and severity level S
    grouped = df.groupby(['corruption_type', 'corruption_amount'])
    mu = grouped[metric_cols].mean()
    sigma = grouped[metric_cols].std()
    
    # 1. Compute Coefficient of Variation (CV)
    cv = sigma / (np.abs(mu) + EPSILON)
    
    # 2. Map CV to [0, 1] where 1 = perfect stability (0 variance)
    bounded_stability = 1.0 / (1.0 + cv)
    
    # 3. Average across all severity levels S
    return bounded_stability.groupby(level='corruption_type')[metric_cols].mean()

def _compute_sensitivity(df, score_cols):
    def _relative_drop(g):
        s_min = g.loc[g['corruption_amount'].idxmin()]
        s_max = g.loc[g['corruption_amount'].idxmax()]
        
        res = {}
        for col in score_cols:
            m_0 = s_min[col]
            m_1 = s_max[col]
            res[col] = (m_0 - m_1) / (np.abs(m_0) + EPSILON)
        return pd.Series(res)

    # Compute per-song fractional drop, then average over songs per corruption type
    per_song = df.groupby(['audio', 'corruption_type'], group_keys=True).apply(_relative_drop)
    return per_song.groupby(level='corruption_type')[score_cols].mean()

def _compute_consistency(df, score_cols):
    def _r2_per_group(g):
        corr_type = g['corruption_type'].iloc[0]
        S = g['corruption_amount'].values
        
        # Applies quadratic mapping for 'onset'
        x = S**2 if 'onset' in str(corr_type).lower() else S
        
        res = {}
        for col in score_cols:
            y = g[col].values
            if np.all(y == y[0]) or np.all(x == x[0]):
                res[col] = 0.0
            else:
                r_val = linregress(x, y).rvalue
                res[col] = r_val**2 if not np.isnan(r_val) else 0.0
        return pd.Series(res)

    per_song = df.groupby(['audio', 'corruption_type'], group_keys=True).apply(_r2_per_group)
    return per_song.groupby(level='corruption_type')[score_cols].mean()


#big main table
def calculate_all_metrics(df, score_cols):
    """
    Calculates Monotonicity, Stability, Sensitivity, and Consistency for all target metrics.
    
    Returns a DataFrame indexed by (corruption_type, metric_name) with columns:
    ['monotonicity', 'stability', 'sensitivity', 'consistency']
    """

    print("Computing Monotonicity (Spearman rho)...")
    mono = _compute_monotonicity(df, score_cols)
    
    print("Computing Stability (CV)...")
    stab = _compute_stability(df, score_cols)
    
    print("Computing Relative Sensitivity ((m0 - m1)/m0)...")
    sens = _compute_sensitivity(df, score_cols)
    
    print("Computing Consistency (R^2 fit)...")
    cons = _compute_consistency(df, score_cols)
    
    # 1. Store results with standard short column names
    eval_dict = {
        'Mono': mono,
        'Stab': stab,
        'Sens': sens,
        'Cons': cons
    }
    
    # 2. Combine into a single MultiIndex DataFrame
    combined = pd.concat(eval_dict, names=['eval_metric', 'corruption_type'])
    
    # 3. Transpose so models are rows, and swap column levels so corruption_type is on top
    final_df = combined.T.swaplevel(0, 1, axis=1)
    
    # 4. Reindex columns to maintain clean order: (corruption_type -> Mono, Stab, Sens, Cons)
    corr_order = [c for c in df['corruption_type'].unique() if c in final_df.columns.levels[0]]
    eval_order = ['Mono', 'Stab', 'Sens', 'Cons']
    
    multi_cols = pd.MultiIndex.from_product(
        [corr_order, eval_order], 
        names=['corruption_type', 'eval_metric']
    )
    
    final_df = final_df.reindex(columns=multi_cols)
    final_df.index.name = 'metric'
    
    return final_df

#monotonicity per dataset table
def calculate_monotonicity_by_dataset(df, score_cols):
    """
    Computes Monotonicity (Spearman rho) for each corruption type, 
    broken down per dataset.
    
    Returns:
      - final_df: MultiIndex DataFrame indexed by scoring metric with columns 
        (dataset, corruption_type)
    """
    datasets = df['dataset'].unique()
    corr_types = df['corruption_type'].unique()
    
    mono_per_dataset = {}

    for ds in datasets:
        df_ds = df[df['dataset'] == ds]
        # Returns DataFrame with index = corruption_type, columns = score_cols
        mono_ds = _compute_monotonicity(df_ds, score_cols)
        # Transpose so rows = metrics, columns = corruption_type
        mono_per_dataset[ds] = mono_ds.T

    # Combine into a single MultiIndex DataFrame
    final_df = pd.concat(mono_per_dataset, axis=1, names=['dataset', 'corruption_type'])
    
    # Enforce clean column order
    multi_cols = pd.MultiIndex.from_product(
        [datasets, corr_types], 
        names=['dataset', 'corruption_type']
    )
    
    final_df = final_df.reindex(columns=multi_cols)
    final_df.index.name = 'metric'
    
    return final_df

#onset fine grained X coarse grained for maps table
def get_onset_comparison(df, score_cols,dataset='maps_synth'):
    """
    Filters for the perfectly-aligned 'maps-synth' dataset and computes 
    Fine-Grained (S <= 30) vs. Coarse-Grained (S >= 40) onset metrics
    for top-performing scoring methods.
    
    Parameters:
      - df: Input DataFrame with evaluation data.
    
    Returns:
      - final_df: MultiIndex DataFrame comparing fine vs. coarse onset performance.
    """

    # 1. Filter for maps-synth dataset and 'onset' corruption
    is_dataset = df['dataset'].astype(str).str.lower() == dataset.lower()
    is_onset = df['corruption_type'] == 'onset'
    df_smd = df[is_dataset & is_onset].copy()

    # 2. Split into fine-grained (S <= 30 ms) and coarse-grained (S >= 40 ms)
    df_fine = df_smd[df_smd['corruption_amount'].round(2) <= 30].copy()

    # 3. Filter for requested methods
    available_methods = [m for m in score_cols if m in df_fine.columns]

    # 4. Compute metrics across subsets
    mono = _compute_monotonicity(df_fine, available_methods)
    stab = _compute_stability(df_fine, available_methods)
    sens = _compute_sensitivity(df_fine, available_methods)
    cons = _compute_consistency(df_fine, available_methods)

    # 5. Format wide MultiIndex DataFrame
    eval_dict = {
        'Mono': mono,
        'Stab': stab,
        'Sens': sens,
        'Cons': cons
    }

    combined = pd.concat(eval_dict, names=['eval_metric', 'corruption_type'])
    final_df = combined.T.swaplevel(0, 1, axis=1)

    corr_order = ['onset']
    eval_order = ['Mono', 'Stab', 'Sens', 'Cons']

    multi_cols = pd.MultiIndex.from_product(
        [corr_order, eval_order], 
        names=['granularity', 'eval_metric']
    )

    final_df = final_df.reindex(columns=multi_cols)
    final_df.index.name = 'metric'

    return final_df


#onset fine grained, 4 metrics, all datasets
def calculate_onset_fine_by_dataset(df, score_cols):
    """
    Computes all four evaluation metrics (Mono, Stab, Sens, Cons) for 
    Fine-Grained Onset corruption (S <= 30 ms), broken down per dataset.
    
    Returns:
      - final_df: MultiIndex DataFrame indexed by scoring metric with columns 
        (dataset, eval_metric)
    """
    # Filter for 'onset' corruption type and fine-grained severity (S <= 30)
    is_onset = df['corruption_type'].astype(str).str.lower() == 'onset'
    is_fine = df['corruption_amount'].round(2) <= 30
    df_fine = df[is_onset & is_fine].copy()

    datasets = df_fine['dataset'].unique()
    eval_order = ['Mono', 'Stab', 'Sens', 'Cons']
    dataset_tables = {}

    # Compute 4 evaluation metrics for each dataset
    for ds in datasets:
        df_ds = df_fine[df_fine['dataset'] == ds]
        if df_ds.empty:
            continue

        available_methods = [m for m in score_cols if m in df_ds.columns]

        mono = _compute_monotonicity(df_ds, available_methods)
        stab = _compute_stability(df_ds, available_methods)
        sens = _compute_sensitivity(df_ds, available_methods)
        cons = _compute_consistency(df_ds, available_methods)

        # Transpose & combine so rows = scoring methods, columns = eval metrics
        ds_metrics = pd.DataFrame({
            'Mono': mono.loc['onset'],
            'Stab': stab.loc['onset'],
            'Sens': sens.loc['onset'],
            'Cons': cons.loc['onset']
        })

        dataset_tables[ds] = ds_metrics

    # 4. Combine into a single wide MultiIndex DataFrame
    final_df = pd.concat(dataset_tables, axis=1, names=['dataset', 'eval_metric'])

    # 5. Enforce clean column order
    multi_cols = pd.MultiIndex.from_product(
        [datasets, eval_order], 
        names=['dataset', 'eval_metric']
    )
    
    final_df = final_df.reindex(columns=multi_cols)
    final_df.index.name = 'metric'

    return final_df

def main():
    load_dotenv()
    results_path = os.path.join(os.environ["EXPERIMENTS_FOLDER"], 'results')
    datasets= ['maestro', 'smd', 'maps_synth']

    dfs = []
    for dataset in datasets:
        df = pd.read_csv(os.path.join(results_path, f"{dataset}_results.csv"))
        #add a column to identify the dataset
        df['dataset'] = dataset
        dfs.append(df)

    #merge into one dataset
    df = pd.concat(dfs, ignore_index=True)

    #drop rows with 'all' corruption type
    df.drop(df[df['corruption_type'] == 'all'].index, inplace=True)

    methods = ['CS-CENS', 'CS-DLNCO', 'CS-SPEC', 'MrMsDTW-CENS', 'MrMsDTW-DLNCO', 'CLAMP3', 'CLAP', 'Onset-F1', 'Onset-MAE', 'AMT-BP', 'AMT-BD', 'AMT-TK', 'AMT-IOI-BP', 'AMT-IOI-BD', 'AMT-IOI-TK']

    #mean and std of each scoring method
    #get_baseline_summary(df, methods).to_csv(os.path.join(results_path, 'table_baseline_summary.csv'))

    #big main table
    calculate_all_metrics(df, methods).to_csv(os.path.join(results_path, 'table_main.csv'))

    #onset fine grained, 4 metrics, all datasets
    calculate_onset_fine_by_dataset(df, methods).to_csv(os.path.join(results_path, 'table_onset_fine_per_dataset.csv'))

    #monotonicity per dataset table
    #calculate_monotonicity_by_dataset(df, methods).to_csv(os.path.join(results_path, 'table_per_dataset.csv'))

    #onset fine grained X coarse grained for maps table
    #dataset = 'maps-s'
    #get_onset_comparison(df, methods, dataset=dataset).to_csv(os.path.join(results_path, f'table_onset_fine_{dataset}.csv'))



if __name__ == "__main__":
    main()
