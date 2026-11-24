# -*- coding: utf-8 -*-
"""
Transfer Learning for Runoff Prediction (W1 -> W2)
--------------------------------------------------
1. Train on Watershed 1 (Keleke) - Source Domain
2. Transfer to Watershed 2 (Manas) - Target Domain
3. Fine-tune using Monthly Observations (Pseudo-labeling)
4. Evaluate using Daily Observations (Hidden Ground Truth)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import warnings

warnings.filterwarnings('ignore')

# =========================================================================================
# 1. Configuration
# =========================================================================================
class Config:
    # Files
    FILE_W1_DAILY = r"D:\Users\mweis\Desktop\论文数据\迁移学习-月度数据降尺度\3.3迁移学习模型验证\目标域模型迁移\迁移\克勒克水文站.xlsx"
    FILE_W2_DAILY = r"D:\Users\mweis\Desktop\论文数据\迁移学习-月度数据降尺度\3.3迁移学习模型验证\目标域模型迁移\迁移\玛纳斯河水文站.xlsx"
    FILE_W2_MONTHLY = r"D:\Users\mweis\Desktop\论文数据\迁移学习-月度数据降尺度\3.3迁移学习模型验证\目标域模型迁移\迁移\月径流-玛纳斯河.xlsx"
    
    OUTPUT_DIR = r"D:\Users\mweis\Desktop\论文数据\迁移学习-月度数据降尺度\3.3迁移学习模型验证\目标域模型迁移\迁移"
    RESULT_EXCEL = os.path.join(OUTPUT_DIR, "Transfer_Learning_Results.xlsx")
    # 只修改这里：改为 PDF
    FIG_DAILY_TS = os.path.join(OUTPUT_DIR, "Transfer_Final_Daily_TimeSeries.pdf")
    FIG_DAILY_SC = os.path.join(OUTPUT_DIR, "Transfer_Final_Daily_Scatter.pdf")
    
    # Model Params (Best from W1 optimization)
    CB_PARAMS = {
        'n_estimators': 3000,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_state': 42,
        'verbose': False,
        'allow_writing_files': False,
        'early_stopping_rounds': 300,
        'bagging_temperature': 0.5,
        'subsample': 0.8
    }

# Plotting Style
rcParams['font.sans-serif'] = ['Times New Roman']
rcParams['axes.unicode_minus'] = False
rcParams['font.weight'] = 'bold'
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titleweight'] = 'bold'
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
plt.rcParams['figure.dpi'] = 300

# =========================================================================================
# 2. Data Processing
# =========================================================================================
def feature_engineering(df):
    """Add meteorological lagged features (Physical mechanism)"""
    df = df.copy()
    
    # Time features
    df['Month'] = df.index.month
    df['DOY'] = df.index.dayofyear
    
    # Rolling Stats
    for window in [3, 7, 15]:
        df[f'Prec_sum_{window}d'] = df['Prec'].rolling(window=window, min_periods=1).sum()
        
    for window in [3, 7]:
        df[f'Temp_mean_{window}d'] = df['Temp'].rolling(window=window, min_periods=1).mean()
        
    for window in [7, 30]:
        df[f'RZSM_mean_{window}d'] = df['RZSM'].rolling(window=window, min_periods=1).mean()

    return df

def load_data():
    print("[Info] Loading Datasets...")
    
    # --- Load W1 (Source) ---
    df1 = pd.read_excel(Config.FILE_W1_DAILY)
    if 'Date' in df1.columns:
        df1['Date'] = pd.to_datetime(df1['Date'])
        df1.set_index('Date', inplace=True)
    df1 = df1.dropna(subset=['Runoff'])
    df1 = feature_engineering(df1)
    
    # --- Load W2 (Target Daily - for Eval) ---
    df2 = pd.read_excel(Config.FILE_W2_DAILY)
    # Standardize W2 columns
    col_map = {'valid_time': 'Date', 'runoff': 'Runoff'}
    df2 = df2.rename(columns=col_map)
    df2['Date'] = pd.to_datetime(df2['Date'])
    df2.set_index('Date', inplace=True)
    
    # Ensure W2 has same features as W1
    # Check for 'Wind2' and remove if present
    if 'Wind2' in df2.columns:
        df2 = df2.drop(columns=['Wind2'])
        
    # Align features (keep only intersection + engineered)
    # Actually, we should use the same features as W1 trained on.
    # Let's define the feature list based on W1.
    common_cols = [c for c in df1.columns if c != 'Runoff']
    
    # Feature Engineering for W2
    df2 = feature_engineering(df2)
    
    # Check if W2 has all columns W1 has
    missing_cols = set(common_cols) - set(df2.columns)
    if missing_cols:
        print(f"[Warning] W2 missing columns: {missing_cols}")
        # Handle missing if critical, or fill 0
    
    # Select features
    X1 = df1[common_cols]
    y1 = df1['Runoff']
    
    X2 = df2[common_cols]
    y2_daily_true = df2['Runoff'] # Hidden ground truth
    
    # --- Load W2 (Target Monthly - for Fine-tuning) ---
    df2_mon = pd.read_excel(Config.FILE_W2_MONTHLY)
    df2_mon['Date'] = pd.to_datetime(df2_mon['Date'])
    df2_mon.set_index('Date', inplace=True)
    # Ensure monthly index is PeriodIndex or normalized to 1st of month
    df2_mon.index = df2_mon.index.to_period('M').to_timestamp()
    
    return X1, y1, X2, y2_daily_true, df2_mon

# =========================================================================================
# 3. Model Training & Transfer
# =========================================================================================
def train_and_transfer():
    X1, y1, X2, y2_daily_true, df2_mon_obs = load_data()
    
    # 1. Train on W1 (Source)
    print("[Step 1] Training Source Model on Watershed 1...")
    model_source = CatBoostRegressor(**Config.CB_PARAMS)
    model_source.fit(X1, y1)
    
    # 2. Initial Predict on W2 (Zero-shot)
    print("[Step 2] Zero-shot Prediction on Watershed 2...")
    pred_w2_init = model_source.predict(X2)
    pred_w2_init = pd.Series(pred_w2_init, index=X2.index, name='Runoff_Init')
    
    # 3. Fine-tuning with Monthly Data (Pseudo-labeling)
    print("[Step 3] Fine-tuning using Monthly Observations...")
    
    # Aggregate predictions to monthly mean
    pred_w2_mon_mean = pred_w2_init.resample('M').mean()
    # Normalize index to start of month to match Obs
    pred_w2_mon_mean.index = pred_w2_mon_mean.index.to_period('M').to_timestamp()
    
    # Align Obs and Pred Monthly
    # df2_mon_obs has 'Monthly_Average_Runoff'
    obs_w2_mon = df2_mon_obs['Monthly_Average_Runoff']
    
    # Create common index
    common_months = pred_w2_mon_mean.index.intersection(obs_w2_mon.index)
    
    # Calculate Correction Factor (Ratio)
    # Factor = Obs / Pred
    # Handle division by zero
    pred_subset = pred_w2_mon_mean.loc[common_months]
    obs_subset = obs_w2_mon.loc[common_months]
    
    ratios = obs_subset / pred_subset.replace(0, 1) # Avoid div/0
    
    # Broadcast ratios to daily
    # Create a Series with daily index and corresponding monthly ratio
    daily_ratios = pd.Series(index=X2.index, dtype=float)
    for month in common_months:
        mask = (X2.index.year == month.year) & (X2.index.month == month.month)
        daily_ratios.loc[mask] = ratios.loc[month]
        
    # Fill missing ratios (months without obs) with 1.0
    daily_ratios = daily_ratios.fillna(1.0)
    
    # Create Pseudo-Labels
    y2_pseudo = pred_w2_init * daily_ratios
    
    # Retrain on W2 using Pseudo-Labels
    # We use the source model parameters, but train on W2 data + Pseudo Labels
    # Since the scale difference is huge (W1~30, W2~670), fine-tuning init_model might be slow/hard.
    # Training from scratch on pseudo-labels is often more robust for large scale shifts.
    print("         Retraining on W2 with Pseudo-Labels (From Scratch)...")
    model_finetuned = CatBoostRegressor(**Config.CB_PARAMS)
    model_finetuned.fit(X2, y2_pseudo) # Removed init_model=model_source
    
    # 4. Final Prediction
    print("[Step 4] Final Prediction on Watershed 2...")
    pred_w2_final = model_finetuned.predict(X2)
    pred_w2_final = pd.Series(pred_w2_final, index=X2.index, name='Runoff_Final')
    
    # Prepare Result DataFrame
    results = pd.DataFrame({
        'Date': X2.index,
        'Observed_Daily': y2_daily_true,
        'Simulated_ZeroShot': pred_w2_init,
        'Simulated_FineTuned': pred_w2_final,
        'Pseudo_Label': y2_pseudo
    })
    
    return results

# =========================================================================================
# 4. Evaluation & Plotting
# =========================================================================================
def calculate_metrics(observed, predicted):
    r2 = r2_score(observed, predicted)
    rmse = np.sqrt(mean_squared_error(observed, predicted))
   # KGE
    if len(observed) > 1 and np.std(observed) > 0 and np.std(predicted) > 0:
        r = np.corrcoef(observed, predicted)[0, 1]
        alpha = np.std(predicted) / np.std(observed)
        beta = np.mean(predicted) / np.mean(observed)
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    else:
        kge = np.nan

    # NSE
    mean_obs = np.mean(observed)
    den = np.sum((observed - mean_obs) ** 2)
    nse = 1 - np.sum((observed - predicted) ** 2) / den if den != 0 else np.nan
    
    threshold = np.percentile(observed, 98)
    high_mask = observed > threshold
    if np.sum(high_mask) > 0:
        fhv = np.sum(predicted[high_mask] - observed[high_mask]) / np.sum(observed[high_mask])
    else:
        fhv = np.nan
        
    # e_t & e_peak (New Definition: Median of top 2 peaks per year)
    df_eval = pd.DataFrame({'Obs': observed, 'Sim': predicted}, index=observed.index)
    df_eval['Year'] = df_eval.index.year
    unique_years = df_eval['Year'].unique()
    
    et_list = []
    epeak_list = []
    
    for yr in unique_years:
        df_yr = df_eval[df_eval['Year'] == yr]
        if len(df_yr) < 30: continue
        
        # Top 2 Obs
        obs_vals = df_yr['Obs'].values
        peaks_idx = []
        for i in range(1, len(obs_vals)-1):
            if obs_vals[i] > obs_vals[i-1] and obs_vals[i] > obs_vals[i+1]:
                peaks_idx.append(i)
        
        if len(peaks_idx) < 2:
            top2_idx = df_yr['Obs'].nlargest(2).index
        else:
            peaks_idx = sorted(peaks_idx, key=lambda i: obs_vals[i], reverse=True)
            top2_local_idx = peaks_idx[:2]
            top2_idx = df_yr.index[top2_local_idx]
            
        # 1. e_peak: Error of Sim at Obs peak dates
        if len(top2_idx) >= 2:
            for i in range(2):
                obs_date = top2_idx[i]
                obs_val = df_yr.loc[obs_date, 'Obs']
                sim_val = df_yr.loc[obs_date, 'Sim']
                epeak_list.append((sim_val - obs_val) / obs_val if obs_val != 0 else 0)

        # 2. e_t: Time difference between Obs Top 2 and Sim Top 2 (Independent)
        # Find Top 2 independent peaks in Sim
        sim_vals = df_yr['Sim'].values
        sim_peaks_idx = []
        for i in range(1, len(sim_vals)-1):
            if sim_vals[i] > sim_vals[i-1] and sim_vals[i] > sim_vals[i+1]:
                sim_peaks_idx.append(i)
        
        if len(sim_peaks_idx) < 2:
             top2_sim_idx = df_yr['Sim'].nlargest(2).index
        else:
             sim_peaks_idx = sorted(sim_peaks_idx, key=lambda i: sim_vals[i], reverse=True)
             top2_sim_local_idx = sim_peaks_idx[:2]
             top2_sim_idx = df_yr.index[top2_sim_local_idx]
             
        # Calculate e_t
        if len(top2_idx) >= 2 and len(top2_sim_idx) >= 2:
            for i in range(2):
                dt = abs((top2_idx[i] - top2_sim_idx[i]).days)
                et_list.append(dt / 365.0)
            
    e_t = np.median(et_list) if et_list else np.nan
    e_peak = np.median(epeak_list) if epeak_list else np.nan

    return {
        'R2': r2, 'RMSE': rmse, 'FHV': fhv, 'NSE': nse, 'KGE': kge,
        'e_t': e_t, 'e_peak': e_peak
    }

def plot_results(df, save_path_ts, save_path_sc):
    """Generate Daily Time Series and Daily Scatter Plot separately"""
    obs = df['Observed_Daily']
    sim = df['Simulated_FineTuned']
    
    # Calculate Metrics
    m = calculate_metrics(obs, sim)
    
    # --- 1. Daily Time Series ---
    print(f"[Plotting] Daily Time Series -> {save_path_ts}")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot Lines
    # Use index (Date) for plotting to show correct X-axis
    ax.plot(obs.index, obs.values, 'k-', lw=1, alpha=0.8, label='Observed')
    ax.plot(sim.index, sim.values, 'r-', lw=1, alpha=0.7, label='Simulated')
    
    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2.0); ax.spines['bottom'].set_linewidth(2.0)
    ax.tick_params(direction='in', width=2.0, length=6)
    
    ax.set_ylabel('Runoff (m$^3$/s)', fontsize=24, fontweight='bold')
    ax.set_title('(a) Transfer learning method', loc='left', fontsize=30, fontweight='bold', fontname='Times New Roman', pad=12)
    
    # Metrics Text
    # 3 lines: NSE KGE / et epeak / FHV
    txt = f"NSE={m['NSE']:.3f}  KGE={m['KGE']:.3f}\ne$_\\Delta$$_t$={m['e_t']:.3f}  e$_p$$_e$$_a$$_k$={m['e_peak']:.3f}\nFHV={m['FHV']:.3f}"
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=24, fontweight='bold', fontname='Times New Roman', va='top', linespacing=1.8)
    
    ax.legend(loc='upper right', fontsize=24, frameon=False)
    ax.set_ylim(0, max(obs.max(), sim.max()) + 1000)
    
    # Tick labels
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(20)
        label.set_fontweight('bold')
        label.set_fontname('Times New Roman')
    
    plt.tight_layout()
    plt.savefig(save_path_ts, format='pdf', bbox_inches='tight')
    plt.close()
    
    # --- 2. Daily Scatter (Daily Error) ---
    print(f"[Plotting] Daily Scatter -> {save_path_sc}")
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter (No label)
    ax.scatter(obs, sim, color='#2ca02c', alpha=0.5, s=15, edgecolors='none')
    
    # 1:1 Line (Add label)
    min_v, max_v = min(obs.min(), sim.min()), max(obs.max(), sim.max())
    ax.plot([min_v, max_v], [min_v, max_v], 'r--', lw=1.5, label='1:1 Line')
    
    # Fit Line (Add label)
    z = np.polyfit(obs, sim, 1)
    ax.plot(obs, np.poly1d(z)(obs), 'b-', lw=1.5, label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')
    
    # Add Legend
    leg = ax.legend(loc='lower right', fontsize=24, frameon=False)
    for text in leg.get_texts():
        text.set_fontname('Times New Roman')
        text.set_fontweight('bold')
    
    # Style
    # ax.spines['top'].set_visible(False) # Keep all spines for scatter
    # ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    ax.tick_params(direction='in', width=2.0, length=6)
    
    # Increase fonts to match Fig 2 style
    ax.set_xlabel('Observed runoff (m$^3$/s)', fontsize=24, fontweight='bold', fontname='Times New Roman')
    ax.set_ylabel('Simulated runoff (m$^3$/s)', fontsize=24, fontweight='bold', fontname='Times New Roman')
    ax.set_title('(a) Transfer learning method', loc='left', fontsize=30, fontweight='bold', fontname='Times New Roman', pad=12)
    
    # Metrics Text
    # 5 lines: NSE / KGE / et / epeak / FHV
    txt = f"NSE={m['NSE']:.3f}\nKGE={m['KGE']:.3f}\ne$_\\Delta$$_t$={m['e_t']:.3f}\ne$_p$$_e$$_a$$_k$={m['e_peak']:.3f}\nFHV={m['FHV']:.3f}"
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=24, fontweight='bold', fontname='Times New Roman', va='top', linespacing=1.8)
    
    # Tick labels
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(20)
        label.set_fontweight('bold')
        label.set_fontname('Times New Roman')
        
    # Limit to 0-3000 as requested
    ax.set_xlim(0, 3000)
    ax.set_ylim(0, 3000)
    
    # Custom ticks
    ticks = [500, 1500, 2500]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    
    plt.tight_layout()
    plt.savefig(save_path_sc, format='pdf', bbox_inches='tight')
    plt.close()

def main():
    print("Starting Transfer Learning Pipeline...")
    results = train_and_transfer()
    
    print("Saving Results...")
    results.to_excel(Config.RESULT_EXCEL)
    
    print("Generating Plots...")
    plot_results(results, Config.FIG_DAILY_TS, Config.FIG_DAILY_SC)
    
    print("Done!")

if __name__ == "__main__":
    main()