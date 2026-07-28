import numpy as np
import pandas as pd

def run_monte_carlo_simulation(df_predictions, num_simulations=1000):
    """
    Simulates 1,000 macroeconomic/market risk scenarios to compute portfolio Value-at-Risk (VaR).
    """
    df = df_predictions.copy()
    
    if 'Churn_Probability' in df.columns:
        base_probs = np.array(df['Churn_Probability'].values, dtype=float)
    elif 'Churn_Probability_%' in df.columns:
        base_probs = np.array(df['Churn_Probability_%'].values / 100.0, dtype=float)
    else:
        base_probs = np.full(len(df), 0.5, dtype=float)
        
    cashbacks = np.array(df['CashBackAmount'].values, dtype=float) if 'CashBackAmount' in df.columns else np.full(len(base_probs), 150.0, dtype=float)
    spends = 800.0 + cashbacks * 4.0
    
    simulated_losses = []
    
    np.random.seed(42)
    for _ in range(num_simulations):
        macro_shock = np.random.normal(1.0, 0.15)
        shocked_probs = np.clip(base_probs * macro_shock, 0.0, 1.0)
        
        churn_occurred = np.random.binomial(1, shocked_probs)
        total_loss = np.sum(churn_occurred * spends)
        simulated_losses.append(total_loss)
        
    simulated_losses = np.array(simulated_losses)
    
    mean_loss = float(np.mean(simulated_losses))
    var_95 = float(np.percentile(simulated_losses, 95))
    max_loss = float(np.max(simulated_losses))
    
    return {
        'Simulations_Run': num_simulations,
        'Mean_Revenue_Loss_USD': round(mean_loss, 2),
        'VaR_95_USD': round(var_95, 2),
        'Worst_Case_Loss_USD': round(max_loss, 2),
        'Loss_Distribution': simulated_losses
    }
