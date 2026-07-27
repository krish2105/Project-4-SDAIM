import numpy as np
import pandas as pd

def run_monte_carlo_simulation(df_predictions, num_simulations=1000):
    """
    Simulates 1,000+ macroeconomic risk scenarios to compute portfolio Value-at-Risk (VaR).
    
    Returns:
        dict containing Mean_Deposit_Loss, VaR_95_%_USD, and simulation distribution array.
    """
    df = df_predictions.copy()
    
    if 'Churn_Probability' in df.columns:
        base_probs = np.array(df['Churn_Probability'].values, dtype=float)
    elif 'Churn_Probability_%' in df.columns:
        base_probs = np.array(df['Churn_Probability_%'].values / 100.0, dtype=float)
    else:
        base_probs = np.full(len(df), 0.5, dtype=float)
        
    balances = np.array(df['Balance'].values, dtype=float) if 'Balance' in df.columns else np.full(len(base_probs), 50000.0, dtype=float)
    
    simulated_losses = []
    
    np.random.seed(42)
    for _ in range(num_simulations):
        # Simulate macroeconomic shocks (+/- 15% shock to churn probabilities)
        macro_shock = np.random.normal(1.0, 0.15)
        shocked_probs = np.clip(base_probs * macro_shock, 0.0, 1.0)
        
        # Monte Carlo Bernoulli trial per customer
        churn_occurred = np.random.binomial(1, shocked_probs)
        total_loss = np.sum(churn_occurred * balances)
        simulated_losses.append(total_loss)
        
    simulated_losses = np.array(simulated_losses)
    
    mean_loss = float(np.mean(simulated_losses))
    var_95 = float(np.percentile(simulated_losses, 95))
    max_loss = float(np.max(simulated_losses))
    
    return {
        'Simulations_Run': num_simulations,
        'Mean_Deposit_Loss_USD': round(mean_loss, 2),
        'VaR_95_USD': round(var_95, 2),  # 95% Confidence maximum deposit loss
        'Worst_Case_Loss_USD': round(max_loss, 2),
        'Loss_Distribution': simulated_losses
    }
