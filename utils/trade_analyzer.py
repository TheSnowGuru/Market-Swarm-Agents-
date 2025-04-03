import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

class TradeAnalyzer:
    """
    Analyze and filter synthetic trades to identify profitable patterns
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the trade analyzer
        
        Args:
            config (dict): Configuration parameters for trade analysis
        """
        self.config = config or {}
        self.default_config = {
            'min_profit_threshold': 1.0,     # Minimum profit percentage to consider a trade profitable
            'min_risk_reward': 1.5,          # Minimum risk/reward ratio
            'max_duration': 100,             # Maximum trade duration in bars
            'min_win_rate': 0.6,             # Minimum win rate for pattern recognition
            'cluster_count': 3,              # Number of clusters for pattern recognition
            'feature_importance_threshold': 0.1  # Threshold for feature importance
        }
        
        # Update default config with provided config
        self.default_config.update(self.config)
        self.config = self.default_config
        
        # Initialize storage
        self.trades_df = None
        self.filtered_trades = None
        self.trade_patterns = None
        self.feature_importance = None
    
    def load_trades(self, file_path: str) -> pd.DataFrame:
        """
        Load trades from CSV file
        
        Args:
            file_path (str): Path to CSV file with trade data
            
        Returns:
            pd.DataFrame: DataFrame with loaded trades
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Trade file not found: {file_path}")
        
        self.trades_df = pd.read_csv(file_path)
        print(f"Loaded {len(self.trades_df)} trades from {file_path}")
        return self.trades_df
    
    def filter_profitable_trades(self, 
                                min_profit: Optional[float] = None,
                                min_rr: Optional[float] = None,
                                max_duration: Optional[int] = None) -> pd.DataFrame:
        """
        Filter trades based on profitability criteria
        
        Args:
            min_profit (float, optional): Minimum profit percentage
            min_rr (float, optional): Minimum risk/reward ratio
            max_duration (int, optional): Maximum trade duration
            
        Returns:
            pd.DataFrame: Filtered trades
        """
        if self.trades_df is None:
            raise ValueError("No trades loaded. Call load_trades() first.")
        
        # Use provided parameters or defaults from config
        min_profit = min_profit if min_profit is not None else self.config['min_profit_threshold']
        min_rr = min_rr if min_rr is not None else self.config['min_risk_reward']
        max_duration = max_duration if max_duration is not None else self.config['max_duration']
        
        # Filter by profit
        filtered = self.trades_df[self.trades_df['pnl_pct'] >= min_profit]
        
        # Filter by risk/reward if the column exists
        if 'risk_reward_realized' in filtered.columns:
            filtered = filtered[filtered['risk_reward_realized'] >= min_rr]
        
        # Filter by duration if the column exists
        if 'duration' in filtered.columns:
            filtered = filtered[filtered['duration'] <= max_duration]
        
        self.filtered_trades = filtered
        
        print(f"Filtered to {len(filtered)} profitable trades (from {len(self.trades_df)} total)")
        return filtered
    
    def identify_trade_patterns(self, n_clusters: Optional[int] = None) -> Dict[str, Any]:
        """
        Identify patterns in profitable trades using clustering
        
        Args:
            n_clusters (int, optional): Number of clusters for pattern recognition
            
        Returns:
            dict: Identified patterns with statistics
        """
        if self.filtered_trades is None:
            raise ValueError("No filtered trades. Call filter_profitable_trades() first.")
        
        n_clusters = n_clusters if n_clusters is not None else self.config['cluster_count']
        
        # Select numeric features for clustering
        feature_cols = [col for col in self.filtered_trades.columns 
                       if col.startswith('entry_') and 
                       col not in ['entry_time', 'entry_price'] and
                       pd.api.types.is_numeric_dtype(self.filtered_trades[col])]
        
        if not feature_cols:
            raise ValueError("No numeric feature columns found for pattern recognition")
        
        # Prepare data for clustering
        X = self.filtered_trades[feature_cols].copy()
        
        # Handle missing values
        X.fillna(X.mean(), inplace=True)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to filtered trades
        self.filtered_trades['cluster'] = clusters
        
        # Analyze patterns by cluster
        patterns = {}
        for cluster_id in range(n_clusters):
            cluster_trades = self.filtered_trades[self.filtered_trades['cluster'] == cluster_id]
            
            # Calculate statistics for this cluster
            pattern = {
                'cluster_id': cluster_id,
                'trade_count': len(cluster_trades),
                'avg_profit': cluster_trades['pnl_pct'].mean(),
                'win_rate': (cluster_trades['pnl_pct'] > 0).mean(),
                'avg_duration': cluster_trades['duration'].mean() if 'duration' in cluster_trades.columns else None,
                'feature_values': {}
            }
            
            # Calculate average feature values for this cluster
            for col in feature_cols:
                pattern['feature_values'][col] = cluster_trades[col].mean()
            
            patterns[f"pattern_{cluster_id}"] = pattern
        
        self.trade_patterns = patterns
        return patterns
    
    def calculate_feature_importance(self) -> Dict[str, float]:
        """
        Calculate importance of features for profitable trades
        
        Returns:
            dict: Feature importance scores
        """
        if self.filtered_trades is None:
            raise ValueError("No filtered trades. Call filter_profitable_trades() first.")
        
        # Select feature columns (entry indicators)
        feature_cols = [col for col in self.filtered_trades.columns 
                       if col.startswith('entry_') and 
                       col not in ['entry_time', 'entry_price'] and
                       pd.api.types.is_numeric_dtype(self.filtered_trades[col])]
        
        if not feature_cols:
            raise ValueError("No feature columns found for importance calculation")
        
        # Calculate correlation with profit
        importance = {}
        for col in feature_cols:
            # Calculate absolute correlation with profit
            corr = abs(self.filtered_trades[col].corr(self.filtered_trades['pnl_pct']))
            importance[col] = corr
        
        # Normalize importance scores
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        # Sort by importance
        importance = {k: v for k, v in sorted(importance.items(), key=lambda item: item[1], reverse=True)}
        
        self.feature_importance = importance
        return importance
    
    def generate_trade_rules(self) -> List[Dict[str, Any]]:
        """
        Generate trading rules based on profitable patterns
        
        Returns:
            list: List of trading rules
        """
        if self.trade_patterns is None:
            raise ValueError("No trade patterns identified. Call identify_trade_patterns() first.")
        
        if self.feature_importance is None:
            self.calculate_feature_importance()
        
        rules = []
        
        # Generate rules for each pattern
        for pattern_id, pattern in self.trade_patterns.items():
            # Skip patterns with low win rate
            if pattern['win_rate'] < self.config['min_win_rate']:
                continue
            
            # Get top features by importance
            top_features = {k: v for k, v in self.feature_importance.items() 
                           if v >= self.config['feature_importance_threshold']}
            
            if not top_features:
                continue
            
            # Create rule conditions based on pattern's feature values
            conditions = []
            for feature, importance in top_features.items():
                feature_value = pattern['feature_values'].get(feature)
                if feature_value is not None:
                    # Determine operator based on feature name
                    if 'rsi' in feature.lower():
                        if feature_value < 30:
                            conditions.append({
                                'feature': feature,
                                'operator': 'below',
                                'threshold': 30,
                                'importance': importance
                            })
                        elif feature_value > 70:
                            conditions.append({
                                'feature': feature,
                                'operator': 'above',
                                'threshold': 70,
                                'importance': importance
                            })
                    elif 'macd' in feature.lower() and 'hist' in feature.lower():
                        if feature_value > 0:
                            conditions.append({
                                'feature': feature,
                                'operator': 'above',
                                'threshold': 0,
                                'importance': importance
                            })
                        else:
                            conditions.append({
                                'feature': feature,
                                'operator': 'below',
                                'threshold': 0,
                                'importance': importance
                            })
                    else:
                        # Generic condition based on average value
                        conditions.append({
                            'feature': feature,
                            'operator': 'close_to',
                            'threshold': feature_value,
                            'importance': importance
                        })
            
            # Create rule
            rule = {
                'pattern_id': pattern_id,
                'conditions': conditions,
                'expected_profit': pattern['avg_profit'],
                'win_rate': pattern['win_rate'],
                'trade_count': pattern['trade_count']
            }
            
            rules.append(rule)
        
        return rules
    
    def visualize_patterns(self, output_dir: str = 'data/analysis') -> str:
        """
        Visualize trade patterns and feature importance
        
        Args:
            output_dir (str): Directory to save visualizations
            
        Returns:
            str: Path to output directory
        """
        if self.filtered_trades is None or self.trade_patterns is None:
            raise ValueError("No patterns to visualize. Run analysis first.")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Profit distribution by pattern
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='cluster', y='pnl_pct', data=self.filtered_trades)
        plt.title('Profit Distribution by Pattern')
        plt.xlabel('Pattern Cluster')
        plt.ylabel('Profit %')
        plt.savefig(os.path.join(output_dir, 'profit_by_pattern.png'))
        
        # 2. Feature importance
        if self.feature_importance:
            plt.figure(figsize=(12, 8))
            features = list(self.feature_importance.keys())
            importance = list(self.feature_importance.values())
            
            # Sort by importance
            sorted_idx = np.argsort(importance)
            plt.barh([features[i] for i in sorted_idx], [importance[i] for i in sorted_idx])
            plt.title('Feature Importance for Profitable Trades')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        
        # 3. Pattern characteristics
        if len(self.trade_patterns) > 0:
            # Create a summary table of patterns
            pattern_summary = pd.DataFrame({
                'Pattern': list(self.trade_patterns.keys()),
                'Trade Count': [p['trade_count'] for p in self.trade_patterns.values()],
                'Avg Profit %': [p['avg_profit'] for p in self.trade_patterns.values()],
                'Win Rate': [p['win_rate'] for p in self.trade_patterns.values()]
            })
            
            pattern_summary.to_csv(os.path.join(output_dir, 'pattern_summary.csv'), index=False)
        
        print(f"Visualizations saved to {output_dir}")
        return output_dir
    
    def save_filtered_trades(self, output_path: str = None) -> str:
        """
        Save filtered profitable trades to CSV
        
        Args:
            output_path (str, optional): Path to save filtered trades
            
        Returns:
            str: Path to saved file
        """
        if self.filtered_trades is None:
            raise ValueError("No filtered trades to save")
        
        if output_path is None:
            # Generate default path
            os.makedirs('data/filtered_trades', exist_ok=True)
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            output_path = f'data/filtered_trades/profitable_trades_{timestamp}.csv'
        
        # Save to CSV
        self.filtered_trades.to_csv(output_path, index=False)
        print(f"Saved {len(self.filtered_trades)} filtered trades to {output_path}")
        
        return output_path
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for filtered trades
        
        Returns:
            dict: Summary statistics
        """
        if self.filtered_trades is None:
            raise ValueError("No filtered trades. Run analysis first.")
        
        stats = {
            'total_trades': len(self.trades_df) if self.trades_df is not None else 0,
            'profitable_trades': len(self.filtered_trades),
            'profit_ratio': len(self.filtered_trades) / len(self.trades_df) if self.trades_df is not None and len(self.trades_df) > 0 else 0,
            'avg_profit': self.filtered_trades['pnl_pct'].mean(),
            'max_profit': self.filtered_trades['pnl_pct'].max(),
            'min_profit': self.filtered_trades['pnl_pct'].min(),
            'std_profit': self.filtered_trades['pnl_pct'].std(),
            'pattern_count': len(self.trade_patterns) if self.trade_patterns is not None else 0
        }
        
        # Add duration statistics if available
        if 'duration' in self.filtered_trades.columns:
            stats.update({
                'avg_duration': self.filtered_trades['duration'].mean(),
                'max_duration': self.filtered_trades['duration'].max(),
                'min_duration': self.filtered_trades['duration'].min()
            })
        
        return stats
