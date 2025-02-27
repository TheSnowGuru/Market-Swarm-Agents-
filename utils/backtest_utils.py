import os
import uuid
import json
from typing import Dict, Any
import webbrowser

class BacktestResultsManager:
    def __init__(self, base_path='backtest_results'):
        """
        Initialize BacktestResultsManager to manage and share backtest results
        
        Args:
            base_path (str): Base directory to store backtest results
        """
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def save_backtest_results(self, results: Dict[str, Any], agent_name: str = None) -> str:
        """
        Save backtest results and generate a unique shareable link
        
        Args:
            results (dict): Backtest performance metrics and details
            agent_name (str, optional): Name of the agent being backtested
        
        Returns:
            str: Unique URL-like identifier for the backtest results
        """
        # Generate unique identifier
        result_id = str(uuid.uuid4())
        
        # Prepare filename
        filename = f"{agent_name or 'backtest'}_{result_id}.json"
        filepath = os.path.join(self.base_path, filename)
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Generate shareable link
        return f"swarm://backtest/{result_id}"
    
    def open_backtest_results(self, result_link: str):
        """
        Open backtest results based on the generated link
        
        Args:
            result_link (str): Unique backtest results link
        """
        # Extract result ID from link
        result_id = result_link.split('/')[-1]
        
        # Find matching file
        for filename in os.listdir(self.base_path):
            if result_id in filename:
                filepath = os.path.join(self.base_path, filename)
                
                # Open results file
                with open(filepath, 'r') as f:
                    results = json.load(f)
                
                # Print or process results
                print(json.dumps(results, indent=2))
                return results
        
        print(f"No backtest results found for link: {result_link}")
        return None

# Global instance for easy access
backtest_results_manager = BacktestResultsManager()
