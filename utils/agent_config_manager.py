import os
import json
import uuid
from datetime import datetime
import joblib

class AgentConfigManager:
    def __init__(self, base_path='agents/configs'):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        os.makedirs('agents/models', exist_ok=True)

    def generate_agent_config(self, 
                               agent_name, 
                               agent_type, 
                               strategy, 
                               features=None, 
                               feature_params=None, 
                               training_params=None):
        """
        Generate a comprehensive agent configuration
        
        Args:
            agent_name (str): Unique name for the agent
            agent_type (str): Type of agent (scalper, trend-follower, etc.)
            strategy (str): Trading strategy name
            features (list, optional): List of features used
            feature_params (dict, optional): Parameters for feature extraction
            training_params (dict, optional): Training hyperparameters
        
        Returns:
            dict: Comprehensive agent configuration
        """
        return {
            'id': str(uuid.uuid4()),
            'name': agent_name,
            'type': agent_type,
            'strategy': strategy,
            'version': '1.0.0',
            'created_at': datetime.now().isoformat(),
            'features': features or [],
            'feature_parameters': feature_params or {},
            'training_parameters': training_params or {},
            'performance_metrics': {},
            'last_trained': None,
            'total_training_runs': 0
        }

    def save_agent_config(self, config):
        """Save agent configuration to JSON"""
        config_path = os.path.join(self.base_path, f"{config['name']}_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        return config_path

    def load_agent_config(self, agent_name):
        """Load agent configuration from JSON"""
        config_path = os.path.join(self.base_path, f"{agent_name}_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return None

    def save_model(self, agent_name, model, model_path=None):
        """Save trained model"""
        if model_path is None:
            model_path = os.path.join('agents/models', f"{agent_name}_model.pkl")
        
        joblib.dump(model, model_path)
        return model_path

    def update_performance_metrics(self, agent_name, metrics):
        """Update agent's performance metrics"""
        config = self.load_agent_config(agent_name)
        if config:
            config['performance_metrics'] = metrics
            config['last_trained'] = datetime.now().isoformat()
            config['total_training_runs'] += 1
            self.save_agent_config(config)

    def list_agents(self):
        """List all saved agents"""
        return [f.replace('_config.json', '') for f in os.listdir(self.base_path) 
                if f.endswith('_config.json')]
