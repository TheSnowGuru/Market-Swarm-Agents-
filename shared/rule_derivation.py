# shared/rule_derivation.py

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split

def derive_optimal_trade_rules(data, feature_columns, label_column='Optimal Trade', max_depth=5):
    """
    Derive trading rules from labeled data using a Decision Tree Classifier.
    
    Parameters:
        data (pd.DataFrame): The DataFrame containing features and the 'Optimal Trade' label.
        feature_columns (list): List of column names to be used as features.
        label_column (str): The name of the label column (default 'Optimal Trade').
        max_depth (int): The maximum depth of the decision tree.
        
    Returns:
        DecisionTreeClassifier: The trained decision tree model.
        list of dict: Extracted rules with their win probabilities.
    """
    # Prepare the data
    X = data[feature_columns]
    y = data[label_column]
    
    # Split the data (you can adjust test_size as needed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train the Decision Tree Classifier
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    
    # Extract rules and their win probabilities
    rules = extract_rules_with_probabilities(clf, feature_columns, X_train, y_train)
    
    return clf, rules

def extract_rules_with_probabilities(tree, feature_names, X_train, y_train):
    """
    Extract rules from the decision tree along with their win probabilities.
    
    Parameters:
        tree (DecisionTreeClassifier): The trained decision tree model.
        feature_names (list): List of feature names.
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training label data.
        
    Returns:
        list of dict: Each dict contains 'rule' and 'win_probability'.
    """
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    paths = []
    
    def recurse(node, current_rule):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            # Left child
            left_rule = current_rule.copy()
            left_rule.append(f"({name} <= {threshold})")
            recurse(tree_.children_left[node], left_rule)
            # Right child
            right_rule = current_rule.copy()
            right_rule.append(f"({name} > {threshold})")
            recurse(tree_.children_right[node], right_rule)
        else:
            # Leaf node
            samples_index = get_samples_in_leaf(tree, node)
            y_samples = y_train.iloc[samples_index]
            win_probability = y_samples.mean()
            rule = " AND ".join(current_rule)
            paths.append({'rule': rule, 'win_probability': win_probability})
    
    recurse(0, [])
    return paths

def get_samples_in_leaf(tree, node_id):
    """
    Get the indices of samples that reach the given leaf node.
    
    Parameters:
        tree: The decision tree.
        node_id: The node ID of the leaf.
        
    Returns:
        list: Indices of samples.
    """
    return tree.tree_.value[node_id].flatten()

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def derive_optimal_trade_rules(labeled_data, test_size=0.2, random_state=42):
    """
    Derive trading rules from labeled data using a Decision Tree Classifier.
    
    Parameters:
        labeled_data (pd.DataFrame): DataFrame with features and 'Optimal Trade' column
        test_size (float): Proportion of data to use for testing
        random_state (int): Seed for reproducibility
    
    Returns:
        dict: A dictionary containing trading rules, model performance, and feature importances
    """
    # Validate input
    if labeled_data is None or labeled_data.empty:
        raise ValueError("Input data cannot be empty")
    
    # Separate features and target
    features = labeled_data.drop('Optimal Trade', axis=1)
    target = labeled_data['Optimal Trade']
    
    # Remove non-numeric columns
    features = features.select_dtypes(include=['int64', 'float64'])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=target
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Decision Tree Classifier
    clf = DecisionTreeClassifier(
        max_depth=5,  # Limit depth to prevent overfitting
        min_samples_split=20,  # Ensure meaningful splits
        random_state=random_state
    )
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    train_score = clf.score(X_train_scaled, y_train)
    test_score = clf.score(X_test_scaled, y_test)
    
    # Extract feature importances
    feature_importances = pd.DataFrame({
        'feature': features.columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Extract rules (simplified representation)
    rules = []
    for feature, importance in zip(feature_importances['feature'], feature_importances['importance']):
        if importance > 0.1:  # Only consider significant features
            rules.append({
                'feature': feature,
                'importance': importance
            })
    
    return {
        'model': clf,
        'scaler': scaler,
        'rules': rules,
        'train_accuracy': train_score,
        'test_accuracy': test_score,
        'feature_importances': feature_importances
    }
