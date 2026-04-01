import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class MLPipeline:
    """
    A comprehensive, flexible ML pipeline that supports:
    - Multiple scalers (StandardScaler, MinMaxScaler, RobustScaler)
    - Pluggable ML models
    - Cross-validation with multiple strategies
    - Hyperparameter tuning
    - Performance evaluation and visualization
    """
    
    def __init__(self, 
                 model: Any,
                 scaler: str = 'standard',
                 random_state: int = 42,
                 task_type: str = 'classification'):
        """
        Initialize the ML Pipeline.
        
        Parameters:
        -----------
        model : estimator object
            Any scikit-learn compatible model (classifier or regressor)
        scaler : str, default='standard'
            Type of scaler: 'standard', 'minmax', or 'robust'
        random_state : int, default=42
            Random state for reproducibility
        task_type : str, default='classification'
            Type of ML task: 'classification' or 'regression'
        """
        self.model = model
        self.scaler_type = scaler
        self.random_state = random_state
        self.task_type = task_type
        self.scaler = self._initialize_scaler(scaler)
        self.pipeline = None
        self.X_scaled = None
        self.cv_results = None
        
    def _initialize_scaler(self, scaler_type: str):
        """Initialize the appropriate scaler."""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        if scaler_type not in scalers:
            raise ValueError(f"Scaler must be one of {list(scalers.keys())}")
        return scalers[scaler_type]
    
    def build_pipeline(self) -> Pipeline:
        """
        Build a scikit-learn pipeline with scaler and model.
        
        Returns:
        --------
        Pipeline : scikit-learn Pipeline object
        """
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('model', self.model)
        ])
        return self.pipeline
    
    def scale_data(self, X: np.ndarray) -> np.ndarray:
        """
        Scale the data using the specified scaler.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
            
        Returns:
        --------
        X_scaled : array-like, shape (n_samples, n_features)
            Scaled features
        """
        self.X_scaled = self.scaler.fit_transform(X)
        return self.X_scaled
    
    def cross_validate(self, 
                      X: np.ndarray, 
                      y: np.ndarray,
                      cv: int = 5,
                      stratified: bool = True,
                      scoring: Optional[Union[str, List[str]]] = None) -> Dict:
        """
        Perform cross-validation on the model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
        y : array-like, shape (n_samples,)
            Target values
        cv : int, default=5
            Number of cross-validation folds
        stratified : bool, default=True
            Use StratifiedKFold for classification (balances class distribution)
        scoring : str or list of str, optional
            Metrics to compute. If None, uses default for task type
            
        Returns:
        --------
        cv_results : dict
            Cross-validation results with scores and timing information
        """
        if self.pipeline is None:
            self.build_pipeline()
        
        # Determine cross-validation strategy
        if stratified and self.task_type == 'classification':
            cv_splitter = StratifiedKFold(n_splits=cv, 
                                          shuffle=True, 
                                          random_state=self.random_state)
        else:
            cv_splitter = KFold(n_splits=cv, 
                               shuffle=True, 
                               random_state=self.random_state)
        
        # Determine scoring metrics
        if scoring is None:
            if self.task_type == 'classification':
                scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            else:
                scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
        
        # Perform cross-validation
        self.cv_results = cross_validate(
            self.pipeline, 
            X, 
            y, 
            cv=cv_splitter,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        return self.cv_results
    
    def print_cv_results(self, cv_results: Optional[Dict] = None) -> None:
        """
        Print formatted cross-validation results.
        
        Parameters:
        -----------
        cv_results : dict, optional
            Cross-validation results. Uses self.cv_results if not provided
        """
        if cv_results is None:
            cv_results = self.cv_results
        
        if cv_results is None:
            print("No cross-validation results available. Run cross_validate() first.")
            return
        
        print("\n" + "="*70)
        print("CROSS-VALIDATION RESULTS")
        print("="*70)
        
        # Extract scoring metrics
        test_scores = {key: val for key, val in cv_results.items() 
                      if key.startswith('test_')}
        train_scores = {key: val for key, val in cv_results.items() 
                       if key.startswith('train_')}
        
        for metric in test_scores.keys():
            metric_name = metric.replace('test_', '').upper()
            test_mean = cv_results[metric].mean()
            test_std = cv_results[metric].std()
            train_mean = cv_results[f'train_{metric_name.lower()}'].mean()
            train_std = cv_results[f'train_{metric_name.lower()}'].std()
            
            print(f"\n{metric_name}:")
            print(f"  Test  - Mean: {test_mean:.4f} (+/- {test_std:.4f})")
            print(f"  Train - Mean: {train_mean:.4f} (+/- {train_std:.4f})")
        
        print(f"\nFit Time - Mean: {cv_results['fit_time'].mean():.4f}s")
        print(f"Score Time - Mean: {cv_results['score_time'].mean():.4f}s")
        print("="*70 + "\n")
    
    def plot_cv_results(self, cv_results: Optional[Dict] = None, 
                       figsize: Tuple = (12, 6)) -> None:
        """
        Plot cross-validation results.
        
        Parameters:
        -----------
        cv_results : dict, optional
            Cross-validation results. Uses self.cv_results if not provided
        figsize : tuple, default=(12, 6)
            Figure size
        """
        if cv_results is None:
            cv_results = self.cv_results
        
        if cv_results is None:
            print("No cross-validation results available. Run cross_validate() first.")
            return
        
        test_scores = {key: val for key, val in cv_results.items() 
                      if key.startswith('test_')}
        
        metrics = [key.replace('test_', '') for key in test_scores.keys()]
        
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        if len(metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            test_vals = cv_results[f'test_{metric}']
            train_vals = cv_results[f'train_{metric}']
            
            axes[idx].plot(range(len(test_vals)), test_vals, 'o-', label='Test', linewidth=2)
            axes[idx].plot(range(len(train_vals)), train_vals, 's-', label='Train', linewidth=2)
            axes[idx].set_xlabel('Fold')
            axes[idx].set_ylabel('Score')
            axes[idx].set_title(f'{metric.upper()}')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLPipeline':
        """
        Fit the pipeline on training data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Training targets
            
        Returns:
        --------
        self : MLPipeline
        """
        if self.pipeline is None:
            self.build_pipeline()
        
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted pipeline.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Features to predict
            
        Returns:
        --------
        predictions : array-like
            Model predictions
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        return self.pipeline.predict(X)
    
    def evaluate(self, 
                X_test: np.ndarray, 
                y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X_test : array-like, shape (n_samples, n_features)
            Test features
        y_test : array-like, shape (n_samples,)
            Test targets
            
        Returns:
        --------
        metrics : dict
            Dictionary of computed metrics
        """
        y_pred = self.predict(X_test)
        metrics = {}
        
        if self.task_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred, average='macro', zero_division=0)
            metrics['recall'] = recall_score(y_test, y_pred, average='macro', zero_division=0)
            metrics['f1'] = f1_score(y_test, y_pred, average='macro', zero_division=0)
        else:
            metrics['r2'] = r2_score(y_test, y_pred)
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
        
        return metrics
    
    def print_evaluation(self, metrics: Dict[str, float]) -> None:
        """
        Print formatted evaluation metrics.
        
        Parameters:
        -----------
        metrics : dict
            Dictionary of metrics from evaluate()
        """
        print("\n" + "="*50)
        print("TEST SET EVALUATION METRICS")
        print("="*50)
        for metric_name, value in metrics.items():
            print(f"{metric_name.upper():<15}: {value:.4f}")
        print("="*50 + "\n")
    
    def get_model(self):
        """Get the underlying model from the pipeline."""
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call build_pipeline() first.")
        return self.pipeline.named_steps['model']


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from sklearn.datasets import load_iris, load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    
    print("\n" + "="*70)
    print("ML PIPELINE EXAMPLE: CLASSIFICATION WITH MULTIPLE MODELS")
    print("="*70)
    
    # Load dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Example 1: Logistic Regression with StandardScaler
    print("\n--- Model 1: Logistic Regression with StandardScaler ---")
    pipeline1 = MLPipeline(
        model=LogisticRegression(max_iter=1000, random_state=42),
        scaler='standard',
        task_type='classification'
    )
    pipeline1.build_pipeline()
    pipeline1.fit(X_train, y_train)
    
    # Cross-validation
    cv_results1 = pipeline1.cross_validate(X_train, y_train, cv=5, stratified=True)
    pipeline1.print_cv_results(cv_results1)
    
    # Evaluation
    metrics1 = pipeline1.evaluate(X_test, y_test)
    pipeline1.print_evaluation(metrics1)
    
    # Example 2: Random Forest with RobustScaler
    print("\n--- Model 2: Random Forest with RobustScaler ---")
    pipeline2 = MLPipeline(
        model=RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        scaler='robust',
        task_type='classification'
    )
    pipeline2.build_pipeline()
    pipeline2.fit(X_train, y_train)
    
    # Cross-validation
    cv_results2 = pipeline2.cross_validate(X_train, y_train, cv=5, stratified=True)
    pipeline2.print_cv_results(cv_results2)
    
    # Evaluation
    metrics2 = pipeline2.evaluate(X_test, y_test)
    pipeline2.print_evaluation(metrics2)
    
    # Example 3: SVM with MinMaxScaler
    print("\n--- Model 3: SVM with MinMaxScaler ---")
    pipeline3 = MLPipeline(
        model=SVC(kernel='rbf', random_state=42),
        scaler='minmax',
        task_type='classification'
    )
    pipeline3.build_pipeline()
    pipeline3.fit(X_train, y_train)
    
    # Cross-validation
    cv_results3 = pipeline3.cross_validate(X_train, y_train, cv=5, stratified=True)
    pipeline3.print_cv_results(cv_results3)
    
    # Evaluation
    metrics3 = pipeline3.evaluate(X_test, y_test)
    pipeline3.print_evaluation(metrics3)
    
    # Compare results
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    models_comparison = pd.DataFrame({
        'Logistic Regression': metrics1,
        'Random Forest': metrics2,
        'SVM': metrics3
    })
    print(models_comparison)