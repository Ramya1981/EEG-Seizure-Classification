# modules/smote_utils.py
from imblearn.over_sampling import SMOTE

def apply_smote(X, y):
    """
    Apply SMOTE to balance classes.
    Input: X (n_samples, n_features), y (n_samples,)
    Output: X_resampled, y_resampled
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


