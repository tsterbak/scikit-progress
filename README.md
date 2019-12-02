# scikit-progress: progressbars for scikit-learn

Scikit-progress contains wrappers and replacements for common scikit-learn classes to add a progressbar. This should help bridging the gap between classical machine learning and deep learning

## Example
```
from sklearn.ensemble import RandomForestClassifier
from skprog.wrappers import TreesProgressor

rf = TreesProgressor(RandomForestClassifier(n_estimators=100, max_depth=5, oob_score=True))

rf.fit(X, y)
```
