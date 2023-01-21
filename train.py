import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Generating Random Data.
X, y = make_blobs(n_samples=10_000, centers=3,
                  n_features=2, cluster_std=1, 
                  center_box=(-3.0, 3.0), random_state=777)

plt.scatter(X[:,0],X[:,1], c=y)
plt.savefig('dataset.png')
plt.show()

# Splitting Data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Training our Model.
xgb = GradientBoostingClassifier(n_estimators=1, learning_rate=0.5, max_depth=1, random_state=0)
xgb.fit(X_train, y_train)

# Results
my_xgb_test_score = xgb.score(X_test, y_test)
print('my_xgb_score is',my_xgb_test_score)

with open('results.txt','w') as my_file:
    my_file.write('Extreme Gradient Boosting Test Score:\n' % my_xgb_test_score)

ConfusionMatrixDisplay.from_estimator(xgb, X_test, y_test)
plt.savefig('Confusion_Matrix.png')
plt.show()

y_pred = xgb.predict(X_test)
class_report = classification_report(y_test, y_pred)
print(class_report)
