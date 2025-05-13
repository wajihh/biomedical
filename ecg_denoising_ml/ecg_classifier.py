from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np

class ECGClassifier:
    """Class to classify ECG beats as normal or arrhythmic."""
    
    def __init__(self):
        """Initialize classifiers."""
        self.models = {
            'SVM': SVC(kernel='rbf'),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'DecisionTree': DecisionTreeClassifier(max_depth=5)
        }
        self.results = {}
        
    def prepare_data(self, features, labels):
        """Prepare data for classification."""
        X = np.array([[f['mean_rr'], f['sdnn'], f['rmssd'], f['pnn50']] for f in features])
        y = np.array(labels)
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train models and evaluate performance."""
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            self.results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
        
        return self.results