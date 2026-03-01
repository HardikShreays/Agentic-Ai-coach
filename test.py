import joblib
import pandas as pd
import numpy as np

models = {
    "Linear": "models/linear.pkl",
    "Polynomial": "models/polynomial.pkl",
    "Logistic": "models/logistic.pkl",
    "KMeans": "models/kmeans.pkl"
}

for name, path in models.items():
    print(f"\nTesting {name}")
    
    model = joblib.load(path)
    feature_names = model.feature_names_in_
    
    print("Expected feature count:", len(feature_names))
    
    # Generate exactly correct feature structure
    test_df = pd.DataFrame(
        np.random.uniform(0, 100, size=(1, len(feature_names))),
        columns=feature_names
    )
    
    print("Test input shape:", test_df.shape)
    
    output = model.predict(test_df)
    print("Output:", output)