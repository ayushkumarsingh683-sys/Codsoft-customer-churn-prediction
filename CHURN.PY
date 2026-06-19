import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score

# ==========================================
# 1. LOAD THE DATASET (USING ABSOLUTE PATH)
# ==========================================
try:
    df = pd.read_csv(r"C:\Users\ayush\CodeAlpha_FAQ_Chatbot\CODSOF\CODSOFT.PY3\customer_churn.csv")
    print("Dataset loaded successfully!")
    print(f"Total records found: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print("\n[!] Error: Python cannot find the file at that path.")
    print("Please verify your path string on line 16.")
    import sys
    sys.exit()

# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================
target_column = 'Exited' 

if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found. Check column capitalization.")

# Drop unique identifier columns that will distort machine learning patterns
drop_cols = [col for col in ['RowNumber', 'CustomerId', 'Surname'] if col in df.columns]
drop_cols.append(target_column)

X = df.drop(columns=drop_cols)
y = df[target_column]

# Split into training and testing sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")

# ==========================================
# 3. AUTOMATED PREPROCESSING PIPELINE
# ==========================================
num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Numeric Scaling Transformer
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical Text Encoder Transformer
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

# ==========================================
# 4. MODEL TRAINING AND COMPARISON
# ==========================================
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

print("\n--- Training Models and Evaluating Performance ---")

# Dictionaries to hold performance metrics for visualization
accuracy_scores = []
auc_scores = []
model_names = list(models.keys())

for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    
    # Run predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Save scores for plotting
    accuracy_scores.append(pipeline.score(X_test, y_test))
    auc_scores.append(roc_auc_score(y_test, y_prob))
    
    print(f"\n================ {name} ================")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# ==========================================
# 5. VISUALIZATION: MODEL COMPARISON GRAPH
# ==========================================
x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
rects1 = ax.bar(x - width/2, accuracy_scores, width, label='Accuracy', color='#2b5c8f')
rects2 = ax.bar(x + width/2, auc_scores, width, label='ROC-AUC', color='#d95f02')

ax.set_ylabel('Scores (0.0 to 1.0)')
ax.set_title('Customer Churn Prediction: Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Adding values on top of the bars for clear presentation
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
print("\n[INFO] Displaying comparison graph window...")
plt.show()