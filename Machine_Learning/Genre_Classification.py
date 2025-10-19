# ============================================
# Classificador de Gênero Musical
# ============================================

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# ============================================
# 1. Carregar o dataset
# ============================================
csv_path = Path(__file__).resolve().parent.parent / 'DataSet' / 'spotify_songs.csv'
df = pd.read_csv(csv_path)

# ============================================
# 2. Engenharia de Features
# ============================================
df['release_year'] = pd.to_datetime(df['track_album_release_date'], errors='coerce').dt.year.fillna(0).astype(int)
subgenre_encoder = LabelEncoder()
df['subgenre_encoded'] = subgenre_encoder.fit_transform(df['playlist_subgenre'])

# ============================================
# 3. Selecionar features e alvo
# ============================================
features = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'duration_ms', 'track_popularity', 'release_year', 'subgenre_encoded'
]
X = df[features]
y = df['playlist_genre']

# ============================================
# 4. Pré-processamento
# ============================================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ============================================
# 5. Modelos leves e rápidos
# ============================================
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=30, class_weight='balanced',
        n_jobs=-1, random_state=42
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300, max_depth=30, class_weight='balanced',
        n_jobs=-1, random_state=42
    ),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight='balanced',
        solver='lbfgs', n_jobs=-1, random_state=42
    )
}

# ============================================
# 6. Validação Cruzada (5-Fold)
# ============================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    scores = cross_val_score(pipeline, X, y_encoded, cv=cv, scoring='accuracy', n_jobs=-1)
    results[name] = {
        "Acurácia Média": np.mean(scores),
        "Desvio Padrão": np.std(scores)
    }

results_df = pd.DataFrame(results).T.sort_values(by="Acurácia Média", ascending=False)
print("=== Resultados da Validação Cruzada (5-Fold) ===")
print(results_df)

# ============================================
# 7. Treinar e avaliar o melhor modelo
# ============================================
melhor_modelo_nome = results_df.index[0]
melhor_modelo = models[melhor_modelo_nome]
print(f"\n🏆 Treinando o melhor modelo: {melhor_modelo_nome}")

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# Pipeline final com SMOTE e escalonamento
final_pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('model', melhor_modelo)
])

final_pipeline.fit(X_train, y_train)
y_pred = final_pipeline.predict(X_test)

# ============================================
# 8. Classification Report
# ============================================
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))
