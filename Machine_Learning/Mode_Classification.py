# ============================================
# Classificador de Modalidade Musical
# Desafio Psicoacústico: Maior vs. Menor
# ============================================
# Hipótese: A percepção humana de positividade (valence) está 
# correlacionada com a escolha teórica de uma escala Maior (mode)?
# ============================================

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# 1. Carregar o dataset
# ============================================
print("=" * 60)
print("🎵 CLASSIFICADOR DE MODALIDADE MUSICAL (Maior vs. Menor)")
print("=" * 60)

csv_path = Path(__file__).resolve().parent.parent / 'DataSet' / 'spotify_songs.csv'
df = pd.read_csv(csv_path)

print(f"\n📊 Dataset carregado: {len(df)} músicas")
print(f"   Colunas disponíveis: {len(df.columns)}")

# ============================================
# 2. Análise Exploratória da Variável Target
# ============================================
print("\n" + "=" * 60)
print("📈 ANÁLISE EXPLORATÓRIA DO MODE")
print("=" * 60)

mode_counts = df['mode'].value_counts()
print(f"\nDistribuição do Mode:")
print(f"   Mode 1 (Maior): {mode_counts.get(1, 0)} ({mode_counts.get(1, 0)/len(df)*100:.2f}%)")
print(f"   Mode 0 (Menor): {mode_counts.get(0, 0)} ({mode_counts.get(0, 0)/len(df)*100:.2f}%)")

# Análise da correlação entre valence e mode
print(f"\n🔍 Correlação entre Valence e Mode: {df['valence'].corr(df['mode']):.4f}")
print("\n📊 Estatísticas de Valence por Mode:")
print(df.groupby('mode')['valence'].describe())

# ============================================
# 3. Selecionar features e alvo
# ============================================
print("\n" + "=" * 60)
print("🎯 PREPARAÇÃO DOS DADOS")
print("=" * 60)

# Features de áudio disponíveis
audio_features = [
    'danceability', 'energy', 'key', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'duration_ms', 'track_popularity'
]

# Remover linhas com valores faltantes nas features ou no target
df_clean = df[audio_features + ['mode']].dropna()
print(f"\n✓ Dados limpos: {len(df_clean)} músicas (removidos {len(df) - len(df_clean)} registros com valores faltantes)")

X = df_clean[audio_features]
y = df_clean['mode']

print(f"\n✓ Features selecionadas ({len(audio_features)}):")
for feat in audio_features:
    print(f"   • {feat}")
print(f"\n✓ Target: mode (0 = Menor, 1 = Maior)")

# ============================================
# 4. Modelos para comparação
# ============================================
print("\n" + "=" * 60)
print("🤖 CONFIGURAÇÃO DOS MODELOS")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, 
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, 
        max_depth=20,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ),
    "KNN (k=5)": KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1
    )
}

print(f"\n✓ {len(models)} modelos configurados:")
for name in models.keys():
    print(f"   • {name}")

# ============================================
# 5. Validação Cruzada Estratificada
# ============================================
print("\n" + "=" * 60)
print("🔄 VALIDAÇÃO CRUZADA (5-Fold Estratificada)")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

# Escalonamento dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nExecutando validação cruzada...")
for name, model in models.items():
    print(f"   → Testando {name}...", end=" ")
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy', n_jobs=-1)
    results[name] = {
        "Acurácia Média": np.mean(scores),
        "Desvio Padrão": np.std(scores),
        "Min": np.min(scores),
        "Max": np.max(scores)
    }
    print(f"✓ {np.mean(scores):.4f} (±{np.std(scores):.4f})")

# ============================================
# 6. Resultados da Validação Cruzada
# ============================================
print("\n" + "=" * 60)
print("📊 RESULTADOS DA VALIDAÇÃO CRUZADA")
print("=" * 60)

results_df = pd.DataFrame(results).T.sort_values(by="Acurácia Média", ascending=False)
print("\n" + results_df.to_string())

# ============================================
# 7. Treinar e avaliar o melhor modelo
# ============================================
melhor_modelo_nome = results_df.index[0]
melhor_acuracia = results_df.loc[melhor_modelo_nome, "Acurácia Média"]

print("\n" + "=" * 60)
print(f"🏆 MELHOR MODELO: {melhor_modelo_nome}")
print(f"   Acurácia Média: {melhor_acuracia:.4f}")
print("=" * 60)

# Divisão treino/teste estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\n✓ Divisão dos dados:")
print(f"   Treino: {len(X_train)} amostras")
print(f"   Teste:  {len(X_test)} amostras")

# Treinar o melhor modelo
melhor_modelo = models[melhor_modelo_nome]
print(f"\n🔧 Treinando {melhor_modelo_nome}...")
melhor_modelo.fit(X_train, y_train)

# Predições
y_pred = melhor_modelo.predict(X_test)
y_pred_proba = melhor_modelo.predict_proba(X_test)[:, 1] if hasattr(melhor_modelo, 'predict_proba') else None

# ============================================
# 8. Classification Report
# ============================================
print("\n" + "=" * 60)
print("📋 CLASSIFICATION REPORT")
print("=" * 60)
print("\nClasses: 0 = Menor (Minor), 1 = Maior (Major)\n")
print(classification_report(y_test, y_pred, target_names=['Menor (0)', 'Maior (1)'], digits=4))

# ============================================
# 9. Métricas Adicionais
# ============================================
print("=" * 60)
print("📊 MÉTRICAS ADICIONAIS")
print("=" * 60)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusão:")
print(f"                 Predito")
print(f"              Menor  Maior")
print(f"Real  Menor  {cm[0][0]:6d} {cm[0][1]:6d}")
print(f"      Maior  {cm[1][0]:6d} {cm[1][1]:6d}")

# ROC-AUC Score
if y_pred_proba is not None:
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n🎯 ROC-AUC Score: {roc_auc:.4f}")

# ============================================
# 10. Análise de Feature Importance
# ============================================
if hasattr(melhor_modelo, 'feature_importances_'):
    print("\n" + "=" * 60)
    print("🔍 IMPORTÂNCIA DAS FEATURES")
    print("=" * 60)
    
    feature_importance = pd.DataFrame({
        'Feature': audio_features,
        'Importance': melhor_modelo.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\n" + feature_importance.to_string(index=False))
    
    valence_importance = feature_importance[feature_importance['Feature'] == 'valence']['Importance'].values[0]
    valence_rank = (feature_importance['Feature'] == 'valence').idxmax() + 1
    print(f"\n🎵 Valence - Ranking: #{valence_rank} | Importância: {valence_importance:.4f}")

elif hasattr(melhor_modelo, 'coef_'):
    print("\n" + "=" * 60)
    print("🔍 COEFICIENTES DO MODELO")
    print("=" * 60)
    
    coef_df = pd.DataFrame({
        'Feature': audio_features,
        'Coefficient': melhor_modelo.coef_[0]
    }).sort_values(by='Coefficient', key=abs, ascending=False)
    
    print("\n" + coef_df.to_string(index=False))
    
    valence_coef = coef_df[coef_df['Feature'] == 'valence']['Coefficient'].values[0]
    print(f"\n🎵 Valence - Coeficiente: {valence_coef:.4f}")

# ============================================
# 11. Comparação de TODOS os Modelos no Teste
# ============================================
print("\n" + "=" * 60)
print("🔬 COMPARAÇÃO DE TODOS OS MODELOS NO CONJUNTO DE TESTE")
print("=" * 60)

test_results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    test_score = model.score(X_test, y_test)
    test_results.append({'Modelo': name, 'Acurácia no Teste': test_score})

test_results_df = pd.DataFrame(test_results).sort_values(by='Acurácia no Teste', ascending=False)
print("\n" + test_results_df.to_string(index=False))

# ============================================
# 12. Conclusão
# ============================================
print("\n" + "=" * 60)
print("CONCLUSÃO")
print("=" * 60)

print(f"\n🏆 Melhor Modelo: {melhor_modelo_nome}")
print(f"   Acurácia: {melhor_acuracia:.4f}")