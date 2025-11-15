# ============================================
# Classificador de Modalidade Musical - VERSÃO APRIMORADA
# Desafio Psicoacústico: Maior vs. Menor
# ============================================
# Hipótese: A percepção humana de positividade (valence) está 
# correlacionada com a escolha teórica de uma escala Maior (mode)?
# ============================================
# Melhorias: Grid Search, SMOTE, Pickle, SHAP
# ============================================

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo dos gráficos
plt.style.use('dark_background')
sns.set_palette("husl")

# ============================================
# 1. Carregar o dataset
# ============================================
print("=" * 80)
print("🎵 CLASSIFICADOR DE MODALIDADE MUSICAL - VERSÃO APRIMORADA")
print("=" * 80)

csv_path = Path(__file__).resolve().parent.parent / 'DataSet' / 'spotify_songs.csv'
df = pd.read_csv(csv_path)

print(f"\n📊 Dataset carregado: {len(df):,} músicas")
print(f"   Colunas disponíveis: {len(df.columns)}")

# ============================================
# 2. Análise Exploratória da Variável Target
# ============================================
print("\n" + "=" * 80)
print("📈 ANÁLISE EXPLORATÓRIA DO MODE")
print("=" * 80)

mode_counts = df['mode'].value_counts()
print(f"\n📊 Distribuição do Mode:")
print(f"   Mode 1 (Maior): {mode_counts.get(1, 0):,} ({mode_counts.get(1, 0)/len(df)*100:.2f}%)")
print(f"   Mode 0 (Menor): {mode_counts.get(0, 0):,} ({mode_counts.get(0, 0)/len(df)*100:.2f}%)")

# Verificar desbalanceamento
desbalanceamento_ratio = mode_counts.max() / mode_counts.min()
print(f"\n⚖️  Razão de Desbalanceamento: {desbalanceamento_ratio:.2f}:1")
if desbalanceamento_ratio > 1.5:
    print("   ⚠️  Dataset desbalanceado detectado! SMOTE será aplicado.")

# Análise da correlação entre valence e mode
correlacao = df['valence'].corr(df['mode'])
print(f"\n🔍 Correlação entre Valence e Mode: {correlacao:.4f}")
print("\n📊 Estatísticas de Valence por Mode:")
print(df.groupby('mode')['valence'].describe().round(4))

# ============================================
# 3. Selecionar features e alvo
# ============================================
print("\n" + "=" * 80)
print("🎯 PREPARAÇÃO DOS DADOS")
print("=" * 80)

# Features de áudio disponíveis
audio_features = [
    'danceability', 'energy', 'key', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'duration_ms', 'track_popularity'
]

# Remover linhas com valores faltantes
df_clean = df[audio_features + ['mode']].dropna()
print(f"\n✓ Dados limpos: {len(df_clean):,} músicas")
print(f"  (Removidos {len(df) - len(df_clean):,} registros com valores faltantes)")

X = df_clean[audio_features]
y = df_clean['mode']

print(f"\n✓ Features selecionadas ({len(audio_features)}):")
for feat in audio_features:
    print(f"   • {feat}")
print(f"\n✓ Target: mode (0 = Menor, 1 = Maior)")

# ============================================
# 4. Escalonamento e Divisão dos Dados
# ============================================
print("\n" + "=" * 80)
print("🔧 ESCALONAMENTO E DIVISÃO DOS DADOS")
print("=" * 80)

# Divisão treino/teste estratificada ANTES do SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\n✓ Divisão inicial:")
print(f"   Treino: {len(X_train):,} amostras")
print(f"   Teste:  {len(X_test):,} amostras")

# Escalonamento
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 5. Aplicar SMOTE para Balanceamento
# ============================================
print("\n" + "=" * 80)
print("⚖️  APLICANDO SMOTE PARA BALANCEAMENTO")
print("=" * 80)

print(f"\n📊 Distribuição ANTES do SMOTE:")
train_counts_before = pd.Series(y_train).value_counts().sort_index()
for mode, count in train_counts_before.items():
    print(f"   Mode {mode}: {count:,} ({count/len(y_train)*100:.2f}%)")

# Aplicar SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"\n📊 Distribuição APÓS o SMOTE:")
train_counts_after = pd.Series(y_train_balanced).value_counts().sort_index()
for mode, count in train_counts_after.items():
    print(f"   Mode {mode}: {count:,} ({count/len(y_train_balanced)*100:.2f}%)")

print(f"\n✓ Amostras sintéticas criadas: {len(X_train_balanced) - len(X_train_scaled):,}")

# ============================================
# 6. Definir Modelos e Grid de Hiperparâmetros
# ============================================
print("\n" + "=" * 80)
print("🤖 CONFIGURAÇÃO DOS MODELOS E GRID SEARCH")
print("=" * 80)

# Dicionário de modelos e seus grids
models_and_grids = {
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42, n_jobs=-1),
        "params": {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    "Extra Trees": {
        "model": ExtraTreesClassifier(random_state=42, n_jobs=-1),
        "params": {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
    },
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        "params": {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
    }
}

print(f"\n✓ {len(models_and_grids)} modelos configurados para Grid Search")

# ============================================
# 7. Grid Search com Validação Cruzada
# ============================================
print("\n" + "=" * 80)
print("🔍 EXECUTANDO GRID SEARCH COM VALIDAÇÃO CRUZADA")
print("=" * 80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_models = {}
grid_results = []

for name, config in models_and_grids.items():
    print(f"\n🔄 Grid Search para {name}...")
    print(f"   Combinações a testar: {np.prod([len(v) for v in config['params'].values()])}")
    
    grid_search = GridSearchCV(
        estimator=config['model'],
        param_grid=config['params'],
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train_balanced, y_train_balanced)
    
    best_models[name] = grid_search.best_estimator_
    
    # Avaliar no conjunto de teste
    test_score = grid_search.best_estimator_.score(X_test_scaled, y_test)
    
    grid_results.append({
        'Modelo': name,
        'Melhor Score (CV)': grid_search.best_score_,
        'Score no Teste': test_score,
        'Melhores Parâmetros': str(grid_search.best_params_)
    })
    
    print(f"   ✓ Melhor Score CV: {grid_search.best_score_:.4f}")
    print(f"   ✓ Score no Teste: {test_score:.4f}")
    print(f"   📋 Melhores Parâmetros: {grid_search.best_params_}")

# ============================================
# 8. Resultados do Grid Search
# ============================================
print("\n" + "=" * 80)
print("📊 RESULTADOS DO GRID SEARCH")
print("=" * 80)

grid_results_df = pd.DataFrame(grid_results).sort_values(by='Score no Teste', ascending=False)
print("\n" + grid_results_df[['Modelo', 'Melhor Score (CV)', 'Score no Teste']].to_string(index=False))

# ============================================
# 9. Selecionar e Avaliar o Melhor Modelo
# ============================================
melhor_modelo_nome = grid_results_df.iloc[0]['Modelo']
melhor_modelo = best_models[melhor_modelo_nome]
melhor_score = grid_results_df.iloc[0]['Score no Teste']

print("\n" + "=" * 80)
print(f"🏆 MELHOR MODELO: {melhor_modelo_nome}")
print(f"   Acurácia no Teste: {melhor_score:.4f}")
print("=" * 80)

# Predições
y_pred = melhor_modelo.predict(X_test_scaled)
y_pred_proba = melhor_modelo.predict_proba(X_test_scaled)[:, 1] if hasattr(melhor_modelo, 'predict_proba') else None

# ============================================
# 10. Classification Report Detalhado
# ============================================
print("\n" + "=" * 80)
print("📋 CLASSIFICATION REPORT")
print("=" * 80)
print("\nClasses: 0 = Menor (Minor), 1 = Maior (Major)\n")
print(classification_report(y_test, y_pred, target_names=['Menor (0)', 'Maior (1)'], digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n📊 Matriz de Confusão:")
print(f"                 Predito")
print(f"              Menor  Maior")
print(f"Real  Menor  {cm[0][0]:6d} {cm[0][1]:6d}")
print(f"      Maior  {cm[1][0]:6d} {cm[1][1]:6d}")

# ROC-AUC Score
if y_pred_proba is not None:
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n🎯 ROC-AUC Score: {roc_auc:.4f}")

# ============================================
# 11. Salvar Modelo e Scaler (Pickle)
# ============================================
print("\n" + "=" * 80)
print("💾 SALVANDO MODELO E SCALER")
print("=" * 80)

# Criar diretório para modelos
models_dir = Path(__file__).resolve().parent / 'saved_models'
models_dir.mkdir(exist_ok=True)

# Timestamp para versionamento
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Salvar modelo
model_path = models_dir / f'mode_classifier_{melhor_modelo_nome.replace(" ", "_")}_{timestamp}.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(melhor_modelo, f)
print(f"\n✓ Modelo salvo: {model_path.name}")

# Salvar scaler
scaler_path = models_dir / f'scaler_{timestamp}.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ Scaler salvo: {scaler_path.name}")

# Salvar metadados
metadata = {
    'modelo': melhor_modelo_nome,
    'acuracia_teste': melhor_score,
    'features': audio_features,
    'timestamp': timestamp,
    'roc_auc': roc_auc if y_pred_proba is not None else None,
    'melhores_parametros': grid_results_df.iloc[0]['Melhores Parâmetros']
}

metadata_path = models_dir / f'metadata_{timestamp}.pkl'
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)
print(f"✓ Metadados salvos: {metadata_path.name}")

# ============================================
# 12. SHAP - Análise de Explicabilidade
# ============================================
print("\n" + "=" * 80)
print("🔍 ANÁLISE SHAP - EXPLICABILIDADE DO MODELO")
print("=" * 80)

# Criar diretório para visualizações
plots_dir = Path(__file__).resolve().parent / 'shap_plots'
plots_dir.mkdir(exist_ok=True)

# Usar amostra menor para SHAP (mais rápido)
sample_size = min(500, len(X_test_scaled))
X_test_sample = X_test_scaled[:sample_size]
y_test_sample = y_test.iloc[:sample_size] if isinstance(y_test, pd.Series) else y_test[:sample_size]

print(f"\n🔄 Calculando SHAP values para {sample_size} amostras...")

# Criar explainer apropriado para o tipo de modelo
if 'Forest' in melhor_modelo_nome or 'Trees' in melhor_modelo_nome:
    explainer = shap.TreeExplainer(melhor_modelo)
    shap_values_raw = explainer.shap_values(X_test_sample)
    # Para modelos de árvore com classificação binária, pode retornar lista ou array 3D
    if isinstance(shap_values_raw, list):
        # Converter lista para array 3D: (n_samples, n_features, n_classes)
        shap_values = np.stack(shap_values_raw, axis=-1)
    else:
        shap_values = shap_values_raw
else:
    # Para modelos lineares ou outros
    explainer = shap.Explainer(melhor_modelo, X_train_balanced)
    shap_values_obj = explainer(X_test_sample)
    shap_values = shap_values_obj.values

print(f"✓ SHAP values calculados! Shape: {shap_values.shape}")

# Para plots summary, usar apenas valores da classe positiva (1)
if len(shap_values.shape) == 3:
    shap_values_for_summary = shap_values[:, :, 1]
else:
    shap_values_for_summary = shap_values

# ============================================
# 13. SHAP PLOT 1: Feature Importance Global
# ============================================
print("\n📊 Gerando Plot 1: Feature Importance Global (Bar Plot)...")

plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values_for_summary, 
    X_test_sample,
    feature_names=audio_features,
    plot_type="bar",
    show=False
)
plt.title('SHAP - Importância Global das Features (Mode Classification)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Impacto Médio no Output do Modelo (|SHAP value|)', fontsize=12)
plt.tight_layout()
plot1_path = plots_dir / f'shap_feature_importance_{timestamp}.png'
plt.savefig(plot1_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print(f"   ✓ Salvo: {plot1_path.name}")

# ============================================
# 14. SHAP PLOT 2: Beeswarm Plot (Densidade)
# ============================================
print("\n📊 Gerando Plot 2: Beeswarm Plot (Densidade)...")

plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values_for_summary,
    X_test_sample,
    feature_names=audio_features,
    show=False
)
plt.title('SHAP - Beeswarm Plot: Impacto das Features por Valor', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('SHAP Value (Impacto na Predição)', fontsize=12)
plt.tight_layout()
plot2_path = plots_dir / f'shap_beeswarm_{timestamp}.png'
plt.savefig(plot2_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print(f"   ✓ Salvo: {plot2_path.name}")

# ============================================
# 15. SHAP PLOT 3: Waterfall Plot (Explicação Local - MELHOR VISUALIZAÇÃO)
# ============================================
print("\n📊 Gerando Plot 3: Waterfall Plot (Explicação Local)...")

# Selecionar uma amostra interessante (uma predição de cada classe)
idx_menor = np.where(y_test_sample == 0)[0][0] if 0 in y_test_sample else 0
idx_maior = np.where(y_test_sample == 1)[0][0] if 1 in y_test_sample else 1

# Pegar valores reais usando .iloc para garantir indexação posicional
valor_real_menor = y_test_sample.iloc[idx_menor] if hasattr(y_test_sample, 'iloc') else y_test_sample[idx_menor]
valor_real_maior = y_test_sample.iloc[idx_maior] if hasattr(y_test_sample, 'iloc') else y_test_sample[idx_maior]

# Se shap_values tem 3 dimensões (n_samples, n_features, n_classes), pegar apenas classe 1
if len(shap_values.shape) == 3:
    shap_values_class1 = shap_values[:, :, 1]  # Classe positiva (Maior)
else:
    shap_values_class1 = shap_values

print(f"   → Usando shap_values com shape: {shap_values_class1.shape}")

# Waterfall plot para classe Menor (0) - MUITO MAIS LEGÍVEL
fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
plt.rcParams['font.size'] = 11

# Criar objeto Explanation para waterfall
if isinstance(explainer.expected_value, np.ndarray):
    base_value = explainer.expected_value[1]
else:
    base_value = explainer.expected_value

# Criar explanation object
explanation_menor = shap.Explanation(
    values=shap_values_class1[idx_menor],
    base_values=base_value,
    data=X_test_sample.iloc[idx_menor] if hasattr(X_test_sample, 'iloc') else X_test_sample[idx_menor],
    feature_names=audio_features
)

shap.plots.waterfall(explanation_menor, max_display=12, show=False)
plt.title(f'SHAP Waterfall - Mode Menor (Real: {valor_real_menor})', 
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plot3a_path = plots_dir / f'shap_waterfall_menor_{timestamp}.png'
plt.savefig(plot3a_path, dpi=100, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   ✓ Salvo: {plot3a_path.name}")

# Waterfall plot para classe Maior (1)
fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
plt.rcParams['font.size'] = 11

explanation_maior = shap.Explanation(
    values=shap_values_class1[idx_maior],
    base_values=base_value,
    data=X_test_sample.iloc[idx_maior] if hasattr(X_test_sample, 'iloc') else X_test_sample[idx_maior],
    feature_names=audio_features
)

shap.plots.waterfall(explanation_maior, max_display=12, show=False)
plt.title(f'SHAP Waterfall - Mode Maior (Real: {valor_real_maior})', 
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plot3b_path = plots_dir / f'shap_waterfall_maior_{timestamp}.png'
plt.savefig(plot3b_path, dpi=100, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   ✓ Salvo: {plot3b_path.name}")

# ============================================
# 16. Análise de Valence
# ============================================
if hasattr(melhor_modelo, 'feature_importances_'):
    print("\n" + "=" * 80)
    print("🎵 ANÁLISE ESPECÍFICA: VALENCE")
    print("=" * 80)
    
    feature_importance = pd.DataFrame({
        'Feature': audio_features,
        'Importance': melhor_modelo.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\n" + feature_importance.to_string(index=False))
    
    valence_importance = feature_importance[feature_importance['Feature'] == 'valence']['Importance'].values[0]
    valence_rank = feature_importance.index[feature_importance['Feature'] == 'valence'].tolist()[0] + 1
    print(f"\n🎵 Valence:")
    print(f"   Ranking: #{valence_rank}")
    print(f"   Importância: {valence_importance:.4f}")
    print(f"   Correlação com Mode: {correlacao:.4f}")

# ============================================
# 17. Resumo Final
# ============================================
print("\n" + "=" * 80)
print("✅ RESUMO FINAL")
print("=" * 80)

print(f"\n🏆 Melhor Modelo: {melhor_modelo_nome}")
print(f"   Acurácia no Teste: {melhor_score:.4f}")
if y_pred_proba is not None:
    print(f"   ROC-AUC Score: {roc_auc:.4f}")

print(f"\n💾 Arquivos Salvos:")
print(f"   • Modelo: {model_path.name}")
print(f"   • Scaler: {scaler_path.name}")
print(f"   • Metadados: {metadata_path.name}")

print(f"\n📊 Visualizações SHAP:")
print(f"   • Feature Importance: {plot1_path.name}")
print(f"   • Beeswarm Plot: {plot2_path.name}")
print(f"   • Force Plot (Menor): {plot3a_path.name}")
print(f"   • Force Plot (Maior): {plot3b_path.name}")

print("\n" + "=" * 80)
print("🎉 ANÁLISE CONCLUÍDA COM SUCESSO!")
print("=" * 80)
