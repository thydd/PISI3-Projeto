# ============================================
# Classificador de Gênero Musical
# Implementação de SHAP (Explicabilidade), Exportação (Pickle) e GRID SEARCH (Tuning)
# ============================================
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pickle # Adicionado para exportação/importação
import shap # Adicionado para explicabilidade
import os # Necessário para funções de exportação
import time

# Configurações para evitar warnings desnecessários do SHAP e do Matplotlib
shap.initjs()
plt.rcParams['figure.dpi'] = 100
pd.set_option('display.max_columns', None)

# ============================================
# 1. Funções Auxiliares
# ============================================

def exportar_pipeline(pipeline, nome_arquivo="genero_musical_pipeline.pkl"):
    """Salva o pipeline completo (SMOTE, Scaling e Modelo) usando Pickle."""
    try:
        with open(nome_arquivo, 'wb') as file:
            pickle.dump(pipeline, file)
        print(f"\nPipeline completo exportado com sucesso para '{nome_arquivo}'.")
    except Exception as e:
        print(f"\nErro ao exportar o pipeline: {e}")

def carregar_pipeline(nome_arquivo="genero_musical_pipeline.pkl"):
    """Carrega o pipeline completo."""
    try:
        with open(nome_arquivo, 'rb') as file:
            pipeline_carregado = pickle.load(file)
        print(f"\nPipeline carregado com sucesso de '{nome_arquivo}'.")
        return pipeline_carregado
    except FileNotFoundError:
        print(f"\nArquivo '{nome_arquivo}' não encontrado.")
        return None

# ============================================
# 2. Carregar e Pré-processar o dataset
# ============================================
csv_path = Path(__file__).resolve().parent.parent / 'DataSet' / 'spotify_songs.csv'

# Criando um caminho fake para garantir que o código rode se o dataset não estiver lá
if not csv_path.exists():
    csv_path = Path('spotify_songs.csv') 
    if not csv_path.exists():
        print("Aviso: O arquivo 'spotify_songs.csv' não foi encontrado no caminho esperado. Usando dados fictícios.")
        # Dados fictícios com 500 amostras para ter um X_test de tamanho razoável (100)
        num_samples = 500
        data = {
            'danceability': np.random.rand(num_samples), 'energy': np.random.rand(num_samples),
            'key': np.random.randint(0, 12, num_samples), 'loudness': np.random.uniform(-20, 0, num_samples),
            'mode': np.random.randint(0, 2, num_samples), 'speechiness': np.random.rand(num_samples),
            'acousticness': np.random.rand(num_samples), 'instrumentalness': np.random.rand(num_samples),
            'liveness': np.random.rand(num_samples), 'valence': np.random.rand(num_samples),
            'tempo': np.random.uniform(80, 180, num_samples), 'duration_ms': np.random.randint(150000, 300000, num_samples),
            'track_popularity': np.random.randint(0, 100, num_samples), 
            'track_album_release_date': pd.to_datetime('2020-01-01', format='%Y-%m-%d'),
            'playlist_subgenre': np.random.choice(['trap', 'latin', 'pop_up', 'rock_sub'], num_samples),
            'playlist_genre': np.random.choice(['rap', 'latin', 'pop', 'rock'], num_samples) # Mais classes
        }
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(csv_path)
else:
    df = pd.read_csv(csv_path)


# 2. Engenharia de Features
df['release_year'] = pd.to_datetime(df['track_album_release_date'], errors='coerce').dt.year.fillna(0).astype(int)

# 3. Selecionar features e alvo
features = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'duration_ms', 'track_popularity', 'release_year'
]
X = df[features]
y = df['playlist_genre']

# 4. Pré-processamento
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
class_names = encoder.classes_ # Nomes das classes para o SHAP

# ============================================
# 5. Configuração dos Modelos
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

# 6. Validação Cruzada (5-Fold) para Comparação Inicial
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    # Usando F1-Weighted como métrica principal devido ao desbalanceamento
    scores = cross_val_score(pipeline, X, y_encoded, cv=cv, scoring='f1_weighted', n_jobs=-1) 
    results[name] = {
        "F1-Weighted Média": np.mean(scores),
        "Desvio Padrão": np.std(scores)
    }

results_df = pd.DataFrame(results).T.sort_values(by="F1-Weighted Média", ascending=False)
print("=== Resultados da Validação Cruzada (5-Fold, F1-Weighted) ===")
print(results_df)

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

melhor_modelo_nome = results_df.index[0]
modelo_base = models[melhor_modelo_nome]

"""Modo rápido configurável:
FAST_GRID=1 -> reduz combinações do Grid Search
FAST_SHAP=1 -> usa menos amostras e cálculo SHAP mais leve
Use variáveis de ambiente ou altere abaixo.
"""
FAST_GRID = os.getenv("FAST_GRID", "1") == "1"
FAST_SHAP = os.getenv("FAST_SHAP", "1") == "1"

# ============================================
# 7. Otimização de Hiperparâmetros (Grid Search)
# ============================================
print(f"\nAplicando Grid Search no melhor modelo: {melhor_modelo_nome} (FAST_GRID={FAST_GRID})")

base_pipeline_grid = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('model', modelo_base)
])

if FAST_GRID:
    param_grid = {
        'model__n_estimators': [100],
        'model__max_depth': [20],
        'model__min_samples_split': [2]
    }
else:
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [10, 20, None],
        'model__min_samples_split': [2, 5]
    }

grid_search = GridSearchCV(
    estimator=base_pipeline_grid,
    param_grid=param_grid,
    scoring='f1_weighted',
    cv=cv,
    verbose=0 if FAST_GRID else 1,
    n_jobs=-1
)

inicio_grid = time.time()
grid_search.fit(X_train, y_train)
duracao_grid = time.time() - inicio_grid

final_pipeline = grid_search.best_estimator_
y_pred = final_pipeline.predict(X_test)

print(f"\nMelhor pontuação F1 (Weighted): {grid_search.best_score_:.4f} | Tempo Grid: {duracao_grid:.1f}s")
print(f"Parâmetros Selecionados: {grid_search.best_params_}")


# 8. Avaliação
print("\n=== Classification Report (Modelo Otimizado) ===")
print(classification_report(y_test, y_pred, target_names=class_names))

# ============================================
# 9. Exportar o Modelo Otimizado (Pickle)
# ============================================
exportar_pipeline(final_pipeline)

# ============================================
# 10. Explicabilidade com SHAP
# ============================================

# O SHAP Explainer precisa ser aplicado ao modelo final dentro do pipeline.
modelo_final_treinado = final_pipeline['model']
scaler_treinado = final_pipeline['scaler']

# 10.1 Preparar Dados para SHAP
# Escalonar e subamostrar para acelerar
X_test_scaled = scaler_treinado.transform(X_test)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features)
shap_target_size = 200 if FAST_SHAP else 500
sample_size = min(shap_target_size, len(X_test_scaled_df))
X_test_sample = X_test_scaled_df.sample(n=sample_size, random_state=42)
print(f"[*] SHAP: usando {sample_size} amostras (FAST_SHAP={FAST_SHAP}).")

# 10.2 Inicializar o TreeExplainer (modo rápido usa saída 'raw')
explainer = shap.TreeExplainer(
    modelo_final_treinado,
    feature_perturbation='tree_path_dependent',
    model_output='raw'
)

# Calcular os valores SHAP
print("\n[*] Calculando valores SHAP (pode levar alguns segundos)...")
inicio_shap = time.time()
if FAST_SHAP:
    shap_values = explainer.shap_values(X_test_sample, check_additivity=False)
else:
    shap_values = explainer.shap_values(X_test_sample)
duracao_shap = time.time() - inicio_shap
print(f"[*] SHAP calculado em {duracao_shap:.1f}s")

# ============================================
# 11. Explicações Globais: feature importance, barras multiclasse e beeswarm
# ============================================

print("\n=== Explicações Globais ===")
print("Plotando importância global das features (único gradiente SHAP)...")

try:
    # 1. Recuperar valores originais (não escalonados) das features para colorir o plot
    sample_indices = X_test_sample.index.tolist()
    X_sample_original = X.loc[sample_indices, features].copy().reset_index(drop=True)

    # 2. Processar valores SHAP baseado no formato retornado pelo TreeExplainer
    shap_array = np.array(shap_values)
    
    # Identificar formato: (n_samples, n_features, n_classes) ou lista de (n_samples, n_features)
    if shap_array.ndim == 3:
        # Formato: (n_samples, n_features, n_classes)
        # Precisamos agregar através das classes (último eixo)
        n_samples, n_features_shap, n_classes = shap_array.shape
        
        if n_features_shap == len(features):
            # Caso 1: shape correto (samples, features, classes)
            shap_matrix = np.mean(shap_array, axis=2)  # média através das classes
        elif n_classes == len(features):
            # Caso 2: shape invertido (samples, classes, features) - precisa transpor
            shap_matrix = np.mean(shap_array, axis=1)  # média através das classes
        else:
            raise ValueError(f"Shape inesperado: {shap_array.shape}. Esperado features={len(features)}")
    
    elif isinstance(shap_values, list):
        # Lista de arrays, um por classe: [(n_samples, n_features), ...]
        shap_matrix = np.mean(np.array(shap_values), axis=0)
    else:
        # Já é 2D
        shap_matrix = shap_array

    # 3. Validação final
    if shap_matrix.shape != (len(sample_indices), len(features)):
        raise ValueError(
            f"Shape incompatível após processamento: {shap_matrix.shape}. "
            f"Esperado: ({len(sample_indices)}, {len(features)})"
        )

    # 4. Ordenar features por importância (magnitude absoluta média)
    mean_abs_importance = np.mean(np.abs(shap_matrix), axis=0)
    sorted_indices = np.argsort(mean_abs_importance)[::-1]
    sorted_feature_names = [features[i] for i in sorted_indices]

    # 5. Reordenar dados
    shap_matrix_sorted = shap_matrix[:, sorted_indices]
    X_sample_sorted = X_sample_original.iloc[:, sorted_indices].copy()
    X_sample_sorted.columns = sorted_feature_names

    # 6. Criar gráfico SHAP usando API antiga (mais estável)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('blue_red', ['#0D3B66', '#4A90E2', '#F39C12', '#900C3F'])
    
    # Usar shap.summary_plot tradicional (compatível com todas as versões)
    shap.summary_plot(
        shap_matrix_sorted,
        X_sample_sorted,
        max_display=15,
        show=False,
        cmap=cmap
    )

    # 7. Personalizar figura e ajustar tamanho dos pontos
    fig = plt.gcf()
    fig.set_size_inches(12, 10)  # Aumentei a altura de 8 para 10 para mais espaçamento
    ax = plt.gca()
    
    # Ajustar tamanho dos pontos manualmente
    for collection in ax.collections:
        collection.set_sizes([50])  # Reduzido de 80 para 50
        collection.set_alpha(0.8)   # Transparência

    ax.axvline(0.0, color='#888888', linewidth=0.8, linestyle='--', alpha=0.7, zorder=1)
    ax.set_title("SHAP Summary Plot - Importância Global das Features", 
                fontsize=16, fontweight='bold', pad=16)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=13, labelpad=10)
    
    # Aumentar tamanho dos nomes das features no eixo Y
    ax.tick_params(axis='y', labelsize=12)  # Aumentado de padrão (~10) para 12
    
    # Aumentar espaçamento entre features (ajustar ylim)
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0] - 0.8, ylim[1] + 0.8)  # Aumentei as margens de 0.5 para 0.8

    # Centralizar eixo X com análise de distribuição
    # Calcular percentis para identificar outliers e definir limites adequados
    shap_flat = shap_matrix_sorted.flatten()
    p1 = np.percentile(shap_flat, 1)   # Percentil 1%
    p99 = np.percentile(shap_flat, 99) # Percentil 99%
    p5 = np.percentile(shap_flat, 5)   # Percentil 5%
    p95 = np.percentile(shap_flat, 95) # Percentil 95%
    
    # Opção 1: Usar percentis 1-99 (captura 98% dos dados, remove outliers extremos)
    # max_abs_x = max(abs(p1), abs(p99))
    
    # Opção 2: Usar percentis 5-95 (captura 90% dos dados, mais conservador)
    # max_abs_x = max(abs(p5), abs(p95))
    
    # Opção 3: Usar máximo absoluto real (mostra todos os dados)
    max_abs_x = max(abs(shap_flat.min()), abs(shap_flat.max()))
    
    # Opção 4: Híbrido - usar percentil 99 se outliers forem muito extremos
    p99_val = max(abs(p1), abs(p99))
    max_val = max(abs(shap_flat.min()), abs(shap_flat.max()))
    
    # Se o máximo for muito maior que p99 (outliers extremos), usar p99
    if max_val > p99_val * 2:
        max_abs_x = p99_val * 1.1  # Usa percentil 99 com margem
        print(f"[INFO] Outliers detectados. Usando percentil 99 para limites: ±{max_abs_x:.3f}")
    else:
        max_abs_x = max_val * 1.05  # Usa máximo com margem pequena
        print(f"[INFO] Distribuição regular. Usando máximo absoluto: ±{max_abs_x:.3f}")
    
    # Estatísticas para o usuário decidir
    print(f"\n=== Análise do Eixo X (SHAP values) ===")
    print(f"Mínimo: {shap_flat.min():.4f}")
    print(f"Máximo: {shap_flat.max():.4f}")
    print(f"Percentil 1%:  {p1:.4f}")
    print(f"Percentil 5%:  {p5:.4f}")
    print(f"Percentil 95%: {p95:.4f}")
    print(f"Percentil 99%: {p99:.4f}")
    print(f"Mediana: {np.median(shap_flat):.4f}")
    print(f"Desvio padrão: {np.std(shap_flat):.4f}")
    print(f"\n💡 Sugestões de limites do eixo X:")
    print(f"   - Conservador (95% dados):   ±{max(abs(p5), abs(p95)):.3f}")
    print(f"   - Balanceado (98% dados):    ±{max(abs(p1), abs(p99)):.3f}")
    print(f"   - Completo (100% dados):     ±{max(abs(shap_flat.min()), abs(shap_flat.max())):.3f}")
    print(f"   - Usado atualmente:          ±{max_abs_x:.3f}\n")
    
    if max_abs_x > 0:
        ax.set_xlim(-max_abs_x, max_abs_x)

    # Customizar colorbar
    for extra_ax in fig.axes:
        if extra_ax is not ax:
            try:
                extra_ax.set_ylabel("Feature value", fontsize=11)
                extra_ax.yaxis.set_label_position("right")
                extra_ax.yaxis.tick_right()
            except Exception:
                pass

    plt.tight_layout()
    plt.show()
    print("✓ Plotagem SHAP global concluída com sucesso.")

    # 11.2 Feature Importance Plot (Barras Multiclasse)
    print("\nPlotando importância média por classe (Barras)...")
    shap.summary_plot(
        shap_values,
        X_test_sample,
        plot_type="bar",
        class_names=class_names,
        show=False
    )
    plt.gcf().suptitle("SHAP: Importância Média por Classe (Magnitude)", fontsize=12)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Erro ao gerar plots globais do SHAP (Beeswarm/Barra): {e}")
    import traceback
    traceback.print_exc()


# ============================================
# 12. Explicação Local: Gráfico de Força (Force Plot)
# ============================================
# Escolher uma instância (música) para explicar. Exemplo: A primeira música do sample
instancia_idx = 0
explicacao_instancia = X_test_sample.iloc[[instancia_idx]]

# Fazer a predição para saber qual classe explicar
predicao_instancia = modelo_final_treinado.predict(explicacao_instancia.values)
classe_predita = predicao_instancia[0]
nome_classe_predita = class_names[classe_predita]

print(f"\n=== Explicação Local (Música #{instancia_idx}) ===")
print(f"Predição: {nome_classe_predita} (Classe {classe_predita})")

try:
    print("Plotando gráfico de força (Force Plot) na tela de classificação...")

    # Para multiclass, shap_values é uma lista de arrays [n_classes][n_samples, n_features]
    base_value = explainer.expected_value[classe_predita] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    shap_vector = shap_values[classe_predita][instancia_idx]
    feature_vals = explicacao_instancia.iloc[0]
    feature_names_local = list(feature_vals.index)

    # Criar objeto Explanation para usar a API moderna
    explanation_obj = shap.Explanation(
        values=shap_vector,
        base_values=base_value,
        data=feature_vals.values,
        feature_names=feature_names_local
    )
    
    # Usar waterfall plot que é mais confiável que force plot para visualização matplotlib
    shap.plots.waterfall(explanation_obj, show=False)
    plt.gcf().suptitle(f"SHAP: Gráfico de Força (Explicação Local) - Predição: {nome_classe_predita}", fontsize=10)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Erro ao gerar o plot local do SHAP (Force Plot): {e}. Tentando Waterfall como alternativa...")
    try:
        # Waterfall plot como alternativa em caso de falha
        explanation_obj = shap.Explanation(
            values=shap_vector,
            base_values=base_value,
            data=feature_vals.values,
            feature_names=feature_names_local
        )
        shap.plots.waterfall(explanation_obj, show=False)
        plt.gcf().suptitle(f"SHAP Waterfall (Fallback) - Predição: {nome_classe_predita}", fontsize=10)
        plt.tight_layout()
        plt.show()
    except Exception as e2:
        print(f"Fallback Waterfall também falhou: {e2}")

# Informação sobre a música que foi explicada
print("\nValores da instância explicada (Escalonados):")
print(explicacao_instancia)
