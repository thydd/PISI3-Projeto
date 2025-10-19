"""
Classificação de Sentimentos Musicais - DNA Emocional
======================================================
Classifica músicas em 4 quadrantes emocionais baseado em energy e valence.
Modelos: Random Forest, K-Nearest Neighbors, Extra Trees
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

print("="*80)
print("CLASSIFICAÇÃO DE SENTIMENTOS MUSICAIS - DNA EMOCIONAL".center(80))
print("="*80)

try:
    csv_path = Path(__file__).resolve().parent.parent / 'DataSet' / 'spotify_songs.csv'
    df = pd.read_csv(csv_path)
    print(f"\n✓ Dataset carregado com sucesso!")
    print(f"  Total de músicas: {len(df):,}")
except FileNotFoundError:
    print("\n✗ Erro: Arquivo 'spotify_songs.csv' não encontrado!")
    exit(1)
except Exception as e:
    print(f"\n✗ Erro ao carregar dados: {e}")
    exit(1)

print("\n" + "-"*80)
print("CRIANDO CLASSES DE SENTIMENTO")
print("-"*80)

def classify_mood(row):
    energy = row['energy']
    valence = row['valence']
    if energy > 0.5 and valence > 0.5:
        return 'Feliz/Energético'
    elif energy > 0.5 and valence <= 0.5:
        return 'Intenso/Turbulento'
    elif energy <= 0.5 and valence > 0.5:
        return 'Calmo/Sereno'
    else:
        return 'Triste/Sombrio'

df['mood'] = df.apply(classify_mood, axis=1)

print("\nDistribuição das classes:")
for mood, count in df['mood'].value_counts().sort_index().items():
    pct = (count / len(df)) * 100
    print(f"  {mood}: {count:,} músicas ({pct:.1f}%)")

print("\n" + "-"*80)
print("PREPARANDO DADOS PARA TREINAMENTO")
print("-"*80)

X = df[['energy', 'valence']].copy()
y = df['mood'].copy()

if X.isnull().sum().sum() > 0:
    mask = X.isnull().any(axis=1)
    X = X[~mask]
    y = y[~mask]
    print(f"\n⚠ Linhas removidas por valores faltantes: {mask.sum()}")

print(f"\n✓ Dados preparados:")
print(f"  Total de amostras: {len(X):,}")
print(f"  Features: energy, valence")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Divisão treino/teste:")
print(f"  Treino: {len(X_train):,} amostras ({len(X_train)/len(X)*100:.0f}%)")
print(f"  Teste:  {len(X_test):,} amostras ({len(X_test)/len(X)*100:.0f}%)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*80)
print("TREINAMENTO E AVALIAÇÃO DE MODELOS".center(80))
print("="*80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

modelos = {
    'Random Forest': {
        'modelo': RandomForestClassifier(n_estimators=100, random_state=42),
        'usar_escala': False
    },
    'K-Nearest Neighbors': {
        'modelo': KNeighborsClassifier(n_neighbors=5),
        'usar_escala': True
    },
    'Extra Trees': {
        'modelo': ExtraTreesClassifier(n_estimators=100, random_state=42),
        'usar_escala': False
    }
}

resultados = []

print("\n🔄 Treinando modelos com validação cruzada (5-fold)...")
print("-"*80)

for nome, config in modelos.items():
    print(f"\n📌 {nome}")
    modelo = config['modelo']
    usar_escala = config['usar_escala']
    X_train_use = X_train_scaled if usar_escala else X_train
    X_test_use = X_test_scaled if usar_escala else X_test
    cv_scores = cross_val_score(modelo, X_train_use, y_train, cv=cv, scoring='accuracy')
    modelo.fit(X_train_use, y_train)
    y_pred = modelo.predict(X_test_use)
    acuracia = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    resultados.append({
        'nome': nome,
        'cv_media': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'acuracia_teste': acuracia,
        'f1_score': f1,
        'modelo': modelo,
        'usar_escala': usar_escala
    })
    print(f"  Acurácia (CV): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Acurácia (Teste): {acuracia:.4f} ({acuracia*100:.2f}%)")
    print(f"  F1-Score: {f1:.4f}")

print("\n" + "="*80)
print("RESULTADO FINAL".center(80))
print("="*80)

resultados_sorted = sorted(resultados, key=lambda x: x['acuracia_teste'], reverse=True)
melhor = resultados_sorted[0]

print(f"\n🏆 MELHOR MODELO: {melhor['nome']}")
print(f"   Acurácia: {melhor['acuracia_teste']:.4f} ({melhor['acuracia_teste']*100:.2f}%)")
print(f"   F1-Score: {melhor['f1_score']:.4f}")
print(f"   Validação Cruzada: {melhor['cv_media']:.4f} (+/- {melhor['cv_std']:.4f})")

print(f"\n📊 COMPARAÇÃO DOS MODELOS:")
print("-"*80)
for i, res in enumerate(resultados_sorted, 1):
    print(f"{i}º {res['nome']:<25} Acurácia: {res['acuracia_teste']:.4f} ({res['acuracia_teste']*100:.2f}%)")

print("\n" + "="*80)
print(f"RELATÓRIO DE CLASSIFICAÇÃO - {melhor['nome']}".center(80))
print("="*80)

modelo_final = melhor['modelo']
usar_escala_final = melhor['usar_escala']
X_test_final = X_test_scaled if usar_escala_final else X_test
y_pred_final = modelo_final.predict(X_test_final)

print("\n" + classification_report(y_test, y_pred_final, zero_division=0))

print("="*80)
print("EXEMPLOS DE PREDIÇÃO".center(80))
print("="*80)

exemplos = [
    {'energy': 0.8, 'valence': 0.9, 'descricao': 'Música energética e positiva'},
    {'energy': 0.9, 'valence': 0.2, 'descricao': 'Música intensa e negativa'},
    {'energy': 0.3, 'valence': 0.8, 'descricao': 'Música calma e positiva'},
    {'energy': 0.2, 'valence': 0.1, 'descricao': 'Música de baixa energia e negativa'},
]

print("\n🎵 Testando predições:")
print("-"*80)

for i, ex in enumerate(exemplos, 1):
    X_novo = pd.DataFrame([[ex['energy'], ex['valence']]], columns=['energy', 'valence'])
    if usar_escala_final:
        X_novo = scaler.transform(X_novo)
    predicao = modelo_final.predict(X_novo)[0]
    print(f"\n{i}. {ex['descricao']}")
    print(f"   Energy: {ex['energy']}, Valence: {ex['valence']}")
    print(f"   ➜ Sentimento: {predicao}")

print("\n" + "="*80)
print("CONCLUSÃO".center(80))
print("="*80)

qualidade = 'Excelente' if melhor['acuracia_teste'] > 0.95 else 'Boa' if melhor['acuracia_teste'] > 0.85 else 'Aceitável'

print(f"""
📊 RESUMO:

1. MELHOR MODELO: {melhor['nome']}
   - Acurácia: {melhor['acuracia_teste']*100:.2f}%
   - F1-Score: {melhor['f1_score']:.4f}
   
2. QUALIDADE: {qualidade}

3. APLICAÇÕES:
   ✓ Criar playlists automáticas por humor
   ✓ Recomendar músicas por estado emocional
   ✓ Filtrar biblioteca musical por sentimento
""")

print("="*80)
print("✅ ANÁLISE CONCLUÍDA!")
print("="*80)
