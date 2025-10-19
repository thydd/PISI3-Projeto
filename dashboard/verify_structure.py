"""
Script de verificação da estrutura modular.
Testa imports e estrutura básica sem executar o servidor.
"""

import sys
from pathlib import Path

print("=" * 70)
print("VERIFICAÇÃO DA ESTRUTURA MODULAR DO DASHBOARD")
print("=" * 70)

# 1. Verificar imports do app.py
print("\n[1/6] Testando imports do app.py...")
try:
    from app.config import (
        BACKGROUND_GRADIENT,
        EXTERNAL_STYLESHEETS,
        PRIMARY_TEXT,
    )
    print("    ✅ app.config importado com sucesso")
except Exception as e:
    print(f"    ❌ Erro ao importar app.config: {e}")
    sys.exit(1)

try:
    from app.layouts.overview_tab import create_overview_layout
    from app.layouts.popularity_tab import create_popularity_layout
    from app.layouts.audio_dna_tab import create_audio_dna_layout
    print("    ✅ Layouts importados com sucesso")
except Exception as e:
    print(f"    ❌ Erro ao importar layouts: {e}")
    sys.exit(1)

try:
    from app.callbacks.overview_callbacks import register_overview_callbacks
    from app.callbacks.popularity_callbacks import register_popularity_callbacks
    from app.callbacks.audio_dna_callbacks import register_audio_dna_callbacks
    print("    ✅ Callbacks importados com sucesso")
except Exception as e:
    print(f"    ❌ Erro ao importar callbacks: {e}")
    sys.exit(1)

try:
    from app.utils.common_components import apply_filters, create_range_marks
    from app.utils.data_utils import load_dataset
    from app.utils.model_utils import CLASSIFIER_FEATURES
    print("    ✅ Utilitários importados com sucesso")
except Exception as e:
    print(f"    ❌ Erro ao importar utilitários: {e}")
    sys.exit(1)

# 2. Verificar dataset
print("\n[2/6] Verificando dataset...")
try:
    BASE_DF = load_dataset()
    print(f"    ✅ Dataset carregado: {len(BASE_DF):,} músicas")
    print(f"    ✅ Colunas: {len(BASE_DF.columns)}")
except Exception as e:
    print(f"    ❌ Erro ao carregar dataset: {e}")
    sys.exit(1)

# 3. Verificar estrutura de arquivos
print("\n[3/6] Verificando estrutura de arquivos...")
required_files = [
    "app.py",
    "app/config.py",
    "app/layouts/overview_tab.py",
    "app/layouts/popularity_tab.py",
    "app/layouts/audio_dna_tab.py",
    "app/callbacks/overview_callbacks.py",
    "app/callbacks/popularity_callbacks.py",
    "app/callbacks/audio_dna_callbacks.py",
    "app/utils/common_components.py",
    "app/utils/data_utils.py",
    "app/utils/model_utils.py",
    "app/utils/visualizations.py",
    "next_features.txt",
    "README.md",
]

missing_files = []
for file_path in required_files:
    if not Path(file_path).exists():
        missing_files.append(file_path)

if missing_files:
    print(f"    ❌ Arquivos faltando: {', '.join(missing_files)}")
    sys.exit(1)
else:
    print(f"    ✅ Todos os {len(required_files)} arquivos necessários estão presentes")

# 4. Verificar criação de layouts
print("\n[4/6] Testando criação de layouts...")
try:
    overview_layout = create_overview_layout()
    popularity_layout = create_popularity_layout()
    audio_dna_layout = create_audio_dna_layout(CLASSIFIER_FEATURES)
    print("    ✅ Layouts criados sem erros")
except Exception as e:
    print(f"    ❌ Erro ao criar layouts: {e}")
    sys.exit(1)

# 5. Verificar features do classificador
print("\n[5/6] Verificando features do modelo...")
expected_features = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "duration_ms", "track_popularity"
]
if set(CLASSIFIER_FEATURES) == set(expected_features):
    print(f"    ✅ {len(CLASSIFIER_FEATURES)} features do classificador corretas")
else:
    print(f"    ❌ Features do classificador incorretas")
    sys.exit(1)

# 6. Verificar filtros
print("\n[6/6] Testando aplicação de filtros...")
try:
    filtered = apply_filters(
        BASE_DF,
        genres=None,
        subgenres=None,
        popularity=[0, 100],
        years=[2000, 2024]
    )
    print(f"    ✅ Filtros funcionando: {len(filtered):,} músicas retornadas")
except Exception as e:
    print(f"    ❌ Erro ao aplicar filtros: {e}")
    sys.exit(1)

# Resumo final
print("\n" + "=" * 70)
print("RESULTADO: ✅ TODOS OS TESTES PASSARAM!")
print("=" * 70)
print("\nEstrutura modular verificada com sucesso!")
print("O dashboard está pronto para ser executado com: python app.py")
print("\nFuncionalidades implementadas:")
print("  • Aba Visão Geral (KPIs, sunburst, violin, timeline)")
print("  • Aba Popularidade (network, top artistas, distribuições)")
print("  • Aba Audio DNA (scatter 3D, heatmap, radar)")
print("\nPróximas features estão documentadas em: next_features.txt")
print("=" * 70)
