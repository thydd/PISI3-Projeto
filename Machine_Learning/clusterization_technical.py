import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def carregar_dados(caminho_dataset="../DataSet/spotify_songs.csv"):
    """
    Carrega o dataset e seleciona features técnicas de produção musical.
    
    Features selecionadas:
    - loudness: Intensidade/volume da música (dB)
    - speechiness: Presença de palavras faladas (0-1)
    - instrumentalness: Nível de conteúdo instrumental (0-1)
    - liveness: Indicador de gravação ao vivo (0-1)
    - duration_min: Duração da música em minutos
    
    Esta combinação agrupa músicas por características de produção,
    formato e contexto de performance.
    """
    if not os.path.exists(caminho_dataset):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_dataset}")
    
    print("[*] Carregando dataset...")
    df = pd.read_csv(caminho_dataset)
    
    # Features técnicas de produção
    features = ['loudness', 'speechiness', 'instrumentalness', 'liveness', 'duration_ms']
    df_features = df[features].copy()
    
    # Converter duração de ms para minutos (mais interpretável)
    df_features['duration_min'] = df_features['duration_ms'] / 60000
    df_features = df_features.drop('duration_ms', axis=1)
    
    print(f"[+] Dataset carregado com {len(df)} registros.")
    print(f"[+] Features técnicas selecionadas: loudness, speechiness, instrumentalness, liveness, duration_min")
    return df, df_features

def preprocessar_dados(df_features):
    """
    Normaliza os dados usando StandardScaler.
    Importante para features com escalas diferentes (ex: loudness em dB vs. proporções 0-1).
    """
    print("\n[*] Padronizando dados...")
    scaler = StandardScaler()
    dados_normalizados = scaler.fit_transform(df_features)
    print("[+] Dados normalizados.")
    return dados_normalizados, scaler

def escolher_numero_clusters(dados_normalizados, max_clusters=10):
    """
    Método do Cotovelo (Elbow Method) para determinar número ótimo de clusters.
    """
    print("\n[*] Calculando WCSS para o método do cotovelo...")
    wcss = []
    for i in range(1, max_clusters + 1):
        km = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        km.fit(dados_normalizados)
        wcss.append(km.inertia_)
        print(f"    K={i}: WCSS={km.inertia_:.2f}")
    
    plt.figure(figsize=(10,6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o', linewidth=2, markersize=8)
    plt.xlabel("Número de clusters", fontsize=12)
    plt.ylabel("WCSS (Inércia)", fontsize=12)
    plt.title("Método do Cotovelo - Clusterização Técnica\n(Features de Produção Musical)", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, max_clusters + 1))
    plt.tight_layout()
    plt.show()

def aplicar_kmeans(dados_normalizados, n_clusters=4):
    """
    Aplica K-Means++ com número definido de clusters.
    """
    print(f"\n[*] Aplicando K-Means++ com {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    clusters = kmeans.fit_predict(dados_normalizados)
    print("[+] K-Means++ concluído.")
    print(f"[+] Distribuição dos clusters: {np.bincount(clusters)}")
    return clusters, kmeans

def analisar_clusters(df, df_features, clusters, kmeans, scaler):
    """
    Analisa e visualiza os clusters formados.
    """
    df_resultado = df.copy()
    df_resultado['cluster'] = clusters
    
    # Adicionar as features processadas ao dataframe de resultado
    for col in df_features.columns:
        df_resultado[col] = df_features[col].values
    
    features = df_features.columns.tolist()

    # Médias das features por cluster
    medias = df_resultado.groupby('cluster')[features].mean()
    print("\n" + "="*80)
    print("ANÁLISE DOS CLUSTERS - Características Técnicas de Produção")
    print("="*80)
    print("\nMédias das features por cluster:\n")
    print(medias.round(3))
    print("\n" + "="*80)
    
    # Interpretação dos clusters
    print("\nINTERPRETAÇÃO DOS CLUSTERS:\n")
    for cluster_id in range(len(medias)):
        print(f"\n🎵 CLUSTER {cluster_id}:")
        print(f"   - Loudness: {medias.loc[cluster_id, 'loudness']:.2f} dB")
        print(f"   - Speechiness: {medias.loc[cluster_id, 'speechiness']:.3f}")
        print(f"   - Instrumentalness: {medias.loc[cluster_id, 'instrumentalness']:.3f}")
        print(f"   - Liveness: {medias.loc[cluster_id, 'liveness']:.3f}")
        print(f"   - Duration: {medias.loc[cluster_id, 'duration_min']:.2f} min")
        
        # Classificação automática do tipo
        if medias.loc[cluster_id, 'speechiness'] > 0.33:
            tipo = "🎤 FALADO (Rap/Hip-Hop/Podcast)"
        elif medias.loc[cluster_id, 'instrumentalness'] > 0.5:
            tipo = "🎹 INSTRUMENTAL"
        elif medias.loc[cluster_id, 'liveness'] > 0.3:
            tipo = "🎸 AO VIVO"
        else:
            tipo = "🎧 ESTÚDIO (Produção Profissional)"
        
        print(f"   → Tipo: {tipo}")
    
    print("\n" + "="*80)

    # Visualizações
    criar_visualizacoes(df_features, clusters, kmeans, features, scaler)

    return df_resultado, medias

def criar_visualizacoes(df_features, clusters, kmeans, features, scaler):
    """
    Cria múltiplas visualizações dos clusters.
    """
    # Plot 1: Speechiness vs Instrumentalness (mostra tipo de conteúdo)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Subplot 1: Speechiness vs Instrumentalness
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(
        df_features['speechiness'],
        df_features['instrumentalness'],
        c=clusters,
        cmap='viridis',
        alpha=0.6,
        s=20
    )
    centros_originais = scaler.inverse_transform(kmeans.cluster_centers_)
    ax1.scatter(
        centros_originais[:, features.index('speechiness')],
        centros_originais[:, features.index('instrumentalness')],
        c='red',
        s=300,
        alpha=0.9,
        marker='*',
        edgecolors='black',
        linewidths=2,
        label='Centróides'
    )
    ax1.set_xlabel('Speechiness (Conteúdo Falado)', fontsize=11)
    ax1.set_ylabel('Instrumentalness (Instrumental)', fontsize=11)
    ax1.set_title('Tipo de Conteúdo Musical', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Cluster')
    
    # Subplot 2: Loudness vs Liveness (mostra contexto de produção)
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(
        df_features['loudness'],
        df_features['liveness'],
        c=clusters,
        cmap='viridis',
        alpha=0.6,
        s=20
    )
    ax2.scatter(
        centros_originais[:, features.index('loudness')],
        centros_originais[:, features.index('liveness')],
        c='red',
        s=300,
        alpha=0.9,
        marker='*',
        edgecolors='black',
        linewidths=2,
        label='Centróides'
    )
    ax2.set_xlabel('Loudness (Volume/dB)', fontsize=11)
    ax2.set_ylabel('Liveness (Gravação ao Vivo)', fontsize=11)
    ax2.set_title('Contexto de Produção', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Cluster')
    
    # Subplot 3: Duration vs Loudness (mostra formato)
    ax3 = axes[1, 0]
    scatter3 = ax3.scatter(
        df_features['duration_min'],
        df_features['loudness'],
        c=clusters,
        cmap='viridis',
        alpha=0.6,
        s=20
    )
    ax3.scatter(
        centros_originais[:, features.index('duration_min')],
        centros_originais[:, features.index('loudness')],
        c='red',
        s=300,
        alpha=0.9,
        marker='*',
        edgecolors='black',
        linewidths=2,
        label='Centróides'
    )
    ax3.set_xlabel('Duração (minutos)', fontsize=11)
    ax3.set_ylabel('Loudness (Volume/dB)', fontsize=11)
    ax3.set_title('Formato e Intensidade', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=ax3, label='Cluster')
    
    # Subplot 4: Distribuição dos clusters
    ax4 = axes[1, 1]
    contagens = np.bincount(clusters)
    cores = plt.cm.viridis(np.linspace(0, 1, len(contagens)))
    bars = ax4.bar(range(len(contagens)), contagens, color=cores, edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Cluster', fontsize=11)
    ax4.set_ylabel('Número de Músicas', fontsize=11)
    ax4.set_title('Distribuição das Músicas por Cluster', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(len(contagens)))
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('clusterizacao_tecnica.png', dpi=300, bbox_inches='tight')
    print("\n[*] Visualizações salvas em 'clusterizacao_tecnica.png'")
    plt.show()

def main():
    """
    Pipeline principal de clusterização técnica.
    
    Este modelo agrupa músicas por características de produção e formato,
    diferente do modelo emocional que usa valence, energy e danceability.
    """
    print("="*80)
    print("CLUSTERIZAÇÃO TÉCNICA DE MÚSICAS DO SPOTIFY")
    print("Modelo baseado em características de produção musical")
    print("="*80)
    
    caminho = os.path.join(os.path.dirname(__file__), "..", "DataSet", "spotify_songs.csv")
    caminho = os.path.abspath(caminho)
    print(f"\n[*] Caminho do dataset: {caminho}")

    # Carregar dados com features técnicas
    df, df_features = carregar_dados(caminho)
    features = df_features.columns.tolist()

    # Normalização
    dados_normalizados, scaler = preprocessar_dados(df_features)

    # Método do cotovelo
    escolher_numero_clusters(dados_normalizados, max_clusters=10)
    
    # Baseado na análise do cotovelo, escolher número de clusters
    # Para características técnicas, 4 clusters costuma ser ideal:
    # 1. Músicas instrumentais longas
    # 2. Músicas vocais de estúdio
    # 3. Gravações ao vivo
    # 4. Conteúdo falado (rap/hip-hop)
    n_clusters = 4

    # K-Means++
    clusters, kmeans = aplicar_kmeans(dados_normalizados, n_clusters=n_clusters)

    # Análise e visualizações
    df_resultado, medias = analisar_clusters(df, df_features, clusters, kmeans, scaler)

    # Salvar resultados
    output_file = "resultados_clusterizacao_tecnica.csv"
    df_resultado.to_csv(output_file, index=False)
    print(f"\n[✓] Resultados salvos em '{output_file}'.")
    
    # Adicionar informações sobre os clusters ao arquivo
    with open("interpretacao_clusters_tecnicos.txt", "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("INTERPRETAÇÃO DOS CLUSTERS - Modelo Técnico\n")
        f.write("="*80 + "\n\n")
        f.write("Features utilizadas:\n")
        f.write("- loudness: Volume/intensidade (dB)\n")
        f.write("- speechiness: Presença de conteúdo falado (0-1)\n")
        f.write("- instrumentalness: Nível de instrumentação (0-1)\n")
        f.write("- liveness: Indicador de gravação ao vivo (0-1)\n")
        f.write("- duration_min: Duração em minutos\n\n")
        f.write("="*80 + "\n\n")
        f.write(medias.to_string())
        f.write("\n\n" + "="*80 + "\n")
    
    print("[✓] Interpretação dos clusters salva em 'interpretacao_clusters_tecnicos.txt'.")
    print("\n" + "="*80)
    print("CLUSTERIZAÇÃO CONCLUÍDA COM SUCESSO!")
    print("="*80)

if __name__ == "__main__":
    main()
