import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try:
    csv_path = Path(__file__).resolve().parent.parent / 'DataSet' / 'spotify_songs.csv'
    df = pd.read_csv(csv_path)

    # Define o estilo dos gráficos
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # --- Gráfico 1: Top 10 Artistas Mais Populares ---
    top_artists = df.groupby('track_artist')['track_popularity'].mean().sort_values(ascending=False).head(10)
    plt.figure()
    sns.barplot(x=top_artists.values, y=top_artists.index, color='skyblue')
    plt.title('Top 10 Artistas Mais Populares')
    plt.xlabel('Popularidade Média')
    plt.ylabel('Artista')
    plt.tight_layout()
    plt.savefig('top_10_artistas.png')
    print("Gráfico 'top_10_artistas.png' gerado com sucesso.")

    # --- Gráfico 2: Distribuição de Músicas por Gênero ---
    genre_counts = df['playlist_genre'].value_counts()
    plt.figure()
    sns.barplot(x=genre_counts.values, y=genre_counts.index, color='salmon')
    plt.title('Distribuição de Músicas por Gênero')
    plt.xlabel('Número de Músicas')
    plt.ylabel('Gênero')
    plt.tight_layout()
    plt.savefig('distribuicao_genero.png')
    print("Gráfico 'distribuicao_genero.png' gerado com sucesso.")

    # --- Gráfico 3: Distribuição de Músicas por Tom (Key) ---
    key_map = {0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F', 6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'}
    df['key_name'] = df['key'].map(key_map)
    key_counts = df['key_name'].value_counts()
    plt.figure()
    sns.barplot(x=key_counts.values, y=key_counts.index, color='mediumpurple')
    plt.title('Distribuição de Músicas por Tom')
    plt.xlabel('Número de Músicas')
    plt.ylabel('Tom')
    plt.tight_layout()
    plt.savefig('distribuicao_tom.png')
    print("Gráfico 'distribuicao_tom.png' gerado com sucesso.")

    # --- Gráfico 4: Média de "Danceability" por Gênero ---
    avg_danceability = df.groupby('playlist_genre')['danceability'].mean().sort_values(ascending=False)
    plt.figure()
    sns.barplot(x=avg_danceability.values, y=avg_danceability.index, color='lightgreen')
    plt.title('Média de "Danceability" por Gênero')
    plt.xlabel('Danceability Média')
    plt.ylabel('Gênero')
    plt.tight_layout()
    plt.savefig('danceability_por_genero.png')
    print("Gráfico 'danceability_por_genero.png' gerado com sucesso.")

except FileNotFoundError:
    print("Erro: O arquivo 'spotify_songs.csv' não foi encontrado.")
    print("Por favor, verifique se o nome do arquivo está correto e se ele está no diretório esperado.")