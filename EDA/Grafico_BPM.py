import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try:
    csv_path = Path(__file__).resolve().parent.parent / 'DataSet' / 'spotify_songs.csv'
    df = pd.read_csv(csv_path)
    print("Dataset carregado com sucesso!")

    # --- Ridge Plot da Distribuição de Tempo (BPM) por Gênero ---
    print("\nGerando o Ridge Plot...")

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    
    g = sns.FacetGrid(df, row="playlist_genre", aspect=9, height=1.2)
    
    g.map_dataframe(sns.kdeplot, x="tempo", fill=True, alpha=0.8, color='steelblue')
    
    g.map_dataframe(sns.kdeplot, x="tempo", color='w')

    for i, ax in enumerate(g.axes.flat):
        genre_name = g.row_names[i]
        ax.text(-0.02, 0.2, genre_name, fontweight="bold", color='#333', 
                ha="right", va="center", transform=ax.transAxes)

    g.fig.subplots_adjust(hspace=-0.5)
    
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    plt.suptitle('Distribuição do Tempo (BPM) por Gênero Musical', y=0.98, fontsize=16)
    
    plt.savefig("ridge_plot_BPM", bbox_inches='tight')
    print("--> Gráfico Ridge Plot revisado salvo como 'ridge_plot_tempo_revisado.png'")

except FileNotFoundError:
    print("\nErro: O arquivo 'spotify_songs.csv' não foi encontrado.")
except Exception as e:
    print(f"\nOcorreu um erro inesperado: {e}")