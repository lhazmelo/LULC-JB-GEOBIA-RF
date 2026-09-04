"""
lulc.py
=======

Pipeline em Python para classificação da cobertura da terra (LULC) por GEOBIA
+ Random Forest, integrando dados ópticos (ortofoto RGB) e estruturais (LiDAR
aerotransportado). Desenvolvido para o mapeamento da área CLOUD7, no Jardim
Botânico da UFRRJ (trabalho "lh_final").

Cada função é independente, imprime seu progresso e grava seu(s) próprio(s)
arquivo(s) de saída em disco. Nenhuma delas é executada automaticamente ao
importar este módulo — elas devem ser chamadas manualmente (ex.: em um
notebook Jupyter), na ordem abaixo, usando os caminhos reais gerados nas
etapas anteriores do QGIS/DJI Terra:

    1. propriedades(...)
       Enriquece o vetor de segmentos (superpixels) com atributos LiDAR
       (CHM, TRI, Intensidade, MDT/TPI), radiometria (RGB), índices
       espectrais (VARI, NGBDI) e forma (circularidade).
       -> gera um GeoPackage com todos os atributos.

    2. rf1(...)
       Cruza os superpixels com os polígonos de treinamento, treina o
       RandomForestClassifier e classifica toda a área de estudo.
       -> gera o mapa vetorial classificado (Vetor_LULC).

    3. vetor_tif(...)
       Converte o mapa vetorial classificado em raster (.tif), usando a
       grade do CHM como referência geométrica.
       -> gera o mapa matricial final (Raster_LULC).

    4. estatisticas(...)
       Valida a classificação a partir do CSV de amostras de campo
       (amostragem aleatória estratificada), calculando matriz de
       confusão, acurácia global, F1/Recall/Precision e Índice Kappa.
       -> gera a tabela de métricas (Excel) e a imagem da matriz de confusão.

    5. calcular_areas_finais(...)
       Quantifica a área (em hectares) e o percentual de cada classe no
       mapa final.
       -> gera a planilha de áreas por classe.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
from rasterio.features import rasterize
from rasterstats import zonal_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from tqdm import tqdm


# =============================================================================
# FUNÇÃO DE APOIO — ESTATÍSTICA ZONAL EM BLOCOS (COM BARRA DE PROGRESSO)
# =============================================================================

def zonal_stats_com_progresso(
    gdf: gpd.GeoDataFrame,
    raster_array: np.ndarray,
    transform: Tuple,
    col_stats: List[str],
    nodata_val: Union[int, float],
    nome_tarefa: str,
    tamanho_fatia: int = 5000
) -> List[Dict[str, Any]]:
    """
    Calcula métricas de estatística zonal dividindo o GeoDataFrame em blocos.

    A lógica espacial consiste em sobrepor a matriz do raster (raster_array) à
    geometria vetorial (gdf). Para cada segmento, extrai-se os descritores
    estatísticos dos pixels internos, ignorando valores de borda/fundo (nodata).

    O processamento em blocos (`tamanho_fatia`) existe apenas para controlar o
    uso de memória em GeoDataFrames muito grandes — o resultado é idêntico ao
    de rodar `zonal_stats` de uma vez só sobre todo o `gdf`.

    Args:
        gdf: Base de dados vetorial contendo os segmentos.
        raster_array: Matriz bidimensional com os valores do sensor.
        transform: Transformação afim (affine) do raster.
        col_stats: Descritores estatísticos a extrair (ex.: ['mean', 'std']).
        nodata_val: Valor do pixel a ser ignorado.
        nome_tarefa: Rótulo de exibição para a barra de progresso.
        tamanho_fatia: Nº de feições por iteração, para controle de RAM.
            Padrão: 5000.

    Returns:
        Lista de dicionários, um por polígono, cada um com as estatísticas
        pedidas em `col_stats`.
    """
    resultados = []

    # Usa range diretamente (sem materializar a lista de índices na memória).
    iterador_fatias = range(0, len(gdf), tamanho_fatia)

    for i in tqdm(iterador_fatias, desc=nome_tarefa, unit="bloco"):
        # Extrai o subconjunto (fatia) sob demanda.
        pedaco = gdf.iloc[i: i + tamanho_fatia]

        res = zonal_stats(
            vectors=pedaco,
            raster=raster_array,
            affine=transform,
            stats=col_stats,
            nodata=nodata_val
        )

        resultados.extend(res)

    return resultados


# =============================================================================
# FUNÇÕES DE APOIO — LEITURA DE RASTER + ESTATÍSTICA ZONAL (uso interno)
# =============================================================================
# As duas funções abaixo concentram um padrão que se repetia 4x (CHM, TRI,
# Intensidade, MDT) e 5x (R, G, B, VARI, NGBDI) dentro de `propriedades()`:
# abrir um raster (ou usar um array já calculado), rodar a estatística zonal e
# renomear a(s) coluna(s) resultante(s). Extraí-las evita duplicação de código
# sem alterar nenhum valor calculado.

def _extrair_atributo_raster(
    gdf: gpd.GeoDataFrame,
    caminho_raster: str,
    prefixo_coluna: str,
    nome_tarefa: str,
    estatisticas: Tuple[str, ...] = ('mean', 'std'),
    limiar_nodata: float = -100.0
) -> pd.DataFrame:
    """
    Lê a banda 1 de um raster (CHM, TRI, Intensidade, MDT...), mascara valores
    espúrios abaixo de `limiar_nodata` como NaN e calcula a estatística zonal
    para cada segmento de `gdf`.

    Args:
        gdf: GeoDataFrame com os segmentos (superpixels).
        caminho_raster: Caminho do raster de entrada (banda única).
        prefixo_coluna: Prefixo usado para nomear as colunas resultantes
            (ex.: 'chm' -> 'chm_mean', 'chm_std').
        nome_tarefa: Rótulo exibido na barra de progresso.
        estatisticas: Estatísticas zonais a extrair. Padrão: média e desvio
            padrão.
        limiar_nodata: Valores do raster abaixo deste limiar são tratados
            como nodata (NaN). Padrão: -100, usado para descartar falhas do
            sensor/voo.

    Returns:
        DataFrame com uma coluna por estatística, já renomeada com o prefixo
        informado (ex.: 'chm_mean', 'chm_std').
    """
    with rasterio.open(caminho_raster) as src:
        array = src.read(1).astype(np.float32)
        array[array < limiar_nodata] = np.nan

        stats = zonal_stats_com_progresso(
            gdf, array, src.transform, list(estatisticas), np.nan, nome_tarefa
        )

    renomeio = {stat: f"{prefixo_coluna}_{stat}" for stat in estatisticas}
    return pd.DataFrame(stats).rename(columns=renomeio)


def _extrair_media_zonal(
    gdf: gpd.GeoDataFrame,
    array: np.ndarray,
    transform: Any,
    nome_coluna: str,
    nome_tarefa: str
) -> pd.DataFrame:
    """
    Calcula apenas a média zonal de um array já pronto em memória — usado para
    as bandas RGB e os índices espectrais VARI/NGBDI, que são calculados no
    próprio script (não vêm de um arquivo raster próprio).

    Args:
        gdf: GeoDataFrame com os segmentos.
        array: Matriz 2D já carregada/calculada (ex.: banda R, índice VARI).
        transform: Transformação afim do raster de origem, para
            georreferenciar o array durante a estatística zonal.
        nome_coluna: Nome final da coluna de saída (ex.: 'r_mean').
        nome_tarefa: Rótulo exibido na barra de progresso.

    Returns:
        DataFrame com uma única coluna, `nome_coluna`.
    """
    stats = zonal_stats_com_progresso(gdf, array, transform, ['mean'], np.nan, nome_tarefa)
    return pd.DataFrame(stats).rename(columns={'mean': nome_coluna})


# =============================================================================
# ETAPA 1 — ENRIQUECIMENTO DE ATRIBUTOS DOS SEGMENTOS (GEOBIA)
# =============================================================================

def propriedades(
    saida_poligonos: str,
    caminho_chm: str,
    caminho_tri: str,
    caminho_rgb: str,
    caminho_mdt: str,
    caminho_intensidade: str,
    saida_com_atributos: str
) -> None:
    """
    Enriquece o vetor de segmentos com atributos físicos (LiDAR), radiométricos
    (RGB) e de refletância ativa (Intensidade). Incorpora também um índice de
    forma (circularidade) e a textura espacial (desvio padrão) dos atributos
    LiDAR.

    Args:
        saida_poligonos: Vetor de segmentos gerado pela segmentação
            Mean-Shift (OTB_Segmentation).
        caminho_chm: Raster do Modelo de Altura do Dossel (CHM = DSM - DTM).
        caminho_tri: Raster do Índice de Rugosidade do Terreno (TRI).
        caminho_rgb: Ortofoto RGB (3 bandas).
        caminho_mdt: Modelo Digital do Terreno (usado para o TPI).
        caminho_intensidade: Raster de intensidade de retorno do LiDAR.
        saida_com_atributos: Caminho do GeoPackage de saída, já com todos os
            atributos.
    """
    print("1. Carregando o mosaico de superpixels...")
    gdf = gpd.read_file(saida_poligonos)

    # ==========================================
    # 2. ÍNDICE DE FORMA (CIRCULARIDADE / GESTALT)
    # ==========================================
    print("\nCalculando Índice de Circularidade (Gestalt)...")
    gdf['area_geom'] = gdf.geometry.area
    gdf['perimetro'] = gdf.geometry.length

    # 1.0 = círculo perfeito; valores menores indicam formas mais alongadas
    # ou irregulares. `np.where` evita divisão por zero em segmentos com
    # perímetro nulo (geometrias degeneradas).
    gdf['circularidade'] = np.where(
        gdf['perimetro'] > 0,
        (4 * np.pi * gdf['area_geom']) / (gdf['perimetro'] ** 2),
        0
    )

    # ==========================================
    # 3. ATRIBUTOS LIDAR: CHM, TRI E INTENSIDADE
    # ==========================================
    # As três variáveis seguem o mesmo padrão de extração (abrir raster,
    # mascarar nodata, estatística zonal de média + desvio padrão), por isso
    # usam o helper `_extrair_atributo_raster` em vez de repetir o bloco.
    print("\nExtraindo atributos do dossel, rugosidade e intensidade LiDAR...")
    df_chm = _extrair_atributo_raster(gdf, caminho_chm, 'chm', "CHM")
    df_tri = _extrair_atributo_raster(gdf, caminho_tri, 'tri', "TRI")
    df_int = _extrair_atributo_raster(gdf, caminho_intensidade, 'int', "Intensidade")

    # ==========================================
    # 4. CORES (RGB) E ÍNDICES ESPECTRAIS (VARI, NGBDI)
    # ==========================================
    print("\nExtraindo cores (RGB) e calculando NGBDI e VARI...")
    with rasterio.open(caminho_rgb) as src_rgb:
        r_array = src_rgb.read(1).astype(np.float32)
        g_array = src_rgb.read(2).astype(np.float32)
        b_array = src_rgb.read(3).astype(np.float32)
        transform_rgb = src_rgb.transform

        # Pixel = 0 em qualquer banda normalmente indica borda/vazio da
        # ortofoto (fora da área efetivamente voada) — tratado como nodata.
        r_array[r_array == 0] = np.nan
        g_array[g_array == 0] = np.nan
        b_array[b_array == 0] = np.nan

        with np.errstate(divide='ignore', invalid='ignore'):
            # VARI (Gitelson et al., 2002) — Equação 2 do trabalho:
            # (G - R) / (G + R - B). Realça o vigor vegetativo mitigando
            # ruído atmosférico.
            denominador_vari = g_array + r_array - b_array
            vari_array = np.where(denominador_vari != 0, (g_array - r_array) / denominador_vari, np.nan)

            # NGBDI (Xu et al., 2019) — Equação 3 do trabalho:
            # (G - B) / (G + B). Evidencia feições hídricas.
            denominador_agua = g_array + b_array
            indice_agua = np.where(denominador_agua != 0, (g_array - b_array) / denominador_agua, np.nan)

        # Estatística zonal (média) de cada banda/índice óptico.
        df_r = _extrair_media_zonal(gdf, r_array, transform_rgb, 'r_mean', "RGB (R)")
        df_g = _extrair_media_zonal(gdf, g_array, transform_rgb, 'g_mean', "RGB (G)")
        df_b = _extrair_media_zonal(gdf, b_array, transform_rgb, 'b_mean', "RGB (B)")
        df_agua = _extrair_media_zonal(gdf, indice_agua, transform_rgb, 'indice_agu', "Índice de Água (NGBDI)")
        df_vari = _extrair_media_zonal(gdf, vari_array, transform_rgb, 'vari_mean', "Índice VARI")

    # ==========================================
    # 5. ALTITUDE ABSOLUTA (MDT)
    # ==========================================
    print("\nExtraindo Altitude Absoluta (MDT)...")
    df_mdt = _extrair_atributo_raster(gdf, caminho_mdt, 'mdt', "MDT", estatisticas=('mean',))

    # ==========================================
    # 6. MONTAGEM DA TABELA E CÁLCULO DO TPI
    # ==========================================
    print("\nMontando a tabela principal...")
    gdf_final = pd.concat(
        [gdf, df_chm, df_tri, df_int, df_r, df_g, df_b, df_agua, df_mdt, df_vari],
        axis=1
    )

    # Preenche eventuais falhas de leitura do MDT (segmento fora da cobertura
    # do raster) com a média geral, evitando propagar NaN para o TPI.
    media_geral_mdt = gdf_final['mdt_mean'].mean()
    gdf_final['mdt_mean'] = gdf_final['mdt_mean'].fillna(media_geral_mdt)

    print("\nAnalisando o formato do terreno (Calculando TPI)...")
    # TPI (Weiss, 2001) = altitude do segmento − altitude média dos segmentos
    # vizinhos que compartilham fronteira (Equação 4 do trabalho). Valores
    # positivos indicam relevo elevado em relação ao entorno (ex.: dossel
    # arbóreo); valores negativos indicam depressões (ex.: corpos d'água).
    #
    # Nota de performance: esta busca de vizinhança compara cada um dos N
    # segmentos com todos os outros N (O(N²)). Para a área CLOUD7 o tempo de
    # execução é aceitável; em mosaicos muito maiores, o mesmo resultado pode
    # ser obtido de forma mais rápida usando o índice espacial do
    # GeoDataFrame (`gdf_final.sindex.query(geom, predicate="touches")`) para
    # pré-filtrar os candidatos antes de aplicar `.touches()`. Não alterei
    # essa lógica aqui para não arriscar mudar o resultado já validado.
    altitudes_vizinhas = []
    for indice, linha in tqdm(gdf_final.iterrows(), total=len(gdf_final), desc="TPI (Vizinhança)"):
        vizinhos = gdf_final[gdf_final.geometry.touches(linha['geometry'])]

        if not vizinhos.empty:
            media_vizinhos = vizinhos['mdt_mean'].mean()
        else:
            # Segmento isolado (sem vizinhos topológicos): usa a própria
            # altitude, o que resulta em TPI = 0 para esse caso.
            media_vizinhos = linha['mdt_mean']

        altitudes_vizinhas.append(media_vizinhos)

    gdf_final['tpi_mean'] = gdf_final['mdt_mean'] - altitudes_vizinhas

    # ==========================================
    # 7. SALVAR ARQUIVO FINAL
    # ==========================================
    print("\nFechando os buracos (NaN) e salvando o arquivo...")
    gdf_final = gdf_final.fillna(0)
    gdf_final.to_file(saida_com_atributos, driver="GPKG")

    print(f"✅ SUCESSO! Banco de dados espacial gerado em: {saida_com_atributos}")


# =============================================================================
# ETAPA 2 — TREINAMENTO, CLASSIFICAÇÃO E PÓS-PROCESSAMENTO (RANDOM FOREST)
# =============================================================================

def rf1(
    saida_com_atributos: str,
    caminho_treino: str,
    saida_mapa_obia: str,
    caminho_grafico_imp: str
) -> None:
    """
    Realiza o treinamento espacial, a classificação LULC via Random Forest e o
    pós-processamento do mapa.

    Etapas: (1) cruza os superpixels com os polígonos de treinamento
    rotulados manualmente; (2) treina o RandomForestClassifier; (3) classifica
    todos os superpixels e extrai a confiança (probabilidade máxima) de cada
    predição; (4) plota a importância das variáveis preditoras; (5) aplica uma
    regra física de correção (telhado sem altura mínima plausível vira solo
    exposto); e (6) exporta o mapa vetorial classificado.

    Args:
        saida_com_atributos: GeoPackage gerado por `propriedades()`.
        caminho_treino: Vetor com os polígonos de treinamento rotulados
            (coluna 'id' = classe real).
        saida_mapa_obia: Caminho de saída do mapa vetorial classificado.
        caminho_grafico_imp: Caminho de saída do gráfico de importância das
            variáveis.
    """

    # =====================================================================
    # 1. CRUZAMENTO ESPACIAL (TREINAMENTO)
    # =====================================================================
    print("Carregando os dados e cruzando mapas...")
    gdf_super = gpd.read_file(saida_com_atributos)

    # Garante um identificador único e estável por superpixel, usado depois
    # para remover duplicatas geradas pelo join espacial.
    if 'id_superpixel' not in gdf_super.columns:
        gdf_super['id_superpixel'] = gdf_super.index

    gdf_treino = gpd.read_file(caminho_treino)

    if gdf_treino.crs != gdf_super.crs:
        print(f"Corrigindo diferença de CRS: Convertendo Treino para {gdf_super.crs}...")
        gdf_treino = gdf_treino.to_crs(gdf_super.crs)

    # Junção espacial: cada superpixel recebe a classe ('id') do polígono de
    # treinamento com o qual ele intersecta.
    superpixels_treinados = gpd.sjoin(
        gdf_super,
        gdf_treino[['id', 'geometry']],
        how='inner',
        predicate='intersects'
    )

    # Um superpixel pode intersectar mais de um polígono de treino na borda;
    # mantém-se apenas uma ocorrência por superpixel para não duplicar amostras.
    superpixels_treinados = superpixels_treinados.drop_duplicates(subset=['id_superpixel']).copy()

    if superpixels_treinados.empty:
        raise ValueError("❌ ERRO GRAVE: Nenhum polígono de treinamento intersectou os superpixels!")

    print(f"✅ Sucesso! Identificados {len(superpixels_treinados)} superpixels de treinamento.")

    # =====================================================================
    # 2. TREINAMENTO DO ALGORITMO RANDOM FOREST
    # =====================================================================
    print("\nTreinando o modelo preditivo (Random Forest)...")

    # Variáveis preditoras: atributos LiDAR (CHM, TRI, Intensidade, MDT, TPI),
    # radiometria (R, G, B), índices espectrais (VARI, NGBDI) e forma
    # (circularidade) — a combinação descrita na metodologia do trabalho.
    colunas_atributos: List[str] = [
        'chm_mean', 'chm_std', 'tri_mean', 'tri_std', 'r_mean',
        'g_mean', 'b_mean', 'vari_mean', 'indice_agu',
        'mdt_mean', 'tpi_mean', 'int_mean', 'int_std', 'circularidade'
    ]

    X_treino = superpixels_treinados[colunas_atributos].fillna(0)
    y_treino = superpixels_treinados['id']

    # class_weight='balanced' e os limites de profundidade/folha foram
    # calibrados para reduzir o efeito "sal e pimenta" e compensar o
    # desbalanceamento de classes minoritárias (ver metodologia do trabalho).
    rf_model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        min_samples_leaf=3,
        max_depth=15
    )

    rf_model.fit(X_treino, y_treino)

    # =====================================================================
    # 3. PREDIÇÃO LULC E MAPA DE INCERTEZA ESPACIAL
    # =====================================================================
    print("Classificando o mapa completo e extraindo incertezas...")
    X_total = gdf_super[colunas_atributos].fillna(0)

    gdf_super['classe_predita'] = rf_model.predict(X_total)

    # A confiança de cada predição é a maior probabilidade entre as classes;
    # segmentos abaixo de 60% são sinalizados para revisão manual.
    probabilidades = rf_model.predict_proba(X_total)
    certeza_maxima = probabilidades.max(axis=1)

    gdf_super['confianca_rf_pct'] = np.round(certeza_maxima * 100, 2)
    gdf_super['alerta_incerteza'] = np.where(gdf_super['confianca_rf_pct'] < 60.0, 'ALTA INCERTEZA', 'CONFIÁVEL')

    # =====================================================================
    # 4. GRÁFICO DE IMPORTÂNCIA DE VARIÁVEIS (ATBD Padrão)
    # =====================================================================
    importancias = pd.Series(rf_model.feature_importances_, index=colunas_atributos)
    importancias = importancias.sort_values(ascending=False)

    print("\nGerando gráfico de Importância de Atributos...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 7))

    importancias_plot = importancias.sort_values(ascending=True)
    ax = importancias_plot.plot(kind='barh', color='#2b5d8c', width=0.7)

    plt.xlabel('Importância Relativa', fontsize=12)
    plt.ylabel('Variáveis Preditoras', fontsize=12)

    for i, v in enumerate(importancias_plot):
        ax.text(v + 0.003, i, f"{v:.4f}", color='black', va='center', fontsize=10)

    plt.xlim(0, importancias_plot.max() * 1.15)
    plt.tight_layout()
    plt.savefig(caminho_grafico_imp, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"✅ Gráfico salvo: {caminho_grafico_imp}")

    # =====================================================================
    # 5. FILTRO PÓS-CLASSIFICAÇÃO (REGRA FÍSICA ALTIMÉTRICA)
    # =====================================================================
    print("\nAplicando filtro topológico para mitigação de falsos positivos...")
    CLASSE_TELHADO = 6
    CLASSE_SOLO_EXPOSTO = 4
    ALTURA_MINIMA_TELHADO = 1.5

    # Regra física: um segmento não pode ser "telhado" se a altura do dossel
    # (CHM) for menor que a altura mínima plausível de uma edificação. Nesse
    # caso, é reclassificado como solo exposto e marcado como correção física
    # (não estatística).
    mascara_falso_telhado = (
        (gdf_super['classe_predita'] == CLASSE_TELHADO) &
        (gdf_super['chm_mean'] < ALTURA_MINIMA_TELHADO)
    )

    gdf_super.loc[mascara_falso_telhado, 'classe_predita'] = CLASSE_SOLO_EXPOSTO
    gdf_super.loc[mascara_falso_telhado, 'confianca_rf_pct'] = 100.0
    gdf_super.loc[mascara_falso_telhado, 'alerta_incerteza'] = 'CORREÇÃO FÍSICA'

    print(f"Filtro aplicado! {mascara_falso_telhado.sum()} superpixels reclassificados de Telhado para Solo.")

    # =====================================================================
    # 6. EXPORTAÇÃO DO MAPA VETORIAL
    # =====================================================================
    print("\nSalvando o produto vetorial final...")

    colunas_finais = ['id_superpixel', 'classe_predita', 'confianca_rf_pct', 'alerta_incerteza', 'geometry']
    gdf_final = gdf_super[colunas_finais].copy()

    gdf_final.to_file(saida_mapa_obia, driver="GPKG")
    print(f"✅ Classificação OBIA concluída! Arquivo: '{saida_mapa_obia}'")


# =============================================================================
# ETAPA 3 — CONVERSÃO DO MAPA VETORIAL CLASSIFICADO PARA RASTER
# =============================================================================

def vetor_tif(
    saida_mapa_obia: str,
    caminho_chm: str,
    saida_tif: str
) -> None:
    """
    Converte o mapa vetorial classificado (OBIA) em formato matricial (.tif).

    Usa a grade geométrica do Modelo de Altura do Dossel (CHM) como
    referência espacial (resolução, extensão e projeção), garantindo o
    alinhamento perfeito com os dados originais.

    Args:
        saida_mapa_obia: Caminho do GeoPackage com a classificação vetorial
            (saída de `rf1()`).
        caminho_chm: Caminho do raster de referência geométrica (CHM).
        saida_tif: Caminho do arquivo matricial de saída.
    """
    # =====================================================================
    # 1. LEITURA E VALIDAÇÃO DOS DADOS VETORIAIS
    # =====================================================================
    print("Carregando o mapa vetorial OBIA...")
    gdf = gpd.read_file(saida_mapa_obia)

    # Proteção de integridade: remove falhas topológicas (geometrias nulas)
    # ou polígonos sem classificação, evitando que a rasterização trave.
    gdf = gdf.dropna(subset=['geometry', 'classe_predita'])

    # =====================================================================
    # 2. EXTRAÇÃO DE METADADOS DA REFERÊNCIA
    # =====================================================================
    print("Lendo as dimensões da grade original do LiDAR...")
    with rasterio.open(caminho_chm) as src:
        meta_mestre = src.meta.copy()
        transform_mestre = src.transform
        formato_imagem = (src.height, src.width)

    # =====================================================================
    # 3. RASTERIZAÇÃO ESPACIAL
    # =====================================================================
    print("Rasterizando os polígonos (Vetor -> Matriz)...")

    # Generator expression com conversão explícita para int, garantindo
    # compatibilidade com o dtype np.uint8 da matriz final.
    shapes_para_queimar = (
        (geom, int(valor)) for geom, valor in zip(gdf.geometry, gdf['classe_predita'])
    )

    mapa_rasterizado = rasterize(
        shapes=shapes_para_queimar,
        out_shape=formato_imagem,
        transform=transform_mestre,
        fill=0,  # Valor de nodata / fundo
        dtype=np.uint8
    )

    # =====================================================================
    # 4. SALVAMENTO E ATUALIZAÇÃO CARTOGRÁFICA
    # =====================================================================
    print("Salvando o produto final matricial...")

    # Atualiza os metadados para gerar um arquivo de banda única e leve.
    meta_mestre.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=0,
        compress='lzw'
    )

    with rasterio.open(saida_tif, 'w', **meta_mestre) as dst:
        dst.write(mapa_rasterizado, 1)

    print(f"✅ Sucesso! O mapa matricial final foi gerado em: {saida_tif}")


# =============================================================================
# ETAPA 4 — VALIDAÇÃO ESTATÍSTICA (MATRIZ DE CONFUSÃO, ACURÁCIA, KAPPA)
# =============================================================================

def estatisticas(
    caminho_csv: str,
    caminho_excel: str,
    caminho_fig: str,
) -> None:
    """
    Processa a auditoria espacial, calculando métricas de validação e gerando
    a Matriz de Confusão.

    Lê o CSV de amostras — Ground Truth cruzado com a predição do modelo
    (colunas 'classe_real' e 'classe_predita1', geradas pela amostragem
    aleatória estratificada + "Amostrar valores do raster" no QGIS) —,
    calcula F1-Score, Recall, Precision e o Índice Kappa, exportando um
    relatório tabular (Excel) e um mapa de calor da matriz de confusão.

    Args:
        caminho_csv: Arquivo tabular com as colunas 'classe_real' e
            'classe_predita1'.
        caminho_excel: Caminho de saída para a tabela final de métricas.
        caminho_fig: Caminho de saída para a imagem (.png/.jpg) da matriz de
            confusão.
    """
    print("--- CALCULANDO E EXPORTANDO MÉTRICAS ESTATÍSTICAS ---")

    # =====================================================================
    # 1. LEITURA DE DADOS E CONFIGURAÇÃO
    # =====================================================================
    df_val = pd.read_csv(caminho_csv, sep=',')
    y_real = df_val['classe_real']
    y_pred = df_val['classe_predita1']

    labels_classes = ['Arbórea', 'Arbustiva', 'Gramínea', 'Solo Exposto', 'Asfalto', 'Telhado', 'Água']

    # =====================================================================
    # 2. CÁLCULO E EXPORTAÇÃO DAS MÉTRICAS (EXCEL)
    # =====================================================================
    kappa = cohen_kappa_score(y_real, y_pred)

    report_dict = classification_report(y_real, y_pred, target_names=labels_classes, output_dict=True)

    # Transforma em DataFrame e arredonda para o rigor acadêmico (4 casas).
    df_metricas = pd.DataFrame(report_dict).transpose().round(4)

    # Acrescenta o Índice Kappa como linha extra da tabela. As 4 colunas
    # seguem o padrão do classification_report (precision, recall, f1-score,
    # support); as três últimas ficam em branco por não se aplicarem ao Kappa.
    df_metricas.loc['Índice Kappa'] = [kappa, '', '', '']

    df_metricas.to_excel(caminho_excel, index=True)
    print(f"✅ Tabela de métricas (Excel) salva com sucesso em: {caminho_excel}")

    # =====================================================================
    # 3. GERAÇÃO DA MATRIZ DE CONFUSÃO VISUAL (MAPA DE CALOR)
    # =====================================================================
    cm = confusion_matrix(y_real, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_classes)
    disp.plot(cmap='Blues', ax=ax, values_format='d', xticks_rotation=45)

    plt.ylabel('Classe Real (Verdade Terrestre)', fontsize=12)
    plt.xlabel('Classe Predita (Modelo)', fontsize=12)
    plt.tight_layout()

    # bbox_inches='tight' garante que os rótulos do eixo X (rotacionados) não
    # sejam cortados.
    plt.savefig(caminho_fig, dpi=300, bbox_inches='tight')
    plt.close('all')

    print(f"✅ Matriz de Confusão em alta resolução salva com sucesso em: {caminho_fig}")


# =============================================================================
# ETAPA 5 — QUANTIFICAÇÃO DE ÁREAS POR CLASSE
# =============================================================================

def calcular_areas_finais(
    saida_mapa_obia: str,
    caminho_excel_areas: str,
    dicionario_classes: Optional[Dict[int, str]] = None
) -> None:
    """
    Quantifica a distribuição territorial das classes de uso e cobertura do
    solo.

    Calcula a área geométrica de cada segmento classificado, converte para
    hectares e gera um relatório estatístico com a proporção percentual da
    paisagem.

    Args:
        saida_mapa_obia: Caminho do arquivo vetorial com a classificação
            final.
        caminho_excel_areas: Caminho para salvar a planilha Excel de saída.
        dicionario_classes: Mapeamento opcional de IDs para nomes de classe.
            Se None, usa as 7 macroclasses do trabalho.
    """
    print("\n" + "=" * 50)
    print("🌍 CALCULANDO AS ÁREAS DO MAPA FINAL 🌍")

    # =====================================================================
    # 1. LEITURA E VALIDAÇÃO ESPACIAL
    # =====================================================================
    gdf = gpd.read_file(saida_mapa_obia)

    # Verificação crítica: o cálculo geométrico exige coordenadas projetadas
    # (ex.: UTM). Em graus decimais (WGS 84 puro), gdf.area retorna valores
    # incorretos (graus², não m²).
    if not gdf.crs.is_projected:
        warnings.warn(
            "ALERTA: O sistema de coordenadas não é projetado. "
            "O cálculo de área resultará em graus e não em metros quadrados. "
            "Certifique-se de que o dado original estava em UTM/SIRGAS 2000."
        )

    # Lógica defensiva para encontrar a coluna correta: aceita tanto a saída
    # de rf1() ('classe_predita') quanto um raster vetorizado manualmente no
    # QGIS (coluna 'DN', padrão da ferramenta "Raster para vetor").
    coluna_classe = 'classe_predita' if 'classe_predita' in gdf.columns else 'DN'

    if coluna_classe not in gdf.columns:
        raise KeyError("❌ ERRO: Nenhuma coluna de classificação encontrada no vetor.")

    # =====================================================================
    # 2. CONFIGURAÇÃO DA NOMENCLATURA DAS CLASSES
    # =====================================================================
    if dicionario_classes is None:
        dicionario_classes = {
            1: 'Arbórea',
            2: 'Arbustiva',
            3: 'Gramínea',
            4: 'Solo Exposto',
            5: 'Asfalto',
            6: 'Telhado',
            7: 'Água'
        }

    # =====================================================================
    # 3. CÁLCULO GEOMÉTRICO E ESTATÍSTICO
    # =====================================================================
    # Área planimétrica exata de cada segmento.
    gdf['area_m2'] = gdf.geometry.area

    # Agrupa todos os segmentos que pertencem à mesma classe.
    areas_por_classe = gdf.groupby(coluna_classe)['area_m2'].sum().reset_index()

    # Conversão métrica (1 hectare = 10.000 m²).
    areas_por_classe['Área (hectares)'] = areas_por_classe['area_m2'] / 10000

    # Vincula o código numérico ao nome descritivo da classe.
    areas_por_classe['Nome da Classe'] = areas_por_classe[coluna_classe].map(dicionario_classes)

    # Representatividade percentual de cada classe na paisagem.
    area_total = areas_por_classe['Área (hectares)'].sum()
    areas_por_classe['Porcentagem (%)'] = (areas_por_classe['Área (hectares)'] / area_total) * 100

    # =====================================================================
    # 4. ORGANIZAÇÃO EDITORIAL E EXPORTAÇÃO
    # =====================================================================
    tabela_final = areas_por_classe[['Nome da Classe', 'Área (hectares)', 'Porcentagem (%)']].copy()

    # Ordena as classes da mais abundante para a menos abundante.
    tabela_final = tabela_final.sort_values(by='Área (hectares)', ascending=False)

    # Padroniza a exibição com 2 casas decimais.
    tabela_final = tabela_final.round(2)

    tabela_final.to_excel(caminho_excel_areas, index=False)

    print(f"✅ Cálculo territorial concluído! Tabela exportada para: {caminho_excel_areas}")
    print("\n--- Resumo Sintético das Áreas (ha) ---")
    print(tabela_final.to_string(index=False))
