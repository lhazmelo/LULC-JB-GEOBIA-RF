import geopandas as gpd 
import rasterio 
from rasterio.features import rasterize
from rasterstats import zonal_stats
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm
from typing import List, Dict, Any, Union, Tuple, Optional
import warnings

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
    
    Args:
        gdf (gpd.GeoDataFrame): Base de dados vetorial contendo os segmentos.
        raster_array (np.ndarray): Matriz bidimensional com os valores do sensor.
        transform (Tuple): Parâmetros de afinidade geométrica do raster.
        col_stats (List[str]): Descritores estatísticos a extrair (ex: ['mean', 'std']).
        nodata_val (Union[int, float]): Valor do pixel a ser ignorado.
        nome_tarefa (str): Rótulo de exibição para a barra de progresso.
        tamanho_fatia (int, opcional): N° de feições por iteração para controle de RAM. Padrão é 5000.
        
    Returns:
        List[Dict[str, Any]]: Lista de dicionários, onde cada dicionário contém as 
                              estatísticas de um polígono específico.
    """
    resultados = []
    
    # 1. Uso do range diretamente para evitar a criação prévia de uma lista pesada na memória
    iterador_fatias = range(0, len(gdf), tamanho_fatia)
    
    # 2. Iteração com barra de progresso
    for i in tqdm(iterador_fatias, desc=nome_tarefa, unit="bloco"):
        # Extrai o subconjunto (fatia) sob demanda
        pedaco = gdf.iloc[i : i + tamanho_fatia]
        
        # 3. Execução do processamento zonal
        res = zonal_stats(
            vectors=pedaco, 
            raster=raster_array, 
            affine=transform, 
            stats=col_stats, 
            nodata=nodata_val
        )
        
        resultados.extend(res)
        
    return resultados

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
    Enriquece o vetor de segmentos com atributos físicos (LiDAR), radiométricos (RGB) 
    e de refletância ativa (Intensidade).
    Incorpora índices de forma (Circularidade) e textura espacial (std) baseados no 
    protocolo ATBD do MapBiomas.
    """
    print("1. Carregando o mosaico de superpixels...")
    gdf = gpd.read_file(saida_poligonos)

    # ==========================================
    # 2. ÍNDICES DE FORMA (GEOMETRIA/GESTALT)
    # ==========================================
    print("\nCalculando Índice de Circularidade (Gestalt)...")
    gdf['area_geom'] = gdf.geometry.area
    gdf['perimetro'] = gdf.geometry.length

    # Prevenção rigorosa de divisão por zero
    gdf['circularidade'] = np.where(
        gdf['perimetro'] > 0, 
        (4 * np.pi * gdf['area_geom']) / (gdf['perimetro'] ** 2), 
        0
    )

    # ==========================================
    # 3. EXTRAIR ESTATÍSTICAS DO LIDAR (CHM, TRI e Intensidade)
    # ==========================================
    print("\nExtraindo atributos do dossel, rugosidade e intensidade LiDAR...")
    
    # 3.1 Modelo de Altura do Dossel (CHM)
    with rasterio.open(caminho_chm) as src_chm:
        chm_array = src_chm.read(1).astype(np.float32)
        chm_array[chm_array < -100] = np.nan  
        stats_chm = zonal_stats_com_progresso(gdf, chm_array, src_chm.transform, ['mean', 'std'], np.nan, "CHM")
        df_chm = pd.DataFrame(stats_chm).rename(columns={'mean': 'chm_mean', 'std': 'chm_std'})

    # 3.2 Índice de Rugosidade (TRI)
    with rasterio.open(caminho_tri) as src_tri:
        tri_array = src_tri.read(1).astype(np.float32)
        tri_array[tri_array < -100] = np.nan
        stats_tri = zonal_stats_com_progresso(gdf, tri_array, src_tri.transform, ['mean', 'std'], np.nan, "TRI")
        df_tri = pd.DataFrame(stats_tri).rename(columns={'mean': 'tri_mean', 'std': 'tri_std'})    
        
    # 3.3 Intensidade (Refletância Ativa)
    with rasterio.open(caminho_intensidade) as src_int:
        int_array = src_int.read(1).astype(np.float32)
        int_array[int_array < -100] = np.nan
        stats_int = zonal_stats_com_progresso(gdf, int_array, src_int.transform, ['mean', 'std'], np.nan, "Intensidade")
        df_int = pd.DataFrame(stats_int).rename(columns={'mean': 'int_mean', 'std': 'int_std'})
    
    # ==========================================
    # 4. EXTRAIR CORES E ÍNDICES VISUAIS
    # ==========================================
    print("\nExtraindo cores (RGB) e calculando NGBDI e VARI...")
    with rasterio.open(caminho_rgb) as src_rgb:
        r_array = src_rgb.read(1).astype(np.float32)
        g_array = src_rgb.read(2).astype(np.float32)
        b_array = src_rgb.read(3).astype(np.float32)
        transform_rgb = src_rgb.transform
        
      
        r_array[r_array == 0] = np.nan
        g_array[g_array == 0] = np.nan
        b_array[b_array == 0] = np.nan
        
       
        with np.errstate(divide='ignore', invalid='ignore'):
            # VARI
            denominador_vari = (g_array + r_array - b_array)
            vari_array = np.where(denominador_vari != 0, (g_array - r_array) / denominador_vari, np.nan)
            
            # NGBDI (Água)
            denominador_agua = (g_array + b_array)
            indice_agua = np.where(denominador_agua != 0, (g_array - b_array) / denominador_agua, np.nan)
        
        # --- EXTRAÇÃO DE ESTATÍSTICAS ---
        stats_r = zonal_stats_com_progresso(gdf, r_array, transform_rgb, ['mean'], np.nan, "RGB (R)")
        stats_g = zonal_stats_com_progresso(gdf, g_array, transform_rgb, ['mean'], np.nan, "RGB (G)")
        stats_b = zonal_stats_com_progresso(gdf, b_array, transform_rgb, ['mean'], np.nan, "RGB (B)")
        stats_agua = zonal_stats_com_progresso(gdf, indice_agua, transform_rgb, ['mean'], np.nan, "Índice de Água")
        stats_vari = zonal_stats_com_progresso(gdf, vari_array, transform_rgb, ['mean'], np.nan, "Índice VARI")

    df_r = pd.DataFrame(stats_r).rename(columns={'mean': 'r_mean'})
    df_g = pd.DataFrame(stats_g).rename(columns={'mean': 'g_mean'})
    df_b = pd.DataFrame(stats_b).rename(columns={'mean': 'b_mean'})
    df_agua = pd.DataFrame(stats_agua).rename(columns={'mean': 'indice_agu'})
    df_vari = pd.DataFrame(stats_vari).rename(columns={'mean': 'vari_mean'})
    
    # ==========================================
    # 5. EXTRAIR ALTITUDE ABSOLUTA (MDT)
    # ==========================================
    print("\nExtraindo Altitude Absoluta (MDT)...")
    with rasterio.open(caminho_mdt) as src_mdt:
        mdt_array = src_mdt.read(1).astype(np.float32)
        mdt_array[mdt_array < -100] = np.nan
        stats_mdt = zonal_stats_com_progresso(gdf, mdt_array, src_mdt.transform, ['mean'], np.nan, "MDT")
    
    df_mdt = pd.DataFrame(stats_mdt).rename(columns={'mean': 'mdt_mean'})
    
    # ==========================================
    # 6. MONTAGEM E CÁLCULO DO TPI
    # ==========================================
    print("\nMontando a tabela principal...")
    # ADICIONADO df_int na concatenação final
    gdf_final = pd.concat([gdf, df_chm, df_tri, df_int, df_r, df_g, df_b, df_agua, df_mdt, df_vari], axis=1)
    
    # Preenchimento de falhas no MDT com a média geral
    media_geral_mdt = gdf_final['mdt_mean'].mean()
    gdf_final['mdt_mean'] = gdf_final['mdt_mean'].fillna(media_geral_mdt)
    
    print("\nAnalisando o formato do terreno (Calculando TPI)...")
    altitudes_vizinhas = []
    
    for indice, linha in tqdm(gdf_final.iterrows(), total=len(gdf_final), desc="TPI (Vizinhança)"):
        vizinhos = gdf_final[gdf_final.geometry.touches(linha['geometry'])]
        
        if not vizinhos.empty:
            media_vizinhos = vizinhos['mdt_mean'].mean()
        else:
            media_vizinhos = linha['mdt_mean']
            
        altitudes_vizinhas.append(media_vizinhos)
        
    # TPI: Altitude local - Média do entorno
    gdf_final['tpi_mean'] = gdf_final['mdt_mean'] - altitudes_vizinhas
    
    # ==========================================
    # 7. SALVAR ARQUIVO FINAL
    # ==========================================
    print("\nFechando os buracos (NaN) e salvando o arquivo...")
 
    gdf_final = gdf_final.fillna(0)
    gdf_final.to_file(saida_com_atributos, driver="GPKG")
    
    print(f"✅ SUCESSO! Banco de dados espacial gerado em: {saida_com_atributos}")
    return propriedades

def rf1(
    saida_com_atributos: str, 
    caminho_treino: str, 
    saida_mapa_obia: str,
    caminho_grafico_imp: str
) -> None:
    """
    Realiza o treinamento espacial, classificação LULC via Random Forest e pós-processamento.
    Extrai a incerteza estatística  e aplica filtro físico de dossel.
    """
    
    # =====================================================================
    # 1. CRUZAMENTO ESPACIAL (TREINAMENTO)
    # =====================================================================
    print("Carregando os dados e cruzando mapas...")
    gdf_super = gpd.read_file(saida_com_atributos)
    
    if 'id_superpixel' not in gdf_super.columns:
        gdf_super['id_superpixel'] = gdf_super.index

    gdf_treino = gpd.read_file(caminho_treino)
    
    if gdf_treino.crs != gdf_super.crs:
        print(f"Corrigindo diferença de CRS: Convertendo Treino para {gdf_super.crs}...")
        gdf_treino = gdf_treino.to_crs(gdf_super.crs)
        
    superpixels_treinados = gpd.sjoin(
        gdf_super, 
        gdf_treino[['id', 'geometry']], 
        how='inner', 
        predicate='intersects'
    )
    
    superpixels_treinados = superpixels_treinados.drop_duplicates(subset=['id_superpixel']).copy()
    
    if superpixels_treinados.empty:
        raise ValueError("❌ ERRO GRAVE: Nenhum polígono de treinamento intersectou os superpixels!")
        
    print(f"✅ Sucesso! Identificados {len(superpixels_treinados)} superpixels de treinamento.")

    # =====================================================================
    # 2. TREINAMENTO DO ALGORITMO RANDOM FOREST
    # =====================================================================
    print("\nTreinando o modelo preditivo (Random Forest)...")
    
    colunas_atributos: List[str] = [
        'chm_mean', 'chm_std', 'tri_mean', 'tri_std', 'r_mean', 
        'g_mean', 'b_mean', 'vari_mean', 'indice_agu', 
        'mdt_mean', 'tpi_mean', 'int_mean', 'int_std', 'circularidade'
    ]
    
    X_treino = superpixels_treinados[colunas_atributos].fillna(0)
    y_treino = superpixels_treinados['id']
        
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
    
    # A Predição Bruta
    gdf_super['classe_predita'] = rf_model.predict(X_total)
    
    # Extração de Incerteza
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
    
    mascara_falso_telhado = (
        (gdf_super['classe_predita'] == CLASSE_TELHADO) & 
        (gdf_super['chm_mean'] < ALTURA_MINIMA_TELHADO)
    )
    
    # Aplica o filtro físico cravando "Solo Exposto" na classe e 100% na confiança (pois foi intervenção direta)
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
    return rf1

def vetor_tif(
        saida_mapa_obia: str,
        caminho_chm: str,
        saida_tif: str
)-> None:
    """
    Converte o mapa vetorial classificado (OBIA) em formato matricial (.tif).
    
    A função utiliza a grade geométrica do Modelo de Altura do Dossel (CHM) como 
    referência espacial (resolução, extensão espacial e projeção). Isso garante 
    o alinhamento perfeito  com os dados originais.
    
    Args:
        saida_mapa_obia (str): Caminho do GeoPackage com a classificação vetorial.
        caminho_chm (str): Caminho do raster de referência geométrica.
        saida_tif (str): Caminho do arquivo matricial exportado.
    """
    
    # =====================================================================
    # 1. LEITURA E VALIDAÇÃO DOS DADOS VETORIAIS
    # =====================================================================
    print("Carregando o mapa vetorial OBIA...")
    gdf = gpd.read_file(saida_mapa_obia)
    
    # Proteção de Integridade: Remove falhas topológicas (geometrias nulas)
    # ou polígonos que não receberam classificação, evitando que a rasterização trave.
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
    
    # Generator expression com conversão explícita (int) para garantir compatibilidade
    # com o formato np.uint8 da matriz final.
    shapes_para_queimar = (
        (geom, int(valor)) for geom, valor in zip(gdf.geometry, gdf['classe_predita'])
    )
    
    mapa_rasterizado = rasterize(
        shapes=shapes_para_queimar,
        out_shape=formato_imagem,
        transform=transform_mestre,
        fill=0,  # Valor de Nodata / Fundo
        dtype=np.uint8
    )
    
    # =====================================================================
    # 4. SALVAMENTO E ATUALIZAÇÃO CARTOGRÁFICA
    # =====================================================================
    print("Salvando o produto final matricial...")
    
    # Atualização dos metadados para garantir um arquivo de banda única e leve
    meta_mestre.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=0,
        compress='lzw' 
    )
    
    with rasterio.open(saida_tif, 'w', **meta_mestre) as dst:
        dst.write(mapa_rasterizado, 1)
        
    print(f"✅ Sucesso! O mapa matricial final foi gerado em: {saida_tif}")
    return vetor_tif

def estatisticas(
    caminho_csv: str, 
    caminho_excel: str, 
    caminho_fig: str,
) -> None:
    """
    Processa a auditoria espacial, calculando métricas de validação e gerando a Matriz de Confusão.
    
    A função lê o Ground Truth cruzado com a predição do modelo, 
    calcula F1-Score, Recall, Precision e o Índice Kappa, exportando um relatório
    tabular (Excel) e um gráfico de calor.
    
    Args:
        caminho_csv (str): Arquivo tabular com as colunas 'Classe_real' e 'Camada_predita1'.
        caminho_excel (str): Caminho de saída para a tabela final de métricas.
        caminho_fig (str): Caminho de saída para a imagem (.png/.jpg) da matriz de confusão.
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
    
    # Gera o relatório como dicionário (usando os nomes das classes)
    report_dict = classification_report(y_real, y_pred, target_names=labels_classes, output_dict=True)
    
    # Transforma em DataFrame e arredonda para o rigor acadêmico (4 casas)
    df_metricas = pd.DataFrame(report_dict).transpose().round(4)
    
    # Adiciona a linha do Índice Kappa no rodapé da tabela 
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
    
    # O bbox_inches='tight' garante que os rótulos do eixo X (rotacionados) não sejam cortados
    plt.savefig(caminho_fig, dpi=300, bbox_inches='tight')
    
    plt.close('all') 
    
    print(f"✅ Matriz de Confusão em alta resolução salva com sucesso em: {caminho_fig}")
    return estatisticas

def calcular_areas_finais(
    saida_mapa_obia: str, 
    caminho_excel_areas: str,
    dicionario_classes: Optional[Dict[int, str]] = None
) -> None:
    """
    Quantifica a distribuição territorial das classes de uso e cobertura do solo.
    
    A rotina calcula a área geométrica de cada segmento classificado, converte 
    para hectares e gera um relatório estatístico com a proporção percentual da 
    paisagem.
    
    Args:
        saida_mapa_obia (str): Caminho do arquivo vetorial com a classificação final.
        caminho_excel_areas (str): Caminho para salvar a planilha Excel de saída.
        dicionario_classes (Dict[int, str], opcional): Mapeamento de IDs para nomes.
    """
    print("\n" + "="*50)
    print("🌍 CALCULANDO AS ÁREAS DO MAPA FINAL 🌍")
    
    # =====================================================================
    # 1. LEITURA E VALIDAÇÃO ESPACIAL
    # =====================================================================
    gdf = gpd.read_file(saida_mapa_obia)
    
    # Verificação crítica: O cálculo geométrico exige coordenadas projetadas (ex: UTM)
    # Se estiver em graus decimais (WGS 84 puro), gdf.area retornará resultados incorretos.
    if not gdf.crs.is_projected:
        warnings.warn(
            "ALERTA: O sistema de coordenadas não é projetado. "
            "O cálculo de área resultará em graus e não em metros quadrados. "
            "Certifique-se de que o dado original estava em UTM/SIRGAS 2000."
        )
        
    # Lógica defensiva para encontrar a coluna correta (evita KeyErrors)
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
    # Calcula a área planimétrica exata de cada segmento
    gdf['area_m2'] = gdf.geometry.area
    
    # Agrupa todos os segmentos que pertencem à mesma classe
    areas_por_classe = gdf.groupby(coluna_classe)['area_m2'].sum().reset_index()
    
    # Conversão métrica (1 hectare = 10.000 metros quadrados)
    areas_por_classe['Área (hectares)'] = areas_por_classe['area_m2'] / 10000
    
    # Vincula o código numérico ao nome descritivo da classe
    areas_por_classe['Nome da Classe'] = areas_por_classe[coluna_classe].map(dicionario_classes)
    
    # Calcula a representatividade percentual da paisagem
    area_total = areas_por_classe['Área (hectares)'].sum()
    areas_por_classe['Porcentagem (%)'] = (areas_por_classe['Área (hectares)'] / area_total) * 100
    
    # =====================================================================
    # 4. ORGANIZAÇÃO EDITORIAL E EXPORTAÇÃO
    # =====================================================================
    tabela_final = areas_por_classe[['Nome da Classe', 'Área (hectares)', 'Porcentagem (%)']].copy()
    
    # Ordena as classes da mais abundante para a menos abundante
    tabela_final = tabela_final.sort_values(by='Área (hectares)', ascending=False)
    
    # Padroniza a exibição com 2 casas decimais
    tabela_final = tabela_final.round(2) 
    
    tabela_final.to_excel(caminho_excel_areas, index=False)
    
    print(f"✅ Cálculo territorial concluído! Tabela exportada para: {caminho_excel_areas}")
    print("\n--- Resumo Sintético das Áreas (ha) ---")
    print(tabela_final.to_string(index=False))
    return calcular_areas_finais

