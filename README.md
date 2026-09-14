# Classificação LULC com GEOBIA, LiDAR e Random Forest

Pipeline semiautomatizado para classificação do Uso e Cobertura da Terra
(LULC), combinando ortofoto RGB, LiDAR aerotransportado, Análise de Imagens
Baseada em Objetos Geográficos (GEOBIA) e Random Forest.

> **Área de estudo:** setor CLOUD7 — Jardim Botânico da UFRRJ,
> Seropédica/RJ  
> **Sensor:** DJI Zenmuse L2 embarcado em ARP DJI Matrice 350 RTK  
> **Resultados do estudo:** acurácia global de 92,38% e índice Kappa de 0,911

## Visão geral

A classificação pixel a pixel pode confundir alvos espectralmente semelhantes,
como copas e gramíneas ou asfalto e água. Este projeto trabalha com segmentos
(superpixels) e integra três grupos de informação:

- **Espectral:** médias de R, G e B, VARI e NGBDI;
- **Estrutural e topográfica:** CHM, TRI, intensidade LiDAR, MDT e TPI;
- **Geométrica:** área, perímetro e circularidade dos segmentos.

O Random Forest produz a classe inicial e um indicador relativo de incerteza.
Depois, uma regra física corrige segmentos classificados como telhado cuja
altura no CHM seja inferior a 1,5 m. A classe original do modelo é preservada,
permitindo auditar separadamente a predição estatística e a correção física.

## Produtos

![Carta-imagem da área de estudo](docs/CartaImagem_retrato.png)

![Mapa de uso e cobertura da terra](docs/LULC_retrato.png)

![Matriz de confusão](docs/Matriz_Confusao.png)

## Fluxo metodológico

1. Pré-processamento dos dados no DJI Terra e no QGIS;
2. segmentação Mean-Shift da ortofoto com o Orfeo ToolBox;
3. extração de atributos por segmento com `propriedades(...)`;
4. seleção das amostras por maior sobreposição espacial e treinamento com
   `rf1(...)`;
5. rasterização do mapa com `vetor_tif(...)`;
6. pós-processamento do raster com o Crivo (Sieve) do GDAL, quando aplicável;
7. validação com amostras de referência por `estatisticas(...)`;
8. cálculo de áreas por classe com `calcular_areas_finais(...)`.

A etapa 6 foi realizada no QGIS para o resultado apresentado neste
repositório; ela não é executada por `lulc.py`.

## Pós-processamento usado no mapeamento CLOUD7

Após a rasterização, foi adotada uma Unidade Mapeável Mínima (UMM) de 1 m².
O limiar do Crivo foi convertido para pixels pela relação:

$
\text{limiar em pixels} = \frac{\text{UMM}}{\text{GSD}^2}
$

Com GSD de 0,0541 m, o cálculo resulta em aproximadamente 342 pixels. No
processamento foi usado o valor prático de 350 pixels, com conectividade de
oito vizinhos. O Crivo generaliza regiões conectadas menores que esse limiar;
por isso, deve ser aplicado antes de gerar as amostras de validação do produto
final.

## Classes

| Código | Classe |
|---:|---|
| 1 | Arbórea |
| 2 | Arbustiva |
| 3 | Gramínea |
| 4 | Solo exposto |
| 5 | Asfalto |
| 6 | Telhado |
| 7 | Água |

O valor `0` é reservado para fundo/`nodata` no raster final.

## Requisitos

- Python 3.10 ou superior;
- QGIS com Orfeo ToolBox, para a preparação e segmentação;
- DJI Terra, usado no processamento dos dados do levantamento;
- um ambiente geoespacial capaz de instalar GDAL/PROJ pelas dependências de
  GeoPandas e Rasterio.

Instalação das dependências Python:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Em ambientes onde GDAL/PROJ não estejam disponíveis, recomenda-se instalar as
bibliotecas geoespaciais com Conda/Mamba antes de executar o `pip`.

## Dados de entrada

| Entrada | Conteúdo esperado |
|---|---|
| Segmentos | vetor poligonal gerado pela segmentação Mean-Shift |
| CHM | raster de altura do dossel |
| TRI | raster de rugosidade |
| RGB | ortofoto com três bandas na ordem R, G e B |
| MDT | modelo digital do terreno |
| Intensidade | raster de intensidade LiDAR |
| Treinamento | vetor poligonal com a classe numérica na coluna `id` |
| Validação | CSV com `classe_real` e `classe_predita1` |

Os vetores usados nos cálculos geométricos devem possuir CRS projetado com
unidades em metros.

Os dados usados no mapeamento CLOUD7 não estão versionados neste repositório;
portanto, seus resultados numéricos não podem ser reproduzidos apenas com os
arquivos aqui disponíveis. O código pode ser aplicado a outros dados
compatíveis, mas as métricas dependerão da qualidade dos insumos, das amostras
e do contexto espacial — não se espera necessariamente obter valores
semelhantes aos reportados aqui.

## Exemplo de execução

O módulo não executa etapas automaticamente ao ser importado. Ajuste os
caminhos abaixo e chame as funções na ordem indicada:

```python
from lulc import (
    calcular_areas_finais,
    estatisticas,
    propriedades,
    rf1,
    vetor_tif,
)

propriedades(
    saida_poligonos="dados/segmentos.gpkg",
    caminho_chm="dados/chm.tif",
    caminho_tri="dados/tri.tif",
    caminho_rgb="dados/ortofoto_rgb.tif",
    caminho_mdt="dados/mdt.tif",
    caminho_intensidade="dados/intensidade.tif",
    saida_com_atributos="resultados/segmentos_atributos.gpkg",
)

rf1(
    saida_com_atributos="resultados/segmentos_atributos.gpkg",
    caminho_treino="dados/amostras_treinamento.gpkg",
    saida_mapa_obia="resultados/lulc_vetor.gpkg",
    caminho_grafico_imp="resultados/importancia_variaveis.png",
    fracao_minima_treino=0.5,
)

vetor_tif(
    saida_mapa_obia="resultados/lulc_vetor.gpkg",
    caminho_chm="dados/chm.tif",
    saida_tif="resultados/lulc.tif",
)

estatisticas(
    caminho_csv="dados/validacao.csv",
    caminho_excel="resultados/metricas.xlsx",
    caminho_fig="resultados/matriz_confusao.png",
)

calcular_areas_finais(
    saida_mapa_obia="resultados/lulc_vetor.gpkg",
    caminho_excel_areas="resultados/areas_por_classe.xlsx",
)
```

A classe de treinamento de cada superpixel é escolhida pela maior área de
interseção. Por padrão, pelo menos 50% do superpixel deve estar coberto pela
classe selecionada; esse limite pode ser aumentado para formar amostras mais
puras.

## Saídas e rastreabilidade

A saída vetorial de `rf1(...)` mantém:

| Campo | Significado |
|---|---|
| `classe_rf` | classe originalmente prevista pelo Random Forest |
| `classe_predita` | classe final após a regra física |
| `confianca_rf_pct` | maior valor de `predict_proba`, em porcentagem |
| `alerta_incerteza` | indicação relativa baseada no limite de 60% |
| `correcao_fisica` | informa se a regra altimétrica alterou a classe |
| `motivo_correcao` | justificativa da alteração pós-classificação |

`confianca_rf_pct` é um indicador interno do modelo, não uma probabilidade
necessariamente calibrada de acerto.

## Validação

`estatisticas(...)` calcula precision, recall, F1-score, acurácia global,
índice Kappa e matriz de confusão. No mapeamento CLOUD7, foram distribuídos
30 pontos independentes em cada uma das sete classes, totalizando 210 amostras
de validação. A classe de referência foi interpretada visualmente sobre a
ortofoto de alta resolução, enquanto a classe predita foi extraída do raster
pós-processado no QGIS.

Os valores de 92,38% de acurácia global e 0,911 de Kappa correspondem a esse
desenho amostral. Como os dados de validação não estão versionados aqui, esses
números não podem ser recalculados apenas com o conteúdo deste repositório.

## Limitações

- o TPI usa uma busca de vizinhança com custo aproximadamente quadrático;
- as importâncias apresentadas pelo Random Forest são baseadas em redução de
  impureza e não devem ser interpretadas como causalidade;
- a regra de telhado usa um limiar empírico de CHM de 1,5 m;
- a qualidade do resultado depende da segmentação e das amostras de referência.

## Autoria, citação e licença

Desenvolvido por **Luís Henrique de Azevedo Melo**, no contexto de pesquisa
realizada na **Universidade Federal Rural do Rio de Janeiro (UFRRJ)**.

Para citar, informe o autor, o título do repositório, o ano e a URL:

> MELO, Luís Henrique de Azevedo. *Classificação LULC com GEOBIA, LiDAR e
> Random Forest*. 2026. Disponível em:
> https://github.com/lhazmelo/LULC-JB-GEOBIA-RF.

Distribuído sob a [Licença MIT](LICENSE).
