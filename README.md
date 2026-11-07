# 🌿 Detecção de Doenças em Folhas de Plantas

Diagnóstico automático de doenças foliares com Aprendizagem Profunda.

Projeto da disciplina **Tendências em Inteligência Computacional**.

## 📌 Visão geral

Este projeto propõe um sistema de classificação de imagens de folhas capaz de identificar se a planta está **saudável** ou afetada por uma **doença específica** (manchas bacterianas, requeima, pinta-preta, mosaico, entre outras), utilizando Redes Neurais Convolucionais (CNNs).

Além de construir o classificador, o trabalho investiga **em que medida a geração e o uso de dados sintéticos ajudam a mitigar o desbalanceamento de classes** do dataset de referência e qual o impacto disso no desempenho do modelo.

## 🎯 Problema

- Doenças foliares reduzem drasticamente a produtividade agrícola em todo o mundo.
- O diagnóstico correto depende de agrônomos, profissionais escassos em regiões rurais e em pequenas propriedades.
- Erros de diagnóstico levam à aplicação incorreta de defensivos, gerando prejuízo econômico e ambiental.
- Aplicativos de visão computacional no celular podem levar diagnóstico especializado a qualquer produtor, a qualquer hora.

### Por que é um problema de IA?

- É uma tarefa **perceptiva**: reconhecer manchas, texturas e descolorações sutis sob variações de luz, ângulo e estágio da doença.
- Regras fixas (limiares de cor ou forma) não generalizam entre espécies e condições de captura.
- CNNs aprendem automaticamente a hierarquia de padrões visuais relevantes, sem engenharia manual de atributos.
- Por isso, Aprendizagem Profunda tende a superar o ML clássico (ex.: SVM sobre features de cor/textura) nesse tipo de tarefa.

## 💡 Motivação

| Eixo | Impacto |
|---|---|
| Agricultura de precisão | Diagnóstico rápido e georreferenciado apoia decisões de manejo em tempo hábil. |
| Apoio ao pequeno produtor | Leva diagnóstico especializado a quem não tem acesso fácil a agrônomos. |
| Uso responsável de defensivos | Tratamento direcionado à doença identificada reduz aplicação desnecessária de agrotóxicos. |
| Segurança alimentar | Menos perdas de safra contribuem para o abastecimento e para a renda do produtor. |

## 🧠 Formalização do aprendizado

| Componente | Definição |
|---|---|
| **Tarefa (T)** | Classificar uma imagem de folha em uma categoria: saudável ou um tipo específico de doença (classificação multiclasse). |
| **Experiência (E)** | Aprendizado supervisionado a partir de milhares de imagens de folhas rotuladas por espécie e condição fitossanitária. |
| **Medida de desempenho (P)** | Acurácia, F1-score por classe e matriz de confusão, essenciais diante do desbalanceamento entre classes. |

## 📊 Dados

**Dataset de referência:** subconjunto do [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) com as culturas de pimentão, batata e tomate.

| Imagens rotuladas | Espécies de cultura | Classes (saudável/doença) |
|---|---|---|
| 20.638 | 3 | 15 |

Distribuição por classe:

| Classe | Imagens |
|---|---:|
| Tomato\_\_Tomato\_YellowLeaf\_\_Curl\_Virus | 3.209 |
| Tomato\_Bacterial\_spot | 2.127 |
| Tomato\_Late\_blight | 1.909 |
| Tomato\_Septoria\_leaf\_spot | 1.771 |
| Tomato\_Spider\_mites\_Two\_spotted\_spider\_mite | 1.676 |
| Tomato\_healthy | 1.591 |
| Pepper\_\_bell\_\_\_healthy | 1.478 |
| Tomato\_\_Target\_Spot | 1.404 |
| Potato\_\_\_Early\_blight | 1.000 |
| Potato\_\_\_Late\_blight | 1.000 |
| Tomato\_Early\_blight | 1.000 |
| Pepper\_\_bell\_\_\_Bacterial\_spot | 997 |
| Tomato\_Leaf\_Mold | 952 |
| Tomato\_\_Tomato\_mosaic\_virus | 373 |
| Potato\_\_\_healthy | 152 |

**Pontos de atenção:**

- As imagens foram capturadas em fundo controlado (estúdio), o que gera um *domain gap* em relação a fotos reais de campo.
- As classes são **fortemente desbalanceadas**: a maior classe (`Tomato__Tomato_YellowLeaf__Curl_Virus`, 3.209) tem cerca de **21×** mais imagens que a menor (`Potato___healthy`, 152), e o tomate concentra ~78% do total.
- **Split:** treino/validação/teste em **70/10/20**, aleatório e estratificado por classe (`random_state=42`), gerado por `scripts/split.py` e salvo em `config/splits/`. Validação e teste contêm apenas imagens reais; dados sintéticos entram somente no treino.
- **Limitação:** o PlantVillage não traz um identificador de folha, e o dataset contém várias fotos da mesma folha. Por isso o split não é agrupado por folha e pode haver **vazamento** entre os conjuntos, o que tende a inflar as métricas. Os resultados devem ser lidos com essa ressalva.
- Recomenda-se complementar com imagens reais de campo (ex.: [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset)) para avaliar a generalização.

## ⚠️ Desafio central: desbalanceamento de classes

O desbalanceamento pode enviesar o treinamento em favor das classes mais representadas e comprometer a generalização para as demais. Este trabalho investiga:

> *Em que medida a geração e o uso de dados sintéticos contribuem para mitigar esse desbalanceamento, e qual o impacto dessa estratégia sobre o desempenho do modelo treinado?*

## 🛠️ Solução proposta

- **Transfer Learning:** CNNs pré-treinadas (ResNet, EfficientNet, VGG) ajustadas por *fine-tuning* para as classes de doenças.
- **CNNs leves para mobile:** arquiteturas como MobileNet viabilizam diagnóstico offline direto no smartphone do produtor.
- **Data Augmentation:** rotação, espelhamento e variação de brilho/contraste simulam condições reais de campo e reduzem overfitting.
- **Dados sintéticos:** geração de imagens para balancear as classes sub-representadas, com qualidade medida por FID.
- **Interpretabilidade (Grad-CAM):** visualiza as regiões da folha que mais influenciam a decisão, aumentando a confiança no diagnóstico.

## 🔬 Pipeline previsto

```
Dados (PlantVillage + PlantDoc)
        │
        ▼
Pré-processamento e split 70/10/20 (estratificado)
        │
        ▼
Balanceamento (data augmentation + dados sintéticos, só no treino)
        │
        ▼
Treinamento (Transfer Learning / fine-tuning)
        │
        ▼
Avaliação (acurácia, F1 por classe, matriz de confusão)
        │
        ▼
Interpretabilidade (Grad-CAM) e exportação para mobile
```

## 📁 Estrutura do repositório

```
.
├── config/splits/     # Splits treino/validação/teste (caminhos relativos a data/raw/PlantVillage)
├── data/              # Datasets (não versionados)
│   ├── raw/           #   PlantVillage original
│   ├── processed/     #   Dados pré-processados
│   └── synthetic/     #   Imagens sintéticas geradas
├── notebooks/         # Exploração e experimentos
├── scripts/           # Scripts utilitários (ex.: geração dos splits)
├── src/               # Código-fonte (dados, modelo, treino, avaliação, geração)
├── models/            # Pesos e modelos exportados (não versionados)
├── results/           # Métricas, gráficos e mapas Grad-CAM
├── requirements.txt
└── README.md
```

## 🚀 Como executar

```bash
# 1. Clone o repositório
git clone https://github.com/silvanio45/Projeto_TIC.git
cd Projeto_TIC

# 2. Crie um ambiente virtual e instale as dependências
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Baixe o dataset e coloque as pastas das 15 classes em data/raw/PlantVillage/

# 4. (Opcional) Regere os splits — os de config/splits/ já estão versionados
python scripts/split.py

# 5. (Opcional) Teste os módulos prontos
python src/data.py    # carrega um batch: torch.Size([32, 3, 224, 224])
python src/model.py   # saída do modelo: torch.Size([4, 15])

# 6. Treine o baseline (salva o melhor modelo em models/baseline_resnet50.pth)
python src/train.py
```

> ℹ️ Ajuste os comandos conforme a estrutura final do projeto.

## 📈 Resultados

*Em desenvolvimento.* Esta seção reunirá as métricas (acurácia, F1 por classe, matriz de confusão), a comparação com vs. sem dados sintéticos e exemplos de mapas Grad-CAM.

| Experimento | Acurácia | F1 (macro) |
|---|---|---|
| Baseline (sem balanceamento) | – | – |
| Com data augmentation | – | – |
| Com dados sintéticos | – | – |

## ✅ Roadmap

### Concluído

- [x] Definição do problema e formalização (T, E, P)
- [x] Escolha do dataset e análise de limitações
- [x] Splits treino/validação/teste (70/10/20, estratificado) com caminhos relativos — `scripts/split.py`, `config/splits/`
- [x] Carregamento dos dados — `src/data.py`
  - `carrega_split()`: lê os splits em JSON
  - `FolhasDataset`: abre as imagens sob demanda e aplica os transforms
  - `get_dataloaders()`: devolve os loaders de treino (com augmentation e shuffle), validação e teste
- [x] Modelo — `src/model.py`: `cria_modelo()` carrega a ResNet-50 pré-treinada no ImageNet e troca a camada `fc` por uma de 15 classes
- [x] Treinamento do baseline — `src/train.py`
  - `treinar_epoca()`: uma época de treino com `CrossEntropyLoss` e `Adam` (`lr=1e-4`), devolve a loss média
  - `avaliar()`: loss, acurácia e F1 macro na validação
  - `main()`: treina por `EPOCAS` épocas e salva o melhor modelo (por F1 macro na validação) em `models/baseline_resnet50.pth`

### Em andamento

- [ ] Avaliação no teste — `src/evaluate.py`: acurácia, F1 por classe, matriz de confusão e saída em `results/`

### A fazer

- [ ] Análise exploratória — `notebooks/analise.ipynb`: distribuição das classes e exemplos de imagens
- [ ] Rebalanceamento clássico (`class_weight` / `WeightedRandomSampler`) como comparação
- [ ] Geração de dados sintéticos — `src/generate.py`, com qualidade medida por FID
- [ ] Avaliação comparativa entre os experimentos (com mais de uma seed)
- [ ] Interpretabilidade com Grad-CAM
- [ ] Exportação para modelo leve (MobileNet)

## 👥 Autores

- *Nome(s) da equipe*

## 📚 Referências

- [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
- [PlantDoc Dataset](https://github.com/pratikkayal/PlantDoc-Dataset)
- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization* (2017)
