# 01-LLM - TensorFlow.js aplicado a classificacao de atencao em saude

Este diretorio contem uma aplicacao web academica que usa **TensorFlow.js** no navegador para treinar um modelo simples de classificacao relacionado a fatores de atencao em saude para Alzheimer/demencia.

O objetivo principal do projeto e demonstrar o ciclo completo de machine learning no front-end:

1. carregar uma base historica em CSV;
2. transformar os dados em tensores;
3. treinar uma rede neural com TensorFlow.js;
4. visualizar o treinamento com TensorFlow Vis;
5. usar o modelo treinado para classificar novos dados informados em formulario.

> Importante: a aplicacao nao realiza diagnostico medico. A classificacao gerada e apenas uma estimativa educacional baseada nos dados informados.

## Arquivos principais

| Arquivo | Papel no TensorFlow.js |
| --- | --- |
| `src/index.html` | Carrega TensorFlow.js, TensorFlow Vis e o modulo principal da aplicacao. |
| `src/index.js` | Define pesos, monta o modelo, treina a rede neural e executa predicoes. |
| `src/Services/HistoricalDatabaseService.js` | Le o CSV, normaliza a base historica em arrays numericos e cria os tensores de entrada e saida. |
| `src/Data/Dataset_Alzheimer.csv` | Base usada para treinamento do modelo. |
| `src/style.css` | Estilos da interface, resultado, console e overlay de treinamento. |

Tambem existe a pagina `src/comparison.html`, voltada a comparacao de rankings gerados por diferentes IAs. Ela e complementar ao projeto, mas nao faz parte do fluxo TensorFlow.js.

## Dependencias de ML

As dependencias sao carregadas diretamente por CDN em `src/index.html`:

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@1.5.1/dist/tfjs-vis.umd.min.js"></script>
```

Nao ha etapa de build. O TensorFlow.js roda inteiramente no navegador.

## Fluxo TensorFlow.js

### 1. Carregamento da base

O servico `HistoricalDatabaseService` busca o arquivo:

```js
const CSV_PATH = './Data/Dataset_Alzheimer.csv';
```

Depois ele separa as colunas do CSV em:

- features numericas, usadas como entrada do modelo;
- label de classificacao, convertida para indice numerico.

As colunas usadas como entrada sao:

```js
[
  'Idade',
  'Genero',
  'Diabetes',
  'Depressao',
  'QueixasMemoria',
  'Confusao',
  'Hipertenso',
  'Sono',
  'QualidadeDieta',
  'AtividadeFisica'
]
```

### 2. Criacao dos tensores

A base historica e convertida em dois tensores principais:

```js
const featureTensor = tf.tensor2d(featureData, [featureData.length, FEATURE_COLUMNS.length]);
const labelTensor = tf.tensor1d(labelData, 'int32');
```

O `featureTensor` representa a matriz de entrada do modelo. Cada linha e um registro historico, e cada coluna e uma feature.

O `labelTensor` contem a classe correta de cada registro:

```js
const LABEL_MAP = {
  'Baixo risco': 0,
  'Atencao moderada': 1,
  'Recomendavel avaliacao medica': 2,
  'Procura medica prioritaria': 3
};
```

Antes do treino, os labels sao convertidos para one-hot encoding:

```js
tf.oneHot(result.labelTensor, 4)
```

### 3. Aplicacao de pesos nas features

O projeto aplica pesos manuais para destacar fatores mais relevantes antes de treinar o modelo:

```js
const WEIGHTS = {
  Idade: 0.3,
  AtividadeFisica: 0.2,
  QualidadeDieta: 0.2,
  Sono: 0.2,
  Depressao: 0.1,
  Genero: 0.1,
  Diabetes: 0.1,
  QueixasMemoria: 0.3,
  Confusao: 0.3,
  Hipertenso: 0.1
};
```

Esses pesos sao transformados em tensor e multiplicados pelas features:

```js
const weightTensor = tf.tensor1d(WEIGHT_VECTOR);
const weightedFeatures = result.featureTensor.mul(weightTensor);
weightTensor.dispose();
```

O ponto de atencao aqui e a ordem: `WEIGHT_VECTOR` precisa seguir exatamente a mesma ordem de `FEATURE_COLUMNS`.

### 4. Arquitetura do modelo

O modelo e criado com `tf.sequential()`:

```js
const model = tf.sequential();
model.add(tf.layers.dense({
  inputShape: [inputXs.shape[1]],
  units: 150,
  activation: 'relu'
}));
model.add(tf.layers.dense({
  units: 4,
  activation: 'softmax'
}));
```

A primeira camada densa recebe as 10 features ponderadas. A camada de saida tem 4 neuronios, um para cada classe possivel.

A ativacao `softmax` transforma a saida em uma distribuicao de probabilidades.

### 5. Compilacao e treinamento

O modelo usa:

- otimizador: `adam`;
- funcao de perda: `categoricalCrossentropy`;
- metrica: `accuracy`;
- epocas: `150`;
- tamanho do lote: `32`;
- validacao: `20%` da base.

Trecho principal:

```js
model.compile({
  optimizer: tf.train.adam(),
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy']
});

await model.fit(inputXs, outputYs, {
  shuffle: true,
  epochs: 150,
  batchSize: 32,
  validationSplit: 0.2
});
```

Durante cada epoca, a aplicacao registra `loss` e `accuracy` para mostrar a evolucao do treinamento.

### 6. Visualizacao com TensorFlow Vis

O projeto usa `tfjs-vis` para renderizar graficos de treinamento:

- precisao do modelo;
- erro de treinamento.

Os dados sao armazenados em:

```js
const trainingHistory = { loss: [], acc: [] };
```

E renderizados com:

```js
tfvis.render.linechart(...)
```

O painel pode ser exibido ou ocultado pelo botao da interface.

### 7. Predicao no formulario

Depois do treinamento, o usuario preenche o formulario. Os dados sao convertidos para um tensor 2D com uma unica linha:

```js
const input = tf.tensor2d([[
  idadeNormalizada * WEIGHTS.Idade,
  formData.genero * WEIGHTS.Genero,
  formData.diabetes * WEIGHTS.Diabetes,
  formData.depressao * WEIGHTS.Depressao,
  formData.queixasMemoria * WEIGHTS.QueixasMemoria,
  formData.confusao * WEIGHTS.Confusao,
  formData.hipertenso * WEIGHTS.Hipertenso,
  formData.sono * WEIGHTS.Sono,
  formData.qualidadeDieta * WEIGHTS.QualidadeDieta,
  formData.atividadeFisica * WEIGHTS.AtividadeFisica
]]);
```

Em seguida:

```js
const prediction = model.predict(input);
const probs = prediction.dataSync();
const labelIndex = probs.indexOf(Math.max(...probs));
```

A classe com maior probabilidade e exibida na tela junto com as probabilidades de todas as classes.

## Como executar

Como a aplicacao usa `fetch()` para carregar CSV e JSON, o ideal e servir o diretorio `src` por HTTP.

Uma opcao simples:

```powershell
cd C:\_dev\pos-ia\01-LLM\src
python -m http.server 5500
```

Depois abra:

```text
http://localhost:5500
```

Tambem e possivel usar uma extensao como Live Server no VS Code apontando para `src/index.html`.

## Pontos importantes para evolucao do modelo

- Se uma feature for adicionada ou removida, atualize `FEATURE_COLUMNS`, `WEIGHTS`, `WEIGHT_VECTOR` e o formulario.
- Se uma classe for adicionada ou removida, atualize `LABEL_MAP`, `LABEL_NAMES`, a quantidade de unidades da camada `softmax` e o valor usado em `tf.oneHot`.
- O modelo e treinado a cada carregamento da pagina; atualmente ele nao e salvo em `localStorage`, IndexedDB ou arquivo.
- O projeto usa uma validacao simples com `validationSplit`; para avaliacao mais robusta, separe uma base de teste independente.
- As probabilidades retornadas pelo `softmax` dependem da qualidade, tamanho e distribuicao do dataset.

## Resumo tecnico

```text
CSV historico
  -> parse no HistoricalDatabaseService
  -> featureTensor + labelTensor
  -> pesos manuais nas features
  -> tf.sequential()
  -> dense relu com 150 unidades
  -> dense softmax com 4 classes
  -> treino com model.fit()
  -> graficos com tfjs-vis
  -> predicao com model.predict()
```

