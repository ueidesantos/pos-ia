import { HistoricalDatabaseService } from './Services/HistoricalDatabaseService.js';

(function () {
  const panel = () => document.getElementById('consoleOutput');
  const levels = {
    log:   { label: 'LOG',   cls: 'console-log' },
    info:  { label: 'INFO',  cls: 'console-info' },
    warn:  { label: 'WARN',  cls: 'console-warn' },
    error: { label: 'ERR',   cls: 'console-error' },
  };
  Object.entries(levels).forEach(([method, { label, cls }]) => {
    const orig = console[method].bind(console);
    console[method] = function (...args) {
      orig(...args);
      const p = panel();
      if (!p) return;
      const line = document.createElement('div');
      line.className = 'console-line ' + cls;
      const time = new Date().toLocaleTimeString('pt-BR', { hour12: false });
      line.textContent = `[${time}] ${label}: ` + args.map(a => {
        try { return typeof a === 'object' ? JSON.stringify(a) : String(a); }
        catch { return String(a); }
      }).join(' ');
      p.appendChild(line);
      p.scrollTop = p.scrollHeight;
    };
  });
})();

async function trainModel(inputXs, outputYs) {
    console.log('Iniciando treinamento do modelo...');
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [inputXs.shape[1]], units: 80, activation: 'relu' }));

    //Saída: 4 neuronios para cada classe
    model.add(tf.layers.dense({ units: 4, activation: 'softmax' }));
    model.compile({
        optimizer: tf.train.adam(),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    const surface = { name: 'Treinamento', tab: 'Treinamento' };
    const history = { loss: [], acc: [] };

    await model.fit(inputXs, outputYs, 
        {
            verbose: 1,
            shuffle: true,
            epochs: 100,
            batchSize: 32,
            validationSplit: 0.2,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch + 1}: loss = ${logs.loss}, accuracy = ${logs.acc}`);
                    history.loss.push(logs.loss);
                    history.acc.push(logs.acc);
                    tfvis.render.linechart(
                        { name: 'Precisão do Modelo', tab: 'Treinamento' },
                        { values: history.acc.map((v, i) => ({ x: i, y: v })), series: ['precisão'] },
                        { xLabel: 'Época (Ciclos de Treinamento)', yLabel: 'Precisão (%)', yAxisDomain: [0, 1] }
                    );
                    tfvis.render.linechart(
                        { name: 'Erro de Treinamento', tab: 'Treinamento' },
                        { values: history.loss.map((v, i) => ({ x: i, y: v })), series: ['erros'] },
                        { xLabel: 'Época (Ciclos de Treinamento)', yLabel: 'Valor do Erro', yAxisDomain: [0, 1] }
                    );
                }
            }
    });
    console.log('Treinamento concluído.');
    return model;
}

const historicalDatabaseService = new HistoricalDatabaseService();
const result = await historicalDatabaseService.load();

console.log(`Base histórica carregada. Total de registros: ${result.size}`);
console.log('Primeiro registro:', result.records[0]);
console.log('Result:', result);

const model = await trainModel(result.featureTensor, tf.oneHot(result.labelTensor, 4));

const LABEL_NAMES = result.labelNames;

const CSS_CLASSES = [
    'baixo-risco',
    'atencao-moderada',
    'avaliacao-medica',
    'prioritaria'
];

function predict(model, formData) {
    console.log('Realizando previsão com os dados do formulário:', formData);
    const idadeNormalizada = Math.min(formData.idade / 120, 1);
    const input = tf.tensor2d([[
        idadeNormalizada,
        formData.genero,
        formData.diabetes,
        formData.depressao,
        formData.queixasMemoria,
        formData.confusao,
        formData.hipertenso,
        formData.sono,
        formData.qualidadeDieta,
        formData.atividadeFisica
    ]]);

    const prediction = model.predict(input);
    const probs = prediction.dataSync();
    const labelIndex = probs.indexOf(Math.max(...probs));

    input.dispose();
    prediction.dispose();

    return {
        classificacao: LABEL_NAMES[labelIndex],
        cssClass: CSS_CLASSES[labelIndex],
        probabilidades: LABEL_NAMES.map((nome, i) => ({
            nome,
            prob: (probs[i] * 100).toFixed(1) + '%'
        }))
    };
}

const form = document.getElementById('healthForm');
const resultBox = document.getElementById('resultBox');
const resultadoTitulo = document.getElementById('resultadoTitulo');
const resultadoTexto = document.getElementById('resultadoTexto');
const fatoresLista = document.getElementById('fatoresLista');

form.addEventListener('submit', function (e) {
    e.preventDefault();

    const formData = {
        idade:          Number(document.getElementById('idade').value || 0),
        genero:         Number(document.getElementById('genero').value || 0),
        diabetes:       Number(document.getElementById('diabetes').value || 0),
        depressao:      Number(document.getElementById('depressao').value || 0),
        queixasMemoria: Number(document.getElementById('queixasMemoria').value || 0),
        confusao:       Number(document.getElementById('confusao').value || 0),
        hipertenso:     Number(document.getElementById('hipertenso').value || 0),
        sono:           Number(document.getElementById('sono').value || 0),
        qualidadeDieta: Number(document.getElementById('qualidadeDieta').value || 0),
        atividadeFisica:Number(document.getElementById('atividadeFisica').value || 0)
    };

    const { classificacao, cssClass, probabilidades } = predict(model, formData);

    resultBox.className = `result-box mt-4 ${cssClass}`;
    resultBox.style.display = 'block';
    resultadoTitulo.textContent = `Classificação: ${classificacao}`;
    resultadoTexto.textContent = 'Probabilidades por classe:';

    fatoresLista.innerHTML = '';
    probabilidades.forEach(({ nome, prob }) => {
        const li = document.createElement('li');
        li.textContent = `${nome}: ${prob}`;
        fatoresLista.appendChild(li);
    });
});

form.addEventListener('reset', function () {
    resultBox.style.display = 'none';
    fatoresLista.innerHTML = '';
});

