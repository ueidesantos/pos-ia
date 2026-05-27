import { HistoricalDatabaseService } from './Services/HistoricalDatabaseService.js';
const WEIGHTS = 
{
    Idade: 0.3,
    AtividadeFisica: 0.2,
    QualidadeDieta: 0.2,
    Sono: 0.2,
    Depressao: 0.1,
    Genero: 0.1,
    Diabetes: 0.1,
    QueixasMemoria: 0.3,
    Confusao: 0.3,
    Hipertenso: 0.1,
};

// Vetor de pesos na mesma ordem de FEATURE_COLUMNS do HistoricalDatabaseService
// ['Idade','Genero','Diabetes','Depressao','QueixasMemoria','Confusao','Hipertenso','Sono','QualidadeDieta','AtividadeFisica']
const WEIGHT_VECTOR = [
    WEIGHTS.Idade, WEIGHTS.Genero, WEIGHTS.Diabetes, WEIGHTS.Depressao,
    WEIGHTS.QueixasMemoria, WEIGHTS.Confusao, WEIGHTS.Hipertenso,
    WEIGHTS.Sono, WEIGHTS.QualidadeDieta, WEIGHTS.AtividadeFisica
];

const trainingHistory = { loss: [], acc: [] };
let showTrainingPanel = false;
const TRAINING_EPOCHS = 150;

function renderTrainingCharts() {
    if (!trainingHistory.acc.length && !trainingHistory.loss.length) return;

    tfvis.render.linechart(
        { name: 'Precisão do Modelo', tab: 'Treinamento' },
        { values: trainingHistory.acc.map((v, i) => ({ x: i, y: v })), series: ['precisão'] },
        { xLabel: 'Época (Ciclos de Treinamento)', yLabel: 'Precisão (%)', yAxisDomain: [0, 1] }
    );

    tfvis.render.linechart(
        { name: 'Erro de Treinamento', tab: 'Treinamento' },
        { values: trainingHistory.loss.map((v, i) => ({ x: i, y: v })), series: ['erros'] },
        { xLabel: 'Época (Ciclos de Treinamento)', yLabel: 'Valor do Erro', yAxisDomain: [0, 1] }
    );
}

function updateTrainingPanelToggleButton(button) {
    if (!button) return;

    button.textContent = showTrainingPanel
        ? 'Ocultar painel de treinamento'
        : 'Mostrar painel de treinamento';
    button.classList.toggle('btn-outline-dark', !showTrainingPanel);
    button.classList.toggle('btn-dark', showTrainingPanel);
}

function setTrainingPanelVisibility(visible) {
    showTrainingPanel = visible;

    if (showTrainingPanel) {
        renderTrainingCharts();
        tfvis.visor().open();
    } else {
        tfvis.visor().close();
    }

    updateTrainingPanelToggleButton(document.getElementById('trainingPanelToggle'));
}

function setFormEnabled(enabled) {
    const form = document.getElementById('healthForm');
    if (!form) return;

    const controls = form.querySelectorAll('input, select, button, textarea');
    controls.forEach((element) => {
        element.disabled = !enabled;
    });
}

function getTrainingMessage(progressPercent) {
    if (progressPercent < 35) {
        return 'Estamos organizando os dados para o modelo aprender.';
    }
    if (progressPercent < 80) {
        return 'O modelo está treinando e melhorando a precisão das classificações.';
    }
    return 'Quase pronto! Estamos finalizando os últimos ajustes do treinamento.';
}

function updateTrainingOverlay(progressPercent) {
    const overlay = document.getElementById('trainingOverlay');
    const progressText = document.getElementById('trainingProgressText');
    const statusMessage = document.getElementById('trainingStatusMessage');
    if (!overlay) return;

    overlay.classList.add('is-visible');
    if (progressText) {
        const boundedPercent = Math.max(0, Math.min(100, progressPercent));
        progressText.textContent = `${boundedPercent}%`;
    }
    if (statusMessage) {
        statusMessage.textContent = getTrainingMessage(progressPercent);
    }
    setFormEnabled(false);
}

function hideTrainingOverlay() {
    const overlay = document.getElementById('trainingOverlay');
    if (overlay) {
        overlay.classList.remove('is-visible');
    }
    setFormEnabled(true);
}

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
    updateTrainingOverlay(0);

    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [inputXs.shape[1]], units: 150, activation: 'relu' }));

    // Saída: 4 neurônios para cada classe
    model.add(tf.layers.dense({ units: 4, activation: 'softmax' }));
    model.compile({
        optimizer: tf.train.adam(),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    try {
        await model.fit(inputXs, outputYs, 
            {
                verbose: 1,
                shuffle: true,
                epochs: TRAINING_EPOCHS,
                batchSize: 32,
                validationSplit: 0.2,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        console.log(`Epoch ${epoch + 1}: loss = ${logs.loss}, accuracy = ${logs.acc}`);
                        trainingHistory.loss.push(logs.loss);
                        trainingHistory.acc.push(logs.acc);

                        const percent = Math.round(((epoch + 1) / TRAINING_EPOCHS) * 100);
                        updateTrainingOverlay(percent);

                        if (showTrainingPanel) {
                            renderTrainingCharts();
                        }
                    }
                }
        });
        updateTrainingOverlay(100);
        console.log('Treinamento concluído.');
        return model;
    } finally {
        hideTrainingOverlay();
    }
}

const historicalDatabaseService = new HistoricalDatabaseService();
const result = await historicalDatabaseService.load();

console.log(`Base histórica carregada. Total de registros: ${result.size}`);
console.log('Primeiro registro:', result.records[0]);
console.log('Result:', result);

const weightTensor = tf.tensor1d(WEIGHT_VECTOR);
const weightedFeatures = result.featureTensor.mul(weightTensor);
weightTensor.dispose();

const model = await trainModel(weightedFeatures, tf.oneHot(result.labelTensor, 4));

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

setTrainingPanelVisibility(true);

const form = document.getElementById('healthForm');
const resultBox = document.getElementById('resultBox');
const resultadoTitulo = document.getElementById('resultadoTitulo');
const resultadoTexto = document.getElementById('resultadoTexto');
const fatoresLista = document.getElementById('fatoresLista');
const trainingPanelToggle = document.getElementById('trainingPanelToggle');

if (trainingPanelToggle) {
    trainingPanelToggle.addEventListener('click', function () {
        setTrainingPanelVisibility(!showTrainingPanel);
    });
}

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

