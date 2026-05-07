const tf = globalThis.tf;

if (!tf) {
    throw new Error('TensorFlow.js não encontrado. Carregue o script do TFJS antes dos módulos da aplicação.');
}

const CSV_PATH = '/Data/Dataset_Alzheimer.csv';

const FEATURE_COLUMNS = [
    'Idade', 'Genero', 'Diabetes', 'Depressao',
    'QueixasMemoria', 'Confusao', 'Hipertenso',
    'Sono', 'QualidadeDieta', 'AtividadeFisica'
];

const LABEL_MAP = {
    'Baixo risco': 0,
    'Atencao moderada': 1,
    'Recomendavel avaliacao medica': 2,
    'Procura medica prioritaria': 3
};

const LABEL_NAMES = Object.keys(LABEL_MAP);

export class HistoricalDatabaseService {
    constructor(csvPath = CSV_PATH) {
        this.csvPath = csvPath;
        this.database = null;
        this._loadPromise = null;
    }

    /**
     * Carrega a base histórica na primeira chamada e retorna os dados normalizados.
     * Chamadas subsequentes retornam o cache sem novo fetch.
     * @returns {Promise<{ records: object[], featureTensor: tf.Tensor2D, labelTensor: tf.Tensor1D, featureColumns: string[], labelNames: string[], size: number }>}
     */
    async load() {
        console.log(`HistoricalDatabaseService.load() - iniciando...`);
        if (this.database) return this.database;
        if (this._loadPromise) return this._loadPromise;

        this._loadPromise = this._fetchAndParse()
            .then(data => {
                this.database = data;
                return data;
            });

        return this._loadPromise;
    }

    async _fetchAndParse() {
        const response = await fetch(this.csvPath);
        if (!response.ok) {
            throw new Error(`Falha ao carregar o dataset: ${response.status} ${response.statusText}`);
        }

        const text = await response.text();
        const lines = text.trim().split(/\r?\n/);

        // Remove BOM se presente
        const rawHeader = lines[0].replace(/^\uFEFF/, '');
        const headers = rawHeader.split(',').map(h => h.trim());

        const classColIndex = headers.findIndex(h => h.toLowerCase().includes('classificacao'));
        const nameColIndex = headers.findIndex(h => h.toLowerCase() === 'nome');
        const featureIndices = FEATURE_COLUMNS.map(col => headers.findIndex(h => h.toLowerCase() === col.toLowerCase()));

        console.log('[CSV] headers:', headers);
        console.log('[CSV] classColIndex:', classColIndex, '| nameColIndex:', nameColIndex);
        console.log('[CSV] featureIndices:', featureIndices);

        const records = [];
        const featureData = [];
        const labelData = [];

        for (let i = 1; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line) continue;

            const cols = line.split(',');
            const label = cols[classColIndex]?.trim();
            const labelIndex = LABEL_MAP[label];

            if (i === 1) console.log('[CSV] primeiro label lido:', JSON.stringify(label), '| labelIndex:', labelIndex);

            if (labelIndex === undefined) continue;

            const features = featureIndices.map(idx => parseFloat(cols[idx]));

            if (features.some(isNaN)) continue;

            featureData.push(features);
            labelData.push(labelIndex);

            records.push({
                nome: cols[nameColIndex]?.trim() ?? '',
                classificacao: label,
                labelIndex,
                features: Object.fromEntries(
                    FEATURE_COLUMNS.map((col, j) => [col, features[j]])
                )
            });
        }

        const featureTensor = tf.tensor2d(featureData, [featureData.length, FEATURE_COLUMNS.length]);
        const labelTensor = tf.tensor1d(labelData, 'int32');
        
        console.log(`${labelTensor.size} labels convertidos para tensor.`);
        console.log(`[HistoricalDatabaseService] Base histórica processada. Total de registros válidos: ${records.length}`);

        return {
            records,
            featureTensor,
            labelTensor,
            featureColumns: FEATURE_COLUMNS,
            labelNames: LABEL_NAMES,
            labelMap: LABEL_MAP,
            size: records.length
        };
    }
}
