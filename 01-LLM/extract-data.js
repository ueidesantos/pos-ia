const XLSX = require('xlsx');
const fs = require('fs');
const path = require('path');

const sourceDir = 'c:\\_dev\\pos-ia\\01-LLM\\Miscellaneous\\Fontes\\Cross-Country Alzheimer Prevalence Ranking';

const files = {
    'DEEPSEEK': path.join(sourceDir, 'Ranking_Alzheimer_Dementia_100_Paises_2026__DEEPSEEK.xlsx'),
    'CHATGPT': path.join(sourceDir, 'Ranking_Alzheimer_Dementia_100_Paises_2026__CHATGPT.xlsx'),
    'GEMINI': path.join(sourceDir, 'Ranking_Alzheimer_Dementia_100_Paises_2026_GEMINI.xlsx')
};

const result = {};

Object.entries(files).forEach(([name, filePath]) => {
    try {
        const workbook = XLSX.readFile(filePath);
        const worksheet = workbook.Sheets[workbook.SheetNames[0]];
        const data = XLSX.utils.sheet_to_json(worksheet);
        result[name] = data;
        console.log(`✓ ${name}: ${data.length} registros carregados`);
    } catch (err) {
        console.error(`✗ Erro ao carregar ${name}:`, err.message);
    }
});

const outputPath = 'c:\\_dev\\pos-ia\\01-LLM\\src\\data\\ias-comparison.json';
fs.writeFileSync(outputPath, JSON.stringify(result, null, 2));
console.log(`\n✓ Dados salvos em: ${outputPath}`);
