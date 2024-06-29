document.getElementById('preprocessForm').addEventListener('change', updatePreprocess);
document.getElementById('trainForm').addEventListener('change', updateTrain);
document.getElementById('inferenceForm').addEventListener('change', updateInference);

function updatePreprocess() {
    const folderToAnalyse = document.getElementById('folderToAnalyse').value;
    const descriptorsFolder = document.getElementById('descriptorsFolder').value;

    const options = [];
    document.querySelectorAll('input[name="options"]:checked').forEach(option => {
        options.push(option.value);
    });

    let umapOption = '';
    if (document.getElementById('umap').checked) {
        const umapValue = document.getElementById('umapDims').value;
        umapOption = `--umap_dims ${umapValue}`;
    }

    const verbose = document.getElementById('verbose').checked ? '--verbose' : '';

    let command = `python src/preprocess.py ${folderToAnalyse} ${descriptorsFolder} ${options.join(' ')} ${umapOption} ${verbose}`;
    command = command.replace(/\s{2,}/g, ' '); // Remove extra spaces
    document.getElementById('preprocessOutput').value = command.trim();

    document.getElementById('folderToTrain').value = descriptorsFolder;
    document.getElementById('modelFolder').value = 'models/model-' + folderToAnalyse + '-new';

    document.getElementById('inferenceModelFolder').value = document.getElementById('modelFolder').value;

    updateTrain();
    updateInference();
}

function updateTrain() {
    const folderToTrain = document.getElementById('folderToTrain').value;
    const modelFolder = document.getElementById('modelFolder').value;

    let command = `python src/__main__.py ${folderToTrain} ${modelFolder}`;
    document.getElementById('trainOutput').value = command.trim();

    document.getElementById('inferenceModelFolder').value = modelFolder;

    updateInference();
}

function updateInference() {
    const modelFolder = document.getElementById('inferenceModelFolder').value;

    let command = `python src/inference.py ${modelFolder}`;
    document.getElementById('inferenceOutput').value = command.trim();
}

function copyCommand(id) {
    const commandOutput = document.getElementById(id);
    commandOutput.select();
    document.execCommand('copy');
    // alert('Command copied to clipboard!');
}

// Initial command update
updatePreprocess();
updateTrain();
updateInference();
