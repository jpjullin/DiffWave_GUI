document.getElementById('preprocessForm').addEventListener('change', updatePreprocess);
document.getElementById('trainForm').addEventListener('change', updateTrain);
document.getElementById('inferenceForm').addEventListener('change', updateInference);
document.getElementById('copyForm').addEventListener('change', updateCopy);
document.getElementById('getModelForm').addEventListener('change', updateGetModel);

function updatePreprocess() {
    const folderToAnalyse = document.getElementById('folderToAnalyse').value;
    const descriptorsFolder = document.getElementById('descriptorsFolder').value;

    const options = [];
    let totalDims = 0;
    document.querySelectorAll('input[name="options"]:checked').forEach(option => {
        options.push(option.value);
        switch (option.value) {
                    case 'mfcc':
                        totalDims += 13;
                        break;
                    case 'chroma':
                        totalDims += 12;
                        break;
                    case 'contrast':
                        totalDims += 6;
                        break;
                    default:
                        totalDims += 1;
                        break;
                }
    });

    let umapOption = '';
    if (document.getElementById('umap').checked) {
        const umapValue = document.getElementById('umapDims').value;
        umapOption = `--umap_dims ${umapValue}`;
        totalDims = umapValue;
    }

    document.getElementById('totalDims').value = totalDims;

    const verbose = document.getElementById('verbose').checked ? '--verbose' : '';

    let command = `python src/preprocess.py ${folderToAnalyse} ${descriptorsFolder} ${options.join(' ')} ${umapOption} ${verbose}`;
    command = command.replace(/\s{2,}/g, ' '); // Remove extra spaces
    document.getElementById('preprocessOutput').value = command.trim();

    document.getElementById('folderToTrain').value = descriptorsFolder;
    document.getElementById('modelFolder').value = 'models/model-' + folderToAnalyse;

    document.getElementById('inferenceModelFolder').value = document.getElementById('modelFolder').value;

    updateTrain();
    updateInference();
}

function updateTrain() {
    const folderToTrain = document.getElementById('folderToTrain').value;
    const modelFolder = document.getElementById('modelFolder').value;

    let command = `python src/__main__.py ${folderToTrain} ${modelFolder}`;
    document.getElementById('trainOutput').value = command.trim();

    let commandBg = `nohup python src/__main__.py ${folderToTrain} ${modelFolder} &`;
    document.getElementById('trainBgOutput').value = commandBg.trim();

    document.getElementById('inferenceModelFolder').value = modelFolder;
    document.getElementById('getModelFolder').value = modelFolder;

    updateInference();
    updateGetModel();
}

function updateInference() {
    const modelFolder = document.getElementById('inferenceModelFolder').value;

    let command = `python src/inference.py ${modelFolder}`;
    document.getElementById('inferenceOutput').value = command.trim();
}

function updateCopy() {
    const filePath = document.getElementById('audioPath').value;
    const fileName = document.getElementById('audioFile').value;

    let fullPath = `"${filePath}\\${fileName}"`;
    let command = `ssh mosaique@pop-os-mosaique.musique.umontreal.ca 'rm -rf /home/mosaique/Desktop/DiffWave_v2/audio/*'; scp ${fullPath} mosaique@pop-os-mosaique.musique.umontreal.ca:/home/mosaique/Desktop/DiffWave_v2/audio`;
    document.getElementById('copyOutput').value = command.trim();
}

function updateGetModel() {
    const modelFolder = document.getElementById('getModelFolder').value;
    const copyPath = document.getElementById('getModelFullpath').value;
    let fullPath = `"${copyPath}"`;

    let command = `scp -r mosaique@pop-os-mosaique.musique.umontreal.ca:/home/mosaique/Desktop/DiffWave_v2/${modelFolder} ${fullPath}`;
    document.getElementById('getModelOutput').value = command.trim();

    document.getElementById('inferenceModelFolder').value = modelFolder;
    updateInference();
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
