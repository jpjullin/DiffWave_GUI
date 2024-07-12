let audioBuffer = null;
let audioFilename = null;

let allDescr = null;
let folderName = null;

let model = null;

// --------------------------------- DROP FILE --------------------------------- //

document.addEventListener('DOMContentLoaded', () => {
    const dropArea = document.getElementById('dropArea');
    const fileInput = document.getElementById('audioFileInput');
    let fileDialogOpen = false;

    if (dropArea && fileInput) {
        dropArea.addEventListener('dragover', handleDragOver, false);
        dropArea.addEventListener('dragleave', handleDragLeave, false);
        dropArea.addEventListener('drop', handleFileDrop, false);
        dropArea.addEventListener('click', () => {
            if (!fileDialogOpen) {
                fileDialogOpen = true;
                fileInput.click();
            }
        }, false);
        fileInput.addEventListener('click', (event) => {
            event.stopPropagation();
        }, false);
        fileInput.addEventListener('change', (event) => {
            handleFileSelect(event);
            fileDialogOpen = false;
        }, false);
    }
});

function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    event.dataTransfer.dropEffect = 'copy';
    event.currentTarget.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.classList.remove('dragover');
}

function handleFileDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.classList.remove('dragover');
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function handleFileSelect(event) {
    const files = event.target.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function processFile(file) {
    if (file.type !== 'audio/wav') {
        alert('WAV only!');
        return;
    }
    audioFilename = file.name.split('.').slice(0, -1).join('.');
    const reader = new FileReader();
    reader.onload = function(event) {
        const arrayBuffer = event.target.result;
        loadAudioBuffer(arrayBuffer, file.name);
    };
    reader.readAsArrayBuffer(file);
}

async function loadAudioBuffer(arrayBuffer, fileName) {
    const audioContext = new(window.AudioContext || window.webkitAudioContext)();
    audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    document.getElementById('uploadStatus').innerText = `${fileName} uploaded successfully!`;
}

// --------------------------------- DESCRIPTORS --------------------------------- //

async function preprocess() {
    const sampleRate = parseInt(document.getElementById('sampleRate').value, 10);
    const nFft = parseInt(document.getElementById('nFft').value, 10);
    const hopLength = parseInt(document.getElementById('hopLength').value, 10);
    const winLength = parseInt(document.getElementById('winLength').value, 10);
    const umapDims = parseInt(document.getElementById('umapDims').value, 10);
    const windowType = 'hanning';

    if (!audioBuffer) {
        alert('Upload an audio file first!');
        return;
    }

    const monoBuffer = audioBuffer.numberOfChannels > 1 ? convertToMono(audioBuffer) : audioBuffer;
    const resampledBuffer = await resampleAudio(monoBuffer, sampleRate);
    const resampledAudio = resampledBuffer.getChannelData(0);

    await showProgressBar();
    const totalSteps = Math.ceil((resampledAudio.length - winLength) / hopLength);
    let currentStep = 0;

    const selectedOptions = Array.from(document.querySelectorAll('input[name="options"]:checked')).map(option => option.value);
    const descriptors = Object.fromEntries(selectedOptions.map(option => [option, []]));

    // Meyda needs the buffer size to be a power of 2
    const nextPowerOf2 = Math.pow(2, Math.round(Math.log2(winLength)));
    const paddedSegment = new Float32Array(nextPowerOf2);

    function processSegment(i) {
        if (i < resampledAudio.length - winLength) {
            let segment = resampledAudio.subarray(i, i + winLength);
            paddedSegment.fill(0);
            paddedSegment.set(segment);

            segment = Meyda.windowing(paddedSegment, windowType);

            selectedOptions.forEach(option => {
                let descriptor;
                switch (option) {
                    case 'loudness':
                        descriptor = Meyda.extract('loudness', segment).total;
                        break;
                    case 'pitch':
                        const spectralCentroid = Meyda.extract('spectralCentroid', segment);
                        descriptor = isNaN(spectralCentroid) ? 20 : spectralCentroid * (sampleRate / 2) / (nFft / 2);
                        break;
                    case 'mfcc':
                        descriptor = Meyda.extract('mfcc', segment).map(value => isNaN(value) ? 0 : value);
                        break;
                    case 'chroma':
                        descriptor = Meyda.extract('chroma', segment).map(value => isNaN(value) ? 0 : value);
                        break;
                    case 'centroid':
                        descriptor = isNaN((descriptor = Meyda.extract('spectralCentroid', segment))) ? 20 : descriptor;
                        break;
                    case 'contrast':
                        descriptor = computeSpectralContrast(Meyda.extract('powerSpectrum', segment), nFft);
                        break;
                    case 'bandwidth':
                        descriptor = isNaN((descriptor = Meyda.extract('spectralSpread', segment))) ? 0 : descriptor;
                        break;
                    case 'rolloff':
                        descriptor = isNaN((descriptor = Meyda.extract('spectralRolloff', segment))) ? 0 : descriptor;
                        break;
                    case 'flatness':
                        descriptor = isNaN((descriptor = Meyda.extract('spectralFlatness', segment))) ? 0 : descriptor;
                        break;
                }
                descriptors[option].push(descriptor);
            });

            // Update progress bar
            currentStep++;
            const progress = Math.floor((currentStep / totalSteps) * 100);
            updateProgressBar(progress);

            // Process the next segment after a short delay
            setTimeout(() => processSegment(i + hopLength), 0);
        } else {
            // Concatenate descriptors
            let allDescriptors = Array.from({
                length: descriptors[selectedOptions[0]].length
            }, (_, i) => {
                return selectedOptions.flatMap(option => descriptors[option][i]);
            });

            // UMAP
            if (document.getElementById('umap').checked) {
                const umap = new UMAP.UMAP({
                    nComponents: umapDims
                });
                allDescriptors = umap.fit(allDescriptors);
            }

            // Normalize and Transpose
            allDescriptors = transpose(normalize(allDescriptors));
            allDescr = allDescriptors;

            displayResults(selectedOptions, allDescriptors);

            hideProgressBar();
        }
    }

    // Start processing segments
    processSegment(0);
}

function convertToMono(buffer) {
    const channelData = buffer.numberOfChannels === 1 ?
        buffer.getChannelData(0) :
        Array.from({
            length: buffer.numberOfChannels
        }, (_, i) => buffer.getChannelData(i));

    const monoData = channelData[0].map((_, i) => {
        return channelData.reduce((sum, channel) => sum + channel[i], 0) / buffer.numberOfChannels;
    });

    // Create a new AudioBuffer with mono data
    const audioContext = new(window.AudioContext || window.webkitAudioContext)();
    const monoBuffer = audioContext.createBuffer(1, monoData.length, buffer.sampleRate);
    monoBuffer.copyToChannel(new Float32Array(monoData), 0);
    return monoBuffer;
}

async function resampleAudio(buffer, targetSampleRate) {
    const offlineContext = new OfflineAudioContext(
        1,
        buffer.duration * targetSampleRate,
        targetSampleRate
    );

    const bufferSource = offlineContext.createBufferSource();
    bufferSource.buffer = buffer;
    bufferSource.connect(offlineContext.destination);
    bufferSource.start(0);

    return await offlineContext.startRendering();
}

/* -------------------------------- FUNCTIONS -------------------------------- */

function normalize(data) {
    const numDims = data[0].length;
    const {
        min,
        max
    } = data.reduce((acc, row) => {
        row.forEach((val, dimIndex) => {
            acc.min[dimIndex] = Math.min(acc.min[dimIndex], val);
            acc.max[dimIndex] = Math.max(acc.max[dimIndex], val);
        });
        return acc;
    }, {
        min: Array(numDims).fill(Infinity),
        max: Array(numDims).fill(-Infinity)
    });

    const range = min.map((val, index) => max[index] - val);
    const rangeInverse = range.map(val => val === 0 ? 0 : 1 / val);

    return data.map(row =>
        row.map((value, dimIndex) =>
            rangeInverse[dimIndex] === 0 ? 0.5 : (value - min[dimIndex]) * rangeInverse[dimIndex]
        )
    );
}

function transpose(data) {
    return data[0].map((_, colIndex) => data.map(row => row[colIndex]));
}

function computeSpectralContrast(powerSpectrum, nFft) {
    const numSubBands = 6;
    const subBandSize = Math.floor(nFft / 2 / numSubBands);
    const contrast = new Array(numSubBands).fill(0);

    for (let j = 0; j < numSubBands; j++) {
        const startIdx = j * subBandSize;
        const endIdx = (j + 1) * subBandSize;
        const subBand = powerSpectrum.slice(startIdx, endIdx);

        if (subBand.length === 0) {
            contrast[j] = 0;
            continue;
        }

        subBand.sort((a, b) => a - b);

        const quartileSize = Math.floor(subBandSize / 4);
        let topSum = 0;
        let bottomSum = 0;

        for (let i = 0; i < quartileSize; i++) {
            bottomSum += subBand[i];
            topSum += subBand[subBand.length - 1 - i];
        }

        const topQuantileMean = topSum / quartileSize;
        const bottomQuantileMean = bottomSum / quartileSize;

        contrast[j] = (bottomQuantileMean === 0 || isNaN(topQuantileMean) || isNaN(bottomQuantileMean)) ?
            0 :
            10 * Math.log10(topQuantileMean / bottomQuantileMean);
    }

    return contrast;
}

function formatNumber(value) {
    return isNaN(value) ? 'N/A' : value.toFixed(2);
}

function displayResults(selectedOptions, descriptors) {
    const resultsDiv = document.getElementById('analysisResults');
    let resultsHtml = '';

    const specialDescriptors = {
        chroma: 12,
        mfcc: 13,
        contrast: 6
    };

    const expandedOptions = selectedOptions.flatMap(option => {
        const numChannels = specialDescriptors[option] || 1;
        return Array(numChannels).fill(option);
    });

    function formatName(inputString) {
        let formattedString = inputString.toLowerCase();
        if (formattedString === 'mfcc') {
            return formattedString.toUpperCase();
        } else {
            return formattedString.charAt(0).toUpperCase() + formattedString.slice(1);
        }
    }

    let descriptorStats = {};
    Object.keys(descriptors).forEach((descriptor, index) => {
        let values = descriptors[descriptor];
        let descriptorName = document.getElementById('umap').checked ?
            `Dim ${(index + 1).toString().padStart(2, '0')}` :
            formatName(expandedOptions[index]);

        let numRows = Array.isArray(values[0]) ? values[0].length : 1;
        let numCols = values.length;

        const valuesArray = new Float32Array(values.flat());
        const validValuesArray = valuesArray.filter(value => !isNaN(value));

        let dims;
        if (descriptorName.startsWith('Dim')) {
            dims = 1;
        } else {
            dims = specialDescriptors.hasOwnProperty(expandedOptions[index]) ? specialDescriptors[expandedOptions[index]] : 1;
        }

        if (!descriptorStats[descriptorName]) {
            descriptorStats[descriptorName] = {
                numRows: numRows,
                numCols: numCols,
                numDims: dims,
                numSamps: numCols,
                min: Infinity,
                max: -Infinity,
                median: null,
                mean: 0,
                std: 0,
                count: 0
            };
        } else {
            descriptorStats[descriptorName].numRows = Math.max(descriptorStats[descriptorName].numRows, numRows);
            descriptorStats[descriptorName].numCols += numCols;
        }

        if (validValuesArray.length === 0) {
            descriptorStats[descriptorName].invalid = true;
        } else {
            const sortedArray = validValuesArray.slice().sort((a, b) => a - b);
            const min = sortedArray[0];
            const max = sortedArray[sortedArray.length - 1];
            const median = sortedArray[Math.floor(sortedArray.length / 2)];
            const mean = validValuesArray.reduce((sum, val) => sum + val, 0) / validValuesArray.length;
            const std = Math.sqrt(validValuesArray.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / validValuesArray.length);

            descriptorStats[descriptorName].min = Math.min(descriptorStats[descriptorName].min, min);
            descriptorStats[descriptorName].max = Math.max(descriptorStats[descriptorName].max, max);
            descriptorStats[descriptorName].median = median;
            descriptorStats[descriptorName].mean = (descriptorStats[descriptorName].mean * descriptorStats[descriptorName].count + mean * validValuesArray.length) / (descriptorStats[descriptorName].count + validValuesArray.length);
            descriptorStats[descriptorName].std = (descriptorStats[descriptorName].std * descriptorStats[descriptorName].count + std * validValuesArray.length) / (descriptorStats[descriptorName].count + validValuesArray.length);
            descriptorStats[descriptorName].count += validValuesArray.length;
        }
    });

    resultsHtml = `<p><i><strong>Descr</strong> [Dims, Samps] (min, max, median, std)</i></p>`;
    Object.keys(descriptorStats).forEach(descriptorName => {
        let {
            numDims,
            numSamps,
            min,
            max,
            median,
            std,
            invalid
        } = descriptorStats[descriptorName];
        if (invalid) {
            resultsHtml += `<p><strong>${descriptorName}</strong> [${numDims}, ${numSamps}] (All values are NaN or invalid)</p>`;
        } else {
            resultsHtml += `<p><strong>${descriptorName}</strong> [${numDims}, ${numSamps}] (${formatNumber(min)}, ${formatNumber(max)}, ${formatNumber(median)}, ${formatNumber(std)})</p>`;
        }
    });

    resultsDiv.innerHTML = resultsHtml;
}

/* -------------------------------- PROGRESS BAR -------------------------------- */
function showProgressBar() {
    return new Promise((resolve) => {
        document.getElementById('progress-container').style.display = 'block';
        requestAnimationFrame(resolve);
    });
}

function updateProgressBar(progress) {
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    progressBar.style.width = progress + '%';
    progressText.textContent = progress + '%';
}

function hideProgressBar() {
    document.getElementById('progress-container').style.display = 'none';
}

// --------------------------------- DATASET --------------------------------- //

async function downloadDataset() {
    if (!allDescr) {
        alert('Analyse an audio file first!');
        return;
    }

    // Create the CSV file
    const csv = allDescr.map(row => row.join(',')).join('\n');
    const csvBlob = new Blob([csv], {
        type: 'text/csv'
    });

    // Create the WAV file
    const monoBuffer = convertToMono(audioBuffer);
    const resampledBuffer = await resampleAudio(monoBuffer, 22050);
    const wavBlob = await getWavBlob(resampledBuffer);

    // Create a zip file containing both the CSV and WAV files
    const zip = new JSZip();
    folderName = `dataset-${audioFilename}`;
    zip.file(`${folderName}/Descr.csv`, csvBlob);
    zip.file(`${folderName}/Audio.wav`, wavBlob);

    // Generate the zip file and trigger the download
    const zipBlob = await zip.generateAsync({
        type: 'blob'
    });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(zipBlob);
    link.download = `${folderName}.zip`;

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    updateTrain();
}

async function getWavBlob(audioBuffer) {
    const wavBuffer = audioBufferToWav(audioBuffer);
    return new Blob([wavBuffer], {
        type: 'audio/wav'
    });
}

function audioBufferToWav(buffer) {
    const numOfChannels = 1;
    const length = buffer.length * numOfChannels * 2 + 44;
    const bufferArray = new ArrayBuffer(length);
    const view = new DataView(bufferArray);

    const channels = [];
    for (let i = 0; i < numOfChannels; i++) {
        channels.push(buffer.getChannelData(i));
    }

    let offset = 0;
    const writeString = function(str) {
        for (let i = 0; i < str.length; i++) {
            view.setUint8(offset + i, str.charCodeAt(i));
        }
        offset += str.length;
    };

    // RIFF identifier
    writeString('RIFF');
    view.setUint32(offset, 36 + buffer.length * 2 * numOfChannels, true);
    offset += 4;
    writeString('WAVE');
    writeString('fmt ');
    view.setUint32(offset, 16, true);
    offset += 4;
    view.setUint16(offset, 1, true);
    offset += 2;
    view.setUint16(offset, numOfChannels, true);
    offset += 2;
    view.setUint32(offset, buffer.sampleRate, true);
    offset += 4;
    view.setUint32(offset, buffer.sampleRate * numOfChannels * 2, true);
    offset += 4;
    view.setUint16(offset, numOfChannels * 2, true);
    offset += 2;
    view.setUint16(offset, 16, true);
    offset += 2;
    writeString('data');
    view.setUint32(offset, buffer.length * 2 * numOfChannels, true);
    offset += 4;

    // Interleave channels
    for (let i = 0; i < buffer.length; i++) {
        for (let j = 0; j < numOfChannels; j++) {
            const sample = Math.max(-1, Math.min(1, channels[j][i]));
            const intSample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
            view.setInt16(offset, intSample, true);
            offset += 2;
        }
    }

    return bufferArray;
}


// --------------------------------- UPDATE DESCRIPTORS --------------------------------- //

document.getElementById('preprocessForm').addEventListener('change', updatePreprocess);

function updatePreprocess() {
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

    if (options.length === 0) {
        alert('Select at least one descriptor!');
        return;
    }

    let umapOption = '';
    if (document.getElementById('umap').checked) {
        const umapValue = document.getElementById('umapDims').value;
        umapOption = `--umap_dims ${umapValue}`;
        totalDims = umapValue;
    }

    document.getElementById('totalDims').value = totalDims;
}

function updatePresets(selected) {
    const descriptors = {
        loudness: document.getElementById('loudness'),
        pitch: document.getElementById('pitch'),
        mfcc: document.getElementById('mfcc'),
        chroma: document.getElementById('chroma'),
        centroid: document.getElementById('centroid'),
        contrast: document.getElementById('contrast'),
        bandwidth: document.getElementById('bandwidth'),
        rolloff: document.getElementById('rolloff'),
        flatness: document.getElementById('flatness')
    };

    // Uncheck all checkboxes
    for (let key in descriptors) {
        descriptors[key].checked = false;
    }

    switch (selected) {
        case '00': // All Descriptors
            for (let key in descriptors) {
                descriptors[key].checked = true;
            }
            break;
        case '01': // Classic
            descriptors.loudness.checked = true;
            descriptors.pitch.checked = true;
            descriptors.mfcc.checked = true;
            break;
        case '02': // Timbral
            descriptors.mfcc.checked = true;
            descriptors.contrast.checked = true;
            descriptors.bandwidth.checked = true;
            descriptors.rolloff.checked = true;
            break;
        case '03': // Spectral
            descriptors.centroid.checked = true;
            descriptors.contrast.checked = true;
            descriptors.bandwidth.checked = true;
            descriptors.rolloff.checked = true;
            descriptors.flatness.checked = true;
            break;
        case '04': // Tonal
            descriptors.loudness.checked = true;
            descriptors.pitch.checked = true;
            descriptors.chroma.checked = true;
            break;
        default:
            console.log("Unknown selection");
    }
}

// Initial command update
updatePreprocess();
updateTrain();

// --------------------------------- COMMANDS --------------------------------- //

function updateTrain() {
    const os = document.getElementById('osSelect').value;
    const separator = (os === 'Windows') ? ';' : '&&';

    if (folderName && typeof folderName === 'string') {
        const zipFile = `${folderName}.zip`;

        // Remove the 'dataset-' prefix if it exists
        let modelName = folderName.startsWith('dataset-') ? folderName.replace('dataset-', '') : folderName;
        modelName = `model-${modelName}`;

        let cmd_scp = `cd ~/Downloads \
            ${separator} tar -xf ${zipFile} \
            ${separator} scp -r ${folderName} mosaique@pop-os-mosaique.musique.umontreal.ca:/home/mosaique/Desktop/DiffWave_v2/datasets`;

        let cmd_ssh = `ssh mosaique@pop-os-mosaique.musique.umontreal.ca  # password: mosaique666`;

        let cmd_train = `cd /home/mosaique/Desktop/DiffWave_v2 \
            ${separator} source venv/bin/activate`;

        let cmd_train_blocking = cmd_train + ` \
            ${separator} python src/__main__.py datasets/${folderName} models/${modelName}`;

        let cmd_train_nonblocking = cmd_train + ` \
            ${separator} nohup python src/__main__.py datasets/${folderName} models/${modelName} &`;

        let cmd_getmodel = `cd ~/Downloads \
            ${separator} scp -r mosaique@pop-os-mosaique.musique.umontreal.ca:/home/mosaique/Desktop/DiffWave_v2/models/${modelName} .`;

        // Delete useless spaces
        cmd_scp = cmd_scp.replace(/\s{2,}/g, ' ').trim();
        cmd_ssh = cmd_ssh.replace(/\s{2,}/g, ' ').trim();
        cmd_train_blocking = cmd_train_blocking.replace(/\s{2,}/g, ' ').trim();
        cmd_train_nonblocking = cmd_train_nonblocking.replace(/\s{2,}/g, ' ').trim();
        cmd_getmodel = cmd_getmodel.replace(/\s{2,}/g, ' ').trim();

        document.getElementById('trainOutput_01').value = cmd_scp;
        document.getElementById('trainOutput_02').value = cmd_ssh;
        document.getElementById('trainOutput_03').value = cmd_train_blocking;
        document.getElementById('trainOutput_04').value = cmd_train_nonblocking;
        document.getElementById('model_get').value = cmd_getmodel;

        // Create bash script
        window.scriptCommands = {
            scp: cmd_scp,
            ssh: cmd_ssh,
            train_blocking: cmd_train_blocking,
            train_nonblocking: cmd_train_nonblocking,
            model_get: cmd_getmodel
        };
    }
}

function downloadScripts() {
    if (window.scriptCommands) {
        const {
            scp,
            ssh,
            train_blocking,
            train_nonblocking,
            model_get
        } = window.scriptCommands;

        let script_scp = `${scp}`;
        let script_ssh = `${ssh}`;
        let script_train_blocking = `${train_blocking}`;
        let script_train_nonblocking = `${train_nonblocking}`;
        let script_model_get = `${model_get}`;

        var zip = new JSZip();
        const folder = `dataset-${audioFilename} (cmds)`;

        zip.file(`${folder}/01 - SCP.txt`, script_scp);
        zip.file(`${folder}/02 - SSH.txt`, script_ssh);
        zip.file(`${folder}/03a - Train (Blocking).txt`, script_train_blocking);
        zip.file(`${folder}/03b - Train (Non Blocking).txt`, script_train_nonblocking);
        zip.file(`${folder}/04 - Model (Get).txt`, script_model_get);

        zip.generateAsync({
                type: "blob"
            })
            .then(function(content) {
                const url = URL.createObjectURL(content);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${folderName} (cmds).zip`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            });
    } else {
        alert("Download a dataset first!");
    }
}

function downloadDevice() {
    const url = 'https://github.com/jpjullin/DiffWave_GUI/tree/main/device';
    window.open(url, '_blank');
}

function copyCommand(id) {
    const commandOutput = document.getElementById(id);
    const text = commandOutput.value; // Assuming it's a text input or textarea

    navigator.clipboard.writeText(text).then(function() {
        // alert('Command copied to clipboard!');
    }).catch(function(error) {
        alert(`Failed to copy command: ${error}`);
    });
}



// --------------------------------- TRAIN --------------------------------- //

function trainModel() {
    const epochs = parseInt(document.getElementById('epochs').value, 10);
    train(epochs, 2e-4, 'webgl'); // WebGL for GPU acceleration
}

// --------------------------------- ~ TENSORFLOW.JS ~ --------------------------------- //

async function train(maxEpochs = null, learningRate= 2e-4, backend = 'webgl') {
    tf.setBackend(backend);

    const start = Date.now();
    const [dataset, nParams] = await loadDataset();
    // console.log(`Loaded dataset in ${Date.now() - start} ms`);

    const opt = tf.train.adam(learningRate);
    model = createModel(nParams, dataset.winLength);
    model.compile({
        optimizer: opt,
        loss: 'meanSquaredError'
    });

    await trainLoop(model, dataset, maxEpochs);

    // console.log('Training complete!');
}

async function downloadModel() {
    if (!model) {
        alert('Train a model first!');
        return;
    }

    // folderName = `model-${audioFilename}`;
    // await model.save(`downloads://${folderName}`);
    await model.save(`downloads://model`);
}


async function loadDataset() {
    const dataset = new WavDataset();
    await dataset._init();
    // console.log(`Using ${dataset.length()} samples for training`);
    return [dataset, dataset.conditioningData.length];
}

class WavDataset {
    constructor() {
        this.audioBuffer = audioBuffer;
        this.conditioningData = allDescr;

        this.sampleRate = parseInt(document.getElementById('sampleRate').value, 10);
        this.winLength = parseInt(document.getElementById('winLength').value, 10);
        this.hopLength = parseInt(document.getElementById('hopLength').value, 10);

        this.nExamples = 0;
    }

    async _init() {
        [this.audio, this.conditioning] = await this._initExamples();
        this.nExamples = this.conditioning.shape[0];
    }

    length() {
        return this.nExamples;
    }

    async _initExamples() {
        const monoBuffer = this.audioBuffer.numberOfChannels > 1 ? convertToMono(audioBuffer) : audioBuffer;
        const resampledBuffer = await resampleAudio(monoBuffer, this.sampleRate);

        const nSamples = resampledBuffer.length;
        const nParamsSamples = this.conditioningData[0].length;

        // console.log('---------------------------');
        // console.log(`Loaded ${1} x ${nSamples} samples`);
        // console.log(`Loaded ${this.conditioningData.length} x ${nParamsSamples} conditioning parameters`);

        const wavSamples = tf.tensor(resampledBuffer.getChannelData(0));
        const conditioningParams = tf.tensor(this.conditioningData);

        const numWindows = Math.floor((nSamples - this.winLength) / this.hopLength);

        const windowedSamples = tf.tidy(() => {
            const windowedSamplesArray = [];
            for (let i = 0; i < numWindows; i++) {
                const start = i * this.hopLength;
                const window = wavSamples.slice([start], [this.winLength]);
                windowedSamplesArray.push(window);
            }
            return tf.stack(windowedSamplesArray);
        });

        const windowedConditioning = conditioningParams.slice([0, 0], [this.conditioningData.length, numWindows]).transpose();

        return [windowedSamples, windowedConditioning];
    }
}

function createModel(nParams, winLength) {
    const model = tf.sequential();

    // Add a dense layer for the input, with nParams input shape
    model.add(tf.layers.dense({
        units: 512,
        activation: 'relu',
        inputShape: [nParams]
    }));
    model.add(tf.layers.dense({
        units: 1024,
        activation: 'relu'
    }));
    model.add(tf.layers.dense({
        units: 2048,
        activation: 'relu'
    }));
    model.add(tf.layers.dense({
        units: 4096,
        activation: 'relu'
    }));

    // Output layer to generate winLength samples
    model.add(tf.layers.dense({
        units: winLength,
        activation: 'tanh'
    }));

    return model;
}

async function trainLoop(model, dataset, maxEpochs) {
    let epoch = 0;
    const lossValues = [];

    function transpose(array) {
        return array[0].map((_, colIndex) => array.map(row => row[colIndex]));
    }

    const conditioningValues = transpose(dataset.conditioningData);

    while (maxEpochs === null || epoch < maxEpochs) {
        for (let i = 0; i < dataset.nExamples; i++) {
            const audioTensor = dataset.audio.slice([i, 0], [1, dataset.winLength]);
            const conditioningTensor = tf.tensor(conditioningValues[i]).expandDims(0);

            // console.log('audioTensor', audioTensor)
            // console.log('conditioningTensor', conditioningTensor)

            // Train the model
            const loss = await model.fit(conditioningTensor, audioTensor, {
                epochs: 1,
                batchSize: 1,
                verbose: 0
            });

            lossValues.push(loss.history.loss[0]);
            updatePlot(lossValues);

            // console.log(`Epoch ${epoch}, Example ${i}, Loss: ${loss.history.loss[0]}`);

            epoch++;
            if (maxEpochs !== null && epoch >= maxEpochs) {
                return;
            }
        }
    }
}

function updatePlot(lossValues) {
    const trace = {
        y: lossValues,
        type: 'scatter',
        mode: 'lines+markers',
        marker: {color: 'red'},
    };

    const layout = {
        title: 'Training Loss',
        xaxis: {
            title: 'Epochs',
        },
        yaxis: {
            title: 'Loss',
        }
    };

    Plotly.newPlot('lossPlot', [trace], layout);
}
