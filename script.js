let audioBuffer = null;
let audioFilename = null;

let allDescr = null;
let resampledAudio = null;
let folderName = null;

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
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
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

    const monoBuffer = convertToMono(audioBuffer);
    const resampledBuffer = await resampleAudio(monoBuffer, sampleRate);
    const signal = resampledBuffer.getChannelData(0);
    resampledAudio = signal;

    await showProgressBar();
    const totalSteps = Math.ceil((signal.length - winLength) / hopLength);
    let currentStep = 0;

    const selectedOptions = Array.from(document.querySelectorAll('input[name="options"]:checked')).map(option => option.value);
    const descriptors = Object.fromEntries(selectedOptions.map(option => [option, []]));

    // Meyda needs the buffer size to be a power of 2
    const nextPowerOf2 = Math.pow(2, Math.round(Math.log2(winLength)));
    const paddedSegment = new Float32Array(nextPowerOf2);

    function processSegment(i) {
        if (i < signal.length - winLength) {
            let segment = signal.subarray(i, i + winLength);
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
            let allDescriptors = Array.from({ length: descriptors[selectedOptions[0]].length }, (_, i) => {
                return selectedOptions.flatMap(option => descriptors[option][i]);
            });

            // UMAP
            if (document.getElementById('umap').checked) {
                const umap = new UMAP.UMAP({ nComponents: umapDims });
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

/* -------------------------------- FUNCTIONS -------------------------------- */

function normalize(data) {
    const numDims = data[0].length;
    const { min, max } = data.reduce((acc, row) => {
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

        contrast[j] = (bottomQuantileMean === 0 || isNaN(topQuantileMean) || isNaN(bottomQuantileMean)) 
                      ? 0 
                      : 10 * Math.log10(topQuantileMean / bottomQuantileMean);
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
        var formattedString = inputString.toLowerCase();
        if (formattedString === 'mfcc') {
            return formattedString.toUpperCase();
        } else {
            return formattedString.charAt(0).toUpperCase() + formattedString.slice(1);
        }
    }

    let descriptorStats = {};
    Object.keys(descriptors).forEach((descriptor, index) => {
        let values = descriptors[descriptor];
        let descriptorName = document.getElementById('umap').checked 
            ? `Dim ${(index + 1).toString().padStart(2, '0')}` 
            : formatName(expandedOptions[index]);

        let numRows = Array.isArray(values[0]) ? values[0].length : 1;
        let numCols = values.length;

        const valuesArray = new Float32Array(values.flat());
        const validValuesArray = valuesArray.filter(value => !isNaN(value));

        let dims = null;
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
        let { numDims, numSamps, min, max, median, std, invalid } = descriptorStats[descriptorName];
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
    const csvBlob = new Blob([csv], { type: 'text/csv' });

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
    const zipBlob = await zip.generateAsync({ type: 'blob' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(zipBlob);
    link.download = `${folderName}.zip`;

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    updateTrain();
}

function convertToMono(buffer) {
    const channelData = buffer.numberOfChannels === 1
        ? buffer.getChannelData(0)
        : Array.from({ length: buffer.numberOfChannels }, (_, i) => buffer.getChannelData(i));

    const monoData = channelData[0].map((_, i) => {
        return channelData.reduce((sum, channel) => sum + channel[i], 0) / buffer.numberOfChannels;
    });

    // Create a new AudioBuffer with mono data
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
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

    const renderedBuffer = await offlineContext.startRendering();

    return renderedBuffer;
}

async function getWavBlob(audioBuffer) {
    const wavBuffer = audioBufferToWav(audioBuffer);
    return new Blob([wavBuffer], { type: 'audio/wav' });
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
    const writeString = function (str) {
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

    switch(selected) {
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
            ${separator} scp -r mosaique@pop-os-mosaique.musique.umontreal.ca:/home/mosaique/Desktop/DiffWave_v2/models/${modelName} .` ;

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
        const { scp, ssh, train_blocking, train_nonblocking, model_get } = window.scriptCommands;
        
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

        zip.generateAsync({ type: "blob" })
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
    commandOutput.select();
    document.execCommand('copy');
    // alert('Command copied to clipboard!');
}

// --------------------------------- TRAIN --------------------------------- //

const args = {
    dataDir: 'path/to/your/data',
    modelDir: 'path/to/your/model',
    maxSteps: null,
    fp16: false
};

function trainModel() {
    train(args, params)
}

// --------------------------------- ~LEARNER.JS --------------------------------- //

function _nestedMap(struct, mapFn) {
    if (Array.isArray(struct)) {
        return struct.map(x => _nestedMap(x, mapFn));
    }
    if (struct !== null && typeof struct === 'object') {
        return Object.fromEntries(Object.entries(struct).map(([k, v]) => [k, _nestedMap(v, mapFn)]));
    }
    return mapFn(struct);
}

class DiffWaveLearner {
    constructor(modelDir, model, dataset, optimizer, params, options = {}) {
        this.model = model;
        this.dataset = dataset;
        this.optimizer = optimizer;
        this.params = params;
        this.fp16 = options.fp16 || false;
        this.step = 0;
        this.isMaster = true;
        this.summaryWriter = null;
        
        this.noiseLevel = tf.tensor(this.params.noiseSchedule).cumprod();
        this.lossFn = tf.losses.meanSquaredError;
    }

    async stateDict() {
        return tf.tidy(() => {
            const modelWeights = this.model.getWeights().map(weight => ({
                name: weight.name,
                data: weight.val.arraySync(),
                shape: weight.shape,
            }));
            return {
                step: this.step,
                model: modelWeights,
                optimizer: this.optimizer.getWeights(), // Pseudo-code, adjust accordingly
                params: { ...this.params },
            };
        });
    }

    async loadStateDict(stateDict) {
        const weights = stateDict.model.map(weight => tf.tensor(weight.data, weight.shape));
        await this.model.setWeights(weights);
        await this.optimizer.setWeights(stateDict.optimizer); // Pseudo-code, adjust accordingly
        this.step = stateDict.step;
    }

    async saveModel(saveName, loss = null, filename = 'weights', maxCheckpoints = 20) {
        // Save model state in tfjs format
        const saveBasename = `${filename}-${this.step}`;
        const savePath = `${this.modelDir}/${saveBasename}`;

        // Assuming 'this.model' is a tf.Model instance
        await this.model.save(`downloads://${saveBasename}`);

        // Save additional information to a text file
        if (loss !== null) {
            const txtSaveName = `${this.modelDir}/model.txt`;
            const paramsContent = Object.entries(this.params).map(([key, value]) => `${key}: ${value}`).join('\n');
            const txtContent = `Loss: ${loss}\nNumber of parameters: ${this.model.nParams}\n${paramsContent}`;

            // Create a Blob for the text content and save it
            const txtBlob = new Blob([txtContent], { type: 'text/plain' });
            const link = document.createElement('a');
            link.href = URL.createObjectURL(txtBlob);
            link.download = txtSaveName;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }

    async train(maxSteps = null) {
        const device = 'gpu';
        console.log(`Training on ${device}`);
        let epoch = 0;

        while (true) {
            await this.dataset.forEachAsync(async (features) => {
                if (maxSteps !== null && this.step >= maxSteps) return;
                
                console.log(`Processing new batch at step ${this.step}`);

                const mappedFeatures = _nestedMap(features, x => x.to(device));
                const loss = await this.trainStep(mappedFeatures);

                if (tf.isnan(loss)) {
                    throw new Error(`Detected NaN loss at step ${this.step}.`);
                }

                if (this.isMaster) {
                    console.log(`Epoch ${epoch}, Step ${this.step}, Loss: ${loss}`);

                    if (this.step % this.params.saveSummaryEvery === 0) {
                        await this._writeSummary(this.step, features, loss);
                    }

                    if (this.step % (this.params.saveModelEvery * this.dataset.size) === 0) {
                        try {
                            await this.saveToCheckpoint();
                            this.saveToTxt(loss);
                        } catch (e) {
                            console.error("An error occurred while saving checkpoint:", e);
                        }
                    }
                }

                    this.step++;
            });

            epoch++;
        }
    }

    async trainStep(features) {
        return tf.tidy(async () => {
            const audio = features.audio;
            const conditioning = features.conditioning;

            const N = audio.shape[0];
            const t = tf.randomUniform([N], 0, this.params.noise_schedule.length, 'int32');
            const noiseScale = this.noiseLevel.gather(t).expandDims(1);
            const noiseScaleSqrt = tf.sqrt(noiseScale);
            const noise = tf.randomNormal(audio.shape);
            const noisyAudio = noiseScaleSqrt.mul(audio).add(tf.sqrt(tf.scalar(1.0).sub(noiseScale)).mul(noise));

            const predicted = this.model.predict([noisyAudio, t, conditioning]);
            const loss = this.lossFn(noise, predicted.squeeze(1));

            await this.optimizer.minimize(() => loss, this.model.trainableWeights);

            return loss;
        });
    }

    async _writeSummary(step, features, loss) {
        const writer = this.summaryWriter || new SummaryWriter(this.modelDir, { purgeStep: step });
        writer.addAudio('feature/audio', features.audio[0], step, { sampleRate: this.params.sample_rate });
        writer.addText('feature/conditioning', features.conditioning[0].toString(), step);
        writer.addScalar('train/loss', loss, step);
        // Assuming grad_norm is calculated
        writer.addScalar('train/grad_norm', this.grad_norm, step);
        await writer.flush();
        this.summaryWriter = writer;
    }
}

async function train(args, params) {
    const start = Date.now();
    const [dataset, nParams] = await fromPath(args.dataDir);
    const end = Date.now();

    console.log(`Loaded dataset in ${end - start} ms`);
    console.log(`Batches per epoch: ${dataset.length}`);

    const model = new DiffWave(params, nParams);
    await _trainImpl(0, model, dataset, args, params);
}

async function _trainImpl(replicaId, model, dataset, args, params) {
    const opt = tf.train.adam(params.learning_rate);

    const learner = new DiffWaveLearner(args.modelDir, model, dataset, opt, params, { fp16: args.fp16 });
    learner.isMaster = (replicaId === 0);
    await learner.train(args.maxSteps);
}


// --------------------------------- ~DATASET.JS --------------------------------- //

function decodeWavBuffer(audioBuffer) {
  const numberOfChannels = audioBuffer.numberOfChannels;
  const length = audioBuffer.length;
  const sampleRate = audioBuffer.sampleRate;
  const bitsPerSample = 16; // Assuming 16-bit audio
  const bytesPerSample = bitsPerSample / 8;

  // Create the WAV header
  const header = new ArrayBuffer(44);
  const view = new DataView(header);

  // "RIFF" chunk descriptor
  view.setUint32(0, 0x52494646, false); // "RIFF"
  view.setUint32(4, 36 + length * bytesPerSample, true);
  view.setUint32(8, 0x57415645, false); // "WAVE"

  // "fmt " sub-chunk
  view.setUint32(12, 0x666D7420, false); // "fmt "
  view.setUint32(16, 16, true); // Subchunk1Size
  view.setUint16(20, 1, true); // AudioFormat (PCM)
  view.setUint16(22, numberOfChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numberOfChannels * bytesPerSample, true); // ByteRate
  view.setUint16(32, numberOfChannels * bytesPerSample, true); // BlockAlign
  view.setUint16(34, bitsPerSample, true);

  // "data" sub-chunk
  view.setUint32(36, 0x64617461, false); // "data"
  view.setUint32(40, length * bytesPerSample, true); // Subchunk2Size

  // Audio data
  const audioData = new Int16Array(length);
  const channelData = audioBuffer.getChannelData(0);
  for (let i = 0; i < length; i++) {
    const sample = Math.max(-1, Math.min(1, channelData[i]));
    audioData[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
  }

  // Combine header and audio data
  const wavBuffer = new Uint8Array(header.byteLength + audioData.buffer.byteLength);
  wavBuffer.set(new Uint8Array(header), 0);
  wavBuffer.set(new Uint8Array(audioData.buffer), header.byteLength);

  return wavBuffer;
}

class WavDataset {
    constructor(wavFile, descrData, nExamples, windowSize, hopLength) {
        this.wavFile = wavFile;
        this.conditioningData = descrData;
        this.nExamples = nExamples;
        this.windowSize = windowSize;
        this.hopLength = hopLength;
        const { examples, conditioning } = this._initExamples();
        this.examples = examples;
        this.conditioning = conditioning;
    }

    len() {
        return this.nExamples;
    }

    getItem(index) {
        const audio = this.examples.slice([index, 0], [1, this.windowSize]).flatten();
        const conditioning = this.conditioning.slice([index, 0], [1, -1]).flatten();

        // console.log(` --- Getting item ${index}`)
        // console.log(`Audio shape: ${audio.shape}`)
        // console.log(`Conditioning shape: ${conditioning.shape}`)

        return {
            audio: audio,
            conditioning: conditioning
        };
    }

    *[Symbol.iterator]() {
        for (let i = 0; i < this.len(); i++) {
            yield this.getItem(i);
        }
    }

    _initExamples() {
        return tf.tidy(() => {
            const wavSamples = tf.tensor1d(this.wavFile);
            const nSamples = this.wavFile.length;
            console.log('---------------------------');
            console.log(`Loaded ${nSamples} samples`);

            const conditioningParams = tf.tensor(this.conditioningData.flat(), [this.conditioningData.length, this.conditioningData[0].length], 'float32');
            const nParamsSamples = conditioningParams.shape[1];
            console.log(`Loaded ${this.conditioningData.length} x ${nParamsSamples} conditioning parameters`);

            const maxWavWindowStart = Math.floor((nSamples - this.windowSize) / this.hopLength);
            const windowStartMultiples = Array.from({ length: this.nExamples }, () => Math.floor(Math.random() * maxWavWindowStart));
            const windowStartIndexes = windowStartMultiples.map(multiple => multiple * this.hopLength);

            const maxParamsWindowStart = Math.floor((nParamsSamples - this.windowSize) / this.hopLength);
            const paramStartMultiples = windowStartIndexes.map(idx => Math.floor(idx / (nSamples - this.windowSize) * maxParamsWindowStart));
            const paramStartIndexes = paramStartMultiples.map(multiple => multiple * this.hopLength);

            const windowedSamples = tf.stack(windowStartIndexes.map(start => wavSamples.slice([start], [this.windowSize])));
            const windowedConditioning = tf.stack(paramStartIndexes.map(start => conditioningParams.slice([start, 0], [1, -1])));

            return {
                examples: windowedSamples,
                conditioning: windowedConditioning
            };
        });
    }
}


/* -------------------------------------- REDO ME REDO ME REDO ME -------------------------------------- */
function createDataloader(dataset, params) {
    const collateFn = batch => {
        return tf.tidy(() => {
            if (Array.isArray(batch)) {
                return {
                    audio: tf.stack(batch.map(item => tf.tensor(item.audio))),
                    conditioning: tf.stack(batch.map(item => tf.tensor(item.conditioning)))
                };
            } else if (batch instanceof tf.Tensor) {
                throw new TypeError('Batch as a Tensor is not expected in this context');
            } else if (typeof batch === 'object' && batch !== null) {
                if (Array.isArray(batch.audio) && Array.isArray(batch.conditioning)) {
                    return {
                        audio: tf.stack(batch.audio.map(item => tf.tensor(item))),
                        conditioning: tf.stack(batch.conditioning.map(item => tf.tensor(item)))
                    };
                } else if (batch.audio instanceof tf.Tensor && batch.conditioning instanceof tf.Tensor) {
                    const audioArray = batch.audio.arraySync();
                    const conditioningArray = batch.conditioning.arraySync();
                    return {
                        audio: tf.tensor(audioArray),
                        conditioning: tf.tensor(conditioningArray)
                    };
                } else {
                    throw new TypeError('Expected batch object to have audio and conditioning properties as arrays or tensors');
                }
            } else {
                throw new TypeError('Expected batch to be an array or an object with audio and conditioning properties');
            }
        });
    };

    return tf.data.generator(function*() {
        for (let item of dataset) {
            yield item;
        }
    })
    .shuffle(dataset.len())
    .batch(params.batchSize, true)
    .map(batch => {
        return tf.tidy(() => collateFn(batch));
    });
}

function fromPath(dataDir) {
    const dataset = new WavDataset(
        resampledAudio, 
        allDescr, 
        params.nExamples, 
        params.winLength, 
        params.hopLength
    );

    const dataloader = createDataloader(dataset, params);

    console.log(`Using ${dataset.len()} samples for training`);

    return [dataloader, dataset.conditioning.shape[1]];
}


// --------------------------------- ~MODEL.JS --------------------------------- //

class DiffusionEmbedding extends tf.layers.Layer {
    constructor(maxSteps, params) {
        super({name: 'DiffusionEmbedding'});
        this.params = params;
        this.embedding = this.buildEmbedding(maxSteps);
        this.projection1 = tf.layers.dense({ units: 512 });
        this.projection2 = tf.layers.dense({ units: 512 });
    }

    buildEmbedding(maxSteps) {
        const steps = tf.range(0, maxSteps).expandDims(1);
        const dims = tf.range(0, 64).expandDims(0);
        const table = steps.mul(tf.scalar(10.0).pow(dims.mul(tf.scalar(4.0).div(63.0))));
        return tf.concat([tf.sin(table), tf.cos(table)], 1);
    }

    call(diffusionStep) {
        let x;
        if (diffusionStep.dtype === 'int32' || diffusionStep.dtype === 'int64') {
            x = this.embedding.gather(diffusionStep);
        } else {
            x = this.lerpEmbedding(diffusionStep);
        }
        x = this.projection1.apply(x);
        x = tf.sigmoid(x).mul(x);
        x = this.projection2.apply(x);
        x = tf.sigmoid(x).mul(x);
        return x;
    }

    lerpEmbedding(t) {
        const lowIdx = tf.floor(t).toInt();
        const highIdx = tf.ceil(t).toInt();
        const low = this.embedding.gather(lowIdx);
        const high = this.embedding.gather(highIdx);
        return low.add(high.sub(low).mul(t.sub(lowIdx)));
    }
}

class ResidualBlock extends tf.layers.Layer {
    constructor(residualChannels, dilation, numParams) {
        super({name: 'ResidualBlock'});
        this.dilatedConv = tf.layers.conv1d({
            filters: 2 * residualChannels,
            kernelSize: 3,
            padding: 'same',
            dilationRate: dilation,
            kernelInitializer: 'heNormal'
        });
        this.diffusionProjection = tf.layers.dense({ units: residualChannels });
        this.conditionerProjection = tf.layers.conv1d({
            filters: 2 * residualChannels,
            kernelSize: 1,
            kernelInitializer: 'heNormal'
        });
        this.outputProjection = tf.layers.conv1d({
            filters: 2 * residualChannels,
            kernelSize: 1,
            kernelInitializer: 'heNormal'
        });
    }

    call(inputs) {
        let [x, diffusionStep, conditioner] = inputs;
        if (conditioner) {
            conditioner = conditioner.expandDims(-1);
        }
        diffusionStep = this.diffusionProjection.apply(diffusionStep).expandDims(-1);
        let y = x.add(diffusionStep);
        if (conditioner) {
            conditioner = this.conditionerProjection.apply(conditioner);
            y = this.dilatedConv.apply(y).add(conditioner);
        } else {
            y = this.dilatedConv.apply(y);
        }
        const [gate, filter] = tf.split(y, 2, 1);
        y = tf.sigmoid(gate).mul(tf.tanh(filter));
        y = this.outputProjection.apply(y);
        const [residual, skip] = tf.split(y, 2, 1);
        return [x.add(residual).div(sqrt(2)), skip];
    }
}

class DiffWave extends tf.layers.Layer {
    constructor(params, nParams) {
        super({name: 'DiffWave'});
        this.params = params;
        this.inChannels = 1;
        this.nParams = nParams;
        this.inputProjection = tf.layers.conv1d({
            filters: params.residualChannels,
            kernelSize: 1,
            kernelInitializer: 'heNormal'
        });
        this.diffusionEmbedding = new DiffusionEmbedding(params.noiseSchedule.length, params);

        this.residualLayers = [];
        for (let i = 0; i < params.residualLayers; i++) {
            this.residualLayers.push(new ResidualBlock(params.residualChannels, 2 ** (i % params.dilationCycleLength), nParams));
        }
        this.skipProjection = tf.layers.conv1d({
            filters: params.residualChannels,
            kernelSize: 1,
            kernelInitializer: 'heNormal'
        });
        this.outputProjection = tf.layers.conv1d({
            filters: 1,
            kernelSize: 1,
            kernelInitializer: 'zeros'
        });
    }

    call(inputs) {
        let [audio, diffusionStep, conditioning] = inputs;
        if (audio.shape.length < 2) {
            audio = audio.expandDims(1);
        }
        audio = audio.reshape([audio.shape[0], this.inChannels, -1]);
        let x = this.inputProjection.apply(audio);
        x = tf.relu(x);

        diffusionStep = this.diffusionEmbedding.call(diffusionStep);

        let skip = null;
        for (let layer of this.residualLayers) {
            [x, skipConnection] = layer.call([x, diffusionStep, conditioning]);
            if (!skip) {
                skip = skipConnection;
            } else {
                skip = skip.add(skipConnection);
            }
        }

        x = skip.div(Math.sqrt(this.residualLayers.length));
        x = this.skipProjection.apply(x);
        x = tf.relu(x);
        x = this.outputProjection.apply(x);
        return x;
    }
}


// --------------------------------- ~PARAMS.JS --------------------------------- //

// const device = 'cpu';
const device = 'cuda'

class AttrDict {
    constructor(initialData = {}) {
        Object.assign(this, initialData);
    }

    override(attrs) {
        if (typeof attrs === 'object' && !Array.isArray(attrs)) {
            Object.assign(this, attrs);
        } else if (Array.isArray(attrs)) {
            attrs.forEach(attr => this.override(attr));
        } else if (attrs !== null && attrs !== undefined) {
            throw new Error('NotImplementedError');
        }
        return this;
    }
}

function linspace(a, b, n) {
  if (n <= 1) {
    return [b];
  }

  var every = (b - a) / (n - 1);
  var arr = [];

  for (var i = 0; i < n; i++) {
    arr.push(a + i * every);
  }

  return arr;
}

const params = new AttrDict({
    // Device
    device: device,

    // Training params
    batchSize: 16,
    learningRate: 2e-4,
    maxGradNorm: null,
    saveSummaryEvery: 500,  // Save tensorboard summaries every n steps
    saveModelEvery: 1,  // Save model every n epochs

    // Descriptors params
    sampleRate: 22050,  // 22050
    nFft: 1024,
    hopLength: 16,  // 16
    winLength: 1024,
    window: 'hann',

    // Data params
    nExamples: 10000,  // Number of random audio + descr examples

    // Model params
    residualLayers: 22,  // 30
    residualChannels: 32,  // 64
    dilationCycleLength: 10,  // 10
    noiseSchedule: linspace(1e-4, 0.05, 50),
    inferenceNoiseSchedule: [0.0001, 0.001, 0.01, 0.2, 0.5],
});
