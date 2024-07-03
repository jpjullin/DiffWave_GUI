// --------------------------------- DROP FILE --------------------------------- //

let audioBuffer = null;
let audioFilename = null;

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

let allDescr = null;
let resampledAudio = null;
let folderName = null;

async function preprocess() {
    const sampleRate = parseInt(document.getElementById('sampleRate').value, 10);
    const nFft = parseInt(document.getElementById('nFft').value, 10);
    const hopLength = parseInt(document.getElementById('hopLength').value, 10);
    const winLength = parseInt(document.getElementById('winLength').value, 10);

    const windowType = 'hanning';
    const umapDims = parseInt(document.getElementById('umapDims').value, 10);

    if (!audioBuffer) {
        alert('Upload an audio file first!');
        return;
    }

    const monoBuffer = convertToMono(audioBuffer);
    const resampledBuffer = await resampleAudio(monoBuffer, sampleRate);
    const signal = resampledBuffer.getChannelData(0);

    const selectedOptions = Array.from(document.querySelectorAll('input[name="options"]:checked')).map(option => option.value);

    const descriptors = {};
    selectedOptions.forEach(option => {
        descriptors[option] = [];
    });

    // Meyda needs the buffer size to be a power of 2
    const nextPowerOf2 = Math.pow(2, Math.round(Math.log2(winLength)));
    const paddedSegment = new Float32Array(nextPowerOf2);

    for (let i = 0; i < signal.length - winLength; i += hopLength) {
        let segment = signal.slice(i, i + winLength);
        paddedSegment.fill(0);
        paddedSegment.set(segment);
        
        segment = Meyda.windowing(paddedSegment, windowType);

        selectedOptions.forEach(option => {
            switch (option) {
                case 'loudness':
                    const loudness = Meyda.extract('loudness', segment);
                    descriptors.loudness.push(loudness.total);
                    break;
                case 'pitch':
                    const spectralCentroid = Meyda.extract('spectralCentroid', segment);
                    let pitch = spectralCentroid * (sampleRate / 2) / (nFft / 2);
                    if (isNaN(pitch)) {
                        pitch = 20;
                    }
                    descriptors.pitch.push(pitch);
                    break;
                case 'mfcc':
                    const mfcc = Meyda.extract('mfcc', segment);
                    mfcc.map(value => isNaN(value) ? 0 : value);
                    descriptors.mfcc.push(mfcc);
                    break;
                case 'chroma':
                    const chroma = Meyda.extract('chroma', segment);
                    chroma.map(value => isNaN(value) ? 0 : value);
                    descriptors.chroma.push(chroma);
                    break;
                case 'centroid':
                    let centroid = Meyda.extract('spectralCentroid', segment);
                    if (isNaN(centroid)) {
                        centroid = 20;
                    }
                    descriptors.centroid.push(centroid);
                    break;
                case 'contrast':
                    const powerSpectrum = Meyda.extract('powerSpectrum', segment);
                    const contrast = computeSpectralContrast(powerSpectrum, nFft);
                    descriptors.contrast.push(contrast);
                    break;
                case 'bandwidth':
                    let bandwidth = Meyda.extract('spectralSpread', segment);
                    if (isNaN(bandwidth)) {
                        bandwidth = 0;
                    }
                    descriptors.bandwidth.push(bandwidth);
                    break;
                case 'rolloff':
                    let rolloff = Meyda.extract('spectralRolloff', segment);
                    if (isNaN(rolloff)) {
                        rolloff = 0;
                    }
                    descriptors.rolloff.push(rolloff);
                    break;
                case 'flatness':
                    let flatness = Meyda.extract('spectralFlatness', segment);
                    if (isNaN(flatness)) {
                        flatness = 0;
                    }
                    descriptors.flatness.push(flatness);
                    break;
            }
        });
    }

    // Concatenate descriptors
    let allDescriptors = [];
    const numSegments = descriptors[selectedOptions[0]].length;
    for (let i = 0; i < numSegments; i++) {
        const segmentDescriptors = [];
        selectedOptions.forEach(option => {
            const descriptor = descriptors[option][i];
            if (Array.isArray(descriptor)) {
                segmentDescriptors.push(...descriptor);
            } else {
                segmentDescriptors.push(descriptor);
            }
        });
        allDescriptors.push(segmentDescriptors);
    }

    // UMAP
    if (document.getElementById('umap').checked) {
        const umap = new UMAP.UMAP({ nComponents: umapDims });
        allDescriptors = umap.fit(allDescriptors);
    }
    
    // Normalize
    allDescriptors = normalize(allDescriptors);

    // Transpose
    allDescriptors = allDescriptors[0].map((_, colIndex) => allDescriptors.map(row => row[colIndex]));

    displayResults(selectedOptions, allDescriptors);
    
    allDescr = allDescriptors;

    function normalize(data) {
        const numDims = data[0].length;
        const numSamples = data.length;

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

            if (bottomQuantileMean === 0 || isNaN(topQuantileMean) || isNaN(bottomQuantileMean)) {
                contrast[j] = 0;
            } else {
                contrast[j] = 10 * Math.log10(topQuantileMean / bottomQuantileMean);
            }
        }

        return contrast;
    }

    function mean(array) {
        if (array.length === 0) return 0;
        const sum = array.reduce((acc, val) => acc + val, 0);
        return sum / array.length;
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

        let expandedOptions = [];
        selectedOptions.forEach((option) => {
            const numChannels = specialDescriptors[option] || 1;
            for (let i = 0; i < numChannels; i++) {
                expandedOptions.push(option);
            }
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
}

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

// Initial command update
updatePreprocess();
updateTrain();
