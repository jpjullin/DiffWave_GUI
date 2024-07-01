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
        alert('Please upload a WAV file.');
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
        alert('Please upload an audio file first.');
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

        Object.keys(descriptors).forEach((descriptor, index) => {
            let values = descriptors[descriptor];
            let descriptorName = document.getElementById('umap').checked 
                ? `Dim ${(index + 1).toString().padStart(2, '0')}` 
                : expandedOptions[index].toUpperCase();

            if (Array.isArray(values[0])) {
                // 2D array
                const numRows = values[0].length;
                const numCols = values.length;
                resultsHtml += `<p><strong>${descriptorName}</strong></p>`;
                resultsHtml += `<p>Shape: [${numRows}, ${numCols}]</p>`;
            } else {
                // 1D array
                const numRows = 1;
                const numCols = values.length;
                resultsHtml += `<p><strong>${descriptorName}</strong></p>`;
                resultsHtml += `<p>Shape: [${numRows}, ${numCols}]</p>`;
            }

            // Convert values to Float32Array for numerical calculations
            const valuesArray = Array.isArray(values[0]) ? new Float32Array(values.flat()) : new Float32Array(values);
            const validValuesArray = valuesArray.filter(value => !isNaN(value));

            if (validValuesArray.length === 0) {
                resultsHtml += `<p>All values are NaN or invalid.</p>`;
            } else {
                const min = Math.min(...validValuesArray);
                const max = Math.max(...validValuesArray);
                const median = validValuesArray.sort()[Math.floor(validValuesArray.length / 2)];
                const mean = validValuesArray.reduce((sum, val) => sum + val, 0) / validValuesArray.length;
                const std = Math.sqrt(validValuesArray.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / validValuesArray.length);

                resultsHtml += `<p>Min: ${formatNumber(min)}, Max: ${formatNumber(max)}, Median: ${formatNumber(median)}, Std: ${formatNumber(std)}</p>`;
            }
        });

        resultsDiv.innerHTML = resultsHtml;
    }
}

async function download() {
    if (!allDescr) {
        alert('Please preprocess an audio file first.');
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
        alert('Please select at least one descriptor.');
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

// --------------------------------- UPDATE TRAIN --------------------------------- //

function updateTrain() {
    const os = document.getElementById('osSelect').value;
    const separator = (os === 'Windows') ? ';' : '&&';

    if (folderName && typeof folderName === 'string') {
        const zipFile = `${folderName}.zip`;
        
        // Remove the 'dataset-' prefix if it exists
        let modelName = folderName.startsWith('dataset-') ? folderName.replace('dataset-', '') : folderName;
        modelName = `model-${modelName}`;

        let connectCommand_01 = `cd ~/Downloads \
            ${separator} tar -xf ${zipFile} \
            ${separator} scp -r ${folderName} mosaique@pop-os-mosaique.musique.umontreal.ca:/home/mosaique/Desktop/DiffWave_v2/audio_concat`;

        // Delete useless spaces
        connectCommand_01 = connectCommand_01.replace(/\s{2,}/g, ' ').trim();
        document.getElementById('trainOutput_01').value = connectCommand_01;

        let connectCommand_02 = `ssh mosaique@pop-os-mosaique.musique.umontreal.ca  # password: mosaique666`;

        // Delete useless spaces
        connectCommand_02 = connectCommand_02.replace(/\s{2,}/g, ' ').trim();
        document.getElementById('trainOutput_02').value = connectCommand_02;

        let baseCommand = `cd /home/mosaique/Desktop/DiffWave_v2 \
            ${separator} source venv/bin/activate \
            ${separator} cd /home/mosaique/Desktop/DiffWave_v2/audio_concat/${folderName} \
            ${separator} mv Audio.wav ../ \
            ${separator} mv Descr.csv ../ \
            ${separator} cd .. \
            ${separator} rm -r ${folderName} \
            ${separator} cd .. \
            ${separator} python src/csvConvert.py audio_concat \
            ${separator} rm audio_concat/Descr.csv`;
        
        // Delete useless spaces
        baseCommand = baseCommand.replace(/\s{2,}/g, ' ').trim();

        let trainOutput_03 = baseCommand + ` \
            ${separator} python src/__main__.py audio_concat models/${modelName}`;

        let trainOutput_04 = baseCommand + ` \
            ${separator} nohup python src/__main__.py audio_concat models/${modelName} &`;

        document.getElementById('trainOutput_03').value = trainOutput_03;
        document.getElementById('trainOutput_04').value = trainOutput_04;

        let getModel = `cd ~/Downloads \
            ${separator} scp -r mosaique@pop-os-mosaique.musique.umontreal.ca:/home/mosaique/Desktop/DiffWave_v2/models/${modelName} .`;
        
        getModel = getModel.replace(/\s{2,}/g, ' ').trim();
        document.getElementById('getModel').value = getModel;
    }
}


document.getElementById('inferenceForm').addEventListener('change', updateInference);

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
