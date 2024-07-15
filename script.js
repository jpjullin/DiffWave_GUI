let audioBuffer = null;
let audioFilename = null;

let allDescr = null;
let folderName = null;

let model = null;

// --------------------------------- DROP FILE --------------------------------- //

document.addEventListener('DOMContentLoaded', () => {
    const dropAreaWav = document.getElementById('dropAreaWav');
    const audioFileInput = document.getElementById('audioFileInput');

    const dropAreaDescr = document.getElementById('dropAreaDescr');
    const descrFileInput = document.getElementById('descrFileInput');

    let fileDialogOpen = false;

    const setupDropArea = (dropArea, fileInput, processFile) => {
        if (dropArea && fileInput) {
            dropArea.addEventListener('dragover', handleDragOver, false);
            dropArea.addEventListener('dragleave', handleDragLeave, false);
            dropArea.addEventListener('drop', (event) => handleFileDrop(event, processFile), false);
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
                handleFileSelect(event, processFile);
                fileDialogOpen = false;
            }, false);
        }
    };

    setupDropArea(dropAreaWav, audioFileInput, processAudioFile);
    setupDropArea(dropAreaDescr, descrFileInput, processDescrFile);
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

function handleFileDrop(event, processFile) {
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.classList.remove('dragover');
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function handleFileSelect(event, processFile) {
    const files = event.target.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function processAudioFile(file) {
    if (file.type !== 'audio/wav') {
        alert('WAV files only!');
        return;
    }
    const audioFilename = file.name.split('.').slice(0, -1).join('.');
    const reader = new FileReader();
    reader.onload = async function(event) {
        const arrayBuffer = event.target.result;
        await loadAudioBuffer(arrayBuffer, file.name);
    };
    reader.readAsArrayBuffer(file);
}

async function loadAudioBuffer(arrayBuffer, fileName) {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    try {
        audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        document.getElementById('uploadStatusWav').innerText = `${fileName} uploaded successfully!`;
    } catch (error) {
        console.error('Error decoding audio data:', error);
        alert('Failed to process audio file.');
    }
}

async function processDescrFile(file) {
    const validExtensions = ['.zip'];
    const fileExtension = file.name.split('.').pop().toLowerCase();

    if (!validExtensions.includes(`.${fileExtension}`)) {
        alert('ZIP files only!');
        return;
    }

    document.getElementById('analysisResults').innerText = 'Loading...';
    await showProgressBar();

    const reader = new FileReader();
    reader.onload = async function(event) {
        try {
            const arrayBuffer = event.target.result;
            const zip = new JSZip();
            updateProgressBar(10);

            const unzipped = await zip.loadAsync(arrayBuffer);
            updateProgressBar(20);

            const folderName = file.name.split('.').slice(0, -1).join('.');
            const folder = unzipped.folder(folderName);

            if (!folder) {
                alert(`Folder ${folderName} not found in ZIP!`);
                hideProgressBar();
                document.getElementById('analysisResults').innerText = '';
                return;
            }

            const audioFile = folder.file('Audio.wav');
            const descrFile = folder.file('Descr.csv');

            if (audioFile && descrFile) {
                const audioArrayBuffer = await audioFile.async('arraybuffer');
                updateProgressBar(30);
                await loadAudioBuffer(audioArrayBuffer, 'Audio.wav');
                updateProgressBar(60);

                const descrText = await descrFile.async('text');
                updateProgressBar(70);
                allDescr = parseCSV(descrText);
                updateProgressBar(90);

                document.getElementById('analysisResults').innerText = `${file.name} uploaded successfully!`;
                updateProgressBar(100);
            } else {
                alert('ZIP must contain Audio.wav and Descr.csv!');
                document.getElementById('analysisResults').innerText = '';
            }
        } catch (error) {
            console.error('Error processing ZIP file:', error);
            alert('Failed to process ZIP file.');
            document.getElementById('analysisResults').innerText = '';
        } finally {
            hideProgressBar();
        }
    };
    reader.readAsArrayBuffer(file);
}

function parseCSV(csvText) {
    const rows = csvText.split('\n');
    return rows.map(row => row.split(','));
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
            descriptors.centroid.checked = true;
            descriptors.flatness.checked = true;
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
    const batchSize = parseInt(document.getElementById('batchSize').value, 10);
    const learningRate = parseFloat(document.getElementById('learningRate').value);

    train(epochs, batchSize, learningRate, 'webgl'); // WebGL for GPU acceleration
}

// --------------------------------- ~ TENSORFLOW.JS ~ --------------------------------- //

async function train(maxEpochs = null, batchSize = 16, learningRate= 2e-4, backend = 'webgl') {
    const start = Date.now();
    const [dataset, nParams] = await loadDataset();
    console.log(`Loaded dataset in ${Date.now() - start} ms`);

    tf.setBackend(backend);
    const opt = tf.train.adam(learningRate);

    let modelType = document.getElementById('modelSelect').value;
    if (modelType === 'simple') {
        model = createModel_Simple(nParams, dataset.winLength);
    } else {
        alert('Model not found!');
        return;
    }

    // model = createDiffWave(nParams, dataset.winLength);

    model.compile({
        optimizer: opt,
        loss: 'meanSquaredError',
        metrics: ['mse']
    });

    await trainLoop(model, dataset, maxEpochs, batchSize, learningRate);
    document.getElementById('trainingResults').innerHTML = 'Training complete!';
    console.log('Training complete!');
}


async function loadDataset() {
    const dataset = new WavDataset();
    await dataset._init();
    console.log(`Using ${dataset.length()} samples for training`);
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

        console.log('---------------------------');
        console.log(`Loaded ${1} x ${nSamples} samples`);
        console.log(`Loaded ${this.conditioningData.length} x ${nParamsSamples} conditioning parameters`);

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

function createModel_Simple(nParams, winLength) {
    return tf.tidy(() => {
        const model = tf.sequential();
        const l2Regularizer = tf.regularizers.l2({ l2: 0.01 });

        // Create an array of units
        const numLayers = 5;
        const units = [];
        let stepSize = (winLength - nParams) / (numLayers - 1);

        for (let i = 0; i < numLayers; i++) {
            units.push((nParams + i * stepSize) | 0) ; // Bitwise OR to convert to integer
        }

        console.log('Number of neurons per layer:', units);

        // Input layer
        model.add(tf.layers.dense({ inputShape: [nParams], units: units[1], activation: 'LeakyReLU', kernelInitializer: 'heNormal', kernelRegularizer: l2Regularizer }));
        model.add(tf.layers.batchNormalization());

        // Dense layers
        model.add(tf.layers.dense({ units: units[2], activation: 'LeakyReLU', kernelInitializer: 'heNormal', kernelRegularizer: l2Regularizer }));
        model.add(tf.layers.batchNormalization());

        // Reshape layer
        model.add(tf.layers.reshape({ targetShape: [units[2], 1] }));

        // LSTM layers
        model.add(tf.layers.lstm({ units: units[3], returnSequences: false, kernelInitializer: 'heNormal', recurrentInitializer: 'heNormal', kernelRegularizer: l2Regularizer }));
        model.add(tf.layers.batchNormalization());

        // Output layer
        model.add(tf.layers.dense({ units: winLength, activation: 'tanh', kernelInitializer: 'heNormal', kernelRegularizer: l2Regularizer }));

        return model;
    });
}

// function createDiffWave(nParams, winLength) {
//     const params = {
//         residualLayers: 22,  // 30
//         residualChannels: 32,  // 64
//         dilationCycleLength: 10,  // 10
//         noiseSchedule: Array.from({ length: 50 }, (_, i) => 1e-4 + (0.05 - 1e-4) * i / 49),
//         inferenceNoiseSchedule: [0.0001, 0.001, 0.01, 0.2, 0.5]
//     };
//     const Conv1d = (filters, kernelSize, strides = 1, padding = 'valid', dilationRate = 1) =>
//         tf.layers.conv1d({ filters, kernelSize, strides, padding, dilationRate, kernelInitializer: 'heNormal' });
//
//     const silu = x => x.mul(tf.sigmoid(x));
//
//     const buildEmbedding = (maxSteps) => {
//         const steps = tf.range(0, maxSteps).reshape([-1, 1]);
//         const dims = tf.range(0, 64).reshape([1, -1]);
//         const table = steps.mul(tf.pow(10.0, dims.mul(4.0 / 63.0)));
//         return tf.concat([tf.sin(table), tf.cos(table)], axis=1);
//     };
//
//     const lerpEmbedding = (embedding, t) => {
//         console.log(t);
//
//         if (t instanceof tf.SymbolicTensor) {
//             const inputs = t.sourceLayer.inboundNodes[0].inputTensors;
//             t = inputs[0];
//         } else if (!(t instanceof tf.Tensor)) {
//             t = tf.tensor(t);
//         }
//
//         const lowIdx = tf.floor(t).toInt();
//         const highIdx = tf.ceil(t).toInt();
//         const low = tf.gather(embedding, lowIdx);
//         const high = tf.gather(embedding, highIdx);
//         return low.add(high.sub(low).mul(t.sub(lowIdx).reshape([-1, 1])));
//     };
//
//     const getDiffusionEmbedding = (maxSteps, params, diffusionStep) => {
//         const embedding = buildEmbedding(maxSteps);
//         const projection1 = tf.layers.dense({ units: 512, inputShape: [128], kernelInitializer: 'heNormal' });
//         const projection2 = tf.layers.dense({ units: 512, kernelInitializer: 'heNormal' });
//
//         let x;
//         if (diffusionStep.dtype === 'int32' || diffusionStep.dtype === 'int64') {
//             x = tf.gather(embedding, diffusionStep);
//         } else {
//             x = lerpEmbedding(embedding, diffusionStep);
//         }
//
//         x = projection1.apply(x);
//         x = silu(x);
//         x = projection2.apply(x);
//         x = silu(x);
//
//         return x;
//     };
//
//     const createDense = units => tf.layers.dense({ units, kernelInitializer: 'heNormal' });
//
//     const residualBlock = (x, diffusionStep, conditioner, residualChannels, dilation, numParams) => {
//         const dilatedConv = Conv1d(2 * residualChannels, 3, 1, 'same', dilation);
//         const diffusionProjection = createDense(residualChannels);
//         const conditionerProjection = Conv1d(2 * residualChannels, 1);
//         const outputProjection = Conv1d(2 * residualChannels, 1);
//
//         if (conditioner !== null) {
//             conditioner = conditioner.reshape([conditioner.shape[0], conditioner.shape[1], 1]);
//         }
//
//         diffusionStep = diffusionProjection.apply(diffusionStep).reshape([diffusionStep.shape[0], diffusionStep.shape[1], 1]);
//         let y = x.add(diffusionStep);
//
//         if (conditioner !== null) {
//             conditioner = conditionerProjection.apply(conditioner);
//             y = dilatedConv.apply(y).add(conditioner);
//         } else {
//             y = dilatedConv.apply(y);
//         }
//
//         const [gate, filter] = tf.split(y, 2, -1);
//         y = tf.sigmoid(gate).mul(tf.tanh(filter));
//
//         y = outputProjection.apply(y);
//         const [residual, skip] = tf.split(y, 2, -1);
//         return [x.add(residual).div(tf.sqrt(tf.scalar(2.0))), skip];
//     };
//
//     const inputConditioning = tf.input({ shape: [nParams] });
//     const inputDiffusionStep = tf.input({ shape: [512] });
//
//     let x = tf.randomNormal([1, winLength, 1]); // Start with noise
//
//     const inputProjection = Conv1d(params.residualChannels, 1);
//     x = inputProjection.apply(x);
//     x = tf.relu(x);
//
//     const diffusionEmbedding = getDiffusionEmbedding(params.noiseSchedule.length, params, inputDiffusionStep);
//
//     let skip = null;
//     for (let i = 0; i < params.residualLayers; i++) {
//         const dilation = Math.pow(2, i % params.dilationCycleLength);
//         const [newX, skipConnection] = residualBlock(x, diffusionEmbedding, inputConditioning, params.residualChannels, dilation, nParams);
//         x = newX;
//         skip = skip ? skip.add(skipConnection) : skipConnection;
//     }
//
//     const skipProjection = Conv1d(params.residualChannels, params.residualChannels, 1);
//     x = skip.div(tf.sqrt(tf.scalar(params.residualLayers)));
//     x = skipProjection.apply(x);
//     x = tf.relu(x);
//
//     const outputProjection = Conv1d(params.residualChannels, 1, 1);
//     x = outputProjection.apply(x);
//
//     const output = x;
//
//     return tf.model({
//         inputs: [inputConditioning, inputDiffusionStep],
//         outputs: output
//     });
// }

async function trainLoop(model, dataset, maxEpochs, batchSize, initialLearningRate) {
    let epoch = 0;
    const lossValues = [];

    const decayRate = parseFloat(document.getElementById('decayRate').value);
    const decaySteps = 100;

    function transpose(array) {
        return array[0].map((_, colIndex) => array.map(row => row[colIndex]));
    }

    function shuffle(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
        return array;
    }

    const conditioningValues = transpose(dataset.conditioningData);
    const nExamples = dataset.nExamples;

    while (maxEpochs === null || epoch < maxEpochs) {
        const indices = shuffle([...Array(nExamples).keys()]);

        model.optimizer.learningRate = initialLearningRate * Math.pow(decayRate, Math.floor(epoch / decaySteps));

        for (let batchIndex = 0; batchIndex < Math.ceil(nExamples / batchSize); batchIndex++) {
            const start = batchIndex * batchSize;
            const end = Math.min(start + batchSize, nExamples);
            const audioBatch = [];
            const conditioningBatch = [];

            for (let i = start; i < end; i++) {
                const idx = indices[i];
                const audioTensor = dataset.audio.slice([idx, 0], [1, dataset.winLength]);
                const conditioningTensor = tf.tensor(conditioningValues[idx]).expandDims(0);
                audioBatch.push(audioTensor);
                conditioningBatch.push(conditioningTensor);
            }

            const audioTensorBatch = tf.concat(audioBatch, 0);
            const conditioningTensorBatch = tf.concat(conditioningBatch, 0);

            // console.log(audioTensorBatch.shape, conditioningTensorBatch.shape)

            const loss = await model.fit(conditioningTensorBatch, audioTensorBatch, {
                epochs: 1,
                batchSize: batchSize,
                callbacks: [
                    tf.callbacks.earlyStopping({ monitor: 'loss', patience: 5 })
                ]
            });

            audioTensorBatch.dispose();
            conditioningTensorBatch.dispose();
            audioBatch.forEach(tensor => tensor.dispose());
            conditioningBatch.forEach(tensor => tensor.dispose());

            lossValues.push(loss.history.loss[0]);
            updatePlot(lossValues);

            document.getElementById('trainingResults').innerHTML = 'Loss: ' + loss.history.loss[0].toFixed(3);

            // console.log(`Epoch ${epoch}, Batch ${batchIndex}, Loss: ${loss.history.loss[0]}`);

            if (maxEpochs !== null && ++epoch >= maxEpochs) {
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
        marker: { color: 'red' },
    };

    const layout = {
        title: 'Training Loss',
        xaxis: { title: 'Epochs' },
        yaxis: { title: 'Loss' }
    };

    Plotly.newPlot('lossPlot', [trace], layout);
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
