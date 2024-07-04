const onnx = require('onnxruntime-node');
const path = require('path');
const Max = require('max-api');
const fs = require('fs');

let session;
let modelParams;
let conditioningNumbers = [];

// TESTING
const INFERENCE_NOISE_SCHEDULE = [0.0001, 0.0002, 0.001, 0.01, 0.2, 0.5];

const loadModel = async (model_folder) => {
    try {
        const modelPath = path.resolve(__dirname, `${model_folder}/weights.onnx`);
        const paramsPath = path.resolve(__dirname, `${model_folder}/model.txt`);

        session = await onnx.InferenceSession.create(modelPath);
        const paramsText = fs.readFileSync(paramsPath, 'utf8');
        modelParams = parseParams(paramsText);

        // TESTING
        modelParams.inference_noise_schedule = INFERENCE_NOISE_SCHEDULE;

        conditioningNumbers = Array(modelParams['Number of parameters']).fill(0);
        Max.outlet('dims', modelParams['Number of parameters']);
        Max.outlet('samps', modelParams['win_length']);

        Max.post("Model loaded successfully.");
        
        predict();

    } catch (error) {
        Max.post(`Error loading model: ${error.message}`);
    }
};

const parseParams = (paramsText) => {
    const lines = paramsText.split('\n');
    const params = {};
    
    lines.forEach(line => {
        const [key, value] = line.split(':').map(item => item.trim());
        
        if (value) {
            if (value.startsWith('[') && value.endsWith(']')) {
                // Parse array values
                params[key] = JSON.parse(value.replace(/'/g, '"'));
            } else if (!isNaN(value)) {
                // Parse numeric values
                params[key] = parseFloat(value);
            } else {
                // Parse string values
                params[key] = value;
            }
        }
    });
    return params;
};

const mul = (tensor, scalar) => {
    return new onnx.Tensor(
        tensor.type,
        tensor.data.map(v => v * scalar),
        tensor.dims
    );
};

const add = (tensor1, tensor2) => {
    return new onnx.Tensor(
        tensor1.type,
        tensor1.data.map((v, i) => v + tensor2.data[i]),
        tensor1.dims
    );
};

const sub = (tensor1, tensor2) => {
    return new onnx.Tensor(
        tensor1.type,
        tensor1.data.map((v, i) => v - tensor2.data[i]),
        tensor1.dims
    );
};

const clip = (tensor, min, max) => {
    return new onnx.Tensor(
        tensor.type,
        tensor.data.map(v => Math.max(min, Math.min(v, max))),
        tensor.dims
    );
};

const generateRandomNormal = (length) => {
    const buffer = new Float32Array(length);
    for (let i = 0; i < length; i += 2) {
        const u1 = Math.random();
        const u2 = Math.random();
        const r = Math.sqrt(-2.0 * Math.log(u1));
        const theta = 2.0 * Math.PI * u2;
        buffer[i] = r * Math.cos(theta);
        if (i + 1 < length) {
            buffer[i + 1] = r * Math.sin(theta);
        }
    }
    return buffer;
};

/* ---------------------------- PREDICTIONS ---------------------------- */

const predict = async () => {
    while (true) {
        try {
            const startTime = process.hrtime();

            const conditioningTensor = new onnx.Tensor('float32', new Float32Array(conditioningNumbers), [1, conditioningNumbers.length]);

            const trainingNoiseSchedule = modelParams.noise_schedule;
            const inferenceNoiseSchedule = modelParams.inference_noise_schedule || trainingNoiseSchedule;

            const talpha = trainingNoiseSchedule.map(value => 1 - value);
            const talphaCum = talpha.reduce((acc, value, index) => {
                if (index === 0) {
                    acc.push(value);
                } else {
                    acc.push(acc[acc.length - 1] * value);
                }
                return acc;
            }, []);

            const beta = inferenceNoiseSchedule;
            const alpha = beta.map(value => 1 - value);
            const alphaCum = alpha.reduce((acc, value, index) => {
                if (index === 0) {
                    acc.push(value);
                } else {
                    acc.push(acc[acc.length - 1] * value);
                }
                return acc;
            }, []);

            let T = [];
            for (let s = 0; s < inferenceNoiseSchedule.length; s++) {
                for (let t = 0; t < trainingNoiseSchedule.length - 1; t++) {
                    if (talphaCum[t + 1] <= alphaCum[s] && alphaCum[s] <= talphaCum[t]) {
                        let twiddle = (Math.sqrt(talphaCum[t]) - Math.sqrt(alphaCum[s])) / (Math.sqrt(talphaCum[t]) - Math.sqrt(talphaCum[t + 1]));
                        T.push(t + twiddle);
                        break;
                    }
                }
            }
            T = new Float32Array(T);

            const audioData = generateRandomNormal(1 * modelParams.win_length);
            let audioTensor = new onnx.Tensor('float32', audioData, [1, 1, modelParams.win_length]);

            for (let n = alpha.length - 1; n >= 0; n--) {
                const c1 = 1 / Math.sqrt(alpha[n]);
                const c2 = beta[n] / Math.sqrt(1 - alphaCum[n]);
                const t = new onnx.Tensor('int64', new BigInt64Array([BigInt(Math.floor(T[n]))]), [1]);

                const output = await session.run({
                    audio: audioTensor,
                    diffusion_step: t,
                    conditioning: conditioningTensor
                });

                const modelOutput = mul(output.output, c2);
                const intermediate = sub(audioTensor, modelOutput);
                audioTensor = mul(intermediate, c1);

                if (n > 0) {
                    const noise = generateRandomNormal(audioTensor.data.length);
                    const sigma = Math.sqrt((1.0 - alphaCum[n - 1]) / (1.0 - alphaCum[n]) * beta[n]);
                    const noiseTensor = mul(new onnx.Tensor('float32', noise, audioTensor.dims), sigma);
                    audioTensor = add(audioTensor, noiseTensor);
                }

                audioTensor = clip(audioTensor, -1.0, 1.0);
            }

            const outputFrame = overlapAdd(audioTensor.data);

            const endTime = process.hrtime(startTime);
            const elapsedTimeMs = Math.round(endTime[0] * 1000 + endTime[1] / 1e6);
            Max.outlet('time', elapsedTimeMs);

            Max.outlet('predictions', outputFrame);
        } catch (error) {
            Max.post(`Error running prediction: ${error.message}`);
            await new Promise(resolve => setTimeout(resolve, 1000)); // Wait for 1 second before retrying
        }
    }
};

/* ---------------------------- OLAD ---------------------------- */

var frameSize = 1024;
var hopSize = 512;
var bufferLength = frameSize * 2;
var buffer = new Float32Array(bufferLength);
var bufferPos = 0;

function overlapAdd(input) {
    const audioFrame = input instanceof Float32Array ? input : new Float32Array(input);

    let bufferZeroCrossing = findLastZeroCrossing(buffer, bufferPos, bufferLength);
    let inputZeroCrossing = findFirstZeroCrossing(audioFrame);

    for (let i = 0; i < frameSize; i++) {
        let bufferIndex = (bufferZeroCrossing + i) % bufferLength;
        let inputIndex = (inputZeroCrossing + i) % frameSize;
        buffer[bufferIndex] = audioFrame[inputIndex];
    }

    bufferPos = (bufferZeroCrossing + hopSize) % bufferLength;

    const outputFrame = new Float32Array(frameSize);
    for (let i = 0; i < frameSize; i++) {
        outputFrame[i] = buffer[(bufferPos + i) % bufferLength];
    }

    return outputFrame;
}

function findLastZeroCrossing(arr, startPos, length) {
    for (let i = 1; i < length; i++) {
        let currentIndex = (startPos - i + length) % length;
        let previousIndex = (currentIndex - 1 + length) % length;
        if (arr[previousIndex] * arr[currentIndex] <= 0) {
            return currentIndex;
        }
    }
    return startPos; // If no zero-crossing found, return the start position
}

function findFirstZeroCrossing(arr) {
    for (let i = 1; i < arr.length; i++) {
        if (arr[i-1] * arr[i] <= 0) {
            return i;
        }
    }
    return 0; // If no zero-crossing found, return the start of the array
}

/* ---------------------------- MAX HANDLERS ---------------------------- */

// Load the model
Max.addHandler('load', (model_folder) => {
    loadModel(model_folder);
});

// Predict
Max.addHandler('predict', async (...numbers) => {
    conditioningNumbers = numbers.map(Number);
});