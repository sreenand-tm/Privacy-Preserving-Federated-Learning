// ────────────────────────────────────────────────────────────────────────────
//  EDGE INFERENCE NODE — app.js
//  Runs the full pipeline locally in the browser:
//      Mic / File Upload → Resample → STFT → Mel Filterbank → Log → ONNX Inference
// ────────────────────────────────────────────────────────────────────────────

const CLASSES = [
    "Air Conditioner", "Car Horn", "Children Playing",
    "Dog Bark", "Drilling", "Engine Idling",
    "Gun Shot", "Jackhammer", "Siren", "Street Music"
];

// Must match config.py exactly
const TARGET_SR    = 22050;
const AUDIO_DUR    = 4.0;       // seconds
const N_FFT        = 2048;
const HOP_LENGTH   = 512;
const N_MELS       = 128;
const FMIN         = 0.0;
const FMAX         = TARGET_SR / 2.0;

let session;
const btnListen      = document.getElementById('btn-listen');
const statusText     = document.getElementById('status');
const resultBox      = document.getElementById('result-box');
const classNameText  = document.getElementById('class-name');
const confidenceText = document.getElementById('confidence');
const bars           = document.querySelectorAll('.bar');

const modelSelector  = document.getElementById('model-selector');
const dropzone       = document.getElementById('dropzone');
const fileInput      = document.getElementById('file-input');

// ───── Mel Filterbank (identical to preprocess.py) ──────────────────────────
function hzToMel(hz)  { return 2595.0 * Math.log10(1.0 + hz / 700.0); }
function melToHz(mel)  { return 700.0 * (Math.pow(10.0, mel / 2595.0) - 1.0); }

function createMelFilterbank() {
    const melMin = hzToMel(FMIN);
    const melMax = hzToMel(FMAX);
    const nFftBins = N_FFT / 2 + 1;  // 1025

    const melPoints = new Float64Array(N_MELS + 2);
    for (let i = 0; i < N_MELS + 2; i++) {
        melPoints[i] = melMin + (melMax - melMin) * i / (N_MELS + 1);
    }
    const hzPoints = melPoints.map(m => melToHz(m));

    const freqBins = new Float64Array(nFftBins);
    for (let j = 0; j < nFftBins; j++) {
        freqBins[j] = (TARGET_SR / 2.0) * j / (nFftBins - 1);
    }

    const fb = [];
    for (let i = 0; i < N_MELS; i++) {
        const row = new Float64Array(nFftBins);
        const lower  = hzPoints[i];
        const center = hzPoints[i + 1];
        const upper  = hzPoints[i + 2];
        for (let j = 0; j < nFftBins; j++) {
            const f = freqBins[j];
            if (f >= lower && f <= center && center !== lower) {
                row[j] = (f - lower) / (center - lower);
            } else if (f > center && f <= upper && upper !== center) {
                row[j] = (upper - f) / (upper - center);
            }
        }
        fb.push(row);
    }
    return fb;
}

const MEL_FILTERBANK = createMelFilterbank();

// ───── FFT (Cooley–Tukey radix‑2) ──────────────────────────────────────────
function fft(re, im) {
    const n = re.length;
    if (n <= 1) return;

    for (let i = 1, j = 0; i < n; i++) {
        let bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            [re[i], re[j]] = [re[j], re[i]];
            [im[i], im[j]] = [im[j], im[i]];
        }
    }

    for (let len = 2; len <= n; len *= 2) {
        const ang = -2 * Math.PI / len;
        const wRe = Math.cos(ang);
        const wIm = Math.sin(ang);
        for (let i = 0; i < n; i += len) {
            let curRe = 1, curIm = 0;
            for (let j = 0; j < len / 2; j++) {
                const a = i + j;
                const b = i + j + len / 2;
                const tRe = curRe * re[b] - curIm * im[b];
                const tIm = curRe * im[b] + curIm * re[b];
                re[b] = re[a] - tRe;
                im[b] = im[a] - tIm;
                re[a] += tRe;
                im[a] += tIm;
                const newCurRe = curRe * wRe - curIm * wIm;
                curIm = curRe * wIm + curIm * wRe;
                curRe = newCurRe;
            }
        }
    }
}

// ───── STFT → Power Spectrogram ─────────────────────────────────────────────
function computeSTFT(signal) {
    const nSamples = signal.length;
    const nFrames  = 1 + Math.floor((nSamples - N_FFT) / HOP_LENGTH);
    const nBins    = N_FFT / 2 + 1;

    const window = new Float64Array(N_FFT);
    for (let i = 0; i < N_FFT; i++) {
        window[i] = 0.5 * (1.0 - Math.cos(2.0 * Math.PI * i / N_FFT));
    }

    const power = [];
    for (let b = 0; b < nBins; b++) power.push(new Float64Array(nFrames));

    for (let t = 0; t < nFrames; t++) {
        const start = t * HOP_LENGTH;
        const re = new Float64Array(N_FFT);
        const im = new Float64Array(N_FFT);

        for (let i = 0; i < N_FFT; i++) {
            re[i] = signal[start + i] * window[i];
        }

        fft(re, im);

        for (let b = 0; b < nBins; b++) {
            power[b][t] = re[b] * re[b] + im[b] * im[b];
        }
    }
    return { power, nFrames };
}

// ───── Full Mel-Spectrogram Pipeline ────────────────────────────────────────
function computeMelSpectrogram(signal) {
    const { power, nFrames } = computeSTFT(signal);
    const nBins = N_FFT / 2 + 1;

    const melSpec = [];
    for (let m = 0; m < N_MELS; m++) {
        const row = new Float64Array(nFrames);
        for (let t = 0; t < nFrames; t++) {
            let sum = 0;
            for (let b = 0; b < nBins; b++) {
                sum += MEL_FILTERBANK[m][b] * power[b][t];
            }
            row[t] = Math.max(sum, 1e-10);
        }
        melSpec.push(row);
    }

    let globalMax = -Infinity;
    for (let m = 0; m < N_MELS; m++) {
        for (let t = 0; t < nFrames; t++) {
            melSpec[m][t] = 10.0 * Math.log10(melSpec[m][t]);
            if (melSpec[m][t] > globalMax) globalMax = melSpec[m][t];
        }
    }

    const isSilence = (globalMax < -38.0);
    if (isSilence) {
        for (let m = 0; m < N_MELS; m++) {
            for (let t = 0; t < nFrames; t++) {
                melSpec[m][t] = -80.0;
            }
        }
    } else {
        for (let m = 0; m < N_MELS; m++) {
            for (let t = 0; t < nFrames; t++) {
                melSpec[m][t] -= globalMax;
                if (melSpec[m][t] < -80.0) melSpec[m][t] = -80.0;
            }
        }
    }

    return { melSpec, nFrames, isSilence };
}

// ───── Resample from browser rate to 22050 Hz ───────────────────────────────
function resampleTo22050(audioBuffer) {
    const numSamples = Math.round(audioBuffer.duration * TARGET_SR);
    const offlineCtx = new OfflineAudioContext(1, numSamples, TARGET_SR);
    const source = offlineCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(offlineCtx.destination);
    source.start(0);
    return offlineCtx.startRendering();
}

// ───── Model Loading ────────────────────────────────────────────────────────
async function loadModel() {
    const selectedModelFile = modelSelector ? modelSelector.value : 'model_8clients.onnx';
    btnListen.disabled = true;
    btnListen.innerText = "Loading Model...";
    statusText.innerText = `Loading ${selectedModelFile}...`;
    try {
        session = await ort.InferenceSession.create(selectedModelFile);
        statusText.innerText = `Model Loaded (${selectedModelFile}). Ready.`;
        btnListen.innerText = "Listen & Classify";
        btnListen.disabled = false;
    } catch (e) {
        statusText.innerText = "Failed to load model: " + e.message;
        console.error(e);
    }
}

if (modelSelector) {
    modelSelector.addEventListener('change', loadModel);
}

// ───── Core Audio Inference Function ────────────────────────────────────────
async function runInferenceOnBuffer(rawBuffer) {
    try {
        statusText.innerText = "Resampling to 22050 Hz...";
        const resampledBuffer = await resampleTo22050(rawBuffer);
        let signal = resampledBuffer.getChannelData(0);

        const targetLength = TARGET_SR * AUDIO_DUR;  // 88200
        if (signal.length < targetLength) {
            const padded = new Float32Array(targetLength);
            padded.set(signal);
            signal = padded;
        } else if (signal.length > targetLength) {
            // Find window with peak energy to avoid lead-in silence
            let maxEnergy = -1.0;
            let bestStart = 0;
            const hop = Math.round(TARGET_SR * 0.5);
            for (let s = 0; s <= signal.length - targetLength; s += hop) {
                let energy = 0;
                for (let i = 0; i < targetLength; i += 16) {
                    energy += signal[s + i] * signal[s + i];
                }
                if (energy > maxEnergy) {
                    maxEnergy = energy;
                    bestStart = s;
                }
            }
            signal = signal.slice(bestStart, bestStart + targetLength);
        }

        statusText.innerText = "Computing Mel-Spectrogram...";
        const { melSpec, nFrames, isSilence } = computeMelSpectrogram(signal);

        if (isSilence) {
            classNameText.innerText = "Silence / Ambient Noise";
            confidenceText.innerText = "No active sound detected";
            resultBox.style.display = 'block';
            statusText.innerText = "Detection Complete (100% Local)";
            btnListen.innerText = "Listen & Classify";
            btnListen.disabled = false;
            return;
        }

        const tensorData = new Float32Array(N_MELS * nFrames);
        for (let m = 0; m < N_MELS; m++) {
            for (let t = 0; t < nFrames; t++) {
                tensorData[m * nFrames + t] = melSpec[m][t];
            }
        }

        const tensor = new ort.Tensor('float32', tensorData, [1, 1, N_MELS, nFrames]);

        statusText.innerText = "Running ONNX Inference...";
        const results = await session.run({ 'input': tensor });
        const output = results.output.data;

        let maxIdx = 0, maxVal = -Infinity;
        for (let i = 0; i < output.length; i++) {
            if (output[i] > maxVal) { maxVal = output[i]; maxIdx = i; }
        }
        let expSum = 0;
        for (let i = 0; i < output.length; i++) expSum += Math.exp(output[i] - maxVal);
        const confidence = (1.0 / expSum) * 100;

        classNameText.innerText = CLASSES[maxIdx];
        confidenceText.innerText = `Confidence: ${confidence.toFixed(1)}%`;
        resultBox.style.display = 'block';
        statusText.innerText = "Classification Complete (100% Local)";
        btnListen.innerText = "Listen & Classify";
        btnListen.disabled = false;

    } catch (err) {
        statusText.innerText = "Error: " + err.message;
        console.error(err);
        btnListen.innerText = "Listen & Classify";
        btnListen.disabled = false;
    }
}

// ───── File Upload / Drop Event Handlers ────────────────────────────────────
async function handleAudioFile(file) {
    if (!file || !file.type.startsWith('audio/')) {
        statusText.innerText = "Please select a valid audio file (.wav, .mp3)!";
        return;
    }

    statusText.innerText = `Decoding ${file.name}...`;
    btnListen.disabled = true;
    resultBox.style.display = 'none';

    try {
        const arrayBuffer = await file.arrayBuffer();
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const decodedBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        audioCtx.close();

        await runInferenceOnBuffer(decodedBuffer);
    } catch (err) {
        statusText.innerText = "Failed to decode audio file.";
        console.error(err);
        btnListen.disabled = false;
    }
}

if (dropzone && fileInput) {
    dropzone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleAudioFile(e.target.files[0]);
        }
    });

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleAudioFile(e.dataTransfer.files[0]);
        }
    });
}

// ───── Live Mic Recording ───────────────────────────────────────────────────
async function startListening() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });

        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const nativeSR = audioCtx.sampleRate;
        const recordDuration = AUDIO_DUR;
        const totalSamples = Math.round(nativeSR * recordDuration);

        const source    = audioCtx.createMediaStreamSource(stream);
        const processor = audioCtx.createScriptProcessor(4096, 1, 1);
        const chunks    = [];
        let collected   = 0;

        btnListen.disabled = true;
        btnListen.innerText = "Recording (4s)...";
        statusText.innerText = "Capturing audio locally...";
        resultBox.style.display = 'none';

        const animInterval = setInterval(() => {
            bars.forEach(bar => { bar.style.height = (Math.random() * 40 + 10) + 'px'; });
        }, 100);

        processor.onaudioprocess = (e) => {
            if (collected >= totalSamples) return;
            const data = e.inputBuffer.getChannelData(0);
            chunks.push(new Float32Array(data));
            collected += data.length;
        };

        source.connect(processor);
        processor.connect(audioCtx.destination);

        await new Promise(resolve => setTimeout(resolve, recordDuration * 1000 + 200));

        processor.disconnect();
        source.disconnect();
        clearInterval(animInterval);
        bars.forEach(bar => { bar.style.height = '5px'; });
        stream.getTracks().forEach(track => track.stop());

        const totalLen = chunks.reduce((s, c) => s + c.length, 0);
        const rawPCM = new Float32Array(totalLen);
        let offset = 0;
        for (const chunk of chunks) {
            rawPCM.set(chunk, offset);
            offset += chunk.length;
        }

        const rawBuffer = audioCtx.createBuffer(1, rawPCM.length, nativeSR);
        rawBuffer.copyToChannel(rawPCM, 0);
        audioCtx.close();

        await runInferenceOnBuffer(rawBuffer);

    } catch (err) {
        statusText.innerText = "Microphone error: " + err.message;
        console.error(err);
        btnListen.innerText = "Listen & Classify";
        btnListen.disabled = false;
    }
}

btnListen.addEventListener('click', startListening);

const btnTrain = document.getElementById('btn-train');

if (btnTrain) {
    btnTrain.addEventListener('click', async () => {
        btnTrain.disabled = true;
        btnTrain.innerText = "⏳ Running Backprop & Uploading...";
        statusText.innerText = `Auto-discovering PC Server & executing PyTorch Backprop...`;

        try {
            const resp = await fetch('/train_and_upload', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            });
            const data = await resp.json();
            if (data.success) {
                statusText.innerText = "✅ " + data.message;
            } else {
                statusText.innerText = "❌ " + data.message;
            }
        } catch (e) {
            statusText.innerText = "❌ Backprop/Upload error: " + e.message;
        } finally {
            btnTrain.disabled = false;
            btnTrain.innerText = "⚡ Trigger PyTorch Backprop & Send Δw to PC Server";
        }
    });
}

window.onload = loadModel;
