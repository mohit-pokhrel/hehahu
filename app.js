/**
 * Expression Scan — frontend application logic.
 *
 * Flow:
 *  1. Request camera permission and start the live preview.
 *  2. Load face-api.js models (tiny face detector + expression recognition).
 *  3. On "Start Scan": record 5 seconds of video via MediaRecorder while
 *     simultaneously sampling face-api.js expression predictions ~5x/sec.
 *  4. Aggregate the sampled predictions to find the dominant emotion.
 *  5. Display the result, then upload the recorded video to the backend
 *     as a backup (fire-and-forget relative to the UI, but status is shown).
 */

(() => {
  "use strict";

  // ---------------------------------------------------------------------
  // Config
  // ---------------------------------------------------------------------
  const MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model";
  const BACKEND_UPLOAD_URL = (window.EXPRESSION_APP_CONFIG && window.EXPRESSION_APP_CONFIG.backendUrl)
    ? window.EXPRESSION_APP_CONFIG.backendUrl
    : "http://localhost:4000/api/upload";

  const RECORD_DURATION_MS = 5000;
  const SAMPLE_INTERVAL_MS = 200; // ~5 samples/sec during the 5s window

  const EMOTION_EMOJI = {
    happy: "😄",
    sad: "😢",
    angry: "😠",
    surprised: "😲",
    fearful: "😨",
    disgusted: "🤢",
    neutral: "😐",
  };

  // ---------------------------------------------------------------------
  // DOM refs
  // ---------------------------------------------------------------------
  const videoEl = document.getElementById("video");
  const overlayCanvas = document.getElementById("overlay");

  const brandDot = document.getElementById("brandDot");
  const statusPill = document.getElementById("statusPill");
  const statusText = document.getElementById("statusText");

  const permissionState = document.getElementById("permissionState");
  const deniedState = document.getElementById("deniedState");
  const loadingState = document.getElementById("loadingState");
  const loadingTitle = document.getElementById("loadingTitle");
  const loadingSub = document.getElementById("loadingSub");
  const retryPermissionBtn = document.getElementById("retryPermissionBtn");
  const reloadBtn = document.getElementById("reloadBtn");

  const noFaceBadge = document.getElementById("noFaceBadge");
  const recBadge = document.getElementById("recBadge");
  const recTimer = document.getElementById("recTimer");

  const scanProgressFill = document.getElementById("scanProgressFill");

  const scanBtn = document.getElementById("scanBtn");
  const scanBtnLabel = document.getElementById("scanBtnLabel");
  const controlsHint = document.getElementById("controlsHint");

  const resultMeta = document.getElementById("resultMeta");
  const resultEmpty = document.getElementById("resultEmpty");
  const resultContent = document.getElementById("resultContent");
  const resultEmoji = document.getElementById("resultEmoji");
  const resultEmotion = document.getElementById("resultEmotion");
  const resultConfidence = document.getElementById("resultConfidence");
  const resultBreakdown = document.getElementById("resultBreakdown");
  const resultUpload = document.getElementById("resultUpload");
  const uploadStatusText = document.getElementById("uploadStatusText");

  // ---------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------
  let modelsLoaded = false;
  let cameraReady = false;
  let isScanning = false;
  let detectionLoopHandle = null;
  let stream = null;

  // ---------------------------------------------------------------------
  // Status helpers
  // ---------------------------------------------------------------------
  function setStatus(mode, label) {
    // mode: 'busy' | 'ready' | 'error' | 'idle'
    statusPill.classList.remove("busy", "ready", "error");
    if (mode !== "idle") statusPill.classList.add(mode);
    statusText.textContent = label;
    brandDot.classList.toggle("ready", mode === "ready");
  }

  function showOverlay(which) {
    [permissionState, deniedState, loadingState].forEach((el) => el.classList.add("hidden"));
    if (which) which.classList.remove("hidden");
  }

  function setHint(text) {
    controlsHint.textContent = text;
  }

  // ---------------------------------------------------------------------
  // Step 1: Camera permission + live preview
  // ---------------------------------------------------------------------
  async function startCamera() {
    showOverlay(null);
    setStatus("busy", "Requesting camera");
    setHint("Waiting for camera…");

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user", // front camera
          width: { ideal: 1280 },
          height: { ideal: 960 },
        },
        audio: false,
      });

      videoEl.srcObject = stream;
      await new Promise((resolve) => {
        videoEl.onloadedmetadata = () => resolve();
      });

      cameraReady = true;
      showOverlay(null);
      sizeOverlayCanvas();

      // Once camera is live, kick off model loading (if not already loaded)
      if (!modelsLoaded) {
        await loadModels();
      } else {
        finishReadyState();
      }
    } catch (err) {
      console.error("Camera permission error:", err);
      cameraReady = false;
      setStatus("error", "Camera blocked");
      if (err && (err.name === "NotAllowedError" || err.name === "PermissionDeniedError")) {
        showOverlay(deniedState);
        setHint("Camera permission denied.");
      } else if (err && err.name === "NotFoundError") {
        showOverlay(deniedState);
        document.querySelector("#deniedState .overlay-title").textContent = "No camera found";
        document.querySelector("#deniedState .overlay-sub").textContent =
          "This device doesn't seem to have an accessible camera.";
        setHint("No camera available.");
      } else {
        showOverlay(deniedState);
        setHint("Could not access camera.");
      }
    }
  }

  function sizeOverlayCanvas() {
    const rect = videoEl.getBoundingClientRect();
    overlayCanvas.width = rect.width;
    overlayCanvas.height = rect.height;
  }
  window.addEventListener("resize", () => {
    if (cameraReady) sizeOverlayCanvas();
  });

  // ---------------------------------------------------------------------
  // Step 2: Load face-api.js models
  // ---------------------------------------------------------------------
  async function loadModels() {
    showOverlay(loadingState);
    loadingTitle.textContent = "Loading models";
    loadingSub.textContent = "Downloading the facial expression recognition engine…";
    setStatus("busy", "Loading models");

    try {
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
        faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
      ]);
      modelsLoaded = true;
      finishReadyState();
    } catch (err) {
      console.error("Model loading error:", err);
      setStatus("error", "Model load failed");
      loadingTitle.textContent = "Couldn't load models";
      loadingSub.textContent = "Check your internet connection and reload the page.";
      // Keep loading overlay visible with the error message; offer reload via button injection
      const retry = document.createElement("button");
      retry.className = "btn-ghost";
      retry.textContent = "Reload Page";
      retry.onclick = () => window.location.reload();
      loadingState.appendChild(retry);
    }
  }

  function finishReadyState() {
    showOverlay(null);
    setStatus("ready", "Ready");
    scanBtn.disabled = false;
    setHint("Tap to start a 5-second scan");
    startLivePreviewDetection();
  }

  // ---------------------------------------------------------------------
  // Live preview: lightweight continuous detection just to show a
  // "no face detected" hint and a soft tracking box before recording.
  // ---------------------------------------------------------------------
  let lastFaceSeenAt = 0;

  function startLivePreviewDetection() {
    const ctx = overlayCanvas.getContext("2d");
    const tinyOptions = new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 });

    async function loop() {
      if (videoEl.readyState >= 2 && modelsLoaded) {
        try {
          const detection = await faceapi.detectSingleFace(videoEl, tinyOptions);
          ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

          if (detection) {
            lastFaceSeenAt = Date.now();
            noFaceBadge.classList.add("hidden");
            drawFaceBox(ctx, detection.box);
          } else {
            // Only show the "no face" hint after a brief grace period to avoid flicker
            if (Date.now() - lastFaceSeenAt > 1200) {
              noFaceBadge.classList.remove("hidden");
            }
          }
        } catch (e) {
          // non-fatal; keep looping
        }
      }
      detectionLoopHandle = requestAnimationFrame(loop);
    }
    loop();
  }

  function drawFaceBox(ctx, box) {
    const scaleX = overlayCanvas.width / videoEl.videoWidth;
    const scaleY = overlayCanvas.height / videoEl.videoHeight;
    const x = box.x * scaleX;
    const y = box.y * scaleY;
    const w = box.width * scaleX;
    const h = box.height * scaleY;

    ctx.strokeStyle = "rgba(94, 230, 166, 0.85)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    if (ctx.roundRect) {
      ctx.roundRect(x, y, w, h, 10);
    } else {
      ctx.rect(x, y, w, h);
    }
    ctx.stroke();
  }

  // ---------------------------------------------------------------------
  // Step 3: Scan = MediaRecorder (5s) + sampled expression detection
  // ---------------------------------------------------------------------
  function pickMimeType() {
    const candidates = [
      "video/webm;codecs=vp9",
      "video/webm;codecs=vp8",
      "video/webm",
      "video/mp4",
    ];
    for (const type of candidates) {
      if (window.MediaRecorder && MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(type)) {
        return type;
      }
    }
    return ""; // let the browser pick a default
  }

  async function startScan() {
    if (isScanning || !cameraReady || !modelsLoaded) return;
    isScanning = true;

    scanBtn.classList.add("recording");
    scanBtn.disabled = true;
    scanBtnLabel.textContent = "Stop";
    setHint("Recording…");
    setStatus("busy", "Scanning");
    recBadge.classList.remove("hidden");
    scanProgressFill.style.width = "0%";

    // Reset result board to a fresh "in progress" look
    resultEmpty.classList.add("hidden");
    resultContent.classList.add("hidden");
    resultMeta.textContent = "scanning…";

    const samples = []; // array of expression score objects
    const detectOptions = new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 });

    // --- Set up MediaRecorder ---
    const recordedChunks = [];
    const mimeType = pickMimeType();
    let mediaRecorder;
    try {
      mediaRecorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);
    } catch (err) {
      console.error("MediaRecorder init failed:", err);
      mediaRecorder = null;
    }

    if (mediaRecorder) {
      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) recordedChunks.push(e.data);
      };
      mediaRecorder.start();
    }

    // --- Sampling loop for expressions, every SAMPLE_INTERVAL_MS ---
    const sampleTimer = setInterval(async () => {
      try {
        const detection = await faceapi
          .detectSingleFace(videoEl, detectOptions)
          .withFaceExpressions();
        if (detection && detection.expressions) {
          samples.push(detection.expressions);
        }
      } catch (e) {
        // ignore individual frame errors
      }
    }, SAMPLE_INTERVAL_MS);

    // --- Progress bar update ---
    const startTime = Date.now();
    const progressTimer = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const pct = Math.min(100, (elapsed / RECORD_DURATION_MS) * 100);
      scanProgressFill.style.width = pct + "%";
      const remaining = Math.max(0, (RECORD_DURATION_MS - elapsed) / 1000);
      recTimer.textContent = remaining.toFixed(1) + "s";
    }, 100);

    // --- Wait exactly 5 seconds ---
    await new Promise((resolve) => setTimeout(resolve, RECORD_DURATION_MS));

    clearInterval(sampleTimer);
    clearInterval(progressTimer);
    scanProgressFill.style.width = "100%";
    recTimer.textContent = "0.0s";

    let videoBlob = null;
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      videoBlob = await new Promise((resolve) => {
        mediaRecorder.onstop = () => {
          resolve(new Blob(recordedChunks, { type: mediaRecorder.mimeType || "video/webm" }));
        };
        mediaRecorder.stop();
      });
    }

    // --- Wrap up UI state ---
    recBadge.classList.add("hidden");
    scanBtn.classList.remove("recording");
    scanBtnLabel.textContent = "Start Scan";
    scanBtn.disabled = false;
    isScanning = false;
    setStatus("ready", "Ready");

    // --- Analyze samples ---
    const result = aggregateExpressions(samples);

    if (!result) {
      setHint("No face detected during scan — try again");
      resultMeta.textContent = "no face detected";
      resultEmpty.classList.remove("hidden");
      resultEmpty.querySelector("p").textContent =
        "We couldn't detect a face clearly enough during that scan. Make sure your face is centered and well-lit, then try again.";
      return;
    }

    setHint("Tap to scan again");
    displayResult(result);

    // --- Upload backup video to backend ---
    if (videoBlob && videoBlob.size > 0) {
      uploadVideo(videoBlob, result);
    } else {
      resultUpload.classList.remove("hidden");
      resultUpload.classList.add("error");
      uploadStatusText.textContent = "Recording unavailable — nothing to back up.";
    }
  }

  // ---------------------------------------------------------------------
  // Aggregate per-frame expression samples into a final result
  // ---------------------------------------------------------------------
  function aggregateExpressions(samples) {
    if (!samples || samples.length === 0) return null;

    const totals = {};
    let count = 0;

    samples.forEach((expr) => {
      // expr is a face-api FaceExpressions object: { neutral, happy, sad, angry, fearful, disgusted, surprised }
      Object.keys(expr).forEach((key) => {
        if (typeof expr[key] !== "number") return;
        totals[key] = (totals[key] || 0) + expr[key];
      });
      count++;
    });

    if (count === 0) return null;

    const averages = {};
    Object.keys(totals).forEach((key) => {
      averages[key] = totals[key] / count;
    });

    // Dominant = highest average confidence across the sampling window
    const sorted = Object.entries(averages).sort((a, b) => b[1] - a[1]);
    const [topEmotion, topScore] = sorted[0];

    return {
      emotion: topEmotion,
      confidence: topScore,
      breakdown: sorted, // array of [emotion, score] sorted descending
      sampleCount: count,
    };
  }

  // ---------------------------------------------------------------------
  // Render result board
  // ---------------------------------------------------------------------
  function displayResult(result) {
    resultContent.classList.remove("hidden");
    resultEmpty.classList.add("hidden");
    resultUpload.classList.remove("hidden", "success", "error");
    uploadStatusText.textContent = "Backing up recording…";

    resultMeta.textContent = `${result.sampleCount} samples`;
    resultEmoji.textContent = EMOTION_EMOJI[result.emotion] || "🙂";
    resultEmotion.textContent = result.emotion;
    resultConfidence.textContent = `${Math.round(result.confidence * 100)}% confidence`;

    resultBreakdown.innerHTML = "";
    result.breakdown.forEach(([emotion, score]) => {
      const row = document.createElement("div");
      row.className = "breakdown-row";

      const label = document.createElement("span");
      label.className = "breakdown-label";
      label.textContent = emotion;

      const track = document.createElement("div");
      track.className = "breakdown-track";
      const fill = document.createElement("div");
      fill.className = "breakdown-fill";
      fill.style.width = "0%";
      track.appendChild(fill);

      const pct = document.createElement("span");
      pct.className = "breakdown-pct";
      pct.textContent = Math.round(score * 100) + "%";

      row.appendChild(label);
      row.appendChild(track);
      row.appendChild(pct);
      resultBreakdown.appendChild(row);

      // animate in
      requestAnimationFrame(() => {
        fill.style.width = Math.round(score * 100) + "%";
      });
    });
  }

  // ---------------------------------------------------------------------
  // Step 4: Upload recorded video to backend as a backup
  // ---------------------------------------------------------------------
  async function uploadVideo(blob, result) {
    try {
      const formData = new FormData();
      const ext = blob.type.includes("mp4") ? "mp4" : "webm";
      formData.append("video", blob, `recording.${ext}`);
      formData.append("emotion", result.emotion);
      formData.append("confidence", String(result.confidence));

      const response = await fetch(BACKEND_UPLOAD_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error(`Upload failed (${response.status})`);
      const data = await response.json();

      resultUpload.classList.add("success");
      uploadStatusText.textContent = `Backed up as ${data.filename || "recording"}`;
    } catch (err) {
      console.error("Video upload error:", err);
      resultUpload.classList.add("error");
      uploadStatusText.textContent = "Backup failed — backend unreachable.";
    }
  }

  // ---------------------------------------------------------------------
  // Wire up buttons
  // ---------------------------------------------------------------------
  scanBtn.addEventListener("click", () => {
    if (!isScanning) startScan();
  });
  retryPermissionBtn.addEventListener("click", startCamera);
  reloadBtn.addEventListener("click", () => window.location.reload());

  // ---------------------------------------------------------------------
  // Boot
  // ---------------------------------------------------------------------
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus("error", "Unsupported browser");
    showOverlay(deniedState);
    document.querySelector("#deniedState .overlay-title").textContent = "Browser not supported";
    document.querySelector("#deniedState .overlay-sub").textContent =
      "Your browser doesn't support camera access. Try the latest Chrome, Edge, or Safari.";
  } else {
    startCamera();
  }
})();
