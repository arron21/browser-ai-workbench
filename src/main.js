const DEFAULT_SYS_PROMPT =
  "You are a helpful, concise AI assistant running locally inside the user's web browser via WebGPU.";

const AGENT_TOOL_INSTRUCTIONS = `
You have access to browser tools. When the user asks to change the theme, show a metric/KPI card, or generate/draw an image, include the exact tag in your reply:
- To switch UI theme (midnight, neon, emerald, sunset): [TOOL:set_theme(neon)]
- To display a KPI metric card: [TOOL:render_kpi(Inference Cost|$0.00|-100% vs Cloud)]
- To generate an image with Janus-Pro-1B: [TOOL:imagine(detailed visual prompt)]`;

const JANUS_MODEL_ID = "onnx-community/Janus-Pro-1B-ONNX";

const MODES = {
  text: {
    modelKey: "text",
    placeholder: "Ask a question or ask the agent to switch themes / generate an image...",
    presets: [
      "Switch the UI theme to neon and explain why WebGPU is fast in 2 sentences.",
      "Show a KPI card for Local Browser AI Latency and explain zero-server privacy.",
      "Use the imagine tool to draw a futuristic solarpunk library with floating plants.",
      "Write a short JavaScript function to compute cosine similarity between two vectors.",
    ],
  },
  image: {
    modelKey: "janus",
    placeholder: "Describe an image to synthesize locally on your GPU...",
    presets: [
      "A cute red panda wearing tiny astronaut goggles on Mars, digital art, vibrant lighting",
      "A cozy cyberpunk ramen shop in the rain at night, neon reflections, studio ghibli style",
      "An isometric miniature crystal island floating in the clouds at sunset, 3d render",
    ],
  },
  vision: {
    modelKey: "janus",
    placeholder: "Ask a question about the selected image (OCR, LaTeX, description)...",
    presets: [
      "Convert this mathematical formula into LaTeX code and explain what it solves.",
      "Describe the subject, colors, and composition of this image in detail.",
      "Extract any visible text or symbols from the image.",
    ],
  },
  chain: {
    modelKey: "chain",
    placeholder: "Enter a concept, theme, or style direction for the selected multi-model chain...",
    presets: [
      "A clockwork owl perched in a glowing bioluminescent forest",
      "A retro 1970s sci-fi poster of an ocean city on Europa",
      "Transform the scene into a vibrant synthwave neon cyber-city at midnight",
    ],
  },
};

// State
let currentMode = "text";
let isBusy = false;
let jobCounter = 0;

let selectedTextModelId = "HuggingFaceTB/SmolLM2-360M-Instruct";
let selectedDtype = "auto";

// Track what is currently loaded inside the Worker
let loadedTextModelId = null;
let loadedTextDtype = null;
let loadedJanusModelId = null;

let currentVisionImageUrl =
  "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/quadratic_formula.png";
let currentVisionLabel = "Sample: Quadratic Formula";

let chatHistory = [];
const pendingJobs = new Map();

// DOM Elements
const modeButtons = {
  text: document.getElementById("mode-text-btn"),
  image: document.getElementById("mode-image-btn"),
  vision: document.getElementById("mode-vision-btn"),
  chain: document.getElementById("mode-chain-btn"),
};
const modeTextTag = document.getElementById("mode-text-tag");

const webgpuBadge = document.getElementById("webgpu-badge");
const webgpuText = document.getElementById("webgpu-text");
const toggleInspectorBtn = document.getElementById("toggle-inspector-btn");
const inspectorSidebar = document.getElementById("inspector-sidebar");

// HF Model Selector Elements
const hfModelSelect = document.getElementById("hf-model-select");
const hfDtypeSelect = document.getElementById("hf-dtype-select");
const activeModelIdEl = document.getElementById("active-model-id");
const modelStatusPill = document.getElementById("model-status-pill");
const browseHubBtn = document.getElementById("browse-hub-btn");
const preloadBtn = document.getElementById("preload-btn");
const clearBtn = document.getElementById("clear-btn");

// Live HF Hub Search Drawer Elements
const hfHubDrawer = document.getElementById("hf-hub-drawer");
const hfSearchInput = document.getElementById("hf-search-input");
const hfSearchBtn = document.getElementById("hf-search-btn");
const hfUseCustomBtn = document.getElementById("hf-use-custom-btn");
const closeHubDrawerBtn = document.getElementById("close-hub-drawer-btn");
const hfHubResults = document.getElementById("hf-hub-results");

const visionToolbar = document.getElementById("vision-toolbar");
const visionPreviewImg = document.getElementById("vision-preview-img");
const visionImgLabel = document.getElementById("vision-img-label");
const visionFileInput = document.getElementById("vision-file-input");

const chainToolbar = document.getElementById("chain-toolbar");
const chainSelect = document.getElementById("chain-select");
const chainStepsVisualizer = document.getElementById("chain-steps-visualizer");

const downloadPanel = document.getElementById("download-panel");
const downloadHeaderText = document.getElementById("download-header-text");
const downloadItems = document.getElementById("download-items");
const chatFeed = document.getElementById("chat-feed");
const welcomeCard = document.getElementById("welcome-card");
const presetChips = document.getElementById("preset-chips");

const promptForm = document.getElementById("prompt-form");
const promptInput = document.getElementById("prompt-input");
const sendBtn = document.getElementById("send-btn");
const stopBtn = document.getElementById("stop-btn");

// Inspector Controls & Telemetry Elements
const activeStageBadge = document.getElementById("active-stage-badge");
const metricTtft = document.getElementById("metric-ttft");
const metricTps = document.getElementById("metric-tps");
const metricTokens = document.getElementById("metric-tokens");
const metricLatency = document.getElementById("metric-latency");

const agentToolsToggle = document.getElementById("agent-tools-toggle");
const paramTemp = document.getElementById("param-temp");
const valTemp = document.getElementById("val-temp");
const paramTopP = document.getElementById("param-top-p");
const valTopP = document.getElementById("val-top-p");
const paramMaxTokens = document.getElementById("param-max-tokens");
const valMaxTokens = document.getElementById("val-max-tokens");
const systemPromptInput = document.getElementById("system-prompt-input");
const resetSyspromptBtn = document.getElementById("reset-sysprompt-btn");

const fileProgressRows = new Map();

// Create Worker
const worker = new Worker(new URL("./worker.js", import.meta.url), {
  type: "module",
});

/**
 * Helper to get short display name for a Hugging Face repo ID
 */
function getShortModelName(repoId) {
  if (!repoId) return "LLM";
  const parts = repoId.split("/");
  return (parts[1] || parts[0]).replace(/-Instruct|-ONNX|-q4f16/gi, "");
}

/**
 * Return current hyperparameter config from the Inspector sidebar
 */
function getGenerationConfig() {
  return {
    temperature: parseFloat(paramTemp.value),
    top_p: parseFloat(paramTopP.value),
    max_new_tokens: parseInt(paramMaxTokens.value, 10),
  };
}

/**
 * Build dynamic step labels for the Model Chaining toolbar using the selected HF model
 */
function getChainSteps(chainType) {
  const shortLlm = getShortModelName(selectedTextModelId);
  const map = {
    expand_to_image: [
      `1. ${shortLlm} (Prompt Expansion)`,
      "2. Janus-Pro-1B (WebGPU Image Synthesis)",
    ],
    image_to_critic: [
      "1. Janus-Pro-1B (Generate Image)",
      "2. Janus-Pro-1B Vision (Inspect Generated Image)",
      `3. ${shortLlm} (Creative Micro-Story)`,
    ],
    vision_to_remix: [
      "1. Janus-Pro-1B Vision (Analyze Input Image)",
      `2. ${shortLlm} (Reimagine Prompt)`,
      "3. Janus-Pro-1B (Render Remixed Image)",
    ],
  };
  return map[chainType] || [];
}

/**
 * Update the visual step nodes in the Model Chaining toolbar
 */
function renderChainSteps(activeIndex = -1, completedUpTo = -1) {
  const steps = getChainSteps(chainSelect.value);
  chainStepsVisualizer.innerHTML = "";

  steps.forEach((label, idx) => {
    const node = document.createElement("div");
    node.className = "chain-step-node";
    if (idx === activeIndex) node.classList.add("active");
    if (idx <= completedUpTo) node.classList.add("done");
    node.textContent = label;
    node.dataset.tooltipTitle = `Pipeline Stage ${idx + 1}: ${label}`;
    node.dataset.tooltipBody =
      "Each stage runs sequentially inside the Web Worker. The output of this stage is automatically passed as the input context to the next stage.";
    chainStepsVisualizer.appendChild(node);

    if (idx < steps.length - 1) {
      const arrow = document.createElement("span");
      arrow.className = "chain-arrow";
      arrow.textContent = "➔";
      chainStepsVisualizer.appendChild(arrow);
    }
  });
}

/**
 * Render preset prompt chips for the active mode
 */
function renderPresets() {
  presetChips.innerHTML = "";
  for (const presetText of MODES[currentMode].presets) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip-btn";
    btn.textContent = presetText;
    btn.dataset.tooltipTitle = "One-Click Starter Experiment";
    btn.dataset.tooltipBody =
      "Populates the prompt box and immediately dispatches this inference job to the Web Worker.";
    btn.addEventListener("click", () => {
      promptInput.value = presetText;
      promptForm.requestSubmit();
    });
    presetChips.appendChild(btn);
  }
}

/**
 * Check if the currently active mode's model is already loaded in the Worker
 */
function isCurrentModelLoaded() {
  if (currentMode === "text") {
    return (
      loadedTextModelId === selectedTextModelId &&
      (selectedDtype === "auto" || loadedTextDtype === selectedDtype)
    );
  }
  if (currentMode === "image" || currentMode === "vision") {
    return loadedJanusModelId === JANUS_MODEL_ID;
  }
  if (currentMode === "chain") {
    return loadedTextModelId === selectedTextModelId && loadedJanusModelId === JANUS_MODEL_ID;
  }
  return false;
}

/**
 * Synchronize mode buttons, HF model selector, toolbars, and status indicators
 */
function updateModeUI() {
  for (const [modeKey, btn] of Object.entries(modeButtons)) {
    btn.classList.toggle("active", modeKey === currentMode);
  }

  modeTextTag.textContent = getShortModelName(selectedTextModelId);

  const modeCfg = MODES[currentMode];
  promptInput.placeholder = modeCfg.placeholder;

  // Show HF Text LLM dropdown in Text & Chain modes; show fixed Janus badge in pure Image/Vision modes
  const usesTextSelector = currentMode === "text" || currentMode === "chain";
  hfModelSelect.classList.toggle("hidden", !usesTextSelector);
  hfDtypeSelect.classList.toggle("hidden", !usesTextSelector);
  browseHubBtn.classList.toggle("hidden", !usesTextSelector);
  activeModelIdEl.classList.toggle("hidden", usesTextSelector);

  if (!usesTextSelector) {
    activeModelIdEl.textContent = JANUS_MODEL_ID;
    hfHubDrawer.classList.add("hidden");
  }

  const showVisionBar =
    currentMode === "vision" ||
    (currentMode === "chain" && chainSelect.value === "vision_to_remix");
  visionToolbar.classList.toggle("hidden", !showVisionBar);
  chainToolbar.classList.toggle("hidden", currentMode !== "chain");

  if (currentMode === "chain") {
    renderChainSteps();
  }

  if (isCurrentModelLoaded()) {
    modelStatusPill.textContent = "Ready in GPU / Browser Cache";
    modelStatusPill.className = "pill pill-ready";
    preloadBtn.textContent = "Loaded ✓";
    preloadBtn.disabled = true;
  } else {
    modelStatusPill.textContent = "Not Loaded (CDN Cache)";
    modelStatusPill.className = "pill pill-idle";
    preloadBtn.textContent = "⬇ Download / Load";
    preloadBtn.disabled = isBusy;
  }

  renderPresets();
}

/**
 * Add or select a custom Hugging Face model ID in the <select> dropdown
 */
function selectOrAddHfModel(repoId) {
  const cleanId = repoId.trim();
  if (!cleanId) return;

  selectedTextModelId = cleanId;

  let existingOption = Array.from(hfModelSelect.options).find(
    (opt) => opt.value === cleanId
  );

  if (!existingOption) {
    existingOption = document.createElement("option");
    existingOption.value = cleanId;
    existingOption.textContent = `${cleanId} (Custom HF Hub)`;
    hfModelSelect.insertBefore(existingOption, hfModelSelect.firstChild);
  }

  hfModelSelect.value = cleanId;
  hfHubDrawer.classList.add("hidden");
  updateModeUI();
}

/**
 * Search live Transformers.js compatible text-generation models on Hugging Face Hub API
 */
async function searchHuggingFaceHub(query = "") {
  hfHubResults.innerHTML = `<span class="hub-hint">Searching Hugging Face Hub API...</span>`;
  try {
    const params = new URLSearchParams({
      library: "transformers.js",
      pipeline_tag: "text-generation",
      sort: "downloads",
      direction: "-1",
      limit: "12",
    });
    if (query.trim()) {
      params.set("search", query.trim());
    }

    const res = await fetch(`https://huggingface.co/api/models?${params.toString()}`);
    if (!res.ok) {
      throw new Error(`Hub API HTTP ${res.status}`);
    }
    const models = await res.json();

    if (!Array.isArray(models) || models.length === 0) {
      hfHubResults.innerHTML = `<span class="hub-hint">No matching transformers.js text-generation models found. You can still paste any repo ID and click "Use Repo ID".</span>`;
      return;
    }

    hfHubResults.innerHTML = "";
    for (const m of models) {
      const card = document.createElement("div");
      card.className = "hub-model-card";

      const idEl = document.createElement("div");
      idEl.className = "hub-card-id";
      idEl.textContent = m.id;

      const metaEl = document.createElement("div");
      metaEl.className = "hub-card-meta";
      const downloads = (m.downloads || 0).toLocaleString();
      const likes = m.likes || 0;
      metaEl.innerHTML = `<span>⬇ ${downloads}</span><span>❤️ ${likes}</span>`;

      card.appendChild(idEl);
      card.appendChild(metaEl);

      card.addEventListener("click", () => {
        selectOrAddHfModel(m.id);
        // Trigger immediate download/load from CDN
        worker.postMessage({
          type: "load_model",
          mode: "text",
          modelId: selectedTextModelId,
          dtype: selectedDtype,
        });
      });

      hfHubResults.appendChild(card);
    }
  } catch (err) {
    hfHubResults.innerHTML = `<span class="hub-hint">⚠️ Error querying Hugging Face API: ${err.message}</span>`;
  }
}

/**
 * Update Live Telemetry Sidebar
 */
function updateTelemetry({ stage, ttftMs, tps, numTokens, totalTimeSec }) {
  if (stage !== undefined) {
    activeStageBadge.textContent = stage;
    activeStageBadge.classList.toggle("running", stage !== "Idle" && stage !== "Complete" && stage !== "Ready");
  }
  if (ttftMs !== undefined && ttftMs !== null) {
    metricTtft.textContent = `${ttftMs} ms`;
  }
  if (tps !== undefined && tps !== null) {
    metricTps.textContent = `${tps} tok/s`;
  }
  if (numTokens !== undefined && numTokens !== null) {
    metricTokens.textContent = `${numTokens}`;
  }
  if (totalTimeSec !== undefined && totalTimeSec !== null) {
    metricLatency.textContent = `${totalTimeSec} s`;
  }
}

/**
 * Append a message bubble to the chat feed
 */
function appendMessage(role, text, label = "", imageThumbUrl = null) {
  if (welcomeCard) {
    welcomeCard.classList.add("hidden");
  }

  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const roleHeader = document.createElement("div");
  roleHeader.className = "message-role";
  roleHeader.textContent = label || (role === "user" ? "You" : "AI Assistant");

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";

  if (imageThumbUrl) {
    const thumb = document.createElement("img");
    thumb.src = imageThumbUrl;
    thumb.className = "user-vision-thumb";
    thumb.alt = "Attached vision input";
    bubble.appendChild(thumb);
  }

  const textNode = document.createElement("div");
  textNode.textContent = text;
  bubble.appendChild(textNode);

  const meta = document.createElement("div");
  meta.className = "message-meta";

  wrapper.appendChild(roleHeader);
  wrapper.appendChild(bubble);
  wrapper.appendChild(meta);
  chatFeed.appendChild(wrapper);
  chatFeed.scrollTop = chatFeed.scrollHeight;

  return { wrapper, bubble, textNode, meta };
}

/**
 * Render an image output card inside a bubble, with "Download PNG" and "Send to Vision QA" buttons
 */
function renderImageInBubble(bubbleEl, blob, promptText) {
  const imageUrl = URL.createObjectURL(blob);
  bubbleEl.innerHTML = "";

  const card = document.createElement("div");
  card.className = "image-gen-card";

  const img = document.createElement("img");
  img.src = imageUrl;
  img.alt = promptText;

  const actions = document.createElement("div");
  actions.className = "image-card-actions";

  const downloadLink = document.createElement("a");
  downloadLink.href = imageUrl;
  downloadLink.download = `janus-${Date.now()}.png`;
  downloadLink.className = "download-img-link";
  downloadLink.textContent = "⬇ Download PNG";

  const inspectVisionBtn = document.createElement("button");
  inspectVisionBtn.type = "button";
  inspectVisionBtn.className = "action-chip-btn";
  inspectVisionBtn.textContent = "👁️ Send to Vision QA";
  inspectVisionBtn.addEventListener("click", () => {
    currentVisionImageUrl = imageUrl;
    currentVisionLabel = `Generated: "${promptText.slice(0, 32)}..."`;
    visionPreviewImg.src = imageUrl;
    visionImgLabel.textContent = currentVisionLabel;
    currentMode = "vision";
    updateModeUI();
    promptInput.value = "Describe this image in detail and critique its visual style.";
    promptInput.focus();
  });

  actions.appendChild(downloadLink);
  actions.appendChild(inspectVisionBtn);
  card.appendChild(img);
  card.appendChild(actions);
  bubbleEl.appendChild(card);

  return imageUrl;
}

/**
 * Update or create a download progress row for a model file from Hugging Face CDN
 */
function updateDownloadRow(file, progress) {
  if (!file) return;
  downloadPanel.classList.remove("hidden");

  let row = fileProgressRows.get(file);
  if (!row) {
    const itemEl = document.createElement("div");
    itemEl.className = "download-item";

    const nameEl = document.createElement("span");
    nameEl.className = "download-filename";
    nameEl.textContent = file;

    const trackEl = document.createElement("div");
    trackEl.className = "progress-track";

    const fillEl = document.createElement("div");
    fillEl.className = "progress-fill";
    trackEl.appendChild(fillEl);

    const pctEl = document.createElement("span");
    pctEl.className = "download-pct";
    pctEl.textContent = "0%";

    itemEl.appendChild(nameEl);
    itemEl.appendChild(trackEl);
    itemEl.appendChild(pctEl);
    downloadItems.appendChild(itemEl);

    row = { itemEl, fillEl, pctEl };
    fileProgressRows.set(file, row);
  }

  const pct = Math.min(100, Math.round(progress || 0));
  row.fillEl.style.width = `${pct}%`;
  row.pctEl.textContent = `${pct}%`;
}

/**
 * Dispatch a worker job and return a Promise that resolves when generation completes
 */
function runWorkerJob(type, payload, hooks = {}) {
  jobCounter += 1;
  const jobId = `job_${jobCounter}_${Date.now()}`;

  return new Promise((resolve, reject) => {
    pendingJobs.set(jobId, { resolve, reject, hooks });
    worker.postMessage({ type, jobId, ...payload });
  });
}

// Worker Message Listener
worker.addEventListener("message", (event) => {
  const data = event.data;

  switch (data.status) {
    case "webgpu_status": {
      webgpuBadge.classList.remove("checking");
      webgpuBadge.classList.add(data.supported ? "supported" : "unsupported");
      webgpuText.textContent = data.message;
      break;
    }

    case "loading_start": {
      modelStatusPill.textContent = `Downloading ${getShortModelName(data.modelId)}...`;
      modelStatusPill.className = "pill pill-loading";
      preloadBtn.disabled = true;
      preloadBtn.textContent = "Loading...";
      downloadHeaderText.textContent = `Downloading ${data.modelId} from Hugging Face CDN (cached in browser)...`;
      updateTelemetry({ stage: `Loading ${getShortModelName(data.modelId)}` });
      break;
    }

    case "model_progress": {
      if (data.status === "progress" || typeof data.progress === "number") {
        updateDownloadRow(data.file, data.progress);
      } else if (data.status === "done" && data.file) {
        updateDownloadRow(data.file, 100);
      }
      break;
    }

    case "model_ready": {
      if (data.modelKey === "text") {
        loadedTextModelId = data.modelId;
        loadedTextDtype = data.dtype;
      } else if (data.modelKey === "janus") {
        loadedJanusModelId = data.modelId;
      }
      downloadPanel.classList.add("hidden");
      fileProgressRows.clear();
      downloadItems.innerHTML = "";
      updateModeUI();
      updateTelemetry({ stage: "Ready" });
      break;
    }

    case "generation_start": {
      const job = pendingJobs.get(data.jobId);
      if (job?.hooks?.onStart) job.hooks.onStart(data);
      break;
    }

    case "token": {
      updateTelemetry({
        ttftMs: data.ttftMs,
        tps: data.tps,
        numTokens: data.numTokens,
      });
      const job = pendingJobs.get(data.jobId);
      if (job?.hooks?.onToken) job.hooks.onToken(data);
      break;
    }

    case "image_progress": {
      updateTelemetry({
        ttftMs: data.ttftMs,
        tps: data.tps,
        numTokens: `${data.step}/${data.total}`,
      });
      const job = pendingJobs.get(data.jobId);
      if (job?.hooks?.onImageProgress) job.hooks.onImageProgress(data);
      break;
    }

    case "image_ready": {
      const job = pendingJobs.get(data.jobId);
      if (job?.hooks?.onImageReady) job.hooks.onImageReady(data);
      break;
    }

    case "generation_complete": {
      updateTelemetry({
        stage: "Complete",
        ttftMs: data.ttftMs,
        numTokens: data.numTokens,
        totalTimeSec: data.totalTimeSec,
      });
      const job = pendingJobs.get(data.jobId);
      if (job) {
        pendingJobs.delete(data.jobId);
        job.resolve(data);
      }
      break;
    }

    case "error": {
      updateTelemetry({ stage: "Error" });
      downloadPanel.classList.add("hidden");
      updateModeUI();
      const job = data.jobId ? pendingJobs.get(data.jobId) : null;
      if (job) {
        pendingJobs.delete(data.jobId);
        job.reject(new Error(data.message));
      } else {
        appendMessage("assistant", `⚠️ ${data.message}`, "System");
      }
      break;
    }

    default:
      break;
  }
});

/**
 * Parse and execute any client-side Agent Tool Calls in the response text
 */
async function processAgentTools(rawText, userPrompt, bubbleEl) {
  if (!agentToolsToggle.checked) return;

  const executedTools = new Set();

  const toolRegex = /\[TOOL:([a-zA-Z0-9_]+)\(([^)]*)\)\]/g;
  let match;
  while ((match = toolRegex.exec(rawText)) !== null) {
    const toolName = match[1].trim().toLowerCase();
    const toolArg = match[2].trim();
    executedTools.add(toolName);
    await executeBrowserTool(toolName, toolArg, bubbleEl);
  }

  const lowerPrompt = userPrompt.toLowerCase();
  if (!executedTools.has("set_theme")) {
    for (const theme of ["neon", "emerald", "sunset", "midnight"]) {
      if (lowerPrompt.includes("theme") && lowerPrompt.includes(theme)) {
        await executeBrowserTool("set_theme", theme, bubbleEl);
        break;
      }
    }
  }

  if (
    !executedTools.has("render_kpi") &&
    (lowerPrompt.includes("kpi card") || lowerPrompt.includes("metric card"))
  ) {
    await executeBrowserTool(
      "render_kpi",
      "Local WebGPU Inference|0ms Network RTT|$0.00 Cloud API Cost",
      bubbleEl
    );
  }

  if (
    !executedTools.has("imagine") &&
    (lowerPrompt.includes("use the imagine tool") || lowerPrompt.startsWith("/imagine"))
  ) {
    const cleaned = userPrompt
      .replace(/use the imagine tool to (draw|generate|create)?/i, "")
      .replace(/^\/imagine\s*/i, "")
      .trim();
    if (cleaned) {
      await executeBrowserTool("imagine", cleaned, bubbleEl);
    }
  }
}

/**
 * Execute a client-side JavaScript tool and attach a visual execution card
 */
async function executeBrowserTool(toolName, argString, bubbleEl) {
  const card = document.createElement("div");
  card.className = "agent-tool-card";

  const header = document.createElement("div");
  header.className = "agent-tool-header";

  if (toolName === "set_theme") {
    const validThemes = ["midnight", "neon", "emerald", "sunset"];
    const chosen = validThemes.find((t) => argString.toLowerCase().includes(t)) || "neon";
    document.body.dataset.theme = chosen;
    header.textContent = `🛠️ Executed Tool: set_theme("${chosen}")`;
    const body = document.createElement("div");
    body.textContent = `Switched application CSS theme variables to "${chosen}" mode.`;
    card.appendChild(header);
    card.appendChild(body);
    bubbleEl.appendChild(card);
  } else if (toolName === "render_kpi") {
    const parts = argString.split("|").map((s) => s.trim());
    const title = parts[0] || "Browser AI Telemetry";
    const value = parts[1] || "100% Local";
    const delta = parts[2] || "Zero API Cost";

    header.textContent = `🛠️ Executed Tool: render_kpi("${title}")`;
    const widget = document.createElement("div");
    widget.className = "kpi-widget";
    widget.innerHTML = `
      <div>
        <div class="kpi-title">${title}</div>
        <div class="kpi-value">${value}</div>
      </div>
      <div class="kpi-delta">${delta}</div>
    `;
    card.appendChild(header);
    card.appendChild(widget);
    bubbleEl.appendChild(card);
  } else if (toolName === "imagine") {
    header.textContent = `🔗 Agent Chained Tool: imagine("${argString}") ➔ Janus-Pro-1B`;
    const statusEl = document.createElement("div");
    statusEl.textContent = "Initializing Janus-Pro-1B to synthesize image...";
    card.appendChild(header);
    card.appendChild(statusEl);
    bubbleEl.appendChild(card);

    updateTelemetry({ stage: "Agent ➔ Janus Image" });
    await runWorkerJob(
      "generate_image",
      { prompt: argString, modelId: JANUS_MODEL_ID },
      {
        onImageProgress: (p) => {
          const pct = Math.round((p.step / p.total) * 100);
          statusEl.textContent = `Synthesizing image tokens on WebGPU: ${p.step} / ${p.total} (${pct}%)`;
        },
        onImageReady: (imgData) => {
          renderImageInBubble(statusEl, imgData.blob, argString);
        },
      }
    );
  }
}

/**
 * Run Mode 1: Text & Agent Chat using the selected Hugging Face Text LLM
 */
async function handleTextChat(userText) {
  const sysContent =
    systemPromptInput.value.trim() +
    (agentToolsToggle.checked ? `\n${AGENT_TOOL_INSTRUCTIONS}` : "");

  const messages = [
    { role: "system", content: sysContent },
    ...chatHistory,
    { role: "user", content: userText },
  ];

  const shortName = getShortModelName(selectedTextModelId);
  const { bubble, textNode, meta } = appendMessage(
    "assistant",
    "...",
    `${selectedTextModelId} (${selectedDtype})`
  );

  updateTelemetry({ stage: `${shortName} Decoding` });

  const result = await runWorkerJob(
    "generate_text",
    {
      messages,
      modelId: selectedTextModelId,
      dtype: selectedDtype,
      config: getGenerationConfig(),
    },
    {
      onToken: (t) => {
        textNode.textContent = t.fullText;
        meta.textContent = `${shortName} • TTFT: ${t.ttftMs}ms • ${t.numTokens} tokens • ${t.tps} tok/s`;
        chatFeed.scrollTop = chatFeed.scrollHeight;
      },
    }
  );

  chatHistory.push({ role: "user", content: userText });
  chatHistory.push({ role: "assistant", content: result.fullText });

  await processAgentTools(result.fullText, userText, bubble);
}

/**
 * Run Mode 2: Image Generation (Janus-Pro-1B-ONNX)
 */
async function handleImageGeneration(promptText) {
  const { bubble, textNode, meta } = appendMessage(
    "assistant",
    "Preparing Janus-Pro-1B image synthesis (0 / 576 tokens)...",
    "Janus-Pro-1B-ONNX (Image Generator)"
  );

  updateTelemetry({ stage: "Janus Image Gen" });

  await runWorkerJob(
    "generate_image",
    { prompt: promptText, modelId: JANUS_MODEL_ID },
    {
      onImageProgress: (p) => {
        const pct = Math.round((p.step / p.total) * 100);
        textNode.textContent = `Synthesizing 24×24 visual token grid on WebGPU: ${p.step} / ${p.total} (${pct}%)`;
        meta.textContent = `${p.tps} image tokens/sec`;
      },
      onImageReady: (imgData) => {
        renderImageInBubble(bubble, imgData.blob, promptText);
        meta.textContent = `Generated 384×384 image in ${imgData.totalTimeSec}s (576 tokens)`;
        chatFeed.scrollTop = chatFeed.scrollHeight;
      },
    }
  );
}

/**
 * Run Mode 3: Multimodal Vision Understanding (Janus-Pro-1B-ONNX)
 */
async function handleVisionAnalysis(promptText) {
  const { textNode, meta } = appendMessage(
    "assistant",
    "Encoding image & generating response...",
    "Janus-Pro-1B-ONNX (Multimodal Vision)"
  );

  updateTelemetry({ stage: "Janus Vision QA" });

  await runWorkerJob(
    "analyze_vision",
    {
      imageUrl: currentVisionImageUrl,
      prompt: promptText,
      modelId: JANUS_MODEL_ID,
      config: getGenerationConfig(),
    },
    {
      onToken: (t) => {
        textNode.textContent = t.fullText;
        meta.textContent = `TTFT: ${t.ttftMs}ms • ${t.numTokens} tokens • ${t.tps} tok/s`;
        chatFeed.scrollTop = chatFeed.scrollHeight;
      },
    }
  );
}

/**
 * Run Mode 4: Multi-Model Chaining Workflows (Selected HF Text LLM ⇄ Janus-Pro-1B)
 */
async function handleModelChain(userInput) {
  const chainType = chainSelect.value;
  const shortLlm = getShortModelName(selectedTextModelId);

  if (chainType === "expand_to_image") {
    renderChainSteps(0, -1);
    updateTelemetry({ stage: `Chain 1/2: ${shortLlm}` });

    const step1Msg = appendMessage(
      "assistant",
      "...",
      `Step 1/2 • ${selectedTextModelId} (Prompt Expansion)`
    );

    const expanderMessages = [
      {
        role: "system",
        content:
          "You are an expert AI visual art director. Expand the user's concept into a vivid, single-paragraph image generation prompt describing lighting, colors, camera angle, and artistic style. Output ONLY the expanded prompt.",
      },
      {
        role: "user",
        content: `Create a detailed visual prompt for: ${userInput}`,
      },
    ];

    const step1Result = await runWorkerJob(
      "generate_text",
      {
        messages: expanderMessages,
        modelId: selectedTextModelId,
        dtype: selectedDtype,
        config: { temperature: 0.75, top_p: 0.9, max_new_tokens: 120 },
      },
      {
        onToken: (t) => {
          step1Msg.textNode.textContent = t.fullText;
          step1Msg.meta.textContent = `Step 1 (${shortLlm}) • ${t.numTokens} tokens • ${t.tps} tok/s`;
          chatFeed.scrollTop = chatFeed.scrollHeight;
        },
      }
    );

    const expandedPrompt = step1Result.fullText.trim() || userInput;

    renderChainSteps(1, 0);
    updateTelemetry({ stage: "Chain 2/2: Janus Gen" });

    const step2Msg = appendMessage(
      "assistant",
      "Synthesizing image from expanded prompt (0 / 576)...",
      "Step 2/2 • Janus-Pro-1B (Image Synthesis)"
    );

    await runWorkerJob(
      "generate_image",
      { prompt: expandedPrompt, modelId: JANUS_MODEL_ID },
      {
        onImageProgress: (p) => {
          const pct = Math.round((p.step / p.total) * 100);
          step2Msg.textNode.textContent = `Rendering expanded prompt on WebGPU: ${p.step} / ${p.total} (${pct}%)`;
          step2Msg.meta.textContent = `${p.tps} image tokens/sec`;
        },
        onImageReady: (imgData) => {
          renderImageInBubble(step2Msg.bubble, imgData.blob, expandedPrompt);
          step2Msg.meta.textContent = `Step 2 Complete • 384×384 image rendered in ${imgData.totalTimeSec}s`;
        },
      }
    );

    renderChainSteps(-1, 1);
  } else if (chainType === "image_to_critic") {
    renderChainSteps(0, -1);
    updateTelemetry({ stage: "Chain 1/3: Janus Image" });

    const step1Msg = appendMessage(
      "assistant",
      "Generating initial artwork (0 / 576)...",
      "Step 1/3 • Janus-Pro-1B (Image Generation)"
    );

    let generatedImageUrl = null;
    await runWorkerJob(
      "generate_image",
      { prompt: userInput, modelId: JANUS_MODEL_ID },
      {
        onImageProgress: (p) => {
          const pct = Math.round((p.step / p.total) * 100);
          step1Msg.textNode.textContent = `Synthesizing image tokens: ${p.step} / ${p.total} (${pct}%)`;
        },
        onImageReady: (imgData) => {
          generatedImageUrl = renderImageInBubble(step1Msg.bubble, imgData.blob, userInput);
          step1Msg.meta.textContent = `Step 1 Complete • Generated in ${imgData.totalTimeSec}s`;
        },
      }
    );

    renderChainSteps(1, 0);
    updateTelemetry({ stage: "Chain 2/3: Janus Vision" });

    const step2Msg = appendMessage(
      "assistant",
      "Inspecting generated image with Vision encoder...",
      "Step 2/3 • Janus-Pro-1B Vision (Visual Critique)"
    );

    const step2Result = await runWorkerJob(
      "analyze_vision",
      {
        imageUrl: generatedImageUrl,
        prompt: "Describe what you see in this image, including the colors, mood, and details.",
        modelId: JANUS_MODEL_ID,
        config: { max_new_tokens: 140 },
      },
      {
        onToken: (t) => {
          step2Msg.textNode.textContent = t.fullText;
          step2Msg.meta.textContent = `Step 2 • ${t.numTokens} tokens • ${t.tps} tok/s`;
          chatFeed.scrollTop = chatFeed.scrollHeight;
        },
      }
    );

    renderChainSteps(2, 1);
    updateTelemetry({ stage: `Chain 3/3: ${shortLlm}` });

    const step3Msg = appendMessage(
      "assistant",
      "...",
      `Step 3/3 • ${selectedTextModelId} (Creative Storyteller)`
    );

    await runWorkerJob(
      "generate_text",
      {
        messages: [
          {
            role: "system",
            content:
              "You are an imaginative storyteller. Given a visual description of an artwork, write a captivating 3-sentence micro-story set in that scene.",
          },
          {
            role: "user",
            content: `Artwork description: ${step2Result.fullText}`,
          },
        ],
        modelId: selectedTextModelId,
        dtype: selectedDtype,
        config: { temperature: 0.8, top_p: 0.9, max_new_tokens: 140 },
      },
      {
        onToken: (t) => {
          step3Msg.textNode.textContent = t.fullText;
          step3Msg.meta.textContent = `Step 3 (${shortLlm}) • ${t.numTokens} tokens • ${t.tps} tok/s`;
          chatFeed.scrollTop = chatFeed.scrollHeight;
        },
      }
    );

    renderChainSteps(-1, 2);
  } else if (chainType === "vision_to_remix") {
    renderChainSteps(0, -1);
    updateTelemetry({ stage: "Chain 1/3: Vision Analyze" });

    const step1Msg = appendMessage(
      "assistant",
      "Analyzing source image with Janus Vision...",
      "Step 1/3 • Janus-Pro-1B Vision (Source Analysis)"
    );

    const step1Result = await runWorkerJob(
      "analyze_vision",
      {
        imageUrl: currentVisionImageUrl,
        prompt: "Describe the main subject and composition of this image concisely.",
        modelId: JANUS_MODEL_ID,
        config: { max_new_tokens: 120 },
      },
      {
        onToken: (t) => {
          step1Msg.textNode.textContent = t.fullText;
          step1Msg.meta.textContent = `Step 1 • ${t.numTokens} tokens • ${t.tps} tok/s`;
          chatFeed.scrollTop = chatFeed.scrollHeight;
        },
      }
    );

    renderChainSteps(1, 0);
    updateTelemetry({ stage: `Chain 2/3: ${shortLlm}` });

    const step2Msg = appendMessage(
      "assistant",
      "...",
      `Step 2/3 • ${selectedTextModelId} (Reimagined Prompt)`
    );

    const step2Result = await runWorkerJob(
      "generate_text",
      {
        messages: [
          {
            role: "system",
            content:
              "You are a creative visual remix artist. Combine the source image description with the user's style direction into a single vivid image generation prompt. Output ONLY the new image prompt.",
          },
          {
            role: "user",
            content: `Original image: ${step1Result.fullText}\nRemix style instruction: ${userInput}`,
          },
        ],
        modelId: selectedTextModelId,
        dtype: selectedDtype,
        config: { temperature: 0.75, top_p: 0.9, max_new_tokens: 120 },
      },
      {
        onToken: (t) => {
          step2Msg.textNode.textContent = t.fullText;
          step2Msg.meta.textContent = `Step 2 (${shortLlm}) • ${t.numTokens} tokens • ${t.tps} tok/s`;
          chatFeed.scrollTop = chatFeed.scrollHeight;
        },
      }
    );

    const remixedPrompt = step2Result.fullText.trim() || userInput;

    renderChainSteps(2, 1);
    updateTelemetry({ stage: "Chain 3/3: Janus Render" });

    const step3Msg = appendMessage(
      "assistant",
      "Synthesizing remixed image (0 / 576)...",
      "Step 3/3 • Janus-Pro-1B (Remixed Image Synthesis)"
    );

    await runWorkerJob(
      "generate_image",
      { prompt: remixedPrompt, modelId: JANUS_MODEL_ID },
      {
        onImageProgress: (p) => {
          const pct = Math.round((p.step / p.total) * 100);
          step3Msg.textNode.textContent = `Rendering remixed artwork on WebGPU: ${p.step} / ${p.total} (${pct}%)`;
        },
        onImageReady: (imgData) => {
          renderImageInBubble(step3Msg.bubble, imgData.blob, remixedPrompt);
          step3Msg.meta.textContent = `Step 3 Complete • Remixed in ${imgData.totalTimeSec}s`;
        },
      }
    );

    renderChainSteps(-1, 2);
  }
}

// Mode Switching Listeners
for (const [modeKey, btn] of Object.entries(modeButtons)) {
  btn.addEventListener("click", () => {
    if (isBusy || currentMode === modeKey) return;
    currentMode = modeKey;
    updateModeUI();
  });
}

// Hugging Face Model Selector & Dtype Listeners
hfModelSelect.addEventListener("change", () => {
  if (hfModelSelect.value === "__custom__") {
    hfHubDrawer.classList.remove("hidden");
    hfSearchInput.focus();
    searchHuggingFaceHub(hfSearchInput.value);
    // Restore previous value until user picks a model
    hfModelSelect.value = selectedTextModelId;
    return;
  }
  selectedTextModelId = hfModelSelect.value;
  updateModeUI();
});

hfDtypeSelect.addEventListener("change", () => {
  selectedDtype = hfDtypeSelect.value;
  updateModeUI();
});

browseHubBtn.addEventListener("click", () => {
  const isHidden = hfHubDrawer.classList.toggle("hidden");
  if (!isHidden) {
    hfSearchInput.focus();
    searchHuggingFaceHub(hfSearchInput.value);
  }
});

closeHubDrawerBtn.addEventListener("click", () => {
  hfHubDrawer.classList.add("hidden");
});

hfSearchBtn.addEventListener("click", () => {
  searchHuggingFaceHub(hfSearchInput.value);
});

hfSearchInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    searchHuggingFaceHub(hfSearchInput.value);
  }
});

hfUseCustomBtn.addEventListener("click", () => {
  const raw = hfSearchInput.value.trim();
  if (!raw || !raw.includes("/")) {
    alert('Please enter a valid Hugging Face repo ID in the format "owner/model-name".');
    return;
  }
  selectOrAddHfModel(raw);
  worker.postMessage({
    type: "load_model",
    mode: "text",
    modelId: selectedTextModelId,
    dtype: selectedDtype,
  });
});

chainSelect.addEventListener("change", () => {
  updateModeUI();
});

// Vision Image Upload & Sample Listeners
visionFileInput.addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  currentVisionImageUrl = url;
  currentVisionLabel = `Uploaded: ${file.name}`;
  visionPreviewImg.src = url;
  visionImgLabel.textContent = currentVisionLabel;
});

document.querySelectorAll(".sample-img-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    currentVisionImageUrl = btn.dataset.url;
    currentVisionLabel = btn.dataset.label;
    visionPreviewImg.src = currentVisionImageUrl;
    visionImgLabel.textContent = currentVisionLabel;
  });
});

// Inspector Sliders & Tool Demo Buttons
paramTemp.addEventListener("input", () => {
  valTemp.textContent = Number(paramTemp.value).toFixed(2);
});

paramTopP.addEventListener("input", () => {
  valTopP.textContent = Number(paramTopP.value).toFixed(2);
});

paramMaxTokens.addEventListener("input", () => {
  valMaxTokens.textContent = paramMaxTokens.value;
});

resetSyspromptBtn.addEventListener("click", () => {
  systemPromptInput.value = DEFAULT_SYS_PROMPT;
});

const inspectorBackdrop = document.getElementById("inspector-backdrop");
const closeInspectorBtn = document.getElementById("close-inspector-btn");

function setInspectorOpen(open) {
  inspectorSidebar.classList.toggle("hidden", !open);
  if (inspectorBackdrop) {
    const isMobile = window.innerWidth <= 960;
    inspectorBackdrop.classList.toggle("hidden", !(open && isMobile));
  }
}

// Start with Inspector drawer closed on mobile/tablet so it doesn't cover the chat screen
if (window.innerWidth <= 960) {
  setInspectorOpen(false);
}

toggleInspectorBtn.addEventListener("click", () => {
  const isCurrentlyHidden = inspectorSidebar.classList.contains("hidden");
  setInspectorOpen(isCurrentlyHidden);
});

if (closeInspectorBtn) {
  closeInspectorBtn.addEventListener("click", () => {
    setInspectorOpen(false);
  });
}

if (inspectorBackdrop) {
  inspectorBackdrop.addEventListener("click", () => {
    setInspectorOpen(false);
  });
}

window.addEventListener("resize", () => {
  if (window.innerWidth > 960 && inspectorBackdrop) {
    inspectorBackdrop.classList.add("hidden");
  }
});

document.querySelectorAll(".tool-pill").forEach((btn) => {
  btn.addEventListener("click", () => {
    if (isBusy) return;
    currentMode = "text";
    agentToolsToggle.checked = true;
    updateModeUI();

    if (window.innerWidth <= 960) {
      setInspectorOpen(false);
    }

    const demoType = btn.dataset.demoTool;
    if (demoType === "theme") {
      promptInput.value =
        "Switch the UI theme to neon and explain how WebGPU accelerates browser LLMs.";
    } else if (demoType === "kpi") {
      promptInput.value =
        "Show a KPI card for Browser AI Privacy & Cost and summarize the benefits.";
    } else if (demoType === "imagine") {
      promptInput.value =
        "Use the imagine tool to draw a glowing crystal lighthouse on an alien ocean.";
    }
    promptForm.requestSubmit();
  });
});

// Preload / Download Model from Hugging Face CDN
preloadBtn.addEventListener("click", () => {
  if (currentMode === "text") {
    worker.postMessage({
      type: "load_model",
      mode: "text",
      modelId: selectedTextModelId,
      dtype: selectedDtype,
    });
  } else if (currentMode === "image" || currentMode === "vision") {
    worker.postMessage({
      type: "load_model",
      mode: "janus",
      modelId: JANUS_MODEL_ID,
    });
  } else if (currentMode === "chain") {
    if (loadedTextModelId !== selectedTextModelId) {
      worker.postMessage({
        type: "load_model",
        mode: "text",
        modelId: selectedTextModelId,
        dtype: selectedDtype,
      });
    } else {
      worker.postMessage({
        type: "load_model",
        mode: "janus",
        modelId: JANUS_MODEL_ID,
      });
    }
  }
});

clearBtn.addEventListener("click", () => {
  if (isBusy) return;
  chatHistory = [];
  chatFeed.innerHTML = "";
  welcomeCard.classList.remove("hidden");
  chatFeed.appendChild(welcomeCard);
  renderPresets();
});

stopBtn.addEventListener("click", () => {
  worker.postMessage({ type: "interrupt" });
});

promptInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    promptForm.requestSubmit();
  }
});

// Form Submit Handler
promptForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (isBusy) return;

  const text = promptInput.value.trim();
  if (!text) return;

  promptInput.value = "";
  isBusy = true;
  sendBtn.disabled = true;
  stopBtn.classList.remove("hidden");

  const attachVisionThumb =
    currentMode === "vision" ||
    (currentMode === "chain" && chainSelect.value === "vision_to_remix")
      ? currentVisionImageUrl
      : null;

  appendMessage("user", text, "You", attachVisionThumb);

  try {
    if (currentMode === "text") {
      await handleTextChat(text);
    } else if (currentMode === "image") {
      await handleImageGeneration(text);
    } else if (currentMode === "vision") {
      await handleVisionAnalysis(text);
    } else if (currentMode === "chain") {
      await handleModelChain(text);
    }
  } catch (err) {
    appendMessage("assistant", `⚠️ ${err.message}`, "System");
  } finally {
    isBusy = false;
    sendBtn.disabled = false;
    stopBtn.classList.add("hidden");
    updateModeUI();
  }
});

// Initialize
updateModeUI();
worker.postMessage({ type: "check_webgpu" });

// Educational Floating Tooltip Controller (Desktop Hover + Mobile Touch)
const eduTooltip = document.getElementById("edu-tooltip");
const eduTooltipTitle = document.getElementById("edu-tooltip-title");
const eduTooltipBody = document.getElementById("edu-tooltip-body");
const eduTooltipCode = document.getElementById("edu-tooltip-code");
const eduTooltipClose = document.getElementById("edu-tooltip-close");

function showEduTooltip(targetEl) {
  if (!eduTooltip || !targetEl) return;
  const title = targetEl.dataset.tooltipTitle;
  const body = targetEl.dataset.tooltipBody;
  const code = targetEl.dataset.tooltipCode;
  if (!title && !body) return;

  eduTooltipTitle.textContent = title || "Browser AI Concept";
  eduTooltipBody.textContent = body || "";

  if (code) {
    eduTooltipCode.textContent = code;
    eduTooltipCode.classList.remove("hidden");
  } else {
    eduTooltipCode.classList.add("hidden");
  }

  eduTooltip.classList.remove("hidden");
  eduTooltip.setAttribute("aria-hidden", "false");

  const rect = targetEl.getBoundingClientRect();
  const tipRect = eduTooltip.getBoundingClientRect();
  const margin = 10;

  let top = rect.bottom + margin;
  let left = rect.left + rect.width / 2 - tipRect.width / 2;

  // Flip above if overflowing bottom of viewport
  if (top + tipRect.height > window.innerHeight - margin) {
    top = Math.max(margin, rect.top - tipRect.height - margin);
  }

  // Clamp horizontally inside viewport
  left = Math.max(margin, Math.min(left, window.innerWidth - tipRect.width - margin));

  eduTooltip.style.top = `${Math.round(top)}px`;
  eduTooltip.style.left = `${Math.round(left)}px`;
}

function hideEduTooltip() {
  if (!eduTooltip) return;
  eduTooltip.classList.add("hidden");
  eduTooltip.setAttribute("aria-hidden", "true");
}

if (eduTooltipClose) {
  eduTooltipClose.addEventListener("click", (e) => {
    e.stopPropagation();
    hideEduTooltip();
  });
}

// Desktop hover support
document.addEventListener("mouseover", (e) => {
  if (e.target.closest("#edu-tooltip")) return;
  const trigger = e.target.closest("[data-tooltip-title]");
  if (trigger) {
    showEduTooltip(trigger);
  } else {
    hideEduTooltip();
  }
});

// Touch / Click support for ? educational badges on mobile
document.addEventListener("click", (e) => {
  if (e.target.closest("#edu-tooltip")) return;

  const badge = e.target.closest(".edu-badge, .edu-badge-inline");
  if (badge) {
    const trigger = badge.closest("[data-tooltip-title]");
    if (trigger) {
      e.preventDefault();
      e.stopPropagation();
      showEduTooltip(trigger);
      return;
    }
  }

  // Also allow tapping non-button containers (like telemetry cards or brand header) on touchscreens
  const trigger = e.target.closest("[data-tooltip-title]");
  if (
    trigger &&
    !trigger.matches("button, select, input, textarea, label") &&
    !e.target.closest("button, select, input, textarea, label")
  ) {
    showEduTooltip(trigger);
  } else {
    hideEduTooltip();
  }
});

document.addEventListener("focusin", (e) => {
  const trigger = e.target.closest("[data-tooltip-title]");
  if (trigger) {
    showEduTooltip(trigger);
  }
});

document.addEventListener("focusout", () => {
  hideEduTooltip();
});

window.addEventListener("scroll", hideEduTooltip, true);
