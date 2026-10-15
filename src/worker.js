import {
  pipeline,
  AutoProcessor,
  MultiModalityCausalLM,
  TextStreamer,
  BaseStreamer,
  InterruptableStoppingCriteria,
} from "@huggingface/transformers";

const DEFAULT_TEXT_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct";
const DEFAULT_IMAGE_MODEL_ID = "onnx-community/Janus-Pro-1B-ONNX";

let webgpuSupported = false;
let fp16Supported = false;

let activeTextModelId = null;
let activeTextDtype = null;
let textGenerator = null;

let activeJanusModelId = null;
let imageProcessor = null;
let imageModel = null;

const stoppingCriteria = new InterruptableStoppingCriteria();

/**
 * Custom streamer that reports image token generation progress (e.g., 1..576 tokens)
 */
class ImageProgressStreamer extends BaseStreamer {
  constructor(totalTokens, onProgress) {
    super();
    this.totalTokens = totalTokens;
    this.onProgress = onProgress;
    this.count = 0;
  }

  put(_value) {
    this.count += 1;
    this.onProgress(this.count, this.totalTokens);
  }

  end() {}
}

/**
 * Detect WebGPU and shader-f16 support in the browser
 */
async function checkWebGPU() {
  try {
    if (!("gpu" in navigator)) {
      webgpuSupported = false;
      fp16Supported = false;
      self.postMessage({
        status: "webgpu_status",
        supported: false,
        fp16: false,
        message: "WebGPU unavailable (WASM fallback)",
      });
      return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      webgpuSupported = false;
      fp16Supported = false;
      self.postMessage({
        status: "webgpu_status",
        supported: false,
        fp16: false,
        message: "No WebGPU adapter (WASM fallback)",
      });
      return;
    }

    webgpuSupported = true;
    fp16Supported = adapter.features.has("shader-f16");
    self.postMessage({
      status: "webgpu_status",
      supported: true,
      fp16: fp16Supported,
      message: `WebGPU Active (${fp16Supported ? "FP16" : "FP32"})`,
    });
  } catch (err) {
    webgpuSupported = false;
    fp16Supported = false;
    self.postMessage({
      status: "webgpu_status",
      supported: false,
      fp16: false,
      message: `WebGPU error: ${err.message}`,
    });
  }
}

/**
 * Shared progress callback forwarded to the main thread for CDN download UI
 */
function handleModelProgress(progressEvent) {
  self.postMessage({
    status: "model_progress",
    ...progressEvent,
  });
}

/**
 * Resolve desired dtype for a text model
 */
function resolveTextDtype(requestedDtype) {
  if (requestedDtype && requestedDtype !== "auto") {
    return requestedDtype;
  }
  return webgpuSupported ? (fp16Supported ? "q4f16" : "q4") : "q4";
}

/**
 * Load the selected model pipeline from Hugging Face CDN:
 * - target="text": loads any causal LM via pipeline("text-generation", modelId)
 * - target="janus": loads a MultiModalityCausalLM (e.g. Janus-Pro-1B-ONNX)
 */
async function loadModel({ mode, modelId, dtype = "auto" }) {
  const modelKey = mode === "text" ? "text" : "janus";
  const resolvedModelId =
    modelId || (modelKey === "text" ? DEFAULT_TEXT_MODEL_ID : DEFAULT_IMAGE_MODEL_ID);

  try {
    self.postMessage({
      status: "loading_start",
      modelKey,
      modelId: resolvedModelId,
    });

    if (modelKey === "text") {
      const resolvedDtype = resolveTextDtype(dtype);

      if (
        !textGenerator ||
        activeTextModelId !== resolvedModelId ||
        activeTextDtype !== resolvedDtype
      ) {
        if (textGenerator && typeof textGenerator.dispose === "function") {
          try {
            await textGenerator.dispose();
          } catch (_e) {
            // ignore dispose errors
          }
        }
        textGenerator = null;

        const device = webgpuSupported ? "webgpu" : "wasm";
        try {
          textGenerator = await pipeline("text-generation", resolvedModelId, {
            device,
            dtype: resolvedDtype,
            progress_callback: handleModelProgress,
          });
        } catch (primaryErr) {
          // If q4f16 is not available in the selected HF repo, fall back automatically to q4
          if (resolvedDtype === "q4f16") {
            textGenerator = await pipeline("text-generation", resolvedModelId, {
              device,
              dtype: "q4",
              progress_callback: handleModelProgress,
            });
          } else {
            throw primaryErr;
          }
        }

        activeTextModelId = resolvedModelId;
        activeTextDtype = resolvedDtype;
      }

      self.postMessage({
        status: "model_ready",
        modelKey: "text",
        modelId: activeTextModelId,
        dtype: activeTextDtype,
      });
      return true;
    } else {
      if (!webgpuSupported) {
        throw new Error(
          "Janus-Pro-1B requires a browser with WebGPU enabled (e.g. Chrome or Edge)."
        );
      }

      if (!imageProcessor || !imageModel || activeJanusModelId !== resolvedModelId) {
        const processorPromise = AutoProcessor.from_pretrained(resolvedModelId, {
          progress_callback: handleModelProgress,
        });

        const modelPromise = MultiModalityCausalLM.from_pretrained(resolvedModelId, {
          dtype: fp16Supported
            ? {
                prepare_inputs_embeds: "q4",
                language_model: "q4f16",
                lm_head: "fp16",
                gen_head: "fp16",
                gen_img_embeds: "fp16",
                image_decode: "fp32",
              }
            : {
                prepare_inputs_embeds: "fp32",
                language_model: "q4",
                lm_head: "fp32",
                gen_head: "fp32",
                gen_img_embeds: "fp32",
                image_decode: "fp32",
              },
          device: {
            prepare_inputs_embeds: "wasm",
            language_model: "webgpu",
            lm_head: "webgpu",
            gen_head: "webgpu",
            gen_img_embeds: "webgpu",
            image_decode: "webgpu",
          },
          progress_callback: handleModelProgress,
        });

        [imageProcessor, imageModel] = await Promise.all([processorPromise, modelPromise]);
        activeJanusModelId = resolvedModelId;
      }

      self.postMessage({
        status: "model_ready",
        modelKey: "janus",
        modelId: activeJanusModelId,
      });
      return true;
    }
  } catch (err) {
    self.postMessage({
      status: "error",
      message: err?.message || String(err),
    });
    return false;
  }
}

/**
 * Stream conversational text generation using the selected Hugging Face Text LLM
 */
async function generateText({
  messages,
  modelId = DEFAULT_TEXT_MODEL_ID,
  dtype = "auto",
  config = {},
  jobId = null,
}) {
  try {
    const resolvedDtype = resolveTextDtype(dtype);
    if (
      !textGenerator ||
      activeTextModelId !== modelId ||
      activeTextDtype !== resolvedDtype
    ) {
      const ok = await loadModel({ mode: "text", modelId, dtype });
      if (!ok) return;
    }

    stoppingCriteria.reset();
    const requestStart = performance.now();
    self.postMessage({
      status: "generation_start",
      task: "text",
      modelId: activeTextModelId,
      jobId,
    });

    let numTokens = 0;
    let firstTokenTime = null;
    let fullText = "";
    let ttftMs = null;

    const streamer = new TextStreamer(textGenerator.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (tokenText) => {
        const now = performance.now();
        if (firstTokenTime === null) {
          firstTokenTime = now;
          ttftMs = Math.round(now - requestStart);
        }
        numTokens += 1;
        fullText += tokenText;
        const elapsedSec = (now - firstTokenTime) / 1000;
        const tps =
          elapsedSec > 0.02
            ? (numTokens / elapsedSec).toFixed(1)
            : (numTokens / ((now - requestStart) / 1000)).toFixed(1);

        self.postMessage({
          status: "token",
          task: "text",
          jobId,
          token: tokenText,
          fullText,
          tps,
          numTokens,
          ttftMs,
        });
      },
    });

    const maxNewTokens = Number(config.max_new_tokens) || 512;
    const temperature = Number(config.temperature ?? 0.7);
    const topP = Number(config.top_p ?? 0.9);

    await textGenerator(messages, {
      max_new_tokens: maxNewTokens,
      temperature,
      do_sample: temperature > 0.05,
      top_p: topP,
      streamer,
      stopping_criteria: stoppingCriteria,
    });

    const totalTimeSec = ((performance.now() - requestStart) / 1000).toFixed(2);
    self.postMessage({
      status: "generation_complete",
      task: "text",
      jobId,
      modelId: activeTextModelId,
      fullText,
      numTokens,
      ttftMs,
      totalTimeSec,
    });
  } catch (err) {
    self.postMessage({
      status: "error",
      jobId,
      message: err?.message || String(err),
    });
  }
}

/**
 * Multimodal Vision Understanding (Image + Text -> Text) using Janus-Pro-1B-ONNX
 */
async function analyzeVision({
  imageUrl,
  prompt,
  modelId = DEFAULT_IMAGE_MODEL_ID,
  config = {},
  jobId = null,
}) {
  try {
    if (!imageProcessor || !imageModel || activeJanusModelId !== modelId) {
      const ok = await loadModel({ mode: "janus", modelId });
      if (!ok) return;
    }

    stoppingCriteria.reset();
    const requestStart = performance.now();
    self.postMessage({ status: "generation_start", task: "vision", jobId });

    const conversation = [
      {
        role: "<|User|>",
        content: `<image_placeholder>\n${prompt}`,
        images: [imageUrl],
      },
    ];

    const inputs = await imageProcessor(conversation);

    let numTokens = 0;
    let firstTokenTime = null;
    let fullText = "";
    let ttftMs = null;

    const streamer = new TextStreamer(imageProcessor.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (tokenText) => {
        const now = performance.now();
        if (firstTokenTime === null) {
          firstTokenTime = now;
          ttftMs = Math.round(now - requestStart);
        }
        numTokens += 1;
        fullText += tokenText;
        const elapsedSec = (now - firstTokenTime) / 1000;
        const tps = elapsedSec > 0.02 ? (numTokens / elapsedSec).toFixed(1) : "—";

        self.postMessage({
          status: "token",
          task: "vision",
          jobId,
          token: tokenText,
          fullText,
          tps,
          numTokens,
          ttftMs,
        });
      },
    });

    const maxNewTokens = Math.min(Number(config.max_new_tokens) || 256, 512);

    await imageModel.generate({
      ...inputs,
      max_new_tokens: maxNewTokens,
      do_sample: false,
      streamer,
      stopping_criteria: stoppingCriteria,
    });

    const totalTimeSec = ((performance.now() - requestStart) / 1000).toFixed(2);
    self.postMessage({
      status: "generation_complete",
      task: "vision",
      jobId,
      fullText,
      numTokens,
      ttftMs,
      totalTimeSec,
    });
  } catch (err) {
    self.postMessage({
      status: "error",
      jobId,
      message: err?.message || String(err),
    });
  }
}

/**
 * Generate an image from a text prompt using Janus-Pro-1B-ONNX
 */
async function generateImage({ prompt, modelId = DEFAULT_IMAGE_MODEL_ID, jobId = null }) {
  try {
    if (!imageProcessor || !imageModel || activeJanusModelId !== modelId) {
      const ok = await loadModel({ mode: "janus", modelId });
      if (!ok) return;
    }

    stoppingCriteria.reset();
    const requestStart = performance.now();
    self.postMessage({ status: "generation_start", task: "image", jobId });

    const conversation = [
      {
        role: "<|User|>",
        content: prompt,
      },
    ];

    const inputs = await imageProcessor(conversation, {
      chat_template: "text_to_image",
    });

    const totalTokens = imageProcessor.num_image_tokens || 576;
    let firstTokenTime = null;
    let ttftMs = null;

    const streamer = new ImageProgressStreamer(totalTokens, (currentStep, total) => {
      const now = performance.now();
      if (firstTokenTime === null) {
        firstTokenTime = now;
        ttftMs = Math.round(now - requestStart);
      }
      const elapsedSec = (now - firstTokenTime) / 1000;
      const tps = elapsedSec > 0.02 ? (currentStep / elapsedSec).toFixed(1) : "0.0";
      self.postMessage({
        status: "image_progress",
        jobId,
        step: currentStep,
        total,
        tps,
        ttftMs,
      });
    });

    const outputs = await imageModel.generate_images({
      ...inputs,
      min_new_tokens: totalTokens,
      max_new_tokens: totalTokens,
      do_sample: true,
      streamer,
      stopping_criteria: stoppingCriteria,
    });

    const blob = await outputs[0].toBlob();
    const totalTimeSec = ((performance.now() - requestStart) / 1000).toFixed(1);

    self.postMessage({
      status: "image_ready",
      jobId,
      blob,
      prompt,
      totalTimeSec,
      numTokens: totalTokens,
      ttftMs,
    });

    self.postMessage({
      status: "generation_complete",
      task: "image",
      jobId,
      numTokens: totalTokens,
      ttftMs,
      totalTimeSec,
    });
  } catch (err) {
    self.postMessage({
      status: "error",
      jobId,
      message: err?.message || String(err),
    });
  }
}

self.addEventListener("message", async (event) => {
  const { type, ...payload } = event.data;

  switch (type) {
    case "check_webgpu":
      await checkWebGPU();
      break;
    case "load_model":
      await loadModel(payload);
      break;
    case "generate_text":
      await generateText(payload);
      break;
    case "analyze_vision":
      await analyzeVision(payload);
      break;
    case "generate_image":
      await generateImage(payload);
      break;
    case "interrupt":
      stoppingCriteria.interrupt();
      break;
    default:
      break;
  }
});
