# ⚡ Browser AI Workbench (Transformers.js + WebGPU)

An interactive, 100% client-side AI educational workbench demonstrating how to run Large Language Models (LLMs), Multimodal Vision-Language Models, Agentic Tool Calling, and Multi-Model Chaining directly in the browser using **JavaScript**, **Web Workers**, **WebGPU**, and **[@huggingface/transformers](https://www.npmjs.com/package/@huggingface/transformers)**.

## ✨ Features

- **🤗 Hugging Face CDN Model Selector & Hub Explorer**:
  - Choose from curated open-weights browser LLMs (`SmolLM2-135M`, `SmolLM2-360M`, `SmolLM2-1.7B`, `Qwen2.5-0.5B`, `Qwen2.5-Coder-0.5B`, `Llama-3.2-1B`) or search the live Hugging Face Hub API for any `transformers.js` compatible repository.
  - Configurable quantization precision (`auto`, `q4f16`, `q4`, `q8`, `fp16`).
- **🎨 Local WebGPU Image Generation (`Janus-Pro-1B-ONNX`)**:
  - Synthesizes 384×384 images from text prompts via 576 discrete visual tokens directly on the GPU.
- **👁️ Multimodal Vision Understanding (`Janus-Pro-1B-ONNX`)**:
  - Upload local images or test with math formulas/wildlife photos for OCR, $\LaTeX$ extraction, and visual question answering.
- **🔗 Multi-Model Chaining Studio**:
  - **Chain 1 (AI Art Director)**: Text LLM expands short concept ➔ Janus-Pro-1B synthesizes artwork.
  - **Chain 2 (Generate & Critique Loop)**: Janus-Pro-1B generates image ➔ Janus Vision critiques what was drawn ➔ Text LLM writes a micro-story.
  - **Chain 3 (Visual Style Remix)**: Janus Vision analyzes input photo ➔ Text LLM reimagines style ➔ Janus-Pro-1B renders remixed artwork.
- **🛠️ Client-Side Agentic Tool Calling**:
  - LLMs can invoke browser tools (`set_theme`, `render_kpi`, `imagine`) to mutate CSS themes, render UI cards, or chain into image generation.
- **🎛️ Live LLM Inspector & Educational Tooltips**:
  - Real-time telemetry (**TTFT**, **Decode Speed `tok/s`**, **Output Tokens**, **Total Latency**), sampling sliders (**Temperature**, **Top-P**, **Max New Tokens**), and interactive `?` concept tooltips with underlying code snippets.

## 🚀 Run Locally

```bash
npm install
npm run dev
```
