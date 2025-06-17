---
title: Adversarial Policy Probe
emoji: 🔒
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.44.1
app_file: app.py
pinned: false
---

# Adversarial Policy Probe

A Streamlit application designed to probe large-language-model policy-violation defenses by generating corrupted user prompts and classifying model replies for disallowed content.

> **Credit:** This tool is based on the “Best-of-N Jailbreak” methodology introduced in the Anthropic-led paper *Best-of-N Jailbreaking* by Hughes *et al.* (2024).[^1]  [oai_citation:0‡arxiv.org](https://arxiv.org/abs/2412.03556?utm_source=chatgpt.com)

## 🚀 Features

- **Prompt-corruption helpers**  
  - Word scrambling (`apply_word_scrambling`)  
  - Random capitalization (`apply_random_caps`)  
  - ASCII-level noise injection (`apply_ascii_noise`)  
- **One-token classifier prompt**  
  - Strict **YES/NO** output for policy-violation detection  
- **Model loader with quantization & MPS support**  
  - 8-bit / 4-bit quantization via `BitsAndBytesConfig`  
  - Automatic device mapping on CUDA, fallback to full precision on MPS/CPU  
- **Adversarial attack loop**  
  - Batch-driven corruption of the seed prompt  
  - Generates & classifies replies, tracks successful policy violations  
- **Streamlit UI**  
  - Interactive sidebar controls: model, device, quantization, σ, iterations, batch size, seed  
  - Real-time progress bar & status updates  

---

[^1]: Hughes, J., Price, S., Lynch, A., Schaeffer, R., Barez, F., Koyejo, S., Sleight, H., Jones, E., Perez, E., & Sharma, M. (2024). *Best-of-N Jailbreaking*. arXiv:2412.03556.
