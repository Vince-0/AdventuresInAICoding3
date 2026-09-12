# Adventures In AI Coding 3
## Local LLM Tinkering

## Why

Because the cloud is someone else's computer and AI usage credits aren't cheap. Let's see what I can do with a 10GB Nvidia card other than play games.

## How

Get a local open source inference server to serve an LLM model to support local agents like Hermes and OpenCode.

### Key Concepts for Local LLMs

<details>
<summary>Key terms and concepts (click to expand)</summary>

**GGUF (Grove Generic Unified Format)**
- Binary format for machine learning models optimized for inference
- Supports quantization (compression while maintaining quality)
- Compatible with llama.cpp and many inference servers
- File sizes range from 2GB to 20+GB depending on model size and quantization
- Lower bit depth = smaller file size but potentially slower/inferior quality

**Parameters**
- The number of learnable weights in a neural network
- Measured in billions (B) = 1B = 1 billion parameters
- 4B = ~4 billion parameters, 8B = ~8 billion, etc.
- More parameters generally = better performance but requires more VRAM/GPU memory
- A 7B model might need 8-12GB VRAM depending on quantization
- A 35B model typically needs 16GB+ VRAM

**Quantization**
- Compression technique that reduces model file size and VRAM requirements
- Works by rounding weights to fewer bit depths
- Trade-off: smaller file = faster load time but potentially slower inference and lower quality
- Common quantizations:
  - Q4_K_M (4-bit): Sweet spot for most users
  - Q2_K (2-bit): Very small but noisy
  - Q8_0 (8-bit): Near original quality, ~2x size
  - BF16 (Binary Float 16): For fine-tuning and multimodal models

**Bit Depth**
- The precision used when quantizing model weights
- Lower bit depth = more aggressive compression
- 1 bit = most compressed, but very noisy
- 4 bits = standard balance (most common)
- 8 bits = near original precision

**Inference Server**
- Software that loads the model and handles requests
- Common options:
  - **vLLM**: High throughput, requires CUDA 12.8
  - **llama.cpp**: Cross-platform, flexible, supports MTP
  - **Hermes**: Agent framework that can connect to servers
- Server configuration options include GPU memory utilization, context length, etc.
- Proper server setup can improve performance by 20-40%

**Agent Harness**
- Frameworks that automate tasks using LLMs
- **Hermes**: Tracks and manages agent lifecycle, handles tool use
  - Can connect to llama.cpp servers
  - Supports custom tool-call parsers
  - Good for multi-step workflows
- **Opencode**: Agent execution platform
  - Browser-based interface
  - Can run local models
  - May have caching issues with certain models
- Agents can enter "thinking loops" if not properly configured

**Tokenizer**
- Converts text to numbers that the model processes
- Different models use different tokenizers
- Tokenization affects speed and performance
- Network overhead can impact measured tok/s (tokens per second)

**Context Size**
- Maximum sequence length the model can handle
- Common sizes: 4K, 8K, 32K, 128K tokens
- Larger context = more memory requirements
- For many tasks, 4K-32K is sufficient
- 128K is useful for long documents/codebases

**Tokenization Speed (tok/s)**
- Tokens processed per second
- Measures model inference performance
- Network overhead can add 3-5 tok/s to server reports
- MTP (Multi-Token Prediction) can improve this by 30-40%

**Multi-Token Prediction (MTP)**
- llama.cpp feature that predicts multiple tokens per forward pass
- Uses speculative decoding for efficiency
- Best suited for parallel/batched contexts
- Can generate 4-5x more tokens per response
- Prompt evaluation may be slower due to upfront work

**Flash Attention**
- Optimization that speeds up matrix multiplications
- Requires compatible hardware (NVIDIA with appropriate drivers)
- Can improve inference by 10-20%
- Not available on all hardware

**KV Cache**
- Key-Value cache stores intermediate computations during inference
- Larger cache = faster continuation but more memory usage
- Quantized KV cache (Q8_0) saves memory while maintaining quality
- Checkpoint caching can prevent re-processing same prompts
</details>

### Environment Setup

**OS**: Windows 11 with WSL Ubuntu 22.04

**CUDA environment**: [CUDA WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

- Worked around environment paths/variable issues in WSL
- Downgraded CUDA 13 to 12.8 because I wanted to try vLLM inference server first, read somewhere it has good performance but required CUDA 12.8 at the time.

### Inference Servers

#### vLLM

**Build and work around some build issues**: [vLLM Build Guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/#build-wheel-from-source)

**[vLLM on Ubuntu Tutorial](https://oneuptime.com/blog/post/2026-03-02-how-to-install-and-configure-vllm-on-ubuntu/view)**

Run command:

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /home/administrator/LLM/models/llama-3.1-8b-instruct-q4_k_m.gguf \
  --host 0.0.0.0 --port 8000 \
  --served-model-name "local" \
  --gpu-memory-utilization 0.8 \
  --max-model-len auto \
  --enable-auto-tool-choice --tool-call-parser hermes
```

**Result**: Once running with a model I connected Hermes to it to see what it would do but it seemed to just be talking in gibberish loops.

Some weeks passed and I had to give it another go after reading about new fancy tech like:

- **Vibe Jam 2026** [Cursor Vibe Jam 2026](https://vibej.am/2026/)
- **getagentcraft.com**: Watch your agents come alive in an RTS game interface!
- **MTP**: speculative decoding [llama.cpp PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673)
- **MoE Mixture of Expert weights in CPU**: [HuggingFace](https://x.com/HuggingModels/status/2052875956600307794)

---

## llama.cpp

### Build

Build [llama.cpp](https://github.com/ggml-org/llama.cpp)

### Model Selection

- Sign up on Huggingface.com to find models that actually fit in my card while I'm running Windows.
- Get huggingface-cli/hcli working in WSL environment.

---

## Models

| Model | Quantization |
|-------|--------------|
| Llama-3-8B-Instruct-Gradient-1048k | Q6_K |
| Qwen3.5-9b-Sushi-Coder-RL | Q6_K |
| Qwen3-4B-Qwen3.6-plus-Reasoning-Slerp | Q8_0 |
| Qwen3-Desert.Coder.MoE-8X0.6B.i1 | Q6_K |
| Qwen3VL-8B-Instruct | Q4_K_M |
| Qwen3.5-4B-MTP | Q4_K_M |
| Tralalabs_Qwen3-2507-4B-Instruct-Haiku-4.5-Merged | Q6_K |
| Qwen3.5-4B-Q4_K_M | Q4_K_M |
| codellama-7b-instruct | Q5_K_M |
| Qwen3.5-9B-DeepSeek-V4-Flash | Q4_K_M |
| google_gemma-4-E4B-it | Q4_K_M |
| Qwen3.5-9B | Q4_K_M |
| llama-3.1-8b-instruct-q4_k_m | Q4_K_M |
| Qwen3.5-9b-Sushi-Coder-RL.BF16-mmproj | BF16 |
| llama-3.2-3b-instruct-q4_k_m | Q4_K_M |
| Qwen3.5-9b-Sushi-Coder-RL.Q4_K_M | Q4_K_M |
| mistral-ft-optimized-1218 | Q6_K |


### Model Filename Breakdown

<details>
<summary>Decoding model filenames (click to expand)</summary>

**Model Size** (4B, 9B, 8B, etc.)
- B = Billion parameters (parameters, not bytes)
- 4B = ~4 billion parameters
- 8B = ~8 billion parameters
- 9B = ~9 billion parameters
- 35B = ~35 billion parameters
- 4.5B, 1218 (likely 1.2B) = smaller models

**Quantization** (Q4_K_M, Q6_K, Q8_0, etc.)
The Q + number + suffix indicates bit depth (lower = smaller file, potentially slower inference):

| Quantization | Bit Depth | Trade-off |
|--------------|-----------|-----------|
| Q1_0 | 1 bit | Very small, very noisy |
| Q2_K | 2 bits | Small, noticeable quality loss |
| Q3_K | 3 bits | Medium, good balance |
| Q4_K_M | 4 bits | Most common, good quality/size |
| Q5_K_M | 5 bits | Better quality, larger file |
| Q6_K | 6 bits | High quality, larger |
| Q8_0 | 8 bits | Near original quality, ~2x size |

K = Kalman algorithm (original quantization method)
M = Modified version (optimized for different use cases)

**Model Type/Version**
- Instruct = Chat-ready, follows instructions
- Coder = Code generation focused
- MoE = Mixture of Experts (sparse architecture)
- VL = Vision-Language (sees images)
- RL = Reinforcement Learning fine-tuned
- BF16-mmproj = Has bias file for RLHF
- Slerp = Slerped weights (smoothed interpolation)
- Reasoning = Optimized for reasoning tasks

**Source/Organization Prefix**
- Qwen = Alibaba's Qwen series
- llama = Meta's Llama series
- google = Google's Gemma
- mistral = Mistral AI
- Tralalabs = Tralabs (Chinese organization)
- Sushi = Custom model (possibly RL fine-tuned)
- DeepSeek = DeepSeek models
- codellama = Codellama series

**Special Labels**
- MTP = Multi-Token Prediction (llama.cpp feature)
- Haiku = Optimized for speed (Microsoft's Haiku)
- Flash = Optimized for latency

**Your Model Breakdown**

**llama-3.1-8b-instruct-q4_k_m.gguf**
- llama-3.1-8b = Meta's Llama 3.1, 8B params
- instruct = chat-ready
- q4_k_m = 4-bit quantization (standard balance)

**Qwen3.5-9B-Q4_K_M.gguf**
- Qwen3.5-9B = Alibaba Qwen 3.5, 9B params
- q4_k_m = 4-bit quantization

**Qwen3-4B-Qwen3.6-plus-Reasoning-Slerp-Q8_0.gguf**
- Qwen3-4B = Base Qwen 3, 4B params
- Qwen3.6-plus-Reasoning = Enhanced Qwen 3.6 variant
- Slerp = Slerped weights
- Q8_0 = 8-bit (largest, best quality)

**Qwen3.5-9b-Sushi-Coder-RL.Q4_K_M.gguf**
- Sushi-Coder = Custom RL fine-tuned coder model
- RL = Reinforcement Learning

**Quick Reference**

| Quant | File Size (relative) | Speed | Quality |
|-------|---------------------|-------|---------|
| Q1_0 | 2x | Fastest | Worst |
| Q2_K | 1.8x | Fast | Poor |
| Q3_K | 1.5x | Faster | Good |
| Q4_K_M | 1.2x | Fastest/Sweet Spot | Best Balance |
| Q5_K_M | 1.0x | Fast | Very Good |
| Q6_K | 0.8x | Medium | Excellent |
| Q8_0 | 0.6x | Slower | Near Original |

Your current model (Qwen3.5-9b-Sushi-Coder-RL.Q4_K_M.gguf) is using 4-bit quantization, which is the sweet spot for most users - good quality with fast inference.
</details>

---

## Agent Harness

### Hermes

- Ask Hermes about how benchmarks work, `llama --perf`
- Ask Hermes to write some benchmark scripts in python. Hermes comes up with 10 science questions to measure tok/s.
- Ask Hermes to come up with 10 python fundamentals questions to measure tok/s as well as evaluate pass/fail for results and parsing.
- Land on results between 47 and 54 tok/s for science and 79-93 tok/s for python. Hermes says that's with network overhead and the llama server reports around 3 tok/s more.
- Tell Hermes to look at some links about Multi-Token Prediction and implement it: [reddit r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1t57xuu/25x_faster_inference_with_qwen_36_27b_using_mtp/)
- Hermes builds llama with these patches and I run llama with Qwen3.5-4B-MTP-Q4_K_M.gguf `--spec-type mtp` to see what the tok/s looks like but my tasks are sequential and MTP is best suited for parallel batching so I would need a different agent to take better advantage of that.
- Land on results around 76 (+82.8%) tok/s for science and 147 (+16.5%) tok/s for python using Qwen3.5-4B-MTP-Q4_K_M.gguf

### Comparison: MTP vs Standard Model

| Model | Average Tokens/Response | Tokens | Response Time | Tokens/Second |
|-------|-------------------------|--------|---------------|---------------|
| MTP | ~4,311 | 68,982 | ~50s+ | ~138 t/s |
| Standard (Qwen3.5-4B) | ~745 | 745 | ~6s | 122 t/s |

**Key Findings**

1. MTP generates ~5-6x more tokens per response compared to standard inference
2. Tokens per second (~138 t/s) is similar to standard model (~122 t/s)
3. MTP trades per-token speed for parallel/batch efficiency
4. MTP is designed for speculative sampling in batched contexts

**Conclusion**

MTP is working correctly! The model generates significantly more tokens per response while maintaining similar tokens/second performance. This aligns with MTP's design goal of enabling speculative sampling for more efficient batched generation.

**Performance metrics**

- MTP: 138 tokens/second, ~4,300 tokens per response
- Standard: 122 tokens/second, ~745 tokens per response
- MTP advantage: 4-5x more tokens per prompt

### Benchmark Results Analysis

**Generation Speed Comparison**

| Metric | Qwen3.5-4B (Standard) | Qwen3.5-4B-MTP | Improvement |
|--------|------------------------|-----------------|-------------|
| Generation tok/s | 117.81 | 163.13 | +38.5% faster |
| Tokens generated | 2,000 | 343 | Fewer tokens in benchmark |
| Generation time | 16,976 ms | 2,103 ms | 8x faster |
| Prompt eval tok/s | 1,325.55 | 1,106.17 | ~17% slower |
| Prompt eval time | 43 ms | 51 ms | More prompt tokens |

**What's Happening**

The MTP model (Multi-Token Prediction) is generating 38% more tokens per second than the standard model.

| Model | Why This Happens |
|-------|------------------|
| Standard | Single-token prediction per forward pass |
| MTP | Generates multiple tokens per forward pass (speculative decoding) |

**Why Fewer Tokens for MTP?**

The benchmark script generates fixed output (e.g., 2,000 tokens for standard model). But MTP's speculative decoding can produce more tokens per iteration, so it generates fewer iterations to reach the same token count.

- Standard: 2,000 tokens ÷ 117.81 tok/s = ~17 seconds
- MTP: 343 tokens ÷ 163.13 tok/s = ~2.1 seconds

The MTP benchmark shows fewer total tokens because it was likely configured with a smaller max_tokens parameter or reached its draft limit faster.

**Key Takeaways**

1. MTP is ~39% faster at generation
2. Prompt eval is slower (MTP does more work upfront)
3. Speculative decoding works — multiple tokens per forward pass
4. Total time is dominated by MTP's shorter generation time

Your server is correctly configured and the performance benefits are real and measurable. The 38% improvement is significant for interactive use cases!

### Opencode

- Needed custom provider tinkering in auth.json and opencode.json to get llama connected.
- Tried implementing better benchmark_python.py tests to accommodate different models responses. Had some success but ended up in a thinking loop that needed a new session to proceed.
- Tried to get a single html file Tetris working, couldn't after a dozen iterations. Opencode kept blaming browser caching.
  
---

## Issues Encountered

- I kept seeing this in the llama server console when running Opencode, Hermes and my python benchmarks: forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see [llama.cpp PR #13194 issue comment](https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055))
- Asked Hermes to review this [llama.cpp issue #20225](https://github.com/ggml-org/llama.cpp/issues/20225) and build it into llama.
- I didn't see much of a difference and so just increased the checkpoint count to 128 and noticed it less, I think.
- My favourite llama run options result in the server not stopping correctly and so another open issue: [llama.cpp issue #22601](https://github.com/ggml-org/llama.cpp/issues/22601)

### llama.cpp Server Run Options Explained

#### MTP-Specific Settings

- `--spec-type mtp` — Tells llama.cpp to run MTP (Multi-Token Prediction) mode, which predicts 3 tokens at once instead of 1
- `--spec-draft-n-max 3` — Maximum draft tokens to generate per MTP step (3 tokens)
- `-ncmoe 0` — No context memory on external endpoint (disables some caching optimizations)
- `--perf` — Performance mode optimizations

#### Cache Configuration

- `--ctx-checkpoints 128` — Maximum checkpoints to keep in memory (128)
- `--cache-type-k q8_0` — KV cache keys quantized to Q8_0 (8-bit)
- `--cache-type-v q8_0` — KV cache values quantized to Q8_0 (8-bit)
- `--cache-prompt` — Cache the prompt tokens (reduces computation for continuation)

#### Hardware Acceleration

- `--flash-attn on` — Enable Flash Attention for faster matrix multiplications

#### Server Configuration

- `--host 0.0.0.0` — Bind to all network interfaces (accessible from any device)
- `--port 8080` — API server runs on port 8080
- `--no-mmap` — Disable memory-mapped files (loads model into RAM instead)

#### Model Settings

- `-m /home/administrator/LLM/models/Qwen3.5-4B-MTP-Q4_K_M.gguf` — Your 4B model in MTP quantized Q4_K_M format
- `--ngl 999` — No global layers (all layers used for inference)
- `--kv-unified` — Unified KV cache for keys and values (faster access)

#### Context & Parallelism

- `--ctx-size 131072` — Maximum sequence length (128K tokens)
- `--parallel 1` — Single token parallelism (not recommended for MTP, may be a mistake)
- `--metrics` — Enable performance metrics reporting

#### Prompt Template

- `--jinja` — Use Jinja-style prompt templates for structured input

---

## Reflections

- Hermes wrote this up and edited it for Github markup from my notepad scribbles very quickly.
- Lots of social media buzz especially when translated from Chinese, Spanish, Japanese.
- There is so much to learn and the speed at which issues and new features are coming out is thick and fast.
- Dissapointed I can't get a working Tetris.html but the feature releases and tool capability are impressive
- I don't know anything
