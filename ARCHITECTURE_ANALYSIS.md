# LLM Serving Submodules: Architecture Analysis

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [vLLM Architecture](#vllm-architecture)
3. [SGLang Architecture](#sglang-architecture)
4. [TensorRT-LLM Architecture](#tensorrt-llm-architecture)
5. [Transformers Architecture](#transformers-architecture)
6. [Similarities Across Serving Frameworks](#similarities-across-serving-frameworks)
7. [Key Differences](#key-differences)
8. [Relationship Between Transformers and the Serving Frameworks](#relationship-between-transformers-and-the-serving-frameworks)
9. [Summary Comparison Table](#summary-comparison-table)

---

## Executive Summary

This document analyzes the code architecture of three LLM serving frameworks (**vLLM**, **SGLang**, **TensorRT-LLM**) and the **Transformers** library, examining their internal structures, similarities, differences, and interdependencies. All three serving frameworks are purpose-built for high-throughput, low-latency inference of large language models, while Transformers serves as the foundational model-definition and weight-loading ecosystem that all three depend on.

---

## vLLM Architecture

**Repository:** `https://github.com/vllm-project/vllm`

### High-Level Structure

```
vllm/
├── vllm/                    # Core Python package
│   ├── engine/              # Legacy engine (v0)
│   ├── v1/                  # New v1 engine (active development)
│   │   ├── engine/          # EngineCore, AsyncLLM, input/output processors
│   │   ├── core/            # Scheduler, KV cache manager, block pool
│   │   ├── worker/          # GPU/CPU/XPU model runners
│   │   ├── attention/       # Attention backends (FlashAttention, FlashInfer, etc.)
│   │   ├── executor/        # Executor abstraction for distributed execution
│   │   ├── spec_decode/     # Speculative decoding (EAGLE, Medusa, n-gram)
│   │   └── structured_output/  # Grammar-guided generation
│   ├── model_executor/      # Model execution layer
│   │   ├── models/          # ~244 model implementations
│   │   ├── layers/          # Custom inference-optimized layers
│   │   └── model_loader/    # Weight loading (safetensors, GGUF, etc.)
│   ├── entrypoints/         # API servers (OpenAI-compatible, gRPC, Anthropic)
│   ├── distributed/         # Tensor parallelism, pipeline parallelism, KV transfer
│   ├── transformers_utils/  # Bridge to HuggingFace transformers
│   ├── compilation/         # torch.compile integration, CUDA graph
│   ├── lora/                # LoRA adapter support
│   ├── multimodal/          # Vision/audio input processing
│   └── kernels/             # Python-side kernel wrappers
├── csrc/                    # Custom CUDA/C++ kernels
└── requirements/            # Dependency specifications
```

### Core Architecture Pattern

vLLM follows a **multi-process, ZMQ-based IPC** architecture:

1. **AsyncLLM** (main process): The top-level async engine client that handles API requests, input processing, and output processing.
2. **EngineCore** (separate process): The inner loop that owns the scheduler and communicates with GPU workers via the Executor. Communication between AsyncLLM and EngineCore uses ZMQ sockets with msgpack serialization.
3. **Scheduler** (`v1/core/sched/`): Manages request queues, applies scheduling policies (e.g., priority, FCFS), and produces `SchedulerOutput` objects that describe each batch.
4. **Executor** (`v1/executor/`): Abstracts over single-GPU, multi-GPU (tensor/pipeline parallel), and Ray-based distributed execution.
5. **Worker / ModelRunner** (`v1/worker/`): Runs the actual model forward pass on GPU, manages KV cache block tables, and handles CUDA graphs.

### Key Design Decisions

- **PagedAttention**: vLLM pioneered paged KV cache management, splitting the KV cache into fixed-size blocks that can be allocated non-contiguously, similar to OS virtual memory.
- **Continuous batching**: New requests can be inserted into running batches at any iteration.
- **Prefix caching**: Block-level hash-based deduplication of shared prefixes across requests.
- **torch.compile integration**: The `compilation/` module provides decorators and backends for compiling model forward passes.
- **Pluggable attention backends**: Registry-based selection of FlashAttention2, FlashInfer, Triton, FlexAttention, and platform-specific backends.

---

## SGLang Architecture

**Repository:** `https://github.com/sgl-project/sglang`

### High-Level Structure

```
sglang/
├── python/sglang/
│   ├── srt/                     # SGLang Runtime (the serving engine)
│   │   ├── entrypoints/         # Engine, HTTP server, gRPC server, OpenAI API
│   │   ├── managers/            # Core process managers
│   │   │   ├── tokenizer_manager.py     # Tokenization (main process)
│   │   │   ├── scheduler.py             # Scheduling + model execution (subprocess)
│   │   │   ├── detokenizer_manager.py   # Detokenization (subprocess)
│   │   │   └── data_parallel_controller.py  # DP coordination
│   │   ├── model_executor/      # Model runner, CUDA graph runner
│   │   ├── models/              # ~161 model implementations
│   │   ├── layers/              # Inference-optimized layers
│   │   │   ├── attention/       # Attention backends (FlashInfer, FlashAttention, etc.)
│   │   │   ├── moe/             # Mixture-of-Experts layers
│   │   │   └── quantization/    # Quantization methods
│   │   ├── model_loader/        # Weight loading utilities
│   │   ├── distributed/         # TP/PP/DP support
│   │   ├── speculative/         # Speculative decoding (EAGLE, n-gram)
│   │   ├── disaggregation/      # Prefill-decode disaggregation
│   │   ├── lora/                # LoRA support
│   │   ├── multimodal/          # Multimodal input processing
│   │   └── constrained/         # Grammar-constrained generation
│   ├── lang/                    # SGLang programming language frontend
│   │   ├── ir.py                # Intermediate representation
│   │   ├── interpreter.py       # IR interpreter
│   │   ├── tracer.py            # Program tracing
│   │   └── backend/             # Execution backends
│   └── sgl-kernel/              # Custom CUDA kernel library (separate package)
│       ├── csrc/                # CUDA/C++ sources
│       └── python/              # Python bindings
```

### Core Architecture Pattern

SGLang uses a **three-process pipeline** with ZMQ-based IPC:

1. **TokenizerManager** (main process): Receives HTTP/gRPC requests, tokenizes input, forwards to scheduler via ZMQ.
2. **Scheduler** (subprocess): The central orchestrator that owns the radix cache, schedules batches, invokes the ModelRunner (tp_worker), and sends output tokens to the detokenizer. Notably, the scheduler and model execution run in the **same process** (unlike vLLM which separates them).
3. **DetokenizerManager** (subprocess): Receives generated token IDs, detokenizes them, and returns results to the TokenizerManager.

### Key Design Decisions

- **RadixAttention**: SGLang's signature innovation. Uses a radix tree for prefix caching that enables automatic, fine-grained prefix sharing at the token level (vs. vLLM's block-level approach).
- **Unified scheduler + worker**: The scheduler directly manages the model runner in-process, reducing IPC overhead.
- **SGLang frontend language** (`lang/`): A unique programming model that allows composing LLM calls with Python control flow, enabling multi-turn conversations, branching, and constrained generation in a structured program.
- **sgl-kernel**: A separately packaged CUDA kernel library, making kernel development and testing independent from the runtime.
- **Heavily forked from vLLM**: Many model implementations and layers explicitly note they were "Adapted from vLLM", indicating shared lineage in the model executor layer.

---

## TensorRT-LLM Architecture

**Repository:** `https://github.com/NVIDIA/TensorRT-LLM`

### High-Level Structure

```
tensorrt_llm/
├── tensorrt_llm/                # Core Python package
│   ├── models/                  # ~35 model families (directory-per-model)
│   │   ├── llama/               # config.py, convert.py, model.py pattern
│   │   ├── modeling_utils.py    # Base classes (DecoderModelForCausalLM)
│   │   └── model_weights_loader.py  # Weight conversion from HF
│   ├── _torch/                  # PyTorch-native execution path (newer)
│   │   ├── models/              # ~40 torch-native model implementations
│   │   ├── modules/             # Torch-native inference layers
│   │   ├── attention_backend/   # Attention backends (TRT, FlashInfer, vanilla)
│   │   ├── pyexecutor/          # Python-based executor
│   │   ├── speculative/         # Speculative decoding
│   │   ├── disaggregation/      # Disaggregated serving
│   │   └── compilation/         # Compilation and graph optimization
│   ├── layers/                  # TRT-native layers (Attention, MoE, MLP, etc.)
│   ├── executor/                # C++ executor bindings + Python wrapper
│   ├── runtime/                 # TRT engine runtime, KV cache management
│   ├── llmapi/                  # High-level LLM API (user-facing)
│   ├── serve/                   # OpenAI-compatible HTTP server
│   ├── builder.py               # TensorRT engine builder
│   ├── quantization/            # Quantization tools (ModelOpt integration)
│   ├── scaffolding/             # Agentic/tool-use scaffolding
│   └── functional.py            # TRT functional operations
├── cpp/                         # C++ runtime and kernels
│   ├── tensorrt_llm/            # C++ library core
│   └── kernels/                 # CUDA kernels (fmha_v2, xqa)
└── triton_backend/              # Triton inference server backend
```

### Core Architecture Pattern

TensorRT-LLM has a **dual execution path**:

#### Path 1: TensorRT Engine Path (Traditional)
1. **Build Phase**: Converts a HuggingFace model to a TensorRT graph representation using TRT-LLM's own `Module`/`Layer` abstraction (defined in `layers/`), then compiles it to a TensorRT engine.
2. **Runtime Phase**: The C++ executor (`executor/`, backed by `cpp/`) loads the TRT engine and runs inference with the C++ runtime managing KV cache, scheduling, and batching.
3. Models in `models/` (e.g., `models/llama/model.py`) use TRT-LLM's own `Module` class (not `nn.Module`) with TRT functional ops.

#### Path 2: PyTorch-Native Path (Newer, `_torch/`)
1. Models in `_torch/models/` (e.g., `modeling_llama.py`) use standard PyTorch `nn.Module` with TRT-LLM optimized modules.
2. The `pyexecutor/` manages scheduling and batching in Python.
3. This path provides faster iteration and broader model support without needing the TRT compilation step.

### Key Design Decisions

- **Two-phase build + deploy**: The traditional path requires an offline build step to compile the model into a TRT engine, unlike vLLM/SGLang which load weights directly.
- **C++ core runtime**: The executor, scheduler, and KV cache management are implemented in C++ for maximum performance, with Python bindings.
- **Deep NVIDIA integration**: Uses NVIDIA-specific optimizations: TensorRT, cuBLAS, custom XQA/FMHA CUDA kernels, NVLink-aware communication.
- **Triton Inference Server integration**: Includes a backend for NVIDIA Triton, enabling enterprise deployment workflows.
- **Weight conversion pipeline**: Each model has a dedicated `convert.py` that transforms HuggingFace weights into TRT-LLM's internal format.
- **Directory-per-model** organization (vs. single-file-per-model in vLLM/SGLang).

---

## Transformers Architecture

**Repository:** `https://github.com/huggingface/transformers`

### High-Level Structure

```
transformers/
├── src/transformers/
│   ├── models/                  # ~422 model families
│   │   └── llama/               # Per-model: configuration, modeling, tokenization
│   │       ├── configuration_llama.py   # LlamaConfig
│   │       ├── modeling_llama.py        # LlamaForCausalLM (nn.Module)
│   │       └── tokenization_llama.py    # LlamaTokenizer
│   ├── modeling_utils.py        # PreTrainedModel base class
│   ├── configuration_utils.py   # PreTrainedConfig base class
│   ├── generation/              # Text generation infrastructure
│   │   ├── utils.py             # GenerationMixin (greedy, beam, sampling)
│   │   ├── continuous_batching/ # New: continuous batching support
│   │   ├── logits_process.py    # Logits processors
│   │   └── stopping_criteria.py
│   ├── pipelines/               # High-level task pipelines
│   ├── tokenization_utils*.py   # Tokenizer infrastructure
│   ├── cache_utils.py           # KV cache abstractions (Dynamic, Static, Paged)
│   ├── quantizers/              # Quantization framework (~24 backends)
│   ├── integrations/            # Third-party integrations (FlashAttention, PEFT, etc.)
│   ├── distributed/             # Basic distributed utilities
│   ├── modeling_rope_utils.py   # RoPE implementations
│   ├── modeling_attn_mask_utils.py  # Attention mask helpers
│   └── utils/                   # General utilities
```

### Core Architecture Pattern

Transformers follows a **model-centric library** pattern (not a serving framework):

1. **PreTrainedConfig**: Serializable configuration class that defines model hyperparameters. Each model has its own config (e.g., `LlamaConfig`). Configs are stored as `config.json` in HuggingFace Hub repositories.
2. **PreTrainedModel**: Base class providing weight loading (`from_pretrained`), saving, device management, and generation capabilities. Built on `nn.Module`.
3. **GenerationMixin**: Provides `generate()` method with greedy, beam search, sampling, contrastive search, and speculative decoding strategies.
4. **AutoModel / AutoConfig / AutoTokenizer**: Registry-based dispatch that maps `model_type` strings to concrete classes, enabling `AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")`.
5. **Pipelines**: High-level task-oriented API (text-generation, question-answering, etc.).

### Key Design Decisions

- **Training + inference**: Designed for both fine-tuning and inference, with full gradient computation support.
- **Readability over performance**: Model code prioritizes clarity and educational value over raw inference speed.
- **Ecosystem hub**: The HuggingFace Hub integration makes it the de facto standard for distributing model weights and configs.
- **Pluggable attention**: Supports `sdpa`, `flash_attention_2`, `flex_attention`, and `eager` attention implementations via the `attn_implementation` parameter.
- **Quantization-agnostic**: The `quantizers/` framework supports 24+ quantization backends (GPTQ, AWQ, bitsandbytes, etc.).
- **New: Continuous batching** (`generation/continuous_batching/`): A recent addition bringing basic continuous batching, paged attention cache, and scheduling to Transformers itself. This signals Transformers is beginning to incorporate serving-framework concepts.

---

## Similarities Across Serving Frameworks

### 1. Continuous Batching with Paged KV Cache
All three serving frameworks implement continuous (or iteration-level) batching, where new requests can join and completed requests can leave a running batch at each decoding step. All use some form of paged/block-based KV cache management.

### 2. OpenAI-Compatible API Layer
All three provide OpenAI API-compatible HTTP endpoints:
- **vLLM**: `entrypoints/openai/` with full chat/completion/embedding support
- **SGLang**: `entrypoints/openai/` with similar coverage
- **TensorRT-LLM**: `serve/openai_server.py`

### 3. Scheduler -> Worker Architecture
Each has a scheduler that selects which requests to batch, and a worker/model runner that executes the batch:
- **vLLM**: `v1/core/sched/scheduler.py` -> `v1/worker/gpu_model_runner.py`
- **SGLang**: `managers/scheduler.py` -> `model_executor/model_runner.py`
- **TensorRT-LLM**: `_torch/pyexecutor/scheduler/` -> `_torch/pyexecutor/model_engine.py`

### 4. Re-implemented Model Layers for Inference
All three re-implement core neural network layers optimized for inference:
- **Custom Attention**: Fused multi-head/grouped-query attention with KV cache
- **Parallel Linear**: Tensor-parallel column/row linear layers
- **Fused MoE**: Custom mixture-of-experts kernels
- **Rotary Embedding**: Optimized RoPE implementations
- **Layer Normalization**: Fused RMSNorm/LayerNorm

### 5. Speculative Decoding Support
All three support speculative decoding to improve throughput:
- **vLLM**: EAGLE, Medusa, n-gram, suffix decoding
- **SGLang**: EAGLE (v1, v2, multi-layer), n-gram
- **TensorRT-LLM**: EAGLE3, MTP, n-gram, model drafter

### 6. Disaggregated Serving (Prefill-Decode Separation)
All three have support for separating prefill and decode phases across different GPU groups:
- **vLLM**: `distributed/kv_transfer/`
- **SGLang**: `srt/disaggregation/` (Mooncake, NIXL backends)
- **TensorRT-LLM**: `_torch/disaggregation/` (NIXL, native)

### 7. LoRA / Adapter Support
All three support serving multiple LoRA adapters simultaneously:
- **vLLM**: `lora/` module
- **SGLang**: `srt/lora/`
- **TensorRT-LLM**: `lora_manager.py` and `_torch/peft/`

### 8. ZMQ / IPC-Based Communication
Both vLLM and SGLang use ZMQ for inter-process communication between engine components. TensorRT-LLM uses a C++ executor with RPC for distributed coordination.

### 9. Multimodal Input Support
All three support vision-language and audio-language models:
- **vLLM**: `multimodal/` module with registry
- **SGLang**: `srt/multimodal/`
- **TensorRT-LLM**: `_torch/models/modeling_multimodal_*.py`

---

## Key Differences

### 1. Language and Runtime Core

| Aspect | vLLM | SGLang | TensorRT-LLM |
|--------|------|--------|---------------|
| **Core language** | Python (with CUDA kernels) | Python (with CUDA kernels) | C++ core + Python wrapper |
| **Execution** | PyTorch-native | PyTorch-native | Dual: TensorRT engine OR PyTorch-native |
| **Build step** | None (direct weight loading) | None (direct weight loading) | Optional TRT engine compilation |

### 2. Scheduling Architecture

- **vLLM**: Scheduler runs in a **separate process** from the API layer, communicating via ZMQ. The EngineCore (scheduler + executor) is fully decoupled from the AsyncLLM (API handler).
- **SGLang**: Scheduler and model execution run in the **same subprocess**, reducing IPC latency. The three-process pipeline is: TokenizerManager | Scheduler+ModelRunner | DetokenizerManager.
- **TensorRT-LLM**: In the traditional path, scheduling is in the **C++ executor**; in the PyTorch path, a Python-based scheduler in `pyexecutor/` coordinates execution.

### 3. KV Cache Management

- **vLLM**: **PagedAttention** with block-level hashing for prefix caching. Blocks are fixed-size and managed by a block pool.
- **SGLang**: **RadixAttention** using a radix tree for token-level prefix sharing. More fine-grained than vLLM's block-level approach, enabling better cache reuse for multi-turn conversations.
- **TensorRT-LLM**: Block-based paged KV cache managed primarily in C++, with a Python wrapper in `runtime/kv_cache_manager.py`.

### 4. Model Count and Organization

| Framework | Model Count | Organization |
|-----------|------------|--------------|
| **vLLM** | ~244 models | Single file per model (`models/llama.py`) |
| **SGLang** | ~161 models | Single file per model (many forked from vLLM) |
| **TensorRT-LLM** | ~35 families (TRT) + ~40 (PyTorch) | Directory per model family (`models/llama/`) |
| **Transformers** | ~422 families | Directory per model family |

### 5. Custom Kernel Strategy

- **vLLM**: Custom CUDA kernels in `csrc/` compiled via PyTorch's extension mechanism. Extensive CUTLASS integration for quantized GEMM.
- **SGLang**: Separate `sgl-kernel` package with its own build system (CMake). Clean separation allows kernel development independently.
- **TensorRT-LLM**: Kernels in `cpp/kernels/` (FMHA, XQA) plus deep integration with TensorRT's internal kernel libraries and cuBLAS.

### 6. Unique Features

- **vLLM**: `torch.compile` integration for model compilation (`compilation/`); plugin system; most extensive model support.
- **SGLang**: SGLang programming language (`lang/`) for structured LLM programs; RadixAttention for superior prefix caching; data-parallel attention.
- **TensorRT-LLM**: TensorRT engine compilation for peak performance; C++ runtime for minimal overhead; Triton Inference Server integration; scaffolding module for agentic workflows.

### 7. Attention Backend Ecosystem

- **vLLM** (21 backends): FlashAttention, FlashInfer, Triton, FlexAttention, ROCm, MLA, Mamba, linear attention, tree attention, etc.
- **SGLang** (30+ backends): FlashAttention, FlashInfer, FlashMLA, CUTLASS MLA, double sparsity, NSA, wave, TBO, hybrid, TRT-LLM backends, etc.
- **TensorRT-LLM** (7 backends): TRT-LLM native, FlashInfer, Star-FlashInfer, vanilla, sparse variants.

### 8. Distributed Execution

- **vLLM**: Tensor parallelism, pipeline parallelism, Ray-based distributed, expert parallelism for MoE, KV cache transfer for disaggregation.
- **SGLang**: TP, PP, data parallelism with a dedicated `data_parallel_controller`, expert parallelism load balancing (`eplb/`), disaggregated prefill-decode.
- **TensorRT-LLM**: MPI-based + NCCL for multi-GPU, multi-node support with leader/worker pattern, C++-level communication primitives.

---

## Relationship Between Transformers and the Serving Frameworks

### Transformers as the Foundation Layer

The HuggingFace Transformers library serves as the **foundational ecosystem** for all three serving frameworks in several critical ways:

#### 1. Model Configuration (Critical Dependency)
All three frameworks load model configurations using `transformers.AutoConfig.from_pretrained()`:

```python
# vLLM (transformers_utils/config.py)
from transformers import AutoConfig, PretrainedConfig

# SGLang (model_loader/loader.py)
from transformers import AutoConfig, AutoModelForCausalLM

# TensorRT-LLM (models/automodel.py)
hf_config = transformers.AutoConfig.from_pretrained(hf_model_or_dir)
```

The `config.json` format defined by Transformers is the universal interface for describing model architectures.

#### 2. Model Weights and Format (Critical Dependency)
All three load weights from HuggingFace Hub repositories using the safetensors/pytorch format defined by Transformers:

- **vLLM**: `model_executor/model_loader/weight_utils.py` uses `huggingface_hub` + `safetensors`
- **SGLang**: `model_loader/weight_utils.py` (largely forked from vLLM's equivalent)
- **TensorRT-LLM**: `models/model_weights_loader.py` + per-model `convert.py` scripts

#### 3. Tokenizer (Critical Dependency)
All three frameworks use Transformers tokenizers:

```python
# All three use:
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

#### 4. Per-Model Config Classes (Direct Import)
Model implementations in all three serving frameworks directly import Transformers config classes:

```python
# vLLM llama.py
from transformers import LlamaConfig

# SGLang llama.py
from transformers import LlamaConfig

# TensorRT-LLM modeling_llama.py
from transformers import LlamaConfig, PretrainedConfig
```

This means each model's hyperparameters (hidden size, number of layers, vocab size, etc.) are defined by Transformers config classes.

#### 5. Architecture Registry (Implicit Dependency)
The `architectures` field in `config.json` (defined by Transformers) is used by all three frameworks to dispatch to the correct model implementation:

```json
{"architectures": ["LlamaForCausalLM"], "model_type": "llama", ...}
```

Each framework maintains its own mapping from these architecture strings to its re-implemented model classes.

### How They Re-implement Transformers Models

All three serving frameworks **re-implement** model architectures (they do NOT reuse `transformers.models.*` for inference). The pattern is:

1. **Read** the HuggingFace `config.json` using `transformers.AutoConfig`
2. **Instantiate** their own model class with custom inference-optimized layers
3. **Load** the HuggingFace weights and remap them into their own model structure
4. **Execute** with optimized attention, KV caching, and continuous batching

The re-implementations differ from Transformers' originals in:
- Fused operations (QKV projection, gate+up projection)
- Tensor-parallel linear layers
- Paged KV cache integration in attention
- Removal of training-only features (dropout, gradient checkpointing)
- Quantization-aware weight loading

### Transformers' New Serving Capabilities

Interestingly, Transformers itself is beginning to incorporate serving-framework concepts:

- **`generation/continuous_batching/`**: A new continuous batching scheduler with paged attention cache
- **`cache_utils.py`**: Increasingly sophisticated KV cache abstractions (Static, Dynamic, Paged, Quantized, Offloaded)
- **`integrations/eager_paged.py`, `flash_paged.py`, `sdpa_paged.py`**: Paged attention implementations

This suggests a convergence where Transformers is evolving from a pure model library toward basic serving capabilities, potentially simplifying integration with the serving frameworks in the future.

### Dependency Version Alignment

All three frameworks pin specific Transformers versions:
- **vLLM**: `transformers >= 4.56.0, < 5` (tests pin `4.57.5`)
- **SGLang**: `transformers == 4.57.1`
- **TensorRT-LLM**: `transformers == 4.57.1`

This tight version pinning reflects how deeply the serving frameworks depend on Transformers' config formats, tokenizer APIs, and weight-loading utilities. Breaking changes in Transformers can cascade into all three.

---

## Summary Comparison Table

| Feature | vLLM | SGLang | TensorRT-LLM | Transformers |
|---------|------|--------|---------------|--------------|
| **Primary Purpose** | LLM serving | LLM serving + programming | LLM serving (NVIDIA-optimized) | Model library + training |
| **Language** | Python + CUDA | Python + CUDA | C++ + Python + CUDA | Python |
| **Model Count** | ~244 | ~161 | ~35-75 | ~422 |
| **KV Cache Strategy** | PagedAttention (blocks) | RadixAttention (radix tree) | Paged (C++ managed) | Dynamic/Static/Paged |
| **Scheduling** | Separate process | Same process as worker | C++ executor or Python | New: basic continuous batching |
| **IPC** | ZMQ + msgpack | ZMQ | C++ RPC / MPI | N/A |
| **API Server** | OpenAI, Anthropic, gRPC | OpenAI, Ollama, gRPC | OpenAI (via serve/) | Pipelines (not serving) |
| **Compilation** | torch.compile, CUDA graph | CUDA graph | TensorRT engine, CUDA graph | torch.compile |
| **Speculative Decoding** | EAGLE, Medusa, n-gram | EAGLE, n-gram | EAGLE3, MTP, n-gram | Assisted generation |
| **Quantization** | ~30 methods | ~30 methods | ModelOpt + native | ~24 backends |
| **LoRA** | Yes (multi-adapter) | Yes (multi-adapter) | Yes | Via PEFT integration |
| **Disaggregated Serving** | KV transfer | Prefill-decode split | NIXL / native | No |
| **Unique Strength** | Largest model coverage, torch.compile | RadixAttention, SGLang language | Peak NVIDIA performance, C++ core | Ecosystem standard, training support |
| **Transformers Dependency** | Config + tokenizer + weights | Config + tokenizer + weights | Config + tokenizer + weights + conversion | Self (is the dependency) |

---

## Conclusion

The three LLM serving frameworks share a common architectural blueprint -- continuous batching with paged KV cache, re-implemented model layers, pluggable attention backends, and OpenAI-compatible APIs -- while differentiating on execution strategy and key innovations. vLLM leads in model breadth and community adoption, SGLang innovates with RadixAttention and its programming frontend, and TensorRT-LLM delivers peak NVIDIA hardware performance through its C++ core and TensorRT compilation.

Transformers sits beneath all three as the indispensable foundation, defining the model configuration format, tokenizer interface, and weight distribution ecosystem that the entire LLM serving landscape depends on. The relationship is symbiotic: Transformers defines the models and distributes the weights; the serving frameworks re-implement those models for production-grade inference.
