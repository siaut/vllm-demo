#!/usr/bin/env python3
"""
LMCache TTFT demonstration script.

Methodology:
  PHASE 1 — COLD baseline:
    Sends prompts with unique never-seen prefixes. GPU cache empty, disk cache
    empty. Measures full prefill latency as the baseline.

  PHASE 2 — WARM (LMCache disk hits):
    Sends the same prompts again. GPU cache empty (enablePrefixCaching=false),
    but LMCache disk has all chunks. Measures TTFT with disk cache hits.

  The speedup = cold_median / warm_median. With ~8000 token shared context
  and only a short unique question, expect 8-15X TTFT improvement.

Usage:
    python3 test_lmcache.py --base-url http://<router-ip>/v1
    python3 test_lmcache.py --base-url http://<router-ip>/v1 --duration 120
"""

import argparse
import time
import statistics
import json
import urllib.request
import urllib.error
from datetime import datetime

# ---------------------------------------------------------------------------
# SHARED_CONTEXT is ~8000 tokens — sent identically on every request.
# This is the portion LMCache caches on disk after phase 1.
# On phase 2, all ~8000 token prefill is skipped; only the short question
# (~20 tokens) is computed.
# ---------------------------------------------------------------------------
SHARED_CONTEXT = """\
You are an expert AI assistant and technical architect. Below is a comprehensive
technical reference spanning machine learning, distributed systems, Kubernetes,
storage, and LLM inference optimization. Read it carefully and answer questions
based solely on the content below.

================================================================================
PART I: TRANSFORMER ARCHITECTURE AND ATTENTION MECHANISMS
================================================================================

1.1 SCALED DOT-PRODUCT ATTENTION
The fundamental operation in transformers is scaled dot-product attention:

    Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

where Q (query), K (key), and V (value) matrices are obtained by learned linear
projections of the input sequence X: Q = X * W_Q, K = X * W_K, V = X * W_V,
with W_Q, W_K, W_V in R^(d_model x d_k). The scaling factor 1/sqrt(d_k) prevents
the dot products from growing large in magnitude, which would push the softmax
into regions with extremely small gradients and slow learning.

For a sequence of length n, the attention matrix is n x n, giving O(n^2 * d)
time and memory complexity. This quadratic scaling is the primary challenge for
long-context models and motivates approximate attention variants such as
FlashAttention (IO-aware exact attention via tiling), Sparse Attention, and
Linear Attention approximations.

1.2 MULTI-HEAD ATTENTION (MHA)
Multi-head attention runs h independent attention functions in parallel:

    MultiHead(Q,K,V) = Concat(head_1,...,head_h) * W_O
    head_i = Attention(Q * W_Q_i, K * W_K_i, V * W_V_i)

with W_Q_i, W_K_i, W_V_i in R^(d_model x d_k), d_k = d_model / h.
This allows the model to jointly attend to information from different
representation subspaces at different positions, capturing diverse syntactic
and semantic relationships simultaneously.

1.3 GROUPED QUERY ATTENTION (GQA) AND MULTI-QUERY ATTENTION (MQA)
Standard MHA has h query, key, and value heads. MQA uses a single K,V head
shared across all h query heads — reducing KV cache memory by factor h but
at some accuracy cost. GQA is a compromise: g groups each with one K,V head
and h/g query heads. Mistral-7B uses GQA with 8 KV heads and 32 query heads.

KV cache size for Mistral-7B per token:
  32 layers * 2 (K+V) * 8 KV heads * 128 head_dim * 2 bytes (fp16) = 131,072 bytes = 128 KB/token

At 16,384 token context: 128 KB * 16384 = 2 GB KV cache per sequence.

1.4 POSITIONAL ENCODINGS
The original transformer uses fixed sinusoidal encodings:
  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Modern LLMs use Rotary Position Embeddings (RoPE), which apply a rotation
matrix to query and key vectors based on position. RoPE is relative (encodes
position difference rather than absolute position), compatible with KV caching,
and naturally extends to longer sequences via frequency scaling (YaRN, NTK).

Mistral-7B uses RoPE with base frequency 10000 and supports up to 32768 tokens
via dynamic NTK-aware scaling.

================================================================================
PART II: LARGE LANGUAGE MODEL TRAINING AT SCALE
================================================================================

2.1 TOKENIZATION
Byte Pair Encoding (BPE) iteratively merges the most frequent byte pair in the
corpus into a new token, building a vocabulary bottom-up from characters.
SentencePiece implements BPE and unigram language model tokenization directly on
raw Unicode text without pre-tokenization, enabling language-agnostic tokenizers.
Modern LLMs use vocabularies of 32K-128K tokens. Mistral uses a 32K BPE vocab.

2.2 OPTIMIZER AND LEARNING RATE SCHEDULE
AdamW extends Adam by decoupling weight decay from the gradient update:
  m_t = beta1 * m_{t-1} + (1-beta1) * g_t
  v_t = beta2 * v_{t-1} + (1-beta2) * g_t^2
  theta_t = theta_{t-1} - lr * m_t / (sqrt(v_t) + eps) - lr * lambda * theta_{t-1}

Typical hyperparameters: beta1=0.9, beta2=0.95, eps=1e-8, lambda=0.1.
Learning rate schedule: linear warmup over 2000 steps to peak lr, then cosine
decay to 10% of peak. Gradient clipping at global norm 1.0 prevents exploding
gradients during early training instability.

2.3 PARALLELISM STRATEGIES
Tensor Parallelism (Megatron-LM): Column-parallel linear splits weight matrix
W along columns across N GPUs; row-parallel linear splits along rows with an
all-reduce of the result. Each GPU holds 1/N of each weight matrix. Communication
cost: two all-reduce operations per transformer layer, each of size batch*seq*d_model.

Pipeline Parallelism: Layer groups assigned to pipeline stages. GPipe uses
micro-batching to fill the pipeline bubble (idle time = (stages-1)/stages).
PipeDream-Flush uses 1F1B (one forward, one backward) scheduling to reduce
peak activation memory while maintaining throughput.

ZeRO (Zero Redundancy Optimizer) shards across data parallel ranks:
  ZeRO-1: optimizer states only (4x memory reduction for Adam)
  ZeRO-2: + gradients (8x reduction)
  ZeRO-3: + parameters (linear reduction with DP degree)
ZeRO-Infinity offloads to CPU and NVMe for trillion-parameter models.

2.4 ALIGNMENT: RLHF, DPO, AND CONSTITUTIONAL AI
RLHF (Reinforcement Learning from Human Feedback):
  Step 1 — SFT: fine-tune on high-quality demonstrations.
  Step 2 — RM: train reward model r(x,y) on human preference pairs (y_w > y_l).
  Step 3 — PPO: maximize E[r(x,y)] - beta * KL(pi_theta || pi_ref).
  The KL penalty prevents the policy from collapsing to reward-hacking outputs.

DPO (Direct Preference Optimization) eliminates the RL loop by reparameterizing
the RL objective and directly optimizing on preference data:
  L_DPO = -E[log sigma(beta * log(pi(y_w|x)/pi_ref(y_w|x)) - beta * log(pi(y_l|x)/pi_ref(y_l|x)))]

Constitutional AI (CAI, Anthropic): uses AI feedback guided by a written
constitution (set of principles) to generate critique-revision pairs for SFT,
followed by RL from AI feedback (RLAIF) without requiring human labelers.

================================================================================
PART III: KUBERNETES STORAGE AND NETWORKING
================================================================================

3.1 PERSISTENT STORAGE ARCHITECTURE
Kubernetes storage abstractions:
  PersistentVolume (PV): cluster-scoped resource representing physical storage.
  PersistentVolumeClaim (PVC): namespace-scoped request binding to a PV.
  StorageClass: defines provisioner, reclaim policy, binding mode, and parameters.

Access modes determine how a volume may be mounted:
  ReadWriteOnce (RWO): single node read-write. Supported by most block storage.
  ReadOnlyMany (ROX): multiple nodes read-only.
  ReadWriteMany (RWX): multiple nodes read-write. Requires distributed storage:
    NFS, CephFS, GlusterFS, IBM Spectrum Scale (GPFS), Azure Files, EFS (AWS).

Volume binding modes:
  Immediate: PVC binds at creation time, before pod scheduling.
  WaitForFirstConsumer: defers binding until a pod is scheduled, enabling
  topology-aware provisioning that places storage co-located with the pod.

3.2 CSI (CONTAINER STORAGE INTERFACE)
CSI separates storage driver development from Kubernetes core. CSI drivers
implement gRPC services: Identity, Controller (CreateVolume, DeleteVolume,
AttachVolume, CreateSnapshot), and Node (NodeStageVolume, NodePublishVolume).
The external-provisioner sidecar watches PVCs and calls the CSI Controller.
The node-driver-registrar registers the driver socket with kubelet.

3.3 NFS AND RWX STORAGE FOR AI WORKLOADS
NFS-based RWX storage is the most common backend for shared AI workloads:
  - nfs-subdir-external-provisioner: creates per-PVC subdirectories on NFS.
  - Performance characteristics: high latency for small random I/O, good for
    large sequential reads (model weights, KV cache blobs).
  - Limitations: no POSIX flock on some servers, symlinks require careful
    handling (NFS v4 supports them; some deployments disable them).

IBM Spectrum Scale (GPFS) provides enterprise RWX with:
  - High bandwidth via parallel I/O to multiple NSD servers.
  - POSIX compliance including advisory and mandatory file locks.
  - Native Kubernetes CSI driver with RWX and RWO support.
  - Sub-millisecond metadata operations via centralized inode table.

================================================================================
PART IV: LLM INFERENCE OPTIMIZATION
================================================================================

4.1 CONTINUOUS BATCHING
Traditional static batching waits for a batch to complete before starting new
requests, wasting GPU cycles during decode-heavy sequences. Continuous batching
(Orca, vLLM) treats the scheduler at iteration granularity: at each forward pass
step, finished sequences are removed and new sequences are added from the waiting
queue. This achieves near-100% GPU utilization and is the foundation of all
modern LLM serving systems.

4.2 PAGEDATTENTION AND KV CACHE MANAGEMENT
vLLM's PagedAttention manages KV cache as fixed-size blocks (pages), analogous
to OS virtual memory. Each block holds KV tensors for `block_size` tokens
(default 16). The block table maps logical sequence positions to physical block
addresses, enabling:
  - No fragmentation: blocks allocated on demand.
  - Copy-on-write sharing: forked sequences (beam search, parallel sampling)
    share prefix blocks until they diverge.
  - Prefix caching: blocks for common prefixes (system prompts, few-shot examples)
    are reused across requests by hashing block content.

4.3 QUANTIZATION: AWQ AND GPTQ
GPTQ (Post-Training Quantization via Second-Order Information): minimizes
layer-wise reconstruction error using the inverse Hessian. Quantizes weights
to INT4 with per-group scales. Fast quantization but susceptible to outlier
activations that degrade accuracy.

AWQ (Activation-aware Weight Quantization): analyzes activation magnitudes to
identify the 1% of weight channels that are most salient (high activation scale).
These channels are scaled up before quantization (effectively protecting them)
and scaled down during inference. AWQ achieves 1-2% higher accuracy than GPTQ
at 4-bit with similar inference speed.

At 4-bit AWQ, Mistral-7B model weights: 7B params * 0.5 bytes = 3.5 GB.
Combined with CUDA kernels (~1 GB) on a 16 GB GPU: ~11.5 GB available for KV cache.
At 128 KB/token: ~90,000 token KV cache capacity (well above 16384 maxModelLen).

4.4 CHUNKED PREFILL
Long prompts monopolize the GPU during prefill, stalling decode for concurrent
requests and causing latency spikes. Chunked prefill splits prompt processing
into chunks of C tokens (e.g., C=512), interleaving decode steps between chunks.
This bounds the latency impact of any single prefill at the cost of slightly
lower prefill throughput. vLLM's `enable_chunked_prefill=True` activates this.

================================================================================
PART V: LMCACHE — DISTRIBUTED KV CACHE PERSISTENCE
================================================================================

5.1 ARCHITECTURE AND STORAGE TIERS
LMCache implements a multi-tier KV cache hierarchy extending vLLM:

  GPU HBM (L1): managed by vLLM's PagedAttention. Fastest access (~1 TB/s HBM3).
  CPU DRAM (L2): pinned memory for zero-copy DMA transfer. ~50-100 GB/s PCIe.
  Local NVMe/PVC (L3): persistent disk cache via local_disk. ~3-7 GB/s NVMe.
  Remote cache server (L4): shared across replicas via TCP. Network-bound.

Eviction is LRU within each tier. The atomic caching unit is a chunk of
chunk_size tokens (default 256). save_unfull_chunk=true caches partial trailing
chunks, required for 100% hit rate when prompt lengths are not multiples of 256.

5.2 DISK CACHE CONFIGURATION
local_disk must be specified as a file:// URI pointing to a mounted PVC.
max_local_disk_size specifies the disk cache size cap in GiB. The CPU staging
buffer (max_local_cpu_size) cannot be zero — it acts as the DMA intermediary
between GPU and disk. Setting it to a small value (1-2 GB) forces rapid eviction
to disk while retaining the required transfer buffer.

5.3 PERFORMANCE CHARACTERISTICS
TTFT (Time To First Token) reduction with LMCache disk hits:
  - 1000 token prompt, disk hit: ~5X TTFT reduction
  - 4000 token prompt, disk hit: ~10X TTFT reduction
  - 8000 token prompt, disk hit: ~15X TTFT reduction
  - 16000 token prompt, disk hit: ~20X TTFT reduction

The speedup scales with prompt length because:
  - Disk hit cost is fixed (proportional to chunk count, each chunk ~microseconds)
  - Cold prefill cost is O(n^2) attention + O(n) feed-forward per layer
  As n grows, the cold prefill cost dominates and the cached version wins by more.

5.4 SESSION ROUTING FOR MAXIMUM CACHE UTILIZATION
The vllm-stack router supports routingLogic: session, pinning each user
(identified by sessionKey header, default x-user-id) to the same vllm replica.
This maximizes GPU L1 cache hits for single-replica deployments and ensures
LMCache disk hits for cross-session reuse of the same document or system prompt.

================================================================================
END OF REFERENCE DOCUMENT
================================================================================

Based ONLY on the reference document above, answer the following question:
"""

# Each entry: (user_id, unique_question)
# The question is short (~15-25 tokens); SHARED_CONTEXT (~8000 tokens) is what gets cached.
QUESTIONS = [
    ("user1", "What is the formula for scaled dot-product attention and why is sqrt(d_k) used as a scaling factor?")
#    ("user2", "How much KV cache memory does Mistral-7B require per token and per 16384-token sequence?"),
#    ("user3", "What are the three ZeRO optimizer stages and what does each one shard?"),
#    ("user4", "What is the difference between AWQ and GPTQ quantization and which achieves better 4-bit accuracy?"),
#    ("user5", "Explain PagedAttention and how it enables prefix caching and copy-on-write sharing."),
#    ("user6", "What is the minimum max_local_cpu_size for LMCache and why can it not be zero?"),
#    ("user7", "What access modes do Kubernetes PVCs support and which is required for shared LMCache storage?"),
#    ("user8", "How does DPO differ from PPO-based RLHF and what does it optimize directly?"),
#    ("user9", "What is chunked prefill in vLLM and how does it prevent latency spikes from long prompts?"),
#    ("user10", "What TTFT speedup does LMCache provide for an 8000-token prompt on a disk hit?"),
]

PROMPTS = [(uid, SHARED_CONTEXT + q) for uid, q in QUESTIONS]


def get_model_name(base_url: str) -> str:
    try:
        req = urllib.request.Request(f"{base_url}/models")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            model = data["data"][0]["id"]
            print(f"  [auto-discovered model: '{model}']")
            return model
    except Exception as e:
        print(f"  [WARNING] Could not auto-discover model: {e}")
        print(f"  [WARNING] Falling back to 'mistral' — override with --model")
        return "mistral"


def send_request(base_url: str, user_id: str, prompt: str, model: str,
                 max_tokens: int = 1) -> dict:
    url = f"{base_url}/completions"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()

    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json", "x-user-id": user_id},
        method="POST",
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            latency = time.perf_counter() - t0
            body = json.loads(resp.read())
            return {"ok": True, "latency": latency, "body": body}
    except urllib.error.HTTPError as e:
        latency = time.perf_counter() - t0
        try:
            error_body = e.read().decode()
        except Exception:
            error_body = "(could not read error body)"
        return {"ok": False, "latency": latency, "error": f"{e} — {error_body}"}
    except Exception as e:
        return {"ok": False, "latency": time.perf_counter() - t0, "error": str(e)}


def run_phase(base_url: str, model: str, phase_name: str, max_tokens: int,
              interval: float) -> list:
    """Send all prompts once and return list of latency results."""
    results = []
    print(f"\n  {'─'*61}")
    print(f"  {phase_name}")
    print(f"  {'─'*61}")
    print(f"  {'#':>3}  {'User':<8}  {'Status':<7}  {'TTFT':>8}  Question (truncated)")
    print(f"  {'-'*3}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*35}")

    for i, (user_id, prompt) in enumerate(PROMPTS):
        question = QUESTIONS[i][1][:45] + "..."
        result = send_request(base_url, user_id, prompt, model, max_tokens)
        results.append(result)

        status = "OK   " if result["ok"] else "ERROR"
        lat = f"{result['latency']*1000:.0f}ms"
        print(f"  {i+1:>3}  {user_id:<8}  {status:<7}  {lat:>8}  {question}")

        if not result["ok"]:
            print(f"       └─ {result.get('error','')[:100]}")

        time.sleep(interval)

    return results


def summarise(cold: list, warm: list, max_tokens: int) -> None:
    cold_ok = [r["latency"] * 1000 for r in cold if r["ok"]]
    warm_ok = [r["latency"] * 1000 for r in warm if r["ok"]]

    print(f"\n{'='*65}")
    print(f"  LMCACHE PERFORMANCE RESULTS  ({datetime.now().strftime('%H:%M:%S')})")
    print(f"{'='*65}")
    print(f"  Shared context  : ~{len(SHARED_CONTEXT.split()) * 4 // 3} tokens (cached by LMCache on disk)")
    print(f"  Question length : ~20 tokens (unique per request, not cached)")
    print(f"  max_tokens      : {max_tokens} (TTFT measurement mode)")
    print(f"  Requests        : {len(QUESTIONS)} prompts × 2 phases = {len(QUESTIONS)*2} total")
    print(f"  Cold successes  : {len(cold_ok)} / {len(cold)}")
    print(f"  Warm successes  : {len(warm_ok)} / {len(warm)}")

    if cold_ok and warm_ok:
        cold_med = statistics.median(cold_ok)
        warm_med = statistics.median(warm_ok)
        speedup  = cold_med / warm_med if warm_med > 0 else 0

        p95_cold = sorted(cold_ok)[max(0, int(len(cold_ok)*0.95)-1)]
        p95_warm = sorted(warm_ok)[max(0, int(len(warm_ok)*0.95)-1)]

        print(f"\n  {'Metric':<28}  {'COLD (no cache)':>16}  {'WARM (disk hit)':>15}")
        print(f"  {'─'*28}  {'─'*16}  {'─'*15}")
        print(f"  {'Min TTFT':<28}  {min(cold_ok):>14.0f}ms  {min(warm_ok):>13.0f}ms")
        print(f"  {'Median TTFT':<28}  {cold_med:>14.0f}ms  {warm_med:>13.0f}ms")
        print(f"  {'p95 TTFT':<28}  {p95_cold:>14.0f}ms  {p95_warm:>13.0f}ms")
        print(f"  {'Max TTFT':<28}  {max(cold_ok):>14.0f}ms  {max(warm_ok):>13.0f}ms")

        bar_len = 40
        bar = "█" * min(bar_len, int(speedup / 15 * bar_len))
        print(f"\n  SPEEDUP  {bar}  {speedup:.1f}X")
        if speedup >= 10:
            print(f"\n  ✓  {speedup:.1f}X TTFT improvement — LMCache 10X target achieved!")
        elif speedup >= 5:
            print(f"\n  ~  {speedup:.1f}X improvement — increase context length for 10X target.")
            print(f"     Tip: SHARED_CONTEXT needs to be longer relative to the question.")
        else:
            print(f"\n  ✗  {speedup:.1f}X — LMCache disk cache may not be hitting.")
            print(f"     Check: kubectl logs <vllm-pod> | grep 'External prefix cache hit rate'")
            print(f"     Check: kubectl exec -it <vllm-pod> -- du -sh /lmcache-disk")

        print(f"\n  Per-request breakdown:")
        print(f"  {'#':>3}  {'User':<8}  {'Cold TTFT':>10}  {'Warm TTFT':>10}  {'Speedup':>8}")
        print(f"  {'-'*3}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}")
        for i, (c, w) in enumerate(zip(cold, warm)):
            if c["ok"] and w["ok"]:
                cl = c["latency"] * 1000
                wl = w["latency"] * 1000
                sp = cl / wl if wl > 0 else 0
                uid = QUESTIONS[i][0]
                print(f"  {i+1:>3}  {uid:<8}  {cl:>9.0f}ms  {wl:>9.0f}ms  {sp:>7.1f}X")

    print(f"\n  Diagnostic commands:")
    print(f"  kubectl logs <vllm-pod> | grep 'External prefix cache hit rate'")
    print(f"  kubectl exec -it <vllm-pod> -- du -sh /lmcache-disk")
    print(f"{'='*65}\n")


def main():
    parser = argparse.ArgumentParser(
        description="LMCache TTFT demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-url", default="http://localhost:80/v1")
    parser.add_argument("--max-tokens", type=int, default=1,
                        help="Tokens to generate — keep at 1 for pure TTFT measurement")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Seconds between requests (default: 0.5)")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--passes", type=int, default=3,
                        help="Number of warm passes after cold baseline (default: 3)")
    args = parser.parse_args()

    model = args.model if args.model else get_model_name(args.base_url)
    approx_tokens = len(SHARED_CONTEXT.split()) * 4 // 3

    print(f"\n{'='*65}")
    print(f"  LMCache TTFT Demonstration")
    print(f"  URL            : {args.base_url}")
    print(f"  Model          : {model}")
    print(f"  Shared context : ~{approx_tokens} tokens  ({approx_tokens // 256} cache chunks of 256)")
    print(f"  Unique question: ~20 tokens  (changes per user)")
    print(f"  max_tokens     : {args.max_tokens}  (TTFT-only measurement)")
    print(f"  Warm passes    : {args.passes}")
    print(f"")
    print(f"  PHASE 1 — COLD: all {len(PROMPTS)} prompts, no cache anywhere")
    print(f"  PHASE 2-{1+args.passes} — WARM: same prompts, LMCache disk serves the hits")
    print(f"{'='*65}")

    # ── PHASE 1: COLD ──────────────────────────────────────────────────────
    cold_results = run_phase(
        args.base_url, model,
        "PHASE 1 — COLD  (GPU cache disabled, disk cache empty)",
        args.max_tokens, args.interval,
    )

    # ── PHASE 2+: WARM ─────────────────────────────────────────────────────
    all_warm = []
    for p in range(args.passes):
        warm = run_phase(
            args.base_url, model,
            f"PHASE {2+p} — WARM  (LMCache disk hits expected)",
            args.max_tokens, args.interval,
        )
        all_warm.extend(warm)

    # Use median across all warm passes for stable estimate
    # Build per-prompt warm aggregates for per-row breakdown
    warm_per_prompt = []
    for i in range(len(PROMPTS)):
        passes_for_i = [all_warm[i + p * len(PROMPTS)] for p in range(args.passes)
                        if (i + p * len(PROMPTS)) < len(all_warm)]
        ok_latencies = [r["latency"] for r in passes_for_i if r["ok"]]
        if ok_latencies:
            warm_per_prompt.append({
                "ok": True,
                "latency": statistics.median(ok_latencies),
            })
        else:
            warm_per_prompt.append({"ok": False, "latency": 0, "error": "all passes failed"})

    summarise(cold_results, warm_per_prompt, args.max_tokens)


if __name__ == "__main__":
    main()
