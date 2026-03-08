#include "inference_engine.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <sstream>
#include <cstring>
#include <stdexcept>

// ============================================================
// MmapModel - Memory-mapped GGUF file
// Weights remain on storage (UFS/eMMC), kernel pages on demand
// ============================================================

bool MmapModel::load(const char* path) {
    fd = open(path, O_RDONLY);
    if (fd < 0) {
        LOGE("Failed to open model file: %s (errno=%d)", path, errno);
        return false;
    }

    // Get file size
    off_t file_size = lseek(fd, 0, SEEK_END);
    if (file_size <= 0) {
        LOGE("Invalid model file size: %ld", (long)file_size);
        close(fd);
        fd = -1;
        return false;
    }
    size = static_cast<size_t>(file_size);
    lseek(fd, 0, SEEK_SET);

    // Memory-map the file:
    // MAP_SHARED: changes would be written back (we use MAP_PRIVATE for safety)
    // MAP_POPULATE: pre-fault pages to avoid latency spikes during first inference
    // MADV_SEQUENTIAL on first access, MADV_RANDOM after warmup
    data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    if (data == MAP_FAILED) {
        LOGE("mmap failed for model (size=%zu): errno=%d", size, errno);
        close(fd);
        fd   = -1;
        data = nullptr;
        return false;
    }

    // Advise kernel: random access pattern (transformer attention is random)
    madvise(data, size, MADV_RANDOM);

    is_valid = true;
    LOGI("Model mmap'd successfully: path=%s size=%.2fMB", path,
         (double)size / (1024.0 * 1024.0));
    return true;
}

void MmapModel::unload() {
    if (data && data != MAP_FAILED) {
        munmap(data, size);
        data = nullptr;
    }
    if (fd >= 0) {
        close(fd);
        fd = -1;
    }
    size     = 0;
    is_valid = false;
    LOGI("Model unmapped from memory");
}

// ============================================================
// KV Cache
// ============================================================

bool KVCache::allocate(int layers, int heads, int dim, int max_context) {
    n_layers  = layers;
    n_heads   = heads;
    head_dim  = dim;
    max_len   = max_context;

    try {
        layers_.resize(layers);
        for (auto& layer : layers_) {
            // Pre-allocate K and V tensors: [n_heads * max_context * head_dim]
            size_t kv_size = static_cast<size_t>(heads) * max_context * dim;
            layer.key.assign(kv_size, 0.0f);
            layer.value.assign(kv_size, 0.0f);
        }
        current_len = 0;
        LOGI("KV Cache allocated: %d layers x %d heads x %d dim x %d ctx = %.2fMB",
             layers, heads, dim, max_context,
             (double)ram_footprint_bytes() / (1024.0 * 1024.0));
        return true;
    } catch (const std::bad_alloc& e) {
        LOGE("Failed to allocate KV cache: %s", e.what());
        return false;
    }
}

void KVCache::clear() {
    for (auto& layer : layers_) {
        std::fill(layer.key.begin(),   layer.key.end(),   0.0f);
        std::fill(layer.value.begin(), layer.value.end(), 0.0f);
    }
    current_len = 0;
    LOGI("KV Cache cleared");
}

size_t KVCache::ram_footprint_bytes() const {
    // 2 tensors (K, V) x n_layers x n_heads x max_len x head_dim x sizeof(float)
    return 2ULL * n_layers * n_heads * max_len * head_dim * sizeof(float);
}

// ============================================================
// InferenceEngine
// ============================================================

InferenceEngine::InferenceEngine() {
    LOGI("InferenceEngine created");
}

InferenceEngine::~InferenceEngine() {
    unloadModel();
    LOGI("InferenceEngine destroyed");
}

bool InferenceEngine::loadModel(const std::string& model_path, int n_ctx, int n_threads) {
    std::lock_guard<std::mutex> lock(inference_mutex_);

    if (model_loaded_.load()) {
        LOGW("Model already loaded, unloading first");
        unloadModel();
    }

    n_threads_ = n_threads;
    LOGI("Loading model: %s (ctx=%d, threads=%d)", model_path.c_str(), n_ctx, n_threads);

    // Step 1: Memory-map the GGUF model file
    if (!mmap_model_.load(model_path.c_str())) {
        LOGE("Failed to mmap model file");
        return false;
    }

    // Step 2: Parse GGUF header to extract model config
    if (!parseGGUFHeader(mmap_model_.data, mmap_model_.size)) {
        LOGE("Failed to parse GGUF header");
        mmap_model_.unload();
        return false;
    }
    config.n_ctx = n_ctx;

    // Step 3: Allocate KV cache in physical RAM
    // For Llama-3.2-1B: 16 layers, 8 KV heads (GQA), head_dim=64
    // For Gemma-3-1B:   18 layers, 1 KV head  (MQA), head_dim=256
    // Use config values parsed from GGUF
    if (!kv_cache_.allocate(config.n_layer, config.n_head_kv,
                             config.n_embd / config.n_head, n_ctx)) {
        LOGE("Failed to allocate KV cache - insufficient RAM");
        mmap_model_.unload();
        return false;
    }

    model_loaded_.store(true);
    stop_flag_.store(false);

    LOGI("Model loaded successfully. mmap=%.2fMB, KVCache=%.2fMB",
         (double)mmap_model_.size / 1024.0 / 1024.0,
         (double)kv_cache_.ram_footprint_bytes() / 1024.0 / 1024.0);
    return true;
}

void InferenceEngine::unloadModel() {
    model_loaded_.store(false);
    stop_flag_.store(true);
    mmap_model_.unload();
    kv_cache_.clear();
    LOGI("Model unloaded");
}

// Token sampling with temperature, top-p (nucleus), top-k
float InferenceEngine::sampleToken(const std::vector<float>& logits,
                                    float temp, float top_p, int top_k) {
    if (logits.empty()) return 0.0f;

    // Apply temperature scaling
    std::vector<float> scaled(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());

    for (size_t i = 0; i < logits.size(); i++) {
        scaled[i] = (logits[i] - max_logit) / std::max(temp, 1e-6f);
    }

    // Softmax
    float sum = 0.0f;
    for (auto& v : scaled) {
        v   = std::exp(v);
        sum += v;
    }
    for (auto& v : scaled) v /= sum;

    // Top-k filtering
    std::vector<std::pair<float, int>> candidates;
    candidates.reserve(logits.size());
    for (int i = 0; i < (int)scaled.size(); i++) {
        candidates.emplace_back(scaled[i], i);
    }
    std::partial_sort(candidates.begin(),
                      candidates.begin() + std::min(top_k, (int)candidates.size()),
                      candidates.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    candidates.resize(std::min(top_k, (int)candidates.size()));

    // Top-p (nucleus) filtering
    float cumulative = 0.0f;
    float norm_sum   = 0.0f;
    for (auto& c : candidates) norm_sum += c.first;
    int keep = 0;
    for (auto& c : candidates) {
        cumulative += c.first / norm_sum;
        keep++;
        if (cumulative >= top_p) break;
    }
    candidates.resize(keep);

    // Renormalize and sample
    norm_sum = 0.0f;
    for (auto& c : candidates) norm_sum += c.first;

    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, norm_sum);
    float sample = dist(rng);

    cumulative = 0.0f;
    for (auto& c : candidates) {
        cumulative += c.first;
        if (sample <= cumulative) return static_cast<float>(c.second);
    }
    return static_cast<float>(candidates.back().second);
}

bool InferenceEngine::parseGGUFHeader(const void* data, size_t size) {
    if (size < sizeof(GGUFHeader)) {
        LOGE("File too small to be a valid GGUF");
        return false;
    }

    const auto* header = reinterpret_cast<const GGUFHeader*>(data);
    if (header->magic != GGUF_MAGIC) {
        LOGE("Invalid GGUF magic: 0x%08X (expected 0x%08X)",
             header->magic, GGUF_MAGIC);
        return false;
    }

    LOGI("GGUF version=%u, n_tensors=%lu, n_kv=%lu",
         header->version, (unsigned long)header->n_tensors,
         (unsigned long)header->n_kv);

    // For production: walk the KV pairs to extract model config
    // Here we set sensible defaults for Llama-3.2-1B / Gemma-3-1B
    // A real implementation would parse all KV metadata from the GGUF format

    // Default to Llama-3.2-1B profile (safe fallback)
    config.n_layer    = 16;
    config.n_head     = 32;
    config.n_head_kv  = 8;
    config.n_embd     = 2048;
    config.n_ff       = 8192;
    config.n_vocab    = 128256;
    config.arch       = "llama";
    config.rope_freq_base = 500000.0f;

    LOGI("Model config: layers=%d, heads=%d, kv_heads=%d, embd=%d",
         config.n_layer, config.n_head, config.n_head_kv, config.n_embd);
    return true;
}

void InferenceEngine::generate(const std::string& prompt, TokenCallback callback,
                                int max_new_tokens, float temperature,
                                float top_p, int top_k) {
    if (!model_loaded_.load()) {
        LOGE("Cannot generate: model not loaded");
        callback("[ERROR: Model not loaded]", true);
        return;
    }

    std::lock_guard<std::mutex> lock(inference_mutex_);
    stop_flag_.store(false);

    LOGI("Starting generation: prompt_len=%zu, max_tokens=%d, temp=%.2f",
         prompt.size(), max_new_tokens, temperature);

    // In production this calls llama_eval() via JNI
    // This stub demonstrates the pipeline architecture
    // with correct mmap + KV cache memory model

    // Check if context eviction needed before starting
    if (kv_cache_.needs_eviction()) {
        LOGW("KV cache near capacity (%d/%d), clearing oldest entries",
             kv_cache_.current_len, kv_cache_.max_len);
        // In production: evict oldest half, preserve recent context
        kv_cache_.current_len = kv_cache_.max_len / 2;
    }

    // Simulate streaming token generation (replace with llama_eval loop)
    std::string response_text =
        "I am Androbot, your on-device AI assistant powered by a quantized "
        "language model running entirely locally. My weights are memory-mapped "
        "from storage, and my KV cache lives in physical RAM for fast reasoning. "
        "How can I help you?";

    // Stream tokens word by word for demo
    std::istringstream iss(response_text);
    std::string word;
    int tokens_generated = 0;

    while (std::getline(iss, word, ' ') &&
           tokens_generated < max_new_tokens &&
           !stop_flag_.load()) {
        callback(word + " ", false);
        kv_cache_.current_len++;
        tokens_generated++;
    }

    callback("", true);
    LOGI("Generation complete: %d tokens generated", tokens_generated);
}

std::vector<float> InferenceEngine::embed(const std::string& text) {
    if (!model_loaded_.load()) {
        LOGW("Cannot embed: model not loaded, returning zero vector");
        return std::vector<float>(config.n_embd, 0.0f);
    }

    // In production: run a single forward pass without sampling
    // to get the last-token hidden state as embedding vector
    std::vector<float> embedding(config.n_embd, 0.0f);

    // Hash-based mock embedding for compile-time correctness
    // Real implementation: llama_eval() -> extract last hidden state
    uint32_t hash = 5381;
    for (char c : text) {
        hash = ((hash << 5) + hash) + static_cast<uint8_t>(c);
    }
    std::mt19937 rng(hash);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    float norm = 0.0f;
    for (auto& v : embedding) {
        v    = dist(rng);
        norm += v * v;
    }
    norm = std::sqrt(norm);
    if (norm > 0.0f) {
        for (auto& v : embedding) v /= norm;
    }
    return embedding;
}

void InferenceEngine::clearContext() {
    std::lock_guard<std::mutex> lock(inference_mutex_);
    kv_cache_.clear();
    LOGI("Context cleared");
}

// ============================================================
// Global singleton
// ============================================================
InferenceEngine& getGlobalEngine() {
    static InferenceEngine engine;
    return engine;
}
