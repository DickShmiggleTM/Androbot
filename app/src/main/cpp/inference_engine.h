#pragma once

#include <jni.h>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <atomic>
#include <mutex>
#include <cstdint>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <android/log.h>

#define LOG_TAG "AndrobotInference"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

// ============================================================
// GGUF Model Header Structures (INT4/Q4_K_M quantization)
// ============================================================

#define GGUF_MAGIC 0x46554747  // "GGUF"
#define GGUF_VERSION 3
#define GGML_MAX_DIMS 4
#define GGML_MAX_NAME 64

// Quantization types
enum ggml_type {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q4_K = 12,   // K-quant 4-bit (Q4_K_M)
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_I8   = 16,
    GGML_TYPE_I16  = 17,
    GGML_TYPE_I32  = 18,
    GGML_TYPE_COUNT,
};

struct GGUFHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
};

// ============================================================
// Memory-Mapped Model File
// Weights stay on storage; kernel pages them on demand
// ============================================================

struct MmapModel {
    void*   data     = nullptr;
    size_t  size     = 0;
    int     fd       = -1;
    bool    is_valid = false;

    bool load(const char* path);
    void unload();
    ~MmapModel() { unload(); }
};

// ============================================================
// KV Cache - lives in physical RAM
// ============================================================

struct KVCacheLayer {
    std::vector<float> key;    // [n_heads, seq_len, head_dim]
    std::vector<float> value;  // [n_heads, seq_len, head_dim]
};

struct KVCache {
    std::vector<KVCacheLayer> layers;
    int current_len    = 0;
    int max_len        = 2048;   // configurable context window
    int n_layers       = 32;
    int n_heads        = 8;
    int head_dim       = 64;

    bool allocate(int layers, int heads, int dim, int max_context);
    void clear();
    bool needs_eviction() const { return current_len >= max_len * 0.9f; }
    size_t ram_footprint_bytes() const;
};

// ============================================================
// Model Config (parsed from GGUF metadata)
// ============================================================

struct ModelConfig {
    int    n_vocab        = 32000;
    int    n_ctx          = 2048;
    int    n_embd         = 2048;
    int    n_head         = 16;
    int    n_head_kv      = 8;      // GQA
    int    n_layer        = 22;
    int    n_ff           = 8192;
    float  rope_freq_base = 500000.0f;
    float  rope_freq_scale= 1.0f;
    int    n_rot          = 128;
    std::string arch      = "llama";
    std::string model_name= "";
};

// ============================================================
// Inference Engine - Core
// ============================================================

using TokenCallback = std::function<void(const std::string& token, bool is_done)>;

class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();

    // Model lifecycle
    bool    loadModel(const std::string& model_path, int n_ctx = 2048, int n_threads = 4);
    void    unloadModel();
    bool    isModelLoaded() const { return model_loaded_.load(); }

    // Generation
    void    generate(const std::string& prompt, TokenCallback callback,
                     int max_new_tokens = 512, float temperature = 0.7f,
                     float top_p = 0.9f, int top_k = 40);
    void    stopGeneration() { stop_flag_.store(true); }

    // Embedding
    std::vector<float> embed(const std::string& text);

    // Memory info
    size_t  getKVCacheRAMBytes() const { return kv_cache_.ram_footprint_bytes(); }
    size_t  getModelMmapBytes() const  { return mmap_model_.size; }
    int     getContextLength() const   { return kv_cache_.current_len; }

    // Context management
    void    clearContext();
    bool    needsContextEviction() const { return kv_cache_.needs_eviction(); }

    ModelConfig config;

private:
    MmapModel   mmap_model_;
    KVCache     kv_cache_;

    std::atomic<bool> model_loaded_{false};
    std::atomic<bool> stop_flag_{false};
    std::mutex        inference_mutex_;

    int  n_threads_ = 4;
    void* ctx_      = nullptr;  // llama_context* when linked

    // Internal helpers
    std::vector<int>   tokenize(const std::string& text, bool add_bos = true);
    std::string        detokenize(const std::vector<int>& tokens);
    float              sampleToken(const std::vector<float>& logits,
                                   float temp, float top_p, int top_k);
    bool               parseGGUFHeader(const void* data, size_t size);
};

// ============================================================
// Global singleton accessor
// ============================================================
InferenceEngine& getGlobalEngine();
