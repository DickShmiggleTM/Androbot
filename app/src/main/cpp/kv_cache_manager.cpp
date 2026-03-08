#include "inference_engine.h"

// ============================================================
// KV Cache Manager JNI Bindings
// These methods allow Kotlin to monitor and manage the KV cache
// to implement context eviction / summarization triggers.
// ============================================================

extern "C" {

/**
 * Returns a JSON summary of the current KV cache state.
 * Used by ContextManager.kt to decide when to trigger summarization.
 */
JNIEXPORT jstring JNICALL
Java_com_androbot_ai_ContextManager_nativeGetCacheStats(
        JNIEnv* env, jobject /* obj */) {

    InferenceEngine& engine = getGlobalEngine();
    size_t kv_bytes    = engine.getKVCacheRAMBytes();
    size_t mmap_bytes  = engine.getModelMmapBytes();
    int    ctx_len     = engine.getContextLength();
    bool   needs_evict = engine.needsContextEviction();

    char buf[512];
    snprintf(buf, sizeof(buf),
             "{"
             "\"kv_cache_mb\": %.2f, "
             "\"model_mmap_mb\": %.2f, "
             "\"context_len\": %d, "
             "\"needs_eviction\": %s, "
             "\"model_loaded\": %s"
             "}",
             (double)kv_bytes  / (1024.0 * 1024.0),
             (double)mmap_bytes / (1024.0 * 1024.0),
             ctx_len,
             needs_evict ? "true" : "false",
             engine.isModelLoaded() ? "true" : "false");

    return env->NewStringUTF(buf);
}

/**
 * Performs a hard context reset - clears the KV cache entirely.
 * Called when context window is fully saturated.
 */
JNIEXPORT void JNICALL
Java_com_androbot_ai_ContextManager_nativeClearContext(
        JNIEnv* /* env */, jobject /* obj */) {
    getGlobalEngine().clearContext();
    LOGI("Context cleared via Kotlin ContextManager");
}

/**
 * Returns the percentage of context window consumed (0-100).
 * Used to trigger summarization at threshold (e.g., 80%).
 */
JNIEXPORT jfloat JNICALL
Java_com_androbot_ai_ContextManager_nativeGetContextFillPercent(
        JNIEnv* /* env */, jobject /* obj */) {
    InferenceEngine& engine = getGlobalEngine();
    if (!engine.isModelLoaded()) return 0.0f;

    float fill = static_cast<float>(engine.getContextLength()) /
                 static_cast<float>(engine.config.n_ctx) * 100.0f;
    return std::min(100.0f, fill);
}

} // extern "C"
