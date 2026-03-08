#include "inference_engine.h"

// ============================================================
// Context Window Management
// Handles the sliding window and summarization triggers
// ============================================================

extern "C" {

/**
 * Checks total estimated RAM usage and returns a JSON status.
 * Kotlin layer uses this to decide if the model should be
 * temporarily swapped or context compressed.
 */
JNIEXPORT jstring JNICALL
Java_com_androbot_ai_ContextManager_nativeGetMemoryStatus(
        JNIEnv* env, jobject /* obj */) {

    InferenceEngine& engine = getGlobalEngine();

    // Read /proc/meminfo for current system state
    FILE* f = fopen("/proc/meminfo", "r");
    long long total_kb = 0, available_kb = 0;
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            sscanf(line, "MemTotal: %lld kB", &total_kb);
            sscanf(line, "MemAvailable: %lld kB", &available_kb);
        }
        fclose(f);
    }

    double total_mb     = total_kb / 1024.0;
    double available_mb = available_kb / 1024.0;
    double kv_mb        = engine.getKVCacheRAMBytes() / (1024.0 * 1024.0);
    double mmap_mb      = engine.getModelMmapBytes()  / (1024.0 * 1024.0);
    double active_mb    = kv_mb;  // Only KV cache in active RAM
    double pressure     = (total_mb > 0) ?
                          (1.0 - available_mb / total_mb) * 100.0 : 0.0;

    // Thresholds for memory management decisions
    // Green:  < 60% RAM pressure
    // Yellow: 60-80% -> warn, may trigger context compression
    // Red:    > 80% -> must compress context or unload model
    const char* status = (pressure < 60.0) ? "green" :
                         (pressure < 80.0) ? "yellow" : "red";

    char buf[512];
    snprintf(buf, sizeof(buf),
             "{"
             "\"total_ram_mb\": %.0f, "
             "\"available_mb\": %.0f, "
             "\"kv_cache_mb\": %.2f, "
             "\"model_mmap_mb\": %.2f, "
             "\"active_ram_mb\": %.2f, "
             "\"pressure_pct\": %.1f, "
             "\"status\": \"%s\""
             "}",
             total_mb, available_mb, kv_mb,
             mmap_mb, active_mb, pressure, status);

    LOGI("Memory status: pressure=%.1f%% (%s), available=%.0fMB",
         pressure, status, available_mb);
    return env->NewStringUTF(buf);
}

} // extern "C"
