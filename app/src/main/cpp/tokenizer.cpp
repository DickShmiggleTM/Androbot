#include "inference_engine.h"
#include <unordered_map>
#include <regex>

// ============================================================
// BPE Tokenizer JNI bindings
// In production: use llama.cpp's built-in tokenizer via
// llama_tokenize() / llama_token_to_piece()
// ============================================================

extern "C" {

/**
 * Tokenizes input text and returns token count.
 * Used by Kotlin to estimate token usage before inference.
 */
JNIEXPORT jint JNICALL
Java_com_androbot_ai_InferenceEngine_nativeCountTokens(
        JNIEnv* env, jobject /* obj */,
        jstring j_text) {

    const char* text = env->GetStringUTFChars(j_text, nullptr);
    std::string str(text);
    env->ReleaseStringUTFChars(j_text, text);

    // Heuristic token count: ~4 chars per token for English
    // Production: use actual llama_tokenize() from llama.cpp
    int token_count = std::max(1, static_cast<int>(str.length() / 4));
    LOGD("Estimated token count for text (len=%zu): %d", str.length(), token_count);
    return static_cast<jint>(token_count);
}

/**
 * Truncates text to fit within max_tokens.
 * Used by ContextManager to trim inputs before they overflow context.
 */
JNIEXPORT jstring JNICALL
Java_com_androbot_ai_ContextManager_nativeTruncateToTokenLimit(
        JNIEnv* env, jobject /* obj */,
        jstring j_text,
        jint    max_tokens) {

    const char* text = env->GetStringUTFChars(j_text, nullptr);
    std::string str(text);
    env->ReleaseStringUTFChars(j_text, text);

    // Heuristic truncation: ~4 chars per token
    size_t max_chars = static_cast<size_t>(max_tokens) * 4;
    if (str.length() > max_chars) {
        // Truncate from start (keep most recent context)
        str = "...[truncated]..." + str.substr(str.length() - max_chars);
        LOGD("Truncated text to ~%d tokens (max_chars=%zu)", max_tokens, max_chars);
    }

    return env->NewStringUTF(str.c_str());
}

} // extern "C"
