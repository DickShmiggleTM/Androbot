package com.androbot.ai

import android.util.Log
import com.google.gson.Gson
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Manages the active context window for the inference engine.
 *
 * Implements the split memory architecture:
 * - KV Cache → physical RAM (fast access, no I/O bottleneck)
 * - Overflow context → compressed summaries pushed to Memory Graph
 *
 * Context eviction strategy:
 * 1. Monitor context fill percentage via native JNI call
 * 2. At 80% capacity: summarize oldest 50% of context
 * 3. Push summary embedding to MemoryGraph vector DB
 * 4. Clear oldest context entries from KV cache
 */
class ContextManager(private val inferenceEngine: InferenceEngine) {

    companion object {
        private const val TAG = "ContextManager"
        private const val EVICTION_THRESHOLD_PCT = 80f
        private const val SYSTEM_PROMPT = """You are Androbot, an advanced on-device AI assistant.
You run entirely locally on Android using a quantized language model.
You have deep system access: you can read the screen, control the device, browse the web,
read documents, and learn from interactions. Be concise, helpful, and privacy-respecting.
Current capabilities: screen reading, UI automation, file access, web search, OCR."""
    }

    // ── JNI declarations ──────────────────────────────────────────────────────

    private external fun nativeGetCacheStats(): String
    private external fun nativeClearContext()
    private external fun nativeGetContextFillPercent(): Float
    private external fun nativeTruncateToTokenLimit(text: String, maxTokens: Int): String
    private external fun nativeGetMemoryStatus(): String

    // ── State ─────────────────────────────────────────────────────────────────

    private val conversationHistory = ArrayDeque<Message>()
    private var summaryContext: String = ""
    private var onEvictionNeeded: (suspend (olderHistory: List<Message>) -> String)? = null

    data class Message(
        val role: String,  // "system", "user", "assistant"
        val content: String,
        val tokenCount: Int = 0
    )

    data class CacheStats(
        val kvCacheMb: Float,
        val modelMmapMb: Float,
        val contextLen: Int,
        val needsEviction: Boolean,
        val modelLoaded: Boolean
    )

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Registers the summarization callback that compresses old context.
     * Called with the oldest messages; must return a summary string.
     */
    fun setEvictionCallback(
        callback: suspend (olderMessages: List<Message>) -> String
    ) {
        onEvictionNeeded = callback
    }

    /**
     * Adds a message to conversation history and checks if eviction is needed.
     */
    suspend fun addMessage(role: String, content: String): Boolean {
        val tokenCount = inferenceEngine.countTokens(content)
        conversationHistory.addLast(Message(role, content, tokenCount))

        val fillPct = nativeGetContextFillPercent()
        Log.d(TAG, "Context fill: $fillPct% (added $tokenCount tokens for $role)")

        if (fillPct >= EVICTION_THRESHOLD_PCT) {
            Log.w(TAG, "Context fill at $fillPct%, triggering eviction")
            evictOldContext()
        }

        return fillPct < 95f  // false = context critically full
    }

    /**
     * Builds the full prompt string for the inference engine.
     * Format: [system] + [summary if any] + [recent messages]
     */
    fun buildPrompt(userInput: String): String {
        val sb = StringBuilder()

        // System prompt
        sb.append("<|system|>\n$SYSTEM_PROMPT\n")

        // Summary of evicted context (if any)
        if (summaryContext.isNotEmpty()) {
            sb.append("\n[Context summary from earlier conversation]:\n$summaryContext\n")
        }

        // Recent conversation history
        conversationHistory.forEach { msg ->
            when (msg.role) {
                "user"      -> sb.append("\n<|user|>\n${msg.content}\n")
                "assistant" -> sb.append("\n<|assistant|>\n${msg.content}\n")
            }
        }

        // New user turn
        sb.append("\n<|user|>\n$userInput\n<|assistant|>\n")

        return sb.toString()
    }

    /**
     * Performs context eviction:
     * 1. Take the oldest 50% of messages
     * 2. Summarize them via the inference engine
     * 3. Store summary as compressed context
     * 4. Clear the KV cache entries for evicted messages
     */
    private suspend fun evictOldContext() = withContext(Dispatchers.Default) {
        val evictCount = conversationHistory.size / 2
        if (evictCount == 0) {
            // Can't evict - just clear and restart
            nativeClearContext()
            conversationHistory.clear()
            Log.w(TAG, "Hard context reset - conversation cleared")
            return@withContext
        }

        val toEvict = conversationHistory.take(evictCount).toList()

        // Summarize the oldest messages
        val summaryPrompt = buildSummaryPrompt(toEvict)
        val newSummary = try {
            onEvictionNeeded?.invoke(toEvict)
                ?: buildFallbackSummary(toEvict)
        } catch (e: Exception) {
            Log.e(TAG, "Summarization failed: ${e.message}")
            buildFallbackSummary(toEvict)
        }

        // Combine with any existing summary
        summaryContext = if (summaryContext.isEmpty()) {
            newSummary
        } else {
            "$summaryContext\n$newSummary"
        }

        // Truncate summary to avoid it growing unbounded
        if (inferenceEngine.countTokens(summaryContext) > 512) {
            summaryContext = nativeTruncateToTokenLimit(summaryContext, 512)
        }

        // Remove evicted messages from history
        repeat(evictCount) {
            if (conversationHistory.isNotEmpty()) {
                conversationHistory.removeFirst()
            }
        }

        // Instruct native layer to partially clear KV cache
        nativeClearContext()

        Log.i(TAG, "Context eviction complete: removed $evictCount messages, " +
              "summary=${summaryContext.take(100)}...")
    }

    private fun buildSummaryPrompt(messages: List<Message>): String {
        val historyText = messages.joinToString("\n") { "${it.role}: ${it.content}" }
        return "Summarize this conversation in 2-3 sentences:\n$historyText"
    }

    private fun buildFallbackSummary(messages: List<Message>): String {
        // Simple extractive summary as fallback
        val topics = messages
            .filter { it.role == "user" }
            .take(5)
            .joinToString("; ") { it.content.take(60) }
        return "Earlier topics: $topics"
    }

    fun getCacheStats(): CacheStats {
        return try {
            val json = nativeGetCacheStats()
            val data = Gson().fromJson(json, Map::class.java)
            CacheStats(
                kvCacheMb     = (data["kv_cache_mb"] as? Double)?.toFloat() ?: 0f,
                modelMmapMb   = (data["model_mmap_mb"] as? Double)?.toFloat() ?: 0f,
                contextLen    = (data["context_len"] as? Double)?.toInt() ?: 0,
                needsEviction = data["needs_eviction"] as? Boolean ?: false,
                modelLoaded   = data["model_loaded"] as? Boolean ?: false
            )
        } catch (e: Exception) {
            CacheStats(0f, 0f, 0, false, false)
        }
    }

    fun getMemoryStatus(): String = nativeGetMemoryStatus()

    fun clearHistory() {
        conversationHistory.clear()
        summaryContext = ""
        nativeClearContext()
        Log.i(TAG, "Conversation history cleared")
    }

    fun getHistorySize(): Int = conversationHistory.size
}
