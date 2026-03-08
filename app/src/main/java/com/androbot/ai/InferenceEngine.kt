package com.androbot.ai

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Kotlin facade for the C++ InferenceEngine via JNI.
 *
 * Responsibilities:
 * - Load/unload quantized GGUF models via mmap
 * - Stream token generation using callbackFlow
 * - Provide embedding vectors for the Memory Graph
 * - Monitor KV cache memory and trigger context eviction
 */
class InferenceEngine {

    companion object {
        private const val TAG = "InferenceEngine"
        const val DEFAULT_CTX_SIZE    = 2048
        const val DEFAULT_MAX_TOKENS  = 512
        const val DEFAULT_TEMPERATURE = 0.7f
        const val DEFAULT_TOP_P       = 0.9f
        const val DEFAULT_TOP_K       = 40

        init {
            try {
                System.loadLibrary("androbot_inference")
                Log.i(TAG, "Native inference library loaded")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load native library: ${e.message}")
            }
        }
    }

    // ── JNI declarations ──────────────────────────────────────────────────────

    private external fun nativeLoadModel(
        modelPath: String, nCtx: Int, nThreads: Int
    ): Boolean

    private external fun nativeUnloadModel()
    private external fun nativeIsModelLoaded(): Boolean

    private external fun nativeGenerate(
        prompt: String, maxNewTokens: Int,
        temperature: Float, topP: Float, topK: Int
    )

    private external fun nativeStopGeneration()
    private external fun nativeEmbed(text: String): FloatArray
    private external fun nativeCountTokens(text: String): Int

    private external fun nativeGetKvCacheBytes(): Long
    private external fun nativeGetMmapBytes(): Long
    private external fun nativeGetContextLength(): Int
    private external fun nativeNeedsContextEviction(): Boolean
    private external fun nativeClearContext()
    private external fun nativeGetVocabSize(): Int
    private external fun nativeGetEmbedDim(): Int
    private external fun nativeGetArchitecture(): String

    // ── Streaming callback (called from C++ via JNI) ──────────────────────────

    @Volatile
    private var tokenCallback: ((token: String, isDone: Boolean) -> Unit)? = null

    /** Called directly from native thread via JNI */
    @Suppress("unused")
    fun onTokenGenerated(token: String, isDone: Boolean) {
        tokenCallback?.invoke(token, isDone)
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Loads a GGUF model file using memory-mapped I/O.
     * Static weights are mmap'd from storage; only KV cache uses RAM.
     */
    suspend fun loadModel(
        modelFile: File,
        nCtx: Int = DEFAULT_CTX_SIZE,
        nThreads: Int = Runtime.getRuntime().availableProcessors().coerceAtMost(4)
    ): Result<Unit> = withContext(Dispatchers.IO) {
        Log.i(TAG, "Loading model: ${modelFile.absolutePath}")

        if (!modelFile.exists()) {
            return@withContext Result.failure(
                IllegalArgumentException("Model file not found: ${modelFile.absolutePath}")
            )
        }

        val success = nativeLoadModel(modelFile.absolutePath, nCtx, nThreads)
        if (success) {
            Log.i(TAG, "Model loaded. mmap=${getMmapMB()}MB, kvCache=${getKvCacheMB()}MB")
            Result.success(Unit)
        } else {
            Result.failure(RuntimeException("Native model loading failed"))
        }
    }

    fun unloadModel() {
        if (nativeIsModelLoaded()) {
            nativeUnloadModel()
            Log.i(TAG, "Model unloaded")
        }
    }

    val isModelLoaded: Boolean get() = nativeIsModelLoaded()

    /**
     * Streams generated tokens as a Flow<String>.
     * Uses callbackFlow to bridge the JNI token callback to Kotlin coroutines.
     */
    fun generateStream(
        prompt: String,
        maxNewTokens: Int   = DEFAULT_MAX_TOKENS,
        temperature: Float  = DEFAULT_TEMPERATURE,
        topP: Float         = DEFAULT_TOP_P,
        topK: Int           = DEFAULT_TOP_K
    ): Flow<String> = callbackFlow {
        tokenCallback = { token, isDone ->
            if (token.isNotEmpty()) {
                trySend(token)
            }
            if (isDone) {
                close()
            }
        }

        withContext(Dispatchers.Default) {
            nativeGenerate(prompt, maxNewTokens, temperature, topP, topK)
        }

        awaitClose {
            tokenCallback = null
            nativeStopGeneration()
        }
    }

    /**
     * Generates a complete response (non-streaming).
     */
    suspend fun generate(
        prompt: String,
        maxNewTokens: Int  = DEFAULT_MAX_TOKENS,
        temperature: Float = DEFAULT_TEMPERATURE
    ): String = withContext(Dispatchers.Default) {
        if (!isModelLoaded) return@withContext "[Model not loaded]"

        val sb = StringBuilder()
        var completed = false

        tokenCallback = { token, isDone ->
            sb.append(token)
            if (isDone) completed = true
        }

        nativeGenerate(prompt, maxNewTokens, temperature, DEFAULT_TOP_P, DEFAULT_TOP_K)

        tokenCallback = null
        sb.toString().trim()
    }

    /**
     * Computes a normalized embedding vector for use in the Memory Graph.
     * Embedding dimension matches model's hidden size (e.g., 2048 for Llama-3.2-1B).
     */
    suspend fun embed(text: String): FloatArray = withContext(Dispatchers.Default) {
        if (!isModelLoaded) {
            Log.w(TAG, "Model not loaded, returning zero embedding")
            FloatArray(2048)
        } else {
            nativeEmbed(text)
        }
    }

    fun countTokens(text: String): Int =
        if (isModelLoaded) nativeCountTokens(text) else (text.length / 4)

    // ── Memory Management ─────────────────────────────────────────────────────

    fun getKvCacheMB(): Float  = nativeGetKvCacheBytes() / (1024f * 1024f)
    fun getMmapMB(): Float     = nativeGetMmapBytes()    / (1024f * 1024f)
    fun getContextLength(): Int = nativeGetContextLength()
    fun needsContextEviction(): Boolean = nativeNeedsContextEviction()
    fun clearContext() = nativeClearContext()

    // ── Model Info ────────────────────────────────────────────────────────────

    val vocabSize:    Int    get() = if (isModelLoaded) nativeGetVocabSize()   else 0
    val embedDim:     Int    get() = if (isModelLoaded) nativeGetEmbedDim()    else 2048
    val architecture: String get() = if (isModelLoaded) nativeGetArchitecture() else "unknown"

    fun stopGeneration() = nativeStopGeneration()
}
