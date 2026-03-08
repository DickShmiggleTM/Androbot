package com.androbot.ai

import android.content.Context
import android.os.Environment
import android.util.Log
import androidx.work.Constraints
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import com.google.gson.Gson
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Manages the lifecycle of quantized GGUF model files.
 *
 * Responsibilities:
 * - Discover GGUF models in standard storage locations
 * - Validate model files before loading
 * - Manage available RAM budget (1.5-2GB limit)
 * - Recommend optimal model based on available RAM
 * - Schedule LoRA adapter updates via WorkManager
 */
class ModelManager(private val context: Context) {

    companion object {
        private const val TAG = "ModelManager"

        // RAM budget: keep total footprint under 2GB
        private const val MAX_TOTAL_RAM_MB = 1800L  // 1.8GB headroom

        // Recommended model profiles (sorted by RAM use, smallest first)
        val RECOMMENDED_MODELS = listOf(
            ModelProfile(
                name = "Gemma-3-1B-IT-Q4_K_M",
                paramBillions = 1.0f,
                mmapSizeMB = 620f,
                kvCacheMB  = 128f,
                totalRamMB = 750f,
                downloadUrl = "https://huggingface.co/bartowski/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf",
                description = "Best quality 1B model, GQA architecture, excellent reasoning"
            ),
            ModelProfile(
                name = "Llama-3.2-1B-Instruct-Q4_K_M",
                paramBillions = 1.0f,
                mmapSizeMB = 780f,
                kvCacheMB  = 160f,
                totalRamMB = 940f,
                downloadUrl = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                description = "Meta Llama 3.2 1B with instruction tuning"
            ),
            ModelProfile(
                name = "SmolLM2-360M-Instruct-Q4_K_M",
                paramBillions = 0.36f,
                mmapSizeMB = 230f,
                kvCacheMB  = 64f,
                totalRamMB = 294f,
                downloadUrl = "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf",
                description = "Ultra-compact 360M model for severely memory-constrained devices"
            )
        )

        // Standard GGUF model search paths
        private val MODEL_SEARCH_DIRS = listOf(
            "/sdcard/Androbot/models",
            "/sdcard/Download",
            "/sdcard/llm",
            "/sdcard/models",
            "/data/local/tmp"
        )

        @JvmStatic external fun nativeScanForModels(directory: String): Array<String>
        @JvmStatic external fun nativeValidateModel(path: String): String
        @JvmStatic external fun nativeGetFreeRamBytes(): Long
    }

    data class ModelProfile(
        val name: String,
        val paramBillions: Float,
        val mmapSizeMB: Float,
        val kvCacheMB: Float,
        val totalRamMB: Float,
        val downloadUrl: String,
        val description: String
    )

    data class ModelValidationResult(
        val valid: Boolean,
        val version: Int = 0,
        val nTensors: Long = 0,
        val sizeMb: Float = 0f,
        val error: String = ""
    )

    sealed class ModelState {
        object Idle : ModelState()
        data class Loading(val progress: Float) : ModelState()
        data class Loaded(val modelFile: File, val profile: ModelProfile?) : ModelState()
        data class Error(val message: String) : ModelState()
    }

    private val _modelState = MutableStateFlow<ModelState>(ModelState.Idle)
    val modelState: StateFlow<ModelState> = _modelState.asStateFlow()

    private val inferenceEngine = InferenceEngine()
    val engine: InferenceEngine get() = inferenceEngine

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Scans all standard locations for GGUF model files.
     */
    suspend fun discoverModels(): List<File> = withContext(Dispatchers.IO) {
        val found = mutableListOf<File>()

        // Internal app storage (downloaded models)
        val internalModelsDir = File(context.filesDir, "models")
        if (internalModelsDir.exists()) {
            internalModelsDir.listFiles { f -> f.name.endsWith(".gguf") }
                ?.forEach { found.add(it) }
        }

        // External storage paths
        MODEL_SEARCH_DIRS.forEach { dir ->
            try {
                val paths = nativeScanForModels(dir)
                paths.forEach { path -> found.add(File(path)) }
            } catch (e: Exception) {
                Log.d(TAG, "Scan skipped for $dir: ${e.message}")
            }
        }

        Log.i(TAG, "Discovered ${found.size} GGUF model(s)")
        found
    }

    /**
     * Validates a GGUF model file and returns its metadata.
     */
    suspend fun validateModel(file: File): ModelValidationResult =
        withContext(Dispatchers.IO) {
            try {
                val json = nativeValidateModel(file.absolutePath)
                Gson().fromJson(json, ModelValidationResult::class.java)
            } catch (e: Exception) {
                ModelValidationResult(valid = false, error = e.message ?: "unknown")
            }
        }

    /**
     * Selects the best model for available RAM.
     * Prefers maximum quality within the 1.8GB budget.
     */
    fun recommendModel(availableRamMB: Long): ModelProfile? {
        return RECOMMENDED_MODELS
            .filter { it.totalRamMB < availableRamMB * 0.85f }  // 15% safety margin
            .maxByOrNull { it.paramBillions }
    }

    /**
     * Loads a model, checking RAM availability first.
     */
    suspend fun loadModel(modelFile: File): Result<Unit> {
        _modelState.value = ModelState.Loading(0f)

        val freeRamMB = nativeGetFreeRamBytes() / (1024L * 1024L)
        Log.i(TAG, "Available RAM: ${freeRamMB}MB")

        if (freeRamMB < 300L) {
            val msg = "Insufficient RAM: ${freeRamMB}MB available (minimum 300MB required)"
            _modelState.value = ModelState.Error(msg)
            return Result.failure(OutOfMemoryError(msg))
        }

        _modelState.value = ModelState.Loading(0.3f)

        val validation = validateModel(modelFile)
        if (!validation.valid) {
            val msg = "Invalid model file: ${validation.error}"
            _modelState.value = ModelState.Error(msg)
            return Result.failure(IllegalArgumentException(msg))
        }

        _modelState.value = ModelState.Loading(0.6f)

        val nCtx = when {
            freeRamMB > 1500 -> 4096
            freeRamMB > 800  -> 2048
            else             -> 1024
        }

        val result = inferenceEngine.loadModel(modelFile, nCtx = nCtx)

        return result.fold(
            onSuccess = {
                val profile = RECOMMENDED_MODELS.find {
                    modelFile.name.contains(it.name, ignoreCase = true)
                }
                _modelState.value = ModelState.Loaded(modelFile, profile)
                Log.i(TAG, "Model ready: ${modelFile.name}")
                Result.success(Unit)
            },
            onFailure = { e ->
                _modelState.value = ModelState.Error(e.message ?: "Load failed")
                Result.failure(e)
            }
        )
    }

    fun unloadModel() {
        inferenceEngine.unloadModel()
        _modelState.value = ModelState.Idle
    }

    /**
     * Returns the models directory in internal storage, creating if needed.
     */
    fun getModelsDir(): File {
        val dir = File(context.filesDir, "models")
        if (!dir.exists()) dir.mkdirs()
        return dir
    }

    /**
     * Returns a human-readable memory report.
     */
    fun getMemoryReport(): String {
        val freeRamMB  = nativeGetFreeRamBytes() / (1024L * 1024L)
        val kvCacheMB  = inferenceEngine.getKvCacheMB()
        val mmapMB     = inferenceEngine.getMmapMB()
        val ctxLen     = inferenceEngine.getContextLength()

        return buildString {
            appendLine("=== Memory Report ===")
            appendLine("Free RAM:      ${freeRamMB}MB")
            appendLine("KV Cache:      ${"%.1f".format(kvCacheMB)}MB (active RAM)")
            appendLine("Model (mmap):  ${"%.1f".format(mmapMB)}MB (virtual, on storage)")
            appendLine("Context len:   $ctxLen tokens")
            appendLine("Architecture:  ${inferenceEngine.architecture}")
        }
    }
}
