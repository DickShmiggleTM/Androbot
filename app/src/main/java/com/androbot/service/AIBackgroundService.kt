package com.androbot.service

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Binder
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import com.androbot.MainActivity
import com.androbot.R
import com.androbot.action.DeviceController
import com.androbot.action.FileManager
import com.androbot.action.WebBrowser
import com.androbot.ai.ContextManager
import com.androbot.ai.FunctionCallingPipeline
import com.androbot.ai.InferenceEngine
import com.androbot.ai.ModelManager
import com.androbot.memory.LearningScheduler
import com.androbot.memory.MemoryGraph
import com.androbot.perception.DocumentProcessor
import com.androbot.perception.VisionProcessor
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * AI Background Service
 *
 * Core service that hosts the inference engine as a long-lived Android Service.
 * Runs as a Foreground Service to prevent the OS from killing it under
 * memory pressure during active sessions.
 *
 * Hosts and coordinates all subsystems:
 * - InferenceEngine (C++ via JNI, model is mmap'd)
 * - ContextManager (KV cache + context eviction)
 * - MemoryGraph (vector database)
 * - FunctionCallingPipeline (tool dispatch)
 * - DeviceController, FileManager, WebBrowser
 * - VisionProcessor, DocumentProcessor
 * - LearningScheduler (LoRA adapter updates)
 */
class AIBackgroundService : Service() {

    companion object {
        private const val TAG            = "AIBackgroundService"
        private const val NOTIFICATION_ID = 1001
        private const val CHANNEL_ID     = "androbot_service"
        private const val CHANNEL_NAME   = "Androbot AI Service"

        const val ACTION_START   = "com.androbot.START_AI"
        const val ACTION_STOP    = "com.androbot.STOP_AI"
        const val ACTION_CHAT    = "com.androbot.CHAT"
        const val EXTRA_MESSAGE  = "message"
        const val EXTRA_RESPONSE = "response"

        @Volatile
        private var instance: AIBackgroundService? = null
        fun getInstance(): AIBackgroundService? = instance
    }

    // ── Binder ────────────────────────────────────────────────────────────────

    inner class AIBinder : Binder() {
        fun getService(): AIBackgroundService = this@AIBackgroundService
    }

    private val binder = AIBinder()

    // ── Subsystems ────────────────────────────────────────────────────────────

    lateinit var modelManager:       ModelManager          private set
    lateinit var inferenceEngine:    InferenceEngine       private set
    lateinit var contextManager:     ContextManager        private set
    lateinit var memoryGraph:        MemoryGraph           private set
    lateinit var functionPipeline:   FunctionCallingPipeline private set
    lateinit var deviceController:   DeviceController      private set
    lateinit var fileManager:        FileManager           private set
    lateinit var webBrowser:         WebBrowser            private set
    lateinit var visionProcessor:    VisionProcessor       private set
    lateinit var documentProcessor:  DocumentProcessor     private set
    lateinit var learningScheduler:  LearningScheduler     private set

    // ── State ─────────────────────────────────────────────────────────────────

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    sealed class ServiceState {
        object Initializing : ServiceState()
        object Ready : ServiceState()
        data class Thinking(val message: String) : ServiceState()
        data class Error(val message: String) : ServiceState()
    }

    private val _state = MutableStateFlow<ServiceState>(ServiceState.Initializing)
    val state: StateFlow<ServiceState> = _state.asStateFlow()

    // ── Service Lifecycle ─────────────────────────────────────────────────────

    override fun onCreate() {
        super.onCreate()
        instance = this
        Log.i(TAG, "AI Background Service created")

        createNotificationChannel()
        startForeground(NOTIFICATION_ID, buildNotification("Initializing..."))
        initializeSubsystems()
    }

    private fun initializeSubsystems() {
        serviceScope.launch {
            try {
                Log.i(TAG, "Initializing subsystems...")

                // Core AI modules
                modelManager     = ModelManager(applicationContext)
                inferenceEngine  = modelManager.engine
                contextManager   = ContextManager(inferenceEngine)
                memoryGraph      = MemoryGraph(applicationContext, inferenceEngine)

                // Action modules
                deviceController = DeviceController(applicationContext)
                fileManager      = FileManager(applicationContext)
                webBrowser       = WebBrowser(applicationContext)

                // Perception modules
                visionProcessor   = VisionProcessor(applicationContext)
                documentProcessor = DocumentProcessor(applicationContext, visionProcessor)

                // Function calling pipeline
                functionPipeline  = FunctionCallingPipeline()
                setupFunctionCallExecutor()

                // Context eviction: summarize old context via the model
                contextManager.setEvictionCallback { oldMessages ->
                    val summaryPrompt = "Summarize these messages briefly:\n" +
                        oldMessages.joinToString("\n") { "${it.role}: ${it.content}" }
                    inferenceEngine.generate(summaryPrompt, maxNewTokens = 100)
                }

                // Learning scheduler
                learningScheduler = LearningScheduler(applicationContext)
                learningScheduler.scheduleAdapterUpdate()

                // Auto-discover and load model
                autoLoadBestModel()

                _state.value = ServiceState.Ready
                updateNotification("Ready")
                Log.i(TAG, "All subsystems initialized")

            } catch (e: Exception) {
                Log.e(TAG, "Initialization failed: ${e.message}", e)
                _state.value = ServiceState.Error(e.message ?: "Init failed")
                updateNotification("Error: ${e.message?.take(40)}")
            }
        }
    }

    private fun setupFunctionCallExecutor() {
        functionPipeline.setExecutor(object : FunctionCallingPipeline.ToolExecutor {
            override suspend fun screenRead() = deviceController.readScreen()
            override suspend fun clickElement(text: String) = deviceController.clickElement(text)
            override suspend fun typeText(text: String) = deviceController.typeText(text)
            override suspend fun openApp(packageName: String) = deviceController.openApp(packageName)
            override suspend fun scroll(direction: String) = deviceController.scroll(direction)
            override suspend fun webSearch(query: String) = webBrowser.search(query)
            override suspend fun readFile(path: String) = fileManager.readExternalFile(path)
            override suspend fun ocrScreen() = visionProcessor.extractScreenText()
            override suspend fun rememberFact(fact: String): Boolean {
                memoryGraph.remember(fact, "fact", 0.8f)
                return true
            }
            override suspend fun recallFact(query: String): String {
                val results = memoryGraph.recall(query)
                return if (results.isEmpty()) "No memories found for: $query"
                else results.joinToString("\n")
            }
        })
    }

    private suspend fun autoLoadBestModel() {
        val models = modelManager.discoverModels()
        Log.i(TAG, "Discovered ${models.size} model(s)")

        if (models.isEmpty()) {
            Log.i(TAG, "No GGUF models found. Place a .gguf file in /sdcard/Androbot/models/")
            updateNotification("No model found - see setup")
            return
        }

        // Sort by size desc (prefer larger/better model within RAM budget)
        val freeRam = ModelManager.nativeGetFreeRamBytes() / (1024L * 1024L)
        val best    = models.maxByOrNull { it.length() }

        if (best != null) {
            Log.i(TAG, "Auto-loading: ${best.name} (${best.length() / 1024 / 1024}MB)")
            updateNotification("Loading ${best.name}...")
            modelManager.loadModel(best)
        }
    }

    // ── Core Chat API ─────────────────────────────────────────────────────────

    /**
     * Main entry point for processing user messages.
     * Handles:
     * 1. Adding to conversation history
     * 2. Building prompt with context
     * 3. Streaming inference
     * 4. Parsing and executing tool calls
     * 5. Storing interaction in Memory Graph
     */
    suspend fun chat(
        userMessage: String,
        onToken: (String) -> Unit,
        onDone: (String) -> Unit
    ) {
        if (!inferenceEngine.isModelLoaded) {
            onToken("[No model loaded. Place a GGUF file in /sdcard/Androbot/models/]")
            onDone("")
            return
        }

        _state.value = ServiceState.Thinking(userMessage)

        try {
            // Retrieve relevant memories to augment the prompt
            val memories = memoryGraph.recall(userMessage, topK = 3)
            val memoryContext = if (memories.isNotEmpty()) {
                "\n[Relevant memories]:\n" + memories.joinToString("\n") +
                "\n[End memories]\n"
            } else ""

            // Add memory context to message if available
            val augmentedMessage = if (memoryContext.isNotEmpty()) {
                "$memoryContext\nUser: $userMessage"
            } else userMessage

            contextManager.addMessage("user", userMessage)

            // Build full prompt
            val prompt = contextManager.buildPrompt(augmentedMessage)

            val responseSb = StringBuilder()

            // Stream tokens
            inferenceEngine.generateStream(
                prompt,
                maxNewTokens = InferenceEngine.DEFAULT_MAX_TOKENS
            ).collect { token ->
                responseSb.append(token)
                onToken(token)
            }

            val fullResponse = responseSb.toString().trim()

            // Handle tool calls if present
            if (functionPipeline.hasToolCalls(fullResponse)) {
                val toolCalls = functionPipeline.extractToolCalls(fullResponse)
                val results   = functionPipeline.executeToolCalls(toolCalls)
                val toolContext = functionPipeline.formatResultsAsContext(results)

                // Inject tool results and continue generation
                val followUpPrompt = prompt + fullResponse + toolContext +
                                     "\n[Continue based on tool results]:\n"
                val followUpSb = StringBuilder()

                inferenceEngine.generateStream(followUpPrompt, maxNewTokens = 256)
                    .collect { token ->
                        followUpSb.append(token)
                        onToken(token)
                    }

                val finalResponse = followUpSb.toString().trim()
                contextManager.addMessage("assistant", finalResponse)
                storeInteraction(userMessage, finalResponse)
                onDone(finalResponse)
            } else {
                contextManager.addMessage("assistant", fullResponse)
                storeInteraction(userMessage, fullResponse)
                onDone(fullResponse)
            }

        } catch (e: Exception) {
            Log.e(TAG, "Chat error: ${e.message}", e)
            onToken("\n[Error: ${e.message}]")
            onDone("")
        } finally {
            _state.value = ServiceState.Ready
        }
    }

    private suspend fun storeInteraction(userMsg: String, assistantMsg: String) {
        serviceScope.launch {
            try {
                val importance = when {
                    userMsg.length > 200    -> 0.8f
                    assistantMsg.length > 300 -> 0.7f
                    else                    -> 0.5f
                }
                memoryGraph.remember(userMsg, "interaction", importance,
                    mapOf("role" to "user"))
                memoryGraph.remember(assistantMsg, "interaction", importance * 0.9f,
                    mapOf("role" to "assistant"))
            } catch (e: Exception) {
                Log.w(TAG, "Failed to store interaction: ${e.message}")
            }
        }
    }

    // ── Service Control ───────────────────────────────────────────────────────

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> {
                Log.i(TAG, "Stop command received")
                stopSelf()
            }
        }
        return START_STICKY  // Restart if killed by system
    }

    override fun onBind(intent: Intent): IBinder = binder

    override fun onDestroy() {
        instance = null
        serviceScope.cancel()
        inferenceEngine.unloadModel()
        visionProcessor.release()
        Log.i(TAG, "AI Background Service destroyed")
        super.onDestroy()
    }

    // ── Notifications ─────────────────────────────────────────────────────────

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID, CHANNEL_NAME,
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Androbot AI engine running in background"
            setShowBadge(false)
        }
        getSystemService(NotificationManager::class.java)
            .createNotificationChannel(channel)
    }

    private fun buildNotification(status: String) =
        NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Androbot AI")
            .setContentText(status)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setOngoing(true)
            .setSilent(true)
            .setContentIntent(
                PendingIntent.getActivity(
                    this, 0,
                    Intent(this, MainActivity::class.java),
                    PendingIntent.FLAG_IMMUTABLE
                )
            )
            .addAction(
                android.R.drawable.ic_delete,
                "Stop",
                PendingIntent.getService(
                    this, 0,
                    Intent(this, AIBackgroundService::class.java).apply {
                        action = ACTION_STOP
                    },
                    PendingIntent.FLAG_IMMUTABLE
                )
            )
            .build()

    private fun updateNotification(status: String) {
        getSystemService(NotificationManager::class.java)
            .notify(NOTIFICATION_ID, buildNotification(status))
    }
}
