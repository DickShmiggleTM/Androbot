package com.androbot.ui

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.androbot.service.AIBackgroundService
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * Chat ViewModel
 *
 * Bridges the Compose UI with the AIBackgroundService.
 * Manages:
 * - Service binding/unbinding
 * - Message history state
 * - Streaming token accumulation
 * - Memory usage display
 */
class ChatViewModel : ViewModel() {

    companion object {
        private const val TAG = "ChatViewModel"
    }

    sealed class MessageType {
        data class User(val text: String) : MessageType()
        data class Assistant(val text: String, val isStreaming: Boolean = false) : MessageType()
        data class System(val text: String) : MessageType()
        data class Error(val text: String) : MessageType()
    }

    data class MemoryStats(
        val kvCacheMB: Float  = 0f,
        val mmapMB: Float     = 0f,
        val contextLen: Int   = 0,
        val memoryCount: Int  = 0,
        val modelLoaded: Boolean = false,
        val pressure: String  = "unknown"
    )

    // ── State ─────────────────────────────────────────────────────────────────

    private val _messages = MutableStateFlow<List<MessageType>>(
        listOf(MessageType.System(
            "Androbot AI initialized.\n" +
            "Place a GGUF model in /sdcard/Androbot/models/ to start.\n" +
            "Enable Accessibility Service in Settings > Accessibility."
        ))
    )
    val messages: StateFlow<List<MessageType>> = _messages.asStateFlow()

    private val _isThinking = MutableStateFlow(false)
    val isThinking: StateFlow<Boolean> = _isThinking.asStateFlow()

    private val _memoryStats = MutableStateFlow(MemoryStats())
    val memoryStats: StateFlow<MemoryStats> = _memoryStats.asStateFlow()

    private val _inputText = MutableStateFlow("")
    val inputText: StateFlow<String> = _inputText.asStateFlow()

    private val _serviceStatus = MutableStateFlow("Connecting...")
    val serviceStatus: StateFlow<String> = _serviceStatus.asStateFlow()

    // ── Service Binding ────────────────────────────────────────────────────────

    private var aiService: AIBackgroundService? = null

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName, binder: IBinder) {
            aiService = (binder as AIBackgroundService.AIBinder).getService()
            _serviceStatus.value = "Connected"
            Log.i(TAG, "Bound to AIBackgroundService")
            refreshMemoryStats()
        }

        override fun onServiceDisconnected(name: ComponentName) {
            aiService = null
            _serviceStatus.value = "Disconnected"
            Log.w(TAG, "Unbound from AIBackgroundService")
        }
    }

    fun bindService(context: Context) {
        val intent = Intent(context, AIBackgroundService::class.java)
        context.startForegroundService(intent)
        context.bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
    }

    fun unbindService(context: Context) {
        try { context.unbindService(serviceConnection) } catch (_: Exception) {}
    }

    // ── Messaging ─────────────────────────────────────────────────────────────

    fun updateInput(text: String) {
        _inputText.value = text
    }

    fun sendMessage() {
        val message = _inputText.value.trim()
        if (message.isEmpty() || _isThinking.value) return

        _inputText.value = ""
        _isThinking.value = true

        // Add user message immediately
        appendMessage(MessageType.User(message))

        val service = aiService
        if (service == null) {
            appendMessage(MessageType.Error("AI service not connected. Restart the app."))
            _isThinking.value = false
            return
        }

        var responseText = ""

        viewModelScope.launch {
            // Add empty assistant message for streaming
            appendMessage(MessageType.Assistant("", isStreaming = true))

            service.chat(
                userMessage = message,
                onToken = { token ->
                    responseText += token
                    updateLastAssistantMessage(responseText, isStreaming = true)
                },
                onDone = { _ ->
                    updateLastAssistantMessage(responseText.trim(), isStreaming = false)
                    _isThinking.value = false
                    refreshMemoryStats()
                }
            )
        }
    }

    private fun appendMessage(msg: MessageType) {
        _messages.value = _messages.value + msg
    }

    private fun updateLastAssistantMessage(text: String, isStreaming: Boolean) {
        val current = _messages.value.toMutableList()
        val lastIdx = current.indexOfLast { it is MessageType.Assistant }
        if (lastIdx >= 0) {
            current[lastIdx] = MessageType.Assistant(text, isStreaming)
            _messages.value = current
        }
    }

    fun clearHistory() {
        aiService?.contextManager?.clearHistory()
        _messages.value = listOf(
            MessageType.System("Conversation cleared.")
        )
    }

    // ── Memory Stats ──────────────────────────────────────────────────────────

    fun refreshMemoryStats() {
        val service = aiService ?: return
        viewModelScope.launch {
            try {
                val engine = service.inferenceEngine
                val memCount = service.memoryGraph.getMemoryCount()

                _memoryStats.value = MemoryStats(
                    kvCacheMB    = engine.getKvCacheMB(),
                    mmapMB       = engine.getMmapMB(),
                    contextLen   = engine.getContextLength(),
                    memoryCount  = memCount,
                    modelLoaded  = engine.isModelLoaded,
                    pressure     = if (engine.needsContextEviction()) "High" else "Normal"
                )
            } catch (e: Exception) {
                Log.w(TAG, "Failed to refresh memory stats: ${e.message}")
            }
        }
    }

    fun startOverlayService(context: Context) {
        val intent = Intent(context, com.androbot.service.OverlayService::class.java)
        context.startForegroundService(intent)
    }

    fun stopOverlayService(context: Context) {
        context.stopService(Intent(context, com.androbot.service.OverlayService::class.java))
    }

    override fun onCleared() {
        super.onCleared()
        aiService = null
    }
}
