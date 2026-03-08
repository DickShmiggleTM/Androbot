package com.androbot.ai

import android.util.Log
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

/**
 * On-device function-calling pipeline.
 *
 * Parses structured JSON tool-call requests from the model's output
 * and routes them to the appropriate action handler:
 *
 * Available tools:
 * - screen_read       → AccessibilityService node tree
 * - click_element     → AccessibilityService performAction
 * - type_text         → inject text via AccessibilityService
 * - open_app          → launch application by package name
 * - scroll            → scroll in direction
 * - web_search        → headless WebView search
 * - read_file         → SAF file read
 * - write_file        → SAF file write
 * - ocr_screen        → ML Kit OCR on screen capture
 * - remember          → store fact in Memory Graph
 * - recall            → semantic search Memory Graph
 */
class FunctionCallingPipeline {

    companion object {
        private const val TAG = "FunctionCalling"
        private val GSON = Gson()

        // JSON tool schema presented to the model in the system prompt
        const val TOOLS_SCHEMA = """
Available tools (call with JSON):
{"tool": "screen_read"} - Read all visible UI text
{"tool": "click_element", "params": {"text": "element text or id"}} - Click UI element
{"tool": "type_text", "params": {"text": "text to type"}} - Type text into focused field
{"tool": "open_app", "params": {"package": "com.example.app"}} - Open application
{"tool": "scroll", "params": {"direction": "up|down|left|right"}} - Scroll screen
{"tool": "web_search", "params": {"query": "search query"}} - Search the web
{"tool": "read_file", "params": {"path": "/path/to/file"}} - Read file contents
{"tool": "ocr_screen"} - Extract text from screen via OCR
{"tool": "remember", "params": {"fact": "fact to store"}} - Store fact in memory
{"tool": "recall", "params": {"query": "what to recall"}} - Recall from memory

To use a tool, output EXACTLY: <tool_call>{"tool": "...", "params": {...}}</tool_call>
"""
    }

    data class ToolCall(
        val tool: String,
        val params: Map<String, Any> = emptyMap()
    )

    data class ToolResult(
        val tool: String,
        val success: Boolean,
        val result: String,
        val error: String = ""
    )

    interface ToolExecutor {
        suspend fun screenRead(): String
        suspend fun clickElement(text: String): Boolean
        suspend fun typeText(text: String): Boolean
        suspend fun openApp(packageName: String): Boolean
        suspend fun scroll(direction: String): Boolean
        suspend fun webSearch(query: String): String
        suspend fun readFile(path: String): String
        suspend fun ocrScreen(): String
        suspend fun rememberFact(fact: String): Boolean
        suspend fun recallFact(query: String): String
    }

    private var executor: ToolExecutor? = null

    fun setExecutor(exec: ToolExecutor) {
        executor = exec
    }

    /**
     * Extracts all tool calls from model output text.
     * Model is expected to emit: <tool_call>{...}</tool_call>
     */
    fun extractToolCalls(modelOutput: String): List<ToolCall> {
        val calls = mutableListOf<ToolCall>()
        val pattern = Regex("<tool_call>(.*?)</tool_call>", RegexOption.DOT_MATCHES_ALL)

        pattern.findAll(modelOutput).forEach { match ->
            try {
                val json = match.groupValues[1].trim()
                val type = object : TypeToken<Map<String, Any>>() {}.type
                val map: Map<String, Any> = GSON.fromJson(json, type)

                val tool = map["tool"] as? String ?: return@forEach
                @Suppress("UNCHECKED_CAST")
                val params = (map["params"] as? Map<String, Any>) ?: emptyMap()
                calls.add(ToolCall(tool, params))

                Log.d(TAG, "Extracted tool call: $tool with params: $params")
            } catch (e: Exception) {
                Log.w(TAG, "Failed to parse tool call: ${match.groupValues[1]}")
            }
        }

        return calls
    }

    /**
     * Checks if model output contains any tool calls.
     */
    fun hasToolCalls(modelOutput: String): Boolean {
        return modelOutput.contains("<tool_call>") && modelOutput.contains("</tool_call>")
    }

    /**
     * Executes a list of tool calls and returns results.
     */
    suspend fun executeToolCalls(calls: List<ToolCall>): List<ToolResult> {
        val exec = executor ?: return calls.map {
            ToolResult(it.tool, false, "", "No executor registered")
        }

        return calls.map { call ->
            try {
                val result = dispatchCall(exec, call)
                Log.i(TAG, "Tool [${call.tool}] executed: ${result.take(100)}")
                ToolResult(call.tool, true, result)
            } catch (e: Exception) {
                Log.e(TAG, "Tool [${call.tool}] failed: ${e.message}")
                ToolResult(call.tool, false, "", e.message ?: "unknown error")
            }
        }
    }

    private suspend fun dispatchCall(exec: ToolExecutor, call: ToolCall): String {
        return when (call.tool) {
            "screen_read" -> exec.screenRead()

            "click_element" -> {
                val text = call.params["text"] as? String ?: throw IllegalArgumentException("missing text param")
                if (exec.clickElement(text)) "Clicked: $text" else "Failed to click: $text"
            }

            "type_text" -> {
                val text = call.params["text"] as? String ?: throw IllegalArgumentException("missing text param")
                if (exec.typeText(text)) "Typed: $text" else "Failed to type text"
            }

            "open_app" -> {
                val pkg = call.params["package"] as? String ?: throw IllegalArgumentException("missing package param")
                if (exec.openApp(pkg)) "Opened: $pkg" else "Failed to open: $pkg"
            }

            "scroll" -> {
                val dir = call.params["direction"] as? String ?: "down"
                if (exec.scroll(dir)) "Scrolled $dir" else "Failed to scroll"
            }

            "web_search" -> {
                val query = call.params["query"] as? String ?: throw IllegalArgumentException("missing query param")
                exec.webSearch(query)
            }

            "read_file" -> {
                val path = call.params["path"] as? String ?: throw IllegalArgumentException("missing path param")
                exec.readFile(path)
            }

            "ocr_screen" -> exec.ocrScreen()

            "remember" -> {
                val fact = call.params["fact"] as? String ?: throw IllegalArgumentException("missing fact param")
                if (exec.rememberFact(fact)) "Remembered: $fact" else "Failed to remember"
            }

            "recall" -> {
                val query = call.params["query"] as? String ?: throw IllegalArgumentException("missing query param")
                exec.recallFact(query)
            }

            else -> throw IllegalArgumentException("Unknown tool: ${call.tool}")
        }
    }

    /**
     * Formats tool results as context to inject back into the model.
     */
    fun formatResultsAsContext(results: List<ToolResult>): String {
        if (results.isEmpty()) return ""
        return buildString {
            appendLine("\n<tool_results>")
            results.forEach { r ->
                if (r.success) {
                    appendLine("[${r.tool}]: ${r.result.take(2000)}")
                } else {
                    appendLine("[${r.tool}] ERROR: ${r.error}")
                }
            }
            appendLine("</tool_results>")
        }
    }
}
