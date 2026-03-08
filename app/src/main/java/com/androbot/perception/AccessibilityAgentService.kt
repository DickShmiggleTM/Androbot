package com.androbot.perception

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.AccessibilityServiceInfo
import android.accessibilityservice.GestureDescription
import android.content.Intent
import android.graphics.Path
import android.graphics.Rect
import android.os.Bundle
import android.util.Log
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Accessibility Agent Service
 *
 * Provides deep Android OS perception and control:
 * - Parse entire UI node tree into structured text
 * - Perform click, scroll, swipe, and type actions
 * - Monitor active app and window changes
 * - Extract semantic UI information for AI reasoning
 */
class AccessibilityAgentService : AccessibilityService() {

    companion object {
        private const val TAG = "AccessibilityAgent"

        @Volatile
        private var instance: AccessibilityAgentService? = null

        fun getInstance(): AccessibilityAgentService? = instance

        private val _uiState = MutableStateFlow<UIState>(UIState.Idle)
        val uiState: StateFlow<UIState> = _uiState.asStateFlow()
    }

    sealed class UIState {
        object Idle : UIState()
        data class ScreenChanged(val packageName: String, val text: String) : UIState()
        data class ActionCompleted(val action: String, val success: Boolean) : UIState()
    }

    // ── Service Lifecycle ─────────────────────────────────────────────────────

    override fun onServiceConnected() {
        instance = this
        Log.i(TAG, "Accessibility Service connected")

        serviceInfo = serviceInfo.apply {
            eventTypes = AccessibilityEvent.TYPES_ALL_MASK
            feedbackType = AccessibilityServiceInfo.FEEDBACK_GENERIC
            flags = AccessibilityServiceInfo.FLAG_REPORT_VIEW_IDS or
                    AccessibilityServiceInfo.FLAG_RETRIEVE_INTERACTIVE_WINDOWS or
                    AccessibilityServiceInfo.FLAG_REQUEST_ENHANCED_WEB_ACCESSIBILITY or
                    AccessibilityServiceInfo.FLAG_INCLUDE_NOT_IMPORTANT_VIEWS
            notificationTimeout = 100L
        }
    }

    override fun onInterrupt() {
        Log.w(TAG, "Accessibility Service interrupted")
    }

    override fun onDestroy() {
        instance = null
        super.onDestroy()
        Log.i(TAG, "Accessibility Service destroyed")
    }

    // ── Event Handling ─────────────────────────────────────────────────────────

    override fun onAccessibilityEvent(event: AccessibilityEvent) {
        when (event.eventType) {
            AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED,
            AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED -> {
                val pkg = event.packageName?.toString() ?: return
                if (pkg == "com.androbot") return  // Ignore own events

                // Async UI state capture (don't block accessibility thread)
                val uiText = captureUITextQuick()
                if (uiText.isNotBlank()) {
                    _uiState.value = UIState.ScreenChanged(pkg, uiText)
                }
            }
            else -> { /* Only process state/content changes */ }
        }
    }

    // ── UI Text Extraction ─────────────────────────────────────────────────────

    /**
     * Captures full UI text from the accessibility node tree.
     * Returns structured text representation of all visible elements.
     */
    fun captureFullUIText(): String {
        val root = rootInActiveWindow ?: return "[No active window]"
        val sb = StringBuilder()

        try {
            traverseNode(root, sb, depth = 0)
        } finally {
            root.recycle()
        }

        val result = sb.toString().trim()
        Log.d(TAG, "Captured UI text (${result.length} chars, ~${result.count { it == '\n' }} elements)")
        return result
    }

    /**
     * Quick text extraction (non-recursive, top-level only).
     * Used for real-time screen monitoring without performance impact.
     */
    private fun captureUITextQuick(): String {
        val root = rootInActiveWindow ?: return ""
        return try {
            val texts = mutableListOf<String>()
            collectTextNodes(root, texts, maxNodes = 50)
            texts.joinToString(" | ")
        } finally {
            root.recycle()
        }
    }

    private fun collectTextNodes(node: AccessibilityNodeInfo?, texts: MutableList<String>, maxNodes: Int) {
        if (node == null || texts.size >= maxNodes) return

        val text = node.text?.toString() ?: ""
        val desc = node.contentDescription?.toString() ?: ""

        if (text.isNotBlank()) texts.add(text)
        else if (desc.isNotBlank()) texts.add(desc)

        for (i in 0 until node.childCount.coerceAtMost(10)) {
            collectTextNodes(node.getChild(i), texts, maxNodes)
        }
    }

    private fun traverseNode(node: AccessibilityNodeInfo?, sb: StringBuilder, depth: Int) {
        if (node == null || depth > 15) return

        val indent = "  ".repeat(depth.coerceAtMost(8))
        val text    = node.text?.toString() ?: ""
        val desc    = node.contentDescription?.toString() ?: ""
        val hint    = node.hintText?.toString() ?: ""
        val cls     = node.className?.toString()?.substringAfterLast('.') ?: ""
        val id      = node.viewIdResourceName?.substringAfterLast('/') ?: ""

        val isClickable = node.isClickable
        val isEditable  = node.isEditable
        val isChecked   = node.isChecked
        val isSelected  = node.isSelected

        // Build semantic description
        val displayText = when {
            text.isNotBlank() -> text
            desc.isNotBlank() -> "[DESC: $desc]"
            hint.isNotBlank() -> "[HINT: $hint]"
            else              -> null
        }

        if (displayText != null || isClickable || isEditable) {
            sb.append(indent)
            if (id.isNotBlank()) sb.append("[$id] ")
            if (cls.isNotBlank()) sb.append("<$cls> ")
            if (displayText != null) sb.append(displayText)
            if (isClickable) sb.append(" [CLICKABLE]")
            if (isEditable)  sb.append(" [EDITABLE]")
            if (isChecked)   sb.append(" [CHECKED]")
            if (isSelected)  sb.append(" [SELECTED]")
            sb.append('\n')
        }

        for (i in 0 until node.childCount) {
            traverseNode(node.getChild(i), sb, depth + 1)
        }
    }

    // ── UI Actions ────────────────────────────────────────────────────────────

    /**
     * Clicks the first UI element matching the given text.
     */
    fun clickElementByText(text: String): Boolean {
        val root = rootInActiveWindow ?: return false

        return try {
            val node = findNodeByText(root, text)
            if (node != null) {
                val success = node.performAction(AccessibilityNodeInfo.ACTION_CLICK)
                Log.i(TAG, "Clicked '$text': $success")
                _uiState.value = UIState.ActionCompleted("click:$text", success)
                node.recycle()
                success
            } else {
                Log.w(TAG, "Element not found: '$text'")
                false
            }
        } finally {
            root.recycle()
        }
    }

    /**
     * Types text into the currently focused editable field.
     */
    fun typeText(text: String): Boolean {
        val root = rootInActiveWindow ?: return false

        return try {
            val focused = findFocusedEditableNode(root)
            if (focused != null) {
                val args = Bundle()
                args.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text)
                val success = focused.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)
                Log.i(TAG, "Typed text '$text': $success")
                focused.recycle()
                success
            } else {
                // Fallback: inject via clipboard
                injectViaClipboard(text)
            }
        } finally {
            root.recycle()
        }
    }

    /**
     * Scrolls the screen in a given direction.
     */
    fun scroll(direction: String): Boolean {
        val root = rootInActiveWindow ?: return false

        return try {
            val scrollableNode = findScrollableNode(root)
            if (scrollableNode != null) {
                val action = when (direction.lowercase()) {
                    "down"  -> AccessibilityNodeInfo.ACTION_SCROLL_FORWARD
                    "up"    -> AccessibilityNodeInfo.ACTION_SCROLL_BACKWARD
                    "right" -> AccessibilityNodeInfo.ACTION_SCROLL_FORWARD
                    "left"  -> AccessibilityNodeInfo.ACTION_SCROLL_BACKWARD
                    else    -> AccessibilityNodeInfo.ACTION_SCROLL_FORWARD
                }
                val success = scrollableNode.performAction(action)
                scrollableNode.recycle()
                success
            } else {
                performGestureScroll(direction)
            }
        } finally {
            root.recycle()
        }
    }

    /**
     * Performs a gesture-based scroll when no scrollable node is found.
     */
    private fun performGestureScroll(direction: String): Boolean {
        val displayMetrics = resources.displayMetrics
        val w = displayMetrics.widthPixels.toFloat()
        val h = displayMetrics.heightPixels.toFloat()

        val path = Path()
        when (direction.lowercase()) {
            "down"  -> { path.moveTo(w / 2, h * 0.7f); path.lineTo(w / 2, h * 0.3f) }
            "up"    -> { path.moveTo(w / 2, h * 0.3f); path.lineTo(w / 2, h * 0.7f) }
            "right" -> { path.moveTo(w * 0.2f, h / 2); path.lineTo(w * 0.8f, h / 2) }
            "left"  -> { path.moveTo(w * 0.8f, h / 2); path.lineTo(w * 0.2f, h / 2) }
            else    -> return false
        }

        val gesture = GestureDescription.Builder()
            .addStroke(GestureDescription.StrokeDescription(path, 0, 300))
            .build()

        var result = false
        dispatchGesture(gesture, object : GestureResultCallback() {
            override fun onCompleted(gestureDescription: GestureDescription) {
                result = true
            }
            override fun onCancelled(gestureDescription: GestureDescription) {
                result = false
            }
        }, null)
        return result
    }

    private fun injectViaClipboard(text: String): Boolean {
        val cm = getSystemService(CLIPBOARD_SERVICE) as? android.content.ClipboardManager
        cm?.setPrimaryClip(android.content.ClipData.newPlainText("text", text))

        val root = rootInActiveWindow ?: return false
        val focused = findFocusedEditableNode(root)
        val success = focused?.performAction(AccessibilityNodeInfo.ACTION_PASTE) ?: false
        focused?.recycle()
        root.recycle()
        return success
    }

    // ── Node Search Utilities ─────────────────────────────────────────────────

    private fun findNodeByText(root: AccessibilityNodeInfo, text: String): AccessibilityNodeInfo? {
        // Try exact match first
        val exact = root.findAccessibilityNodeInfosByText(text)
        if (exact.isNotEmpty()) return exact.first()

        // Fuzzy match: find node containing the text substring
        return findNodeRecursive(root) { node ->
            val nodeText = node.text?.toString() ?: ""
            val nodeDesc = node.contentDescription?.toString() ?: ""
            nodeText.contains(text, ignoreCase = true) ||
            nodeDesc.contains(text, ignoreCase = true)
        }
    }

    private fun findFocusedEditableNode(root: AccessibilityNodeInfo): AccessibilityNodeInfo? {
        // Try focused node first
        val focused = root.findFocus(AccessibilityNodeInfo.FOCUS_INPUT)
        if (focused?.isEditable == true) return focused

        // Find any editable node
        return findNodeRecursive(root) { it.isEditable }
    }

    private fun findScrollableNode(root: AccessibilityNodeInfo): AccessibilityNodeInfo? {
        return findNodeRecursive(root) { it.isScrollable }
    }

    private fun findNodeRecursive(
        node: AccessibilityNodeInfo,
        predicate: (AccessibilityNodeInfo) -> Boolean
    ): AccessibilityNodeInfo? {
        if (predicate(node)) return node
        for (i in 0 until node.childCount) {
            val child = node.getChild(i) ?: continue
            val found = findNodeRecursive(child, predicate)
            if (found != null) return found
            child.recycle()
        }
        return null
    }

    fun getCurrentPackageName(): String {
        return rootInActiveWindow?.packageName?.toString() ?: "unknown"
    }

    fun getClickableElements(): List<String> {
        val root = rootInActiveWindow ?: return emptyList()
        val elements = mutableListOf<String>()
        try {
            collectClickableTexts(root, elements)
        } finally {
            root.recycle()
        }
        return elements
    }

    private fun collectClickableTexts(node: AccessibilityNodeInfo?, list: MutableList<String>) {
        if (node == null) return
        if (node.isClickable) {
            val text = node.text?.toString() ?: node.contentDescription?.toString() ?: ""
            if (text.isNotBlank()) list.add(text)
        }
        for (i in 0 until node.childCount.coerceAtMost(20)) {
            collectClickableTexts(node.getChild(i), list)
        }
    }
}
