package com.androbot.service

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.PixelFormat
import android.os.IBinder
import android.util.Log
import android.view.*
import android.view.inputmethod.InputMethodManager
import android.widget.*
import androidx.core.app.NotificationCompat
import com.androbot.R

/**
 * Overlay Service
 *
 * Manages the persistent floating UI overlay using
 * WindowManager with TYPE_APPLICATION_OVERLAY.
 *
 * Features:
 * - Draggable floating button to summon/dismiss chat
 * - Expandable chat panel with scrollable history
 * - Non-blocking: passes touch events to underlying apps
 * - Persists across app switches (true system overlay)
 */
class OverlayService : Service() {

    companion object {
        private const val TAG          = "OverlayService"
        private const val NOTIFICATION_ID = 1002
        private const val CHANNEL_ID   = "androbot_overlay"

        @Volatile
        private var instance: OverlayService? = null
        fun getInstance(): OverlayService? = instance
    }

    private var windowManager: WindowManager? = null
    private var fabView:       View?          = null
    private var chatPanelView: View?          = null

    private var isChatVisible  = false
    private var isDragging     = false
    private var initialX       = 0
    private var initialY       = 0
    private var initialTouchX  = 0f
    private var initialTouchY  = 0f

    private var onMessageSentCallback: ((String) -> Unit)? = null

    // ── Service Lifecycle ─────────────────────────────────────────────────────

    override fun onCreate() {
        super.onCreate()
        instance = this
        windowManager = getSystemService(WINDOW_SERVICE) as WindowManager

        createNotificationChannel()
        startForeground(NOTIFICATION_ID, buildNotification())

        setupFloatingButton()
        Log.i(TAG, "Overlay Service started")
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        return START_STICKY
    }

    override fun onBind(intent: Intent): IBinder? = null

    override fun onDestroy() {
        instance = null
        removeOverlays()
        Log.i(TAG, "Overlay Service destroyed")
        super.onDestroy()
    }

    // ── Floating Button ────────────────────────────────────────────────────────

    private fun setupFloatingButton() {
        val inflater = LayoutInflater.from(this)

        // Create a simple programmatic FAB view
        fabView = createFabView()

        val params = WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
            WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.END
            x = 24
            y = 200
        }

        fabView?.setOnTouchListener(object : View.OnTouchListener {
            override fun onTouch(v: View, event: MotionEvent): Boolean {
                return handleFabTouch(v, event, params)
            }
        })

        try {
            windowManager?.addView(fabView, params)
            Log.i(TAG, "Floating button added to WindowManager")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to add overlay: ${e.message}")
        }
    }

    private fun createFabView(): View {
        val size = (60 * resources.displayMetrics.density).toInt()

        return TextView(this).apply {
            text = "🤖"
            textSize = 28f
            gravity = Gravity.CENTER
            setBackgroundResource(android.R.drawable.btn_default)

            val pad = (8 * resources.displayMetrics.density).toInt()
            setPadding(pad, pad, pad, pad)
            layoutParams = ViewGroup.LayoutParams(size, size)
        }
    }

    private fun handleFabTouch(
        view: View,
        event: MotionEvent,
        params: WindowManager.LayoutParams
    ): Boolean {
        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                isDragging    = false
                initialX      = params.x
                initialY      = params.y
                initialTouchX = event.rawX
                initialTouchY = event.rawY
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                val dx = (event.rawX - initialTouchX).toInt()
                val dy = (event.rawY - initialTouchY).toInt()
                if (Math.abs(dx) > 8 || Math.abs(dy) > 8) {
                    isDragging = true
                    params.x   = initialX - dx  // Negative because Gravity.END
                    params.y   = initialY + dy
                    try { windowManager?.updateViewLayout(view, params) }
                    catch (e: Exception) { /* ignore layout races */ }
                }
                return true
            }
            MotionEvent.ACTION_UP -> {
                if (!isDragging) {
                    toggleChatPanel()
                }
                return true
            }
        }
        return false
    }

    // ── Chat Panel ────────────────────────────────────────────────────────────

    private fun toggleChatPanel() {
        if (isChatVisible) {
            hideChatPanel()
        } else {
            showChatPanel()
        }
    }

    private fun showChatPanel() {
        if (chatPanelView != null) return

        chatPanelView = createChatPanelView()

        val metrics   = resources.displayMetrics
        val panelW    = (metrics.widthPixels * 0.92f).toInt()
        val panelH    = (metrics.heightPixels * 0.6f).toInt()

        val params = WindowManager.LayoutParams(
            panelW,
            panelH,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_WATCH_OUTSIDE_TOUCH,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL
            y = (metrics.heightPixels * 0.05f).toInt()
        }

        try {
            windowManager?.addView(chatPanelView, params)
            isChatVisible = true
            (fabView as? TextView)?.text = "✕"
            Log.i(TAG, "Chat panel shown")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to show chat panel: ${e.message}")
            chatPanelView = null
        }
    }

    private fun hideChatPanel() {
        chatPanelView?.let {
            try {
                windowManager?.removeView(it)
            } catch (e: Exception) { /* ignore */ }
        }
        chatPanelView = null
        isChatVisible = false
        (fabView as? TextView)?.text = "🤖"
        Log.i(TAG, "Chat panel hidden")
    }

    private fun createChatPanelView(): LinearLayout {
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(0xEE1E1E2E.toInt())  // Dark navy, 93% opacity

            val dp = resources.displayMetrics.density
            setPadding(
                (12 * dp).toInt(), (12 * dp).toInt(),
                (12 * dp).toInt(), (12 * dp).toInt()
            )
        }

        // Header
        val header = TextView(this).apply {
            text = "Androbot AI"
            textSize = 16f
            setTextColor(0xFFCDD6F4.toInt())
            setPadding(0, 0, 0, (8 * resources.displayMetrics.density).toInt())
        }
        layout.addView(header)

        // Messages scroll view
        val scrollView = ScrollView(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f
            )
            setBackgroundColor(0x1ACDD6F4.toInt())
        }

        val messageContainer = LinearLayout(this).apply {
            id = android.R.id.content
            orientation = LinearLayout.VERTICAL
            val p = (8 * resources.displayMetrics.density).toInt()
            setPadding(p, p, p, p)
        }
        scrollView.addView(messageContainer)
        layout.addView(scrollView)

        // Input row
        val inputRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            val dp = resources.displayMetrics.density
            setPadding(0, (8 * dp).toInt(), 0, 0)
        }

        val input = EditText(this).apply {
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            hint = "Ask anything..."
            setTextColor(0xFFCDD6F4.toInt())
            setHintTextColor(0x88CDD6F4.toInt())
            setBackgroundColor(0x33CDD6F4.toInt())
            textSize = 14f
            maxLines = 3
            val p = (8 * resources.displayMetrics.density).toInt()
            setPadding(p, p, p, p)
        }

        val sendBtn = Button(this).apply {
            text = "Send"
            textSize = 13f
            setTextColor(0xFF1E1E2E.toInt())
            setBackgroundColor(0xFF89B4FA.toInt())

            setOnClickListener {
                val msg = input.text.toString().trim()
                if (msg.isNotEmpty()) {
                    input.setText("")
                    addMessageBubble(messageContainer, "You", msg, isUser = true)
                    scrollView.post { scrollView.fullScroll(ScrollView.FOCUS_DOWN) }
                    onMessageSentCallback?.invoke(msg)
                }
            }
        }

        inputRow.addView(input)
        inputRow.addView(sendBtn)
        layout.addView(inputRow)

        return layout
    }

    // ── Public API ────────────────────────────────────────────────────────────

    fun setOnMessageSentCallback(callback: (String) -> Unit) {
        onMessageSentCallback = callback
    }

    fun appendAssistantMessage(text: String) {
        val container = chatPanelView?.findViewById<LinearLayout>(android.R.id.content)
            ?: return

        container.post {
            addMessageBubble(container, "Androbot", text, isUser = false)
            // Auto-scroll to bottom
            (chatPanelView as? LinearLayout)?.let { panel ->
                for (i in 0 until panel.childCount) {
                    (panel.getChildAt(i) as? ScrollView)?.fullScroll(ScrollView.FOCUS_DOWN)
                }
            }
        }
    }

    fun appendTokenToLastMessage(token: String) {
        val container = chatPanelView?.findViewById<LinearLayout>(android.R.id.content)
            ?: return

        container.post {
            val lastChild = if (container.childCount > 0)
                container.getChildAt(container.childCount - 1) else null
            val lastMsg = lastChild as? TextView

            if (lastMsg != null && lastMsg.tag == "assistant") {
                lastMsg.append(token)
            } else {
                addMessageBubble(container, "Androbot", token, isUser = false)
            }
        }
    }

    private fun addMessageBubble(
        container: LinearLayout,
        sender: String,
        text: String,
        isUser: Boolean
    ) {
        val dp = resources.displayMetrics.density
        val bubble = TextView(this).apply {
            this.text = if (isUser) "You: $text" else "AI: $text"
            textSize  = 13f
            tag       = if (isUser) "user" else "assistant"
            setTextColor(if (isUser) 0xFFA6E3A1.toInt() else 0xFFCDD6F4.toInt())
            setBackgroundColor(if (isUser) 0x22A6E3A1.toInt() else 0x22CDD6F4.toInt())

            val lp = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                setMargins(0, (4 * dp).toInt(), 0, (4 * dp).toInt())
            }
            layoutParams = lp
            val p = (8 * dp).toInt()
            setPadding(p, p, p, p)
        }
        container.addView(bubble)
    }

    private fun removeOverlays() {
        fabView?.let {
            try { windowManager?.removeView(it) } catch (_: Exception) {}
        }
        chatPanelView?.let {
            try { windowManager?.removeView(it) } catch (_: Exception) {}
        }
        fabView       = null
        chatPanelView = null
    }

    // ── Notifications ─────────────────────────────────────────────────────────

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID, "Androbot Overlay",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            setShowBadge(false)
            description = "Floating UI overlay"
        }
        getSystemService(NotificationManager::class.java)
            .createNotificationChannel(channel)
    }

    private fun buildNotification() =
        NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Androbot")
            .setContentText("Floating overlay active")
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .setOngoing(true)
            .setSilent(true)
            .build()
}
