package com.androbot

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.androbot.service.AIBackgroundService
import com.androbot.ui.ChatViewModel
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    private val viewModel: ChatViewModel by viewModels()

    // Permission launcher
    private val overlayPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { _ ->
        if (Settings.canDrawOverlays(this)) {
            Log.i(TAG, "Overlay permission granted")
        }
    }

    private val storagePermissionLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { _ -> Log.i(TAG, "Storage permission result received") }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        permissions.entries.forEach { (perm, granted) ->
            Log.i(TAG, "Permission $perm: $granted")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        requestRequiredPermissions()
        viewModel.bindService(this)

        setContent {
            AndrobotTheme {
                MainScreen(
                    viewModel     = viewModel,
                    onOverlayToggle = { enabled ->
                        if (enabled) viewModel.startOverlayService(this)
                        else         viewModel.stopOverlayService(this)
                    },
                    onOpenAccessibilitySettings = {
                        startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
                    },
                    onRequestOverlayPerm = {
                        val intent = Intent(
                            Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                            Uri.parse("package:$packageName")
                        )
                        overlayPermissionLauncher.launch(intent)
                    }
                )
            }
        }
    }

    override fun onDestroy() {
        viewModel.unbindService(this)
        super.onDestroy()
    }

    private fun requestRequiredPermissions() {
        // Overlay permission
        if (!Settings.canDrawOverlays(this)) {
            val intent = Intent(
                Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                Uri.parse("package:$packageName")
            )
            overlayPermissionLauncher.launch(intent)
        }

        // Storage permission (Android 11+)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (!Environment.isExternalStorageManager()) {
                val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION).apply {
                    data = Uri.parse("package:$packageName")
                }
                storagePermissionLauncher.launch(intent)
            }
        }

        // Notification permission (Android 13+)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
                permissionLauncher.launch(arrayOf(Manifest.permission.POST_NOTIFICATIONS))
            }
        }
    }
}

// ============================================================
// Catppuccin Mocha Color Palette
// ============================================================
object CatppuccinMocha {
    val Base     = Color(0xFF1E1E2E)
    val Mantle   = Color(0xFF181825)
    val Crust    = Color(0xFF11111B)
    val Surface0 = Color(0xFF313244)
    val Surface1 = Color(0xFF45475A)
    val Surface2 = Color(0xFF585B70)
    val Overlay0 = Color(0xFF6C7086)
    val Overlay1 = Color(0xFF7F849C)
    val Text     = Color(0xFFCDD6F4)
    val Subtext0 = Color(0xFFA6ADC8)
    val Subtext1 = Color(0xFFBAC2DE)
    val Blue     = Color(0xFF89B4FA)
    val Green    = Color(0xFFA6E3A1)
    val Yellow   = Color(0xFFF9E2AF)
    val Red      = Color(0xFFF38BA8)
    val Lavender = Color(0xFFB4BEFE)
    val Mauve    = Color(0xFFCBA6F7)
    val Peach    = Color(0xFFFAB387)
    val Teal     = Color(0xFF94E2D5)
}

@Composable
fun AndrobotTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = darkColorScheme(
            primary         = CatppuccinMocha.Blue,
            secondary       = CatppuccinMocha.Mauve,
            tertiary        = CatppuccinMocha.Green,
            background      = CatppuccinMocha.Base,
            surface         = CatppuccinMocha.Mantle,
            surfaceVariant  = CatppuccinMocha.Surface0,
            onBackground    = CatppuccinMocha.Text,
            onSurface       = CatppuccinMocha.Text,
            onPrimary       = CatppuccinMocha.Crust,
            error           = CatppuccinMocha.Red
        ),
        content = content
    )
}

// ============================================================
// Main Screen Composable
// ============================================================

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(
    viewModel: ChatViewModel,
    onOverlayToggle: (Boolean) -> Unit,
    onOpenAccessibilitySettings: () -> Unit,
    onRequestOverlayPerm: () -> Unit
) {
    val messages    by viewModel.messages.collectAsState()
    val isThinking  by viewModel.isThinking.collectAsState()
    val inputText   by viewModel.inputText.collectAsState()
    val memStats    by viewModel.memoryStats.collectAsState()
    val serviceStatus by viewModel.serviceStatus.collectAsState()

    val listState   = rememberLazyListState()
    val scope       = rememberCoroutineScope()
    var showStatsPanel by remember { mutableStateOf(false) }
    var overlayEnabled by remember { mutableStateOf(false) }

    // Auto-scroll on new messages
    LaunchedEffect(messages.size) {
        if (messages.isNotEmpty()) {
            listState.animateScrollToItem(messages.size - 1)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(
                            "Androbot",
                            color     = CatppuccinMocha.Blue,
                            fontWeight = FontWeight.Bold,
                            fontSize  = 20.sp
                        )
                        Text(
                            serviceStatus,
                            color    = CatppuccinMocha.Subtext0,
                            fontSize = 11.sp
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = CatppuccinMocha.Mantle
                ),
                actions = {
                    // Memory stats toggle
                    IconButton(onClick = { showStatsPanel = !showStatsPanel }) {
                        Icon(
                            Icons.Filled.Memory,
                            contentDescription = "Memory Stats",
                            tint = if (memStats.pressure == "High")
                                CatppuccinMocha.Red else CatppuccinMocha.Subtext0
                        )
                    }
                    // Overlay toggle
                    IconButton(onClick = {
                        overlayEnabled = !overlayEnabled
                        onOverlayToggle(overlayEnabled)
                    }) {
                        Icon(
                            Icons.Filled.Layers,
                            contentDescription = "Toggle Overlay",
                            tint = if (overlayEnabled)
                                CatppuccinMocha.Green else CatppuccinMocha.Subtext0
                        )
                    }
                    // More options
                    var showMenu by remember { mutableStateOf(false) }
                    IconButton(onClick = { showMenu = true }) {
                        Icon(Icons.Filled.MoreVert, "Menu",
                             tint = CatppuccinMocha.Subtext0)
                    }
                    DropdownMenu(
                        expanded = showMenu,
                        onDismissRequest = { showMenu = false }
                    ) {
                        DropdownMenuItem(
                            text = { Text("Accessibility Settings") },
                            onClick = {
                                showMenu = false
                                onOpenAccessibilitySettings()
                            }
                        )
                        DropdownMenuItem(
                            text = { Text("Overlay Permission") },
                            onClick = {
                                showMenu = false
                                onRequestOverlayPerm()
                            }
                        )
                        DropdownMenuItem(
                            text = { Text("Clear History") },
                            onClick = {
                                showMenu = false
                                viewModel.clearHistory()
                            }
                        )
                        DropdownMenuItem(
                            text = { Text("Refresh Stats") },
                            onClick = {
                                showMenu = false
                                viewModel.refreshMemoryStats()
                            }
                        )
                    }
                }
            )
        },
        containerColor = CatppuccinMocha.Base
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
        ) {
            // Memory stats panel
            if (showStatsPanel) {
                MemoryStatsPanel(memStats)
            }

            // Model status banner
            if (!memStats.modelLoaded) {
                ModelSetupBanner()
            }

            // Messages
            LazyColumn(
                modifier  = Modifier.weight(1f),
                state     = listState,
                contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                items(messages) { msg ->
                    MessageBubble(msg)
                }
                if (isThinking) {
                    item {
                        ThinkingIndicator()
                    }
                }
            }

            // Input bar
            ChatInputBar(
                text      = inputText,
                onTextChange = viewModel::updateInput,
                onSend    = viewModel::sendMessage,
                enabled   = !isThinking && memStats.modelLoaded,
                isThinking = isThinking
            )
        }
    }
}

@Composable
fun MessageBubble(message: ChatViewModel.MessageType) {
    val (bgColor, textColor, isRight) = when (message) {
        is ChatViewModel.MessageType.User      ->
            Triple(CatppuccinMocha.Surface0, CatppuccinMocha.Green, true)
        is ChatViewModel.MessageType.Assistant ->
            Triple(CatppuccinMocha.Mantle, CatppuccinMocha.Text, false)
        is ChatViewModel.MessageType.System    ->
            Triple(CatppuccinMocha.Crust, CatppuccinMocha.Subtext0, false)
        is ChatViewModel.MessageType.Error     ->
            Triple(CatppuccinMocha.Crust, CatppuccinMocha.Red, false)
    }

    val text = when (message) {
        is ChatViewModel.MessageType.User      -> message.text
        is ChatViewModel.MessageType.Assistant -> message.text +
            if (message.isStreaming) "▋" else ""
        is ChatViewModel.MessageType.System    -> "ℹ ${message.text}"
        is ChatViewModel.MessageType.Error     -> "⚠ ${message.text}"
    }

    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isRight) Arrangement.End else Arrangement.Start
    ) {
        Box(
            modifier = Modifier
                .widthIn(max = 320.dp)
                .background(bgColor, RoundedCornerShape(12.dp))
                .padding(horizontal = 12.dp, vertical = 8.dp)
        ) {
            Text(
                text       = text,
                color      = textColor,
                fontSize   = 14.sp,
                fontFamily = FontFamily.Default,
                lineHeight = 20.sp
            )
        }
    }
}

@Composable
fun ThinkingIndicator() {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp)
    ) {
        CircularProgressIndicator(
            modifier  = Modifier.size(16.dp),
            color     = CatppuccinMocha.Blue,
            strokeWidth = 2.dp
        )
        Spacer(Modifier.width(8.dp))
        Text("Thinking...", color = CatppuccinMocha.Subtext0, fontSize = 13.sp)
    }
}

@Composable
fun ChatInputBar(
    text: String,
    onTextChange: (String) -> Unit,
    onSend: () -> Unit,
    enabled: Boolean,
    isThinking: Boolean
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(CatppuccinMocha.Mantle)
            .padding(8.dp),
        verticalAlignment = Alignment.Bottom
    ) {
        OutlinedTextField(
            value         = text,
            onValueChange = onTextChange,
            modifier      = Modifier.weight(1f),
            placeholder   = {
                Text(
                    if (enabled) "Ask anything..." else "Loading model...",
                    color = CatppuccinMocha.Overlay0,
                    fontSize = 14.sp
                )
            },
            colors = OutlinedTextFieldDefaults.colors(
                focusedTextColor    = CatppuccinMocha.Text,
                unfocusedTextColor  = CatppuccinMocha.Text,
                focusedBorderColor  = CatppuccinMocha.Blue,
                unfocusedBorderColor= CatppuccinMocha.Surface1,
                cursorColor         = CatppuccinMocha.Blue,
                focusedContainerColor   = CatppuccinMocha.Surface0,
                unfocusedContainerColor = CatppuccinMocha.Surface0
            ),
            shape   = RoundedCornerShape(12.dp),
            maxLines = 4,
            enabled  = enabled
        )
        Spacer(Modifier.width(8.dp))
        IconButton(
            onClick  = onSend,
            enabled  = enabled && text.isNotBlank(),
            modifier = Modifier.size(48.dp)
        ) {
            Icon(
                if (isThinking) Icons.Filled.Stop else Icons.Filled.Send,
                contentDescription = "Send",
                tint = if (enabled && text.isNotBlank())
                    CatppuccinMocha.Blue else CatppuccinMocha.Overlay0
            )
        }
    }
}

@Composable
fun MemoryStatsPanel(stats: ChatViewModel.MemoryStats) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp, vertical = 4.dp),
        colors = CardDefaults.cardColors(containerColor = CatppuccinMocha.Mantle)
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Text(
                "Memory & RAM",
                color      = CatppuccinMocha.Blue,
                fontWeight = FontWeight.Bold,
                fontSize   = 13.sp
            )
            Spacer(Modifier.height(4.dp))
            StatRow("Model (mmap)",   "${"%.1f".format(stats.mmapMB)}MB",   CatppuccinMocha.Teal)
            StatRow("KV Cache (RAM)", "${"%.1f".format(stats.kvCacheMB)}MB", CatppuccinMocha.Peach)
            StatRow("Context",        "${stats.contextLen} tokens",          CatppuccinMocha.Lavender)
            StatRow("Memories",       "${stats.memoryCount} stored",         CatppuccinMocha.Mauve)
            StatRow("Pressure",       stats.pressure,
                if (stats.pressure == "High") CatppuccinMocha.Red else CatppuccinMocha.Green)
        }
    }
}

@Composable
fun StatRow(label: String, value: String, valueColor: Color) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(label, color = CatppuccinMocha.Subtext0, fontSize = 12.sp)
        Text(value, color = valueColor, fontSize = 12.sp, fontWeight = FontWeight.Medium)
    }
}

@Composable
fun ModelSetupBanner() {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp, vertical = 4.dp),
        colors = CardDefaults.cardColors(containerColor = CatppuccinMocha.Surface0)
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Text(
                "⚠ No model loaded",
                color = CatppuccinMocha.Yellow,
                fontWeight = FontWeight.Bold,
                fontSize = 13.sp
            )
            Text(
                "Download a GGUF model and place it in:\n" +
                "/sdcard/Androbot/models/\n\n" +
                "Recommended: Gemma-3-1B-IT-Q4_K_M.gguf (~620MB)",
                color    = CatppuccinMocha.Subtext0,
                fontSize = 12.sp,
                lineHeight = 17.sp
            )
        }
    }
}
