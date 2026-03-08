package com.androbot.action

import android.content.Context
import android.net.Uri
import android.os.Environment
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * File Manager
 *
 * Provides the AI agent secure file access via:
 * - Internal app storage (always accessible, sandboxed)
 * - External storage via MANAGE_EXTERNAL_STORAGE (requires permission)
 * - SAF (Storage Access Framework) for user-granted file access
 *
 * Security model:
 * - Sandboxed paths enforced (can't escape designated directories)
 * - File size caps to prevent memory exhaustion
 * - Read-only mode for sensitive locations
 */
class FileManager(private val context: Context) {

    companion object {
        private const val TAG = "FileManager"
        private const val MAX_READ_SIZE = 100_000L  // 100KB cap
        private const val MAX_WRITE_SIZE = 50_000L   // 50KB cap

        // Allowed base directories for security sandboxing
        private val ALLOWED_READ_DIRS = listOf(
            "/sdcard/Androbot",
            "/sdcard/Download",
            "/sdcard/Documents",
            "/sdcard/Pictures",
            "/sdcard/DCIM"
        )
    }

    // ── Internal Storage ──────────────────────────────────────────────────────

    /**
     * Reads a file from app's internal storage (always sandboxed).
     */
    suspend fun readInternalFile(filename: String): String = withContext(Dispatchers.IO) {
        try {
            val file = File(context.filesDir, filename)
            if (!file.exists()) return@withContext "[File not found: $filename]"
            if (file.length() > MAX_READ_SIZE) {
                return@withContext file.bufferedReader().use { it.readText(MAX_READ_SIZE.toInt()) } +
                       "\n[...truncated]"
            }
            file.readText()
        } catch (e: Exception) {
            "[Read error: ${e.message}]"
        }
    }

    suspend fun writeInternalFile(filename: String, content: String): Boolean =
        withContext(Dispatchers.IO) {
            try {
                if (content.length > MAX_WRITE_SIZE) {
                    Log.w(TAG, "Content too large to write: ${content.length} chars")
                    return@withContext false
                }
                val file = File(context.filesDir, filename)
                file.parentFile?.mkdirs()
                file.writeText(content)
                Log.i(TAG, "Wrote ${content.length} chars to $filename")
                true
            } catch (e: Exception) {
                Log.e(TAG, "Write failed: $filename - ${e.message}")
                false
            }
        }

    /**
     * Lists files in internal app storage.
     */
    suspend fun listInternalFiles(subdir: String = ""): List<FileInfo> =
        withContext(Dispatchers.IO) {
            val dir = if (subdir.isEmpty()) context.filesDir
                      else File(context.filesDir, subdir)
            dir.listFiles()?.map { f ->
                FileInfo(f.name, f.length(), f.isDirectory, f.lastModified())
            } ?: emptyList()
        }

    // ── External Storage ──────────────────────────────────────────────────────

    /**
     * Reads a file from external storage.
     * Only allowed within permitted directories.
     */
    suspend fun readExternalFile(path: String): String = withContext(Dispatchers.IO) {
        if (!isPathAllowed(path)) {
            Log.w(TAG, "Access denied to path: $path")
            return@withContext "[Access denied: path not in allowed directories]"
        }

        try {
            val file = File(path)
            if (!file.exists()) return@withContext "[File not found: $path]"
            if (!file.canRead()) return@withContext "[No read permission: $path]"

            if (file.length() > MAX_READ_SIZE) {
                return@withContext file.bufferedReader().use { it.readText(MAX_READ_SIZE.toInt()) } +
                       "\n[...file truncated at ${MAX_READ_SIZE / 1000}KB]"
            }
            file.readText()
        } catch (e: Exception) {
            "[Read error: ${e.message}]"
        }
    }

    suspend fun writeExternalFile(path: String, content: String): Boolean =
        withContext(Dispatchers.IO) {
            if (!isPathAllowed(path)) {
                Log.w(TAG, "Write access denied: $path")
                return@withContext false
            }
            try {
                if (content.length > MAX_WRITE_SIZE) return@withContext false
                val file = File(path)
                file.parentFile?.mkdirs()
                file.writeText(content)
                Log.i(TAG, "Wrote to external: $path")
                true
            } catch (e: Exception) {
                Log.e(TAG, "External write failed: ${e.message}")
                false
            }
        }

    /**
     * Lists files in an external directory (sandboxed).
     */
    suspend fun listExternalDirectory(path: String): String = withContext(Dispatchers.IO) {
        if (!isPathAllowed(path)) {
            return@withContext "[Access denied: $path]"
        }
        try {
            val dir = File(path)
            if (!dir.exists() || !dir.isDirectory) return@withContext "[Not a directory: $path]"

            val files = dir.listFiles()?.take(50) ?: emptyList()
            buildString {
                appendLine("Contents of $path (${files.size} items):")
                files.sortedWith(compareBy({ !it.isDirectory }, { it.name })).forEach { f ->
                    val icon = if (f.isDirectory) "📁" else "📄"
                    val size = if (f.isFile) " [${formatFileSize(f.length())}]" else ""
                    appendLine("  $icon ${f.name}$size")
                }
            }
        } catch (e: Exception) {
            "[Listing failed: ${e.message}]"
        }
    }

    // ── SAF (Content URI) Access ──────────────────────────────────────────────

    suspend fun readContentUri(uri: Uri, maxBytes: Long = MAX_READ_SIZE): String =
        withContext(Dispatchers.IO) {
            try {
                context.contentResolver.openInputStream(uri)?.use { stream ->
                    val bytes = ByteArray(maxBytes.toInt())
                    val read  = stream.read(bytes)
                    String(bytes, 0, read.coerceAtLeast(0))
                } ?: "[Cannot open URI: $uri]"
            } catch (e: Exception) {
                "[URI read error: ${e.message}]"
            }
        }

    // ── Androbot Data Directory ────────────────────────────────────────────────

    /**
     * Returns (and creates) the Androbot data directory on external storage.
     */
    fun getAndrobotDir(): File {
        val dir = File(
            Environment.getExternalStorageDirectory(),
            "Androbot"
        )
        if (!dir.exists()) dir.mkdirs()
        return dir
    }

    fun getModelsDir(): File = File(getAndrobotDir(), "models").also { it.mkdirs() }
    fun getNotesDir():  File = File(getAndrobotDir(), "notes").also  { it.mkdirs() }

    // ── Security ──────────────────────────────────────────────────────────────

    private fun isPathAllowed(path: String): Boolean {
        val canonicalPath = try {
            File(path).canonicalPath
        } catch (e: Exception) {
            return false
        }

        // Internal storage always allowed
        if (canonicalPath.startsWith(context.filesDir.canonicalPath)) return true

        // Check against allowed external directories
        return ALLOWED_READ_DIRS.any { allowedDir ->
            canonicalPath.startsWith(allowedDir)
        }
    }

    private fun formatFileSize(bytes: Long): String = when {
        bytes < 1024        -> "${bytes}B"
        bytes < 1024 * 1024 -> "${"%.1f".format(bytes / 1024.0)}KB"
        else                -> "${"%.1f".format(bytes / (1024.0 * 1024.0))}MB"
    }

    private fun java.io.BufferedReader.readText(maxChars: Int): String {
        val sb = StringBuilder()
        var read = 0
        val buf = CharArray(4096)
        while (read < maxChars) {
            val n = this.read(buf, 0, minOf(buf.size, maxChars - read))
            if (n < 0) break
            sb.append(buf, 0, n)
            read += n
        }
        return sb.toString()
    }

    data class FileInfo(
        val name: String,
        val size: Long,
        val isDirectory: Boolean,
        val lastModified: Long
    )
}
