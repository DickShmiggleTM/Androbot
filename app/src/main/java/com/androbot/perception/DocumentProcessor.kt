package com.androbot.perception

import android.content.Context
import android.graphics.Bitmap
import android.graphics.pdf.PdfRenderer
import android.net.Uri
import android.os.ParcelFileDescriptor
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Document Processor
 *
 * Provides local document understanding via:
 * - PdfRenderer: native Android PDF → Bitmap page rendering
 * - ML Kit Text Recognition: extract text from rendered pages
 * - Plain text file reading for TXT/MD/code files
 *
 * All processing is 100% on-device, no cloud calls.
 */
class DocumentProcessor(
    private val context: Context,
    private val visionProcessor: VisionProcessor
) {
    companion object {
        private const val TAG     = "DocumentProcessor"
        private const val MAX_PDF_PAGES = 20    // Cap to prevent OOM
        private const val PDF_DPI       = 150   // 150 DPI for legibility at low memory
        private const val PDF_WIDTH_PX  = 1200  // Approximate letter-size width
    }

    // ── PDF Processing ────────────────────────────────────────────────────────

    /**
     * Extracts all text from a PDF file using PdfRenderer + OCR.
     * Processes up to MAX_PDF_PAGES pages to stay within memory budget.
     */
    suspend fun extractPdfText(uri: Uri): String = withContext(Dispatchers.IO) {
        Log.i(TAG, "Processing PDF: $uri")

        var pfd: ParcelFileDescriptor? = null
        var renderer: PdfRenderer? = null

        try {
            pfd = context.contentResolver.openFileDescriptor(uri, "r")
                ?: return@withContext "[Cannot open PDF file]"

            renderer = PdfRenderer(pfd)
            val pageCount = renderer.pageCount.coerceAtMost(MAX_PDF_PAGES)
            Log.i(TAG, "PDF has $pageCount pages (capped at $MAX_PDF_PAGES)")

            val allText = StringBuilder()
            allText.appendLine("=== PDF Document ===")

            for (pageIndex in 0 until pageCount) {
                val pageText = extractPdfPage(renderer, pageIndex)
                if (pageText.isNotBlank()) {
                    allText.appendLine("--- Page ${pageIndex + 1} ---")
                    allText.appendLine(pageText)
                }
            }

            if (renderer.pageCount > MAX_PDF_PAGES) {
                allText.appendLine("[... ${renderer.pageCount - MAX_PDF_PAGES} more pages not shown]")
            }

            allText.toString()
        } catch (e: Exception) {
            Log.e(TAG, "PDF extraction failed: ${e.message}", e)
            "[PDF extraction error: ${e.message}]"
        } finally {
            renderer?.close()
            pfd?.close()
        }
    }

    private suspend fun extractPdfPage(
        renderer: PdfRenderer,
        pageIndex: Int
    ): String = withContext(Dispatchers.IO) {
        var page: PdfRenderer.Page? = null
        var bitmap: Bitmap? = null

        try {
            page = renderer.openPage(pageIndex)

            // Calculate dimensions preserving aspect ratio
            val aspectRatio = page.height.toFloat() / page.width.toFloat()
            val bitmapWidth  = PDF_WIDTH_PX
            val bitmapHeight = (PDF_WIDTH_PX * aspectRatio).toInt()

            bitmap = Bitmap.createBitmap(bitmapWidth, bitmapHeight, Bitmap.Config.ARGB_8888)

            // Render the PDF page to bitmap
            // Use RENDER_MODE_FOR_DISPLAY for fast rendering (vs RENDER_MODE_FOR_PRINT)
            page.render(bitmap, null, null, PdfRenderer.Page.RENDER_MODE_FOR_DISPLAY)

            // Extract text from rendered page via OCR
            val text = visionProcessor.runOcr(bitmap)
            Log.d(TAG, "Page ${pageIndex + 1}: extracted ${text.length} chars")
            text
        } catch (e: Exception) {
            Log.e(TAG, "Failed to render page $pageIndex: ${e.message}")
            ""
        } finally {
            page?.close()
            bitmap?.recycle()
        }
    }

    /**
     * Reads a plain text file (TXT, MD, CSV, code files).
     * Caps at 50KB to stay within context window.
     */
    suspend fun readTextFile(uri: Uri, maxBytes: Long = 50_000L): String =
        withContext(Dispatchers.IO) {
            try {
                context.contentResolver.openInputStream(uri)?.use { stream ->
                    val bytes = ByteArray(maxBytes.toInt())
                    val read  = stream.read(bytes)
                    val text  = String(bytes, 0, read.coerceAtLeast(0))

                    Log.i(TAG, "Read text file: $read bytes from $uri")
                    text
                } ?: "[Cannot open file]"
            } catch (e: Exception) {
                Log.e(TAG, "Text file read failed: ${e.message}")
                "[Read error: ${e.message}]"
            }
        }

    /**
     * Reads a file from the filesystem by absolute path.
     */
    suspend fun readFileByPath(path: String, maxBytes: Long = 50_000L): String =
        withContext(Dispatchers.IO) {
            try {
                val file = File(path)
                if (!file.exists() || !file.canRead()) {
                    return@withContext "[File not found or not readable: $path]"
                }

                val text = file.inputStream().use { stream ->
                    val bytes = ByteArray(maxBytes.toInt())
                    val read  = stream.read(bytes)
                    String(bytes, 0, read.coerceAtLeast(0))
                }

                Log.i(TAG, "Read file: ${text.length} chars from $path")
                text
            } catch (e: Exception) {
                Log.e(TAG, "File read failed: $path - ${e.message}")
                "[Read error: ${e.message}]"
            }
        }

    /**
     * Lists files in a directory.
     */
    suspend fun listDirectory(path: String): String = withContext(Dispatchers.IO) {
        try {
            val dir = File(path)
            if (!dir.exists() || !dir.isDirectory) {
                return@withContext "[Not a valid directory: $path]"
            }

            val entries = dir.listFiles()?.take(100) ?: emptyList()
            buildString {
                appendLine("Directory: $path")
                appendLine("${entries.size} items:")
                entries.sortedWith(compareBy({ !it.isDirectory }, { it.name })).forEach { f ->
                    val type = if (f.isDirectory) "DIR " else "FILE"
                    val size = if (f.isFile) " (${formatSize(f.length())})" else ""
                    appendLine("  [$type] ${f.name}$size")
                }
            }
        } catch (e: Exception) {
            "[Directory listing failed: ${e.message}]"
        }
    }

    private fun formatSize(bytes: Long): String = when {
        bytes < 1024        -> "${bytes}B"
        bytes < 1024 * 1024 -> "${"%.1f".format(bytes / 1024.0)}KB"
        else                -> "${"%.1f".format(bytes / (1024.0 * 1024.0))}MB"
    }
}
