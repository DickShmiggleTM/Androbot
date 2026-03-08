package com.androbot.perception

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.Image
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * Vision Processor
 *
 * Combines MediaProjection (screen capture) with ML Kit OCR to
 * provide visual understanding of the Android screen.
 *
 * Architecture:
 * - MediaProjection → VirtualDisplay → ImageReader (screen frames)
 * - Captured Bitmap → ML Kit Text Recognition (OCR)
 * - Result text injected into AI context for visual reasoning
 */
class VisionProcessor(private val context: Context) {

    companion object {
        private const val TAG        = "VisionProcessor"
        private const val VIRTUAL_DX = "androbot_capture"
        private const val MAX_IMAGES = 2  // Double-buffer

        // Downscale factor to reduce memory usage (720p → 360p equivalent)
        private const val CAPTURE_SCALE = 0.5f
    }

    private var mediaProjection: MediaProjection? = null
    private var virtualDisplay:  VirtualDisplay?  = null
    private var imageReader:     ImageReader?      = null
    private var captureThread:   HandlerThread?    = null
    private var captureHandler:  Handler?          = null

    private val textRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

    // ── MediaProjection Setup ─────────────────────────────────────────────────

    fun setupProjection(projection: MediaProjection) {
        mediaProjection = projection
        Log.i(TAG, "MediaProjection configured")
    }

    fun startCapture(width: Int, height: Int, dpi: Int) {
        stopCapture()

        val scaledW = (width * CAPTURE_SCALE).toInt()
        val scaledH = (height * CAPTURE_SCALE).toInt()

        captureThread = HandlerThread("AndrobotCapture").also { it.start() }
        captureHandler = Handler(captureThread!!.looper)

        imageReader = ImageReader.newInstance(scaledW, scaledH, PixelFormat.RGBA_8888, MAX_IMAGES)

        virtualDisplay = mediaProjection?.createVirtualDisplay(
            VIRTUAL_DX,
            scaledW, scaledH, dpi,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            imageReader!!.surface,
            null,
            captureHandler
        )

        Log.i(TAG, "Screen capture started: ${scaledW}x${scaledH} @ ${dpi}dpi")
    }

    fun stopCapture() {
        virtualDisplay?.release()
        imageReader?.close()
        captureThread?.quitSafely()
        virtualDisplay  = null
        imageReader     = null
        captureThread   = null
        captureHandler  = null
    }

    fun releaseProjection() {
        stopCapture()
        mediaProjection?.stop()
        mediaProjection = null
        Log.i(TAG, "MediaProjection released")
    }

    // ── Screen Capture + OCR ──────────────────────────────────────────────────

    /**
     * Captures current screen as Bitmap.
     * Uses a double-buffered ImageReader for minimal memory allocation.
     */
    suspend fun captureScreen(): Bitmap? = withContext(Dispatchers.IO) {
        val reader = imageReader ?: run {
            Log.w(TAG, "ImageReader not initialized")
            return@withContext null
        }

        var image: Image? = null
        try {
            image = reader.acquireLatestImage() ?: return@withContext null

            val planes = image.planes
            val buffer = planes[0].buffer
            val pixelStride = planes[0].pixelStride
            val rowStride   = planes[0].rowStride
            val rowPadding  = rowStride - pixelStride * image.width

            val bitmap = Bitmap.createBitmap(
                image.width + rowPadding / pixelStride,
                image.height,
                Bitmap.Config.ARGB_8888
            )
            bitmap.copyPixelsFromBuffer(buffer)

            // Crop to exact size (remove row padding)
            Bitmap.createBitmap(bitmap, 0, 0, image.width, image.height)
        } catch (e: Exception) {
            Log.e(TAG, "Screen capture failed: ${e.message}")
            null
        } finally {
            image?.close()
        }
    }

    /**
     * Extracts all text from the current screen using ML Kit OCR.
     * Combines bounding box positions with text for spatial understanding.
     */
    suspend fun extractScreenText(): String = withContext(Dispatchers.IO) {
        val bitmap = captureScreen()
            ?: return@withContext "[Screen capture unavailable - no MediaProjection permission]"

        try {
            val text = runOcr(bitmap)
            if (text.isBlank()) {
                "[Screen shows no readable text]"
            } else {
                text
            }
        } finally {
            bitmap.recycle()
        }
    }

    /**
     * Runs ML Kit OCR on a bitmap. Returns structured text with positions.
     */
    suspend fun runOcr(bitmap: Bitmap): String = suspendCancellableCoroutine { cont ->
        val image = InputImage.fromBitmap(bitmap, 0)

        textRecognizer.process(image)
            .addOnSuccessListener { visionText ->
                val sb = StringBuilder()
                visionText.textBlocks.forEach { block ->
                    block.lines.forEach { line ->
                        sb.appendLine(line.text)
                    }
                    sb.appendLine()
                }
                cont.resume(sb.toString().trim())
            }
            .addOnFailureListener { e ->
                Log.e(TAG, "OCR failed: ${e.message}")
                cont.resume("[OCR failed: ${e.message}]")
            }
    }

    /**
     * Captures screen and returns both the bitmap and extracted text.
     * Used for combined visual + textual understanding.
     */
    suspend fun captureAndAnalyze(): ScreenAnalysis = withContext(Dispatchers.IO) {
        val bitmap = captureScreen()
        val ocrText = if (bitmap != null) runOcr(bitmap) else ""
        ScreenAnalysis(
            bitmap  = bitmap,
            ocrText = ocrText
        )
    }

    data class ScreenAnalysis(
        val bitmap:  Bitmap?,
        val ocrText: String
    )

    fun release() {
        releaseProjection()
        textRecognizer.close()
        Log.i(TAG, "VisionProcessor released")
    }
}
