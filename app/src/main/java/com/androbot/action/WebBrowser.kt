package com.androbot.action

import android.content.Context
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.webkit.*
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import okhttp3.OkHttpClient
import okhttp3.Request
import java.util.concurrent.TimeUnit
import java.util.regex.Pattern

/**
 * Headless Web Browser
 *
 * Provides the AI agent with web browsing capability:
 * - Search queries via DuckDuckGo (privacy-first)
 * - Page fetching and HTML → plain text extraction
 * - JavaScript execution in a headless WebView
 * - Content length caps to fit within context window
 *
 * Two implementations:
 * 1. OkHttp-based (fast, for simple pages)
 * 2. WebView-based (for JS-heavy pages requiring rendering)
 */
class WebBrowser(private val context: Context) {

    companion object {
        private const val TAG          = "WebBrowser"
        private const val MAX_CONTENT  = 8_000   // chars to return to AI
        private const val TIMEOUT_SECS = 15L
        private val SEARCH_URL = "https://html.duckduckgo.com/html/?q="

        private val HTTP_CLIENT = OkHttpClient.Builder()
            .connectTimeout(TIMEOUT_SECS, TimeUnit.SECONDS)
            .readTimeout(TIMEOUT_SECS, TimeUnit.SECONDS)
            .followRedirects(true)
            .build()

        // User agent mimicking a mobile browser
        private const val USER_AGENT =
            "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 " +
            "(KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36"
    }

    // ── Web Search ────────────────────────────────────────────────────────────

    /**
     * Performs a web search and returns extracted text results.
     * Uses DuckDuckGo HTML endpoint (no API key required).
     */
    suspend fun search(query: String): String = withContext(Dispatchers.IO) {
        Log.i(TAG, "Web search: $query")

        val encodedQuery = Uri.encode(query)
        val url = "$SEARCH_URL$encodedQuery"

        try {
            val html = fetchUrl(url)
            val text = extractSearchResults(html)
            Log.d(TAG, "Search returned ${text.length} chars")
            text.take(MAX_CONTENT)
        } catch (e: Exception) {
            Log.e(TAG, "Search failed: ${e.message}")
            "[Search failed: ${e.message}]"
        }
    }

    /**
     * Fetches a URL and returns its readable text content.
     */
    suspend fun fetchPage(url: String): String = withContext(Dispatchers.IO) {
        Log.i(TAG, "Fetching page: $url")

        try {
            val html = fetchUrl(url)
            val text = htmlToText(html)
            Log.d(TAG, "Page content: ${text.length} chars")
            text.take(MAX_CONTENT)
        } catch (e: Exception) {
            Log.e(TAG, "Page fetch failed: ${e.message}")
            "[Fetch failed: ${e.message}]"
        }
    }

    // ── HTTP Fetching ─────────────────────────────────────────────────────────

    private fun fetchUrl(url: String): String {
        val request = Request.Builder()
            .url(url)
            .header("User-Agent", USER_AGENT)
            .header("Accept", "text/html,application/xhtml+xml")
            .header("Accept-Language", "en-US,en;q=0.9")
            .build()

        HTTP_CLIENT.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw Exception("HTTP ${response.code}: ${response.message}")
            }
            return response.body?.string() ?: ""
        }
    }

    // ── HTML Parsing ──────────────────────────────────────────────────────────

    /**
     * Converts HTML to readable plain text.
     * Strips tags, scripts, styles while preserving structure.
     */
    fun htmlToText(html: String): String {
        var text = html

        // Remove script and style blocks entirely
        text = text.replace(Regex("<script[^>]*>.*?</script>",
            setOf(RegexOption.DOT_MATCHES_ALL, RegexOption.IGNORE_CASE)), "")
        text = text.replace(Regex("<style[^>]*>.*?</style>",
            setOf(RegexOption.DOT_MATCHES_ALL, RegexOption.IGNORE_CASE)), "")

        // Replace structural tags with newlines
        text = text.replace(Regex("<br\\s*/?>",   RegexOption.IGNORE_CASE), "\n")
        text = text.replace(Regex("<p[^>]*>",     RegexOption.IGNORE_CASE), "\n")
        text = text.replace(Regex("</p>",         RegexOption.IGNORE_CASE), "\n")
        text = text.replace(Regex("<div[^>]*>",   RegexOption.IGNORE_CASE), "\n")
        text = text.replace(Regex("<h[1-6][^>]*>",RegexOption.IGNORE_CASE), "\n## ")
        text = text.replace(Regex("</h[1-6]>",   RegexOption.IGNORE_CASE), "\n")
        text = text.replace(Regex("<li[^>]*>",    RegexOption.IGNORE_CASE), "\n• ")
        text = text.replace(Regex("<tr[^>]*>",    RegexOption.IGNORE_CASE), "\n")
        text = text.replace(Regex("<td[^>]*>",    RegexOption.IGNORE_CASE), " | ")

        // Remove all remaining HTML tags
        text = text.replace(Regex("<[^>]+>"), "")

        // Decode common HTML entities
        text = text
            .replace("&amp;",   "&")
            .replace("&lt;",    "<")
            .replace("&gt;",    ">")
            .replace("&quot;",  "\"")
            .replace("&apos;",  "'")
            .replace("&nbsp;",  " ")
            .replace("&#39;",   "'")
            .replace("&#160;",  " ")

        // Collapse whitespace
        text = text.replace(Regex("[ \t]+"), " ")
        text = text.replace(Regex("\n{3,}"), "\n\n")

        return text.trim()
    }

    /**
     * Extracts DuckDuckGo search result snippets from HTML.
     */
    private fun extractSearchResults(html: String): String {
        val sb = StringBuilder()

        // Extract result titles and snippets
        val resultPattern = Pattern.compile(
            "<a[^>]+class=\"result__a\"[^>]*>(.*?)</a>.*?" +
            "<a[^>]+class=\"result__snippet\"[^>]*>(.*?)</a>",
            Pattern.DOTALL or Pattern.CASE_INSENSITIVE
        )

        val matcher = resultPattern.matcher(html)
        var count = 0

        while (matcher.find() && count < 5) {
            val title   = htmlToText(matcher.group(1) ?: "")
            val snippet = htmlToText(matcher.group(2) ?: "")
            if (title.isNotBlank()) {
                sb.appendLine("**$title**")
                if (snippet.isNotBlank()) sb.appendLine(snippet)
                sb.appendLine()
                count++
            }
        }

        // Fallback: extract all text if pattern didn't match
        if (sb.isEmpty()) {
            return htmlToText(html).take(MAX_CONTENT)
        }

        return sb.toString()
    }

    // ── WebView-based Rendering (for JS-heavy pages) ──────────────────────────

    /**
     * Fetches a page using a headless WebView with JavaScript enabled.
     * Runs on the main thread as required by WebView.
     * Returns DOM text content via JavaScript injection.
     */
    suspend fun fetchWithWebView(url: String): String {
        return withContext(Dispatchers.Main) {
            suspendCancellableCoroutine { cont ->
                val webView = WebView(context).apply {
                    settings.javaScriptEnabled = true
                    settings.domStorageEnabled  = true
                    settings.userAgentString    = USER_AGENT
                    settings.loadWithOverviewMode = true
                }

                webView.webViewClient = object : WebViewClient() {
                    override fun onPageFinished(view: WebView, url: String) {
                        // Inject JS to extract page text
                        view.evaluateJavascript(
                            "(function() { return document.body.innerText || document.body.textContent; })()"
                        ) { result ->
                            val text = result
                                .removePrefix("\"")
                                .removeSuffix("\"")
                                .replace("\\n", "\n")
                                .replace("\\\"", "\"")
                                .take(MAX_CONTENT)

                            webView.destroy()
                            if (!cont.isCompleted) cont.resume(text)
                        }
                    }

                    override fun onReceivedError(
                        view: WebView, request: WebResourceRequest, error: WebResourceError
                    ) {
                        webView.destroy()
                        if (!cont.isCompleted) {
                            cont.resume("[WebView error: ${error.description}]")
                        }
                    }
                }

                webView.loadUrl(url)

                // Timeout after 20 seconds
                Handler(Looper.getMainLooper()).postDelayed({
                    if (!cont.isCompleted) {
                        webView.destroy()
                        cont.resume("[WebView timeout after 20s]")
                    }
                }, 20_000)

                cont.invokeOnCancellation { webView.destroy() }
            }
        }
    }
}
