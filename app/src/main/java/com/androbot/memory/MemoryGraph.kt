package com.androbot.memory

import android.content.Context
import android.util.Log
import com.androbot.ai.InferenceEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

/**
 * Semantic Memory Graph - the AI's long-term persistent memory.
 *
 * Architecture:
 * - Vector store: Room SQLite with binary blob embeddings
 * - Similarity search: cosine similarity computed in Kotlin (BLAS-like)
 * - Knowledge graph: edges between related memory nodes
 * - Importance scoring: TF-IDF-like recency + access frequency decay
 *
 * Memory Types:
 * - "interaction" : user conversation turns
 * - "observation" : screen reads from AccessibilityService
 * - "fact"        : extracted knowledge facts
 * - "document"    : OCR/PDF extracted text chunks
 * - "screen"      : MediaProjection screen summaries
 */
class MemoryGraph(
    private val context: Context,
    private val inferenceEngine: InferenceEngine
) {
    companion object {
        private const val TAG = "MemoryGraph"
        private const val EMBED_DIM = 2048         // Llama-3.2-1B hidden dim
        private const val TOP_K_RESULTS = 5
        private const val MAX_MEMORIES = 10_000    // Prune beyond this
        private const val LOW_IMPORTANCE_PRUNE_DAYS = 30L
    }

    private val db: MemoryDatabase by lazy { MemoryDatabase.getInstance(context) }
    private val memoryDao:   MemoryDao         get() = db.memoryDao()
    private val graphDao:    KnowledgeGraphDao get() = db.knowledgeGraphDao()

    // ── Embedding utilities ────────────────────────────────────────────────────

    private fun floatsToBytes(floats: FloatArray): ByteArray {
        val buffer = ByteBuffer.allocate(floats.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        floats.forEach { buffer.putFloat(it) }
        return buffer.array()
    }

    private fun bytesToFloats(bytes: ByteArray): FloatArray {
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(bytes.size / 4) { buffer.getFloat() }
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        if (a.size != b.size) return 0f
        var dot = 0f
        var normA = 0f
        var normB = 0f
        for (i in a.indices) {
            dot   += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        val denom = sqrt(normA) * sqrt(normB)
        return if (denom < 1e-9f) 0f else dot / denom
    }

    private fun computeNorm(embedding: FloatArray): Float {
        var sum = 0f
        for (v in embedding) sum += v * v
        return sqrt(sum)
    }

    // ── Store ─────────────────────────────────────────────────────────────────

    /**
     * Stores a new memory. Computes embedding and persists to vector DB.
     */
    suspend fun remember(
        content: String,
        memoryType: String = "interaction",
        importanceScore: Float = 0.5f,
        metadata: Map<String, String> = emptyMap()
    ): Long = withContext(Dispatchers.IO) {

        // Compute embedding using the inference engine
        val embedding = inferenceEngine.embed(content)
        val norm = computeNorm(embedding)

        val metaJson = metadata.entries.joinToString(",", "{", "}") {
            "\"${it.key}\": \"${it.value}\""
        }

        val entity = MemoryEntity(
            content         = content.take(4096),  // Cap at 4K chars
            memoryType      = memoryType,
            embeddingBlob   = floatsToBytes(embedding),
            embeddingNorm   = norm,
            importanceScore = importanceScore,
            metadataJson    = metaJson
        )

        val id = memoryDao.insert(entity)
        Log.d(TAG, "Stored memory #$id (type=$memoryType, score=$importanceScore): ${content.take(80)}...")

        // Auto-prune if over limit
        val count = memoryDao.getCount()
        if (count > MAX_MEMORIES) {
            pruneOldMemories()
        }

        id
    }

    /**
     * Recalls the top-K semantically similar memories for a query.
     */
    suspend fun recall(
        query: String,
        topK: Int = TOP_K_RESULTS,
        memoryType: String? = null
    ): List<RecallResult> = withContext(Dispatchers.IO) {

        val queryEmbedding = inferenceEngine.embed(query)
        val candidates = memoryDao.getEmbeddingsForSearch(500)

        val scored = candidates
            .filter { memoryType == null || it.memoryType == memoryType }
            .mapNotNull { row ->
                try {
                    val candidateEmbed = bytesToFloats(row.embeddingBlob)
                    val similarity = cosineSimilarity(queryEmbedding, candidateEmbed)
                    ScoredMemory(row, similarity)
                } catch (e: Exception) {
                    null
                }
            }
            .sortedByDescending { it.similarity }
            .take(topK)

        // Update access counts for retrieved memories
        scored.forEach { sm ->
            memoryDao.incrementAccessCount(sm.row.id)
        }

        Log.d(TAG, "Recall for '${query.take(50)}': found ${scored.size} results")

        scored.map { sm ->
            RecallResult(
                id         = sm.row.id,
                content    = sm.row.content,
                memoryType = sm.row.memoryType,
                similarity = sm.similarity,
                importance = sm.row.importanceScore
            )
        }
    }

    /**
     * Links two memories with a semantic edge in the knowledge graph.
     */
    suspend fun linkMemories(
        sourceId: Long,
        targetId: Long,
        relation: String,
        weight: Float = 1.0f
    ) = withContext(Dispatchers.IO) {
        val edge = KnowledgeEdge(
            sourceId = sourceId,
            targetId = targetId,
            relation = relation,
            weight   = weight
        )
        graphDao.insertEdge(edge)
        Log.d(TAG, "Linked memory #$sourceId --[$relation]--> #$targetId")
    }

    /**
     * Returns the most recent memories as context snippets.
     */
    suspend fun getRecentContext(limit: Int = 10): String = withContext(Dispatchers.IO) {
        val memories = memoryDao.getTopMemories(limit)
        if (memories.isEmpty()) return@withContext ""

        memories.joinToString("\n") { m ->
            "[${m.memoryType.uppercase()}] ${m.content.take(200)}"
        }
    }

    /**
     * Prunes old, low-importance memories to keep DB size manageable.
     */
    suspend fun pruneOldMemories() = withContext(Dispatchers.IO) {
        val cutoffMs = System.currentTimeMillis() - LOW_IMPORTANCE_PRUNE_DAYS * 86_400_000L
        val deleted = memoryDao.pruneOldLowImportance(cutoffMs, scoreThreshold = 0.3f)
        Log.i(TAG, "Pruned $deleted old low-importance memories")
    }

    /**
     * Extracts and stores key facts from a block of text.
     */
    suspend fun extractAndStoreFacts(text: String, source: String = "observation") {
        // Split text into sentences and store each as a separate memory
        val sentences = text.split(Regex("[.!?]")).filter { it.trim().length > 20 }
        sentences.take(10).forEach { sentence ->
            remember(
                content       = sentence.trim(),
                memoryType    = "fact",
                importanceScore = 0.6f,
                metadata      = mapOf("source" to source)
            )
        }
    }

    suspend fun getMemoryCount(): Int = withContext(Dispatchers.IO) {
        memoryDao.getCount()
    }

    // ── Internal data classes ─────────────────────────────────────────────────

    private data class ScoredMemory(val row: MemorySearchRow, val similarity: Float)

    data class RecallResult(
        val id: Long,
        val content: String,
        val memoryType: String,
        val similarity: Float,
        val importance: Float
    ) {
        override fun toString(): String =
            "[${(similarity * 100).toInt()}%] $content"
    }
}
