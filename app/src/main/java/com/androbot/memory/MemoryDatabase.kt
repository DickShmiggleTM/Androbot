package com.androbot.memory

import androidx.room.*
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase
import kotlinx.coroutines.flow.Flow

// ============================================================
// Room Entities
// ============================================================

@Entity(
    tableName = "memories",
    indices = [
        Index("memory_type"),
        Index("created_at"),
        Index("importance_score")
    ]
)
data class MemoryEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,

    @ColumnInfo(name = "content")
    val content: String,

    @ColumnInfo(name = "memory_type")
    val memoryType: String,  // "interaction", "observation", "fact", "screen", "document"

    @ColumnInfo(name = "embedding_blob")
    val embeddingBlob: ByteArray,  // float[] stored as raw bytes for fast retrieval

    @ColumnInfo(name = "embedding_norm")
    val embeddingNorm: Float = 1.0f,

    @ColumnInfo(name = "importance_score")
    val importanceScore: Float = 0.5f,

    @ColumnInfo(name = "access_count")
    val accessCount: Int = 0,

    @ColumnInfo(name = "created_at")
    val createdAt: Long = System.currentTimeMillis(),

    @ColumnInfo(name = "last_accessed")
    val lastAccessed: Long = System.currentTimeMillis(),

    @ColumnInfo(name = "metadata_json")
    val metadataJson: String = "{}"
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is MemoryEntity) return false
        return id == other.id
    }
    override fun hashCode(): Int = id.hashCode()
}

@Entity(
    tableName = "knowledge_graph_edges",
    foreignKeys = [
        ForeignKey(entity = MemoryEntity::class, parentColumns = ["id"],
                   childColumns = ["source_id"], onDelete = ForeignKey.CASCADE),
        ForeignKey(entity = MemoryEntity::class, parentColumns = ["id"],
                   childColumns = ["target_id"], onDelete = ForeignKey.CASCADE)
    ],
    indices = [Index("source_id"), Index("target_id")]
)
data class KnowledgeEdge(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,

    @ColumnInfo(name = "source_id") val sourceId: Long,
    @ColumnInfo(name = "target_id") val targetId: Long,
    @ColumnInfo(name = "relation")  val relation: String,   // "related_to", "caused_by", "learned_from"
    @ColumnInfo(name = "weight")    val weight: Float = 1.0f,
    @ColumnInfo(name = "created_at") val createdAt: Long = System.currentTimeMillis()
)

@Entity(tableName = "lora_checkpoints")
data class LoraCheckpoint(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,

    @ColumnInfo(name = "checkpoint_path") val checkpointPath: String,
    @ColumnInfo(name = "base_model_name") val baseModelName: String,
    @ColumnInfo(name = "training_steps")  val trainingSteps: Int,
    @ColumnInfo(name = "loss")            val loss: Float,
    @ColumnInfo(name = "created_at")      val createdAt: Long = System.currentTimeMillis(),
    @ColumnInfo(name = "is_active")       val isActive: Boolean = false
)

// ============================================================
// DAOs
// ============================================================

@Dao
interface MemoryDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(memory: MemoryEntity): Long

    @Update
    suspend fun update(memory: MemoryEntity)

    @Delete
    suspend fun delete(memory: MemoryEntity)

    @Query("SELECT * FROM memories ORDER BY importance_score DESC, created_at DESC LIMIT :limit")
    suspend fun getTopMemories(limit: Int = 100): List<MemoryEntity>

    @Query("SELECT * FROM memories WHERE memory_type = :type ORDER BY created_at DESC LIMIT :limit")
    suspend fun getMemoriesByType(type: String, limit: Int = 50): List<MemoryEntity>

    @Query("SELECT COUNT(*) FROM memories")
    suspend fun getCount(): Int

    @Query("SELECT * FROM memories ORDER BY created_at DESC LIMIT :limit")
    fun observeRecentMemories(limit: Int = 20): Flow<List<MemoryEntity>>

    // Retrieve all embeddings for vector similarity search
    // (done in Kotlin since SQLite doesn't support cosine similarity natively)
    @Query("SELECT id, embedding_blob, embedding_norm, content, memory_type, importance_score FROM memories ORDER BY importance_score DESC LIMIT :candidatePool")
    suspend fun getEmbeddingsForSearch(candidatePool: Int = 500): List<MemorySearchRow>

    @Query("UPDATE memories SET access_count = access_count + 1, last_accessed = :timestamp WHERE id = :id")
    suspend fun incrementAccessCount(id: Long, timestamp: Long = System.currentTimeMillis())

    @Query("DELETE FROM memories WHERE created_at < :cutoffMs AND importance_score < :scoreThreshold")
    suspend fun pruneOldLowImportance(cutoffMs: Long, scoreThreshold: Float): Int

    @Query("SELECT * FROM memories WHERE id IN (:ids)")
    suspend fun getByIds(ids: List<Long>): List<MemoryEntity>
}

@DatabaseView(
    "SELECT id, embedding_blob, embedding_norm, content, memory_type, importance_score FROM memories"
)
data class MemorySearchRow(
    val id: Long,
    @ColumnInfo(name = "embedding_blob") val embeddingBlob: ByteArray,
    @ColumnInfo(name = "embedding_norm") val embeddingNorm: Float,
    val content: String,
    @ColumnInfo(name = "memory_type") val memoryType: String,
    @ColumnInfo(name = "importance_score") val importanceScore: Float
)

@Dao
interface KnowledgeGraphDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertEdge(edge: KnowledgeEdge): Long

    @Query("SELECT * FROM knowledge_graph_edges WHERE source_id = :nodeId OR target_id = :nodeId")
    suspend fun getEdgesForNode(nodeId: Long): List<KnowledgeEdge>
}

@Dao
interface LoraDao {
    @Insert
    suspend fun insert(checkpoint: LoraCheckpoint): Long

    @Query("SELECT * FROM lora_checkpoints WHERE is_active = 1 ORDER BY created_at DESC LIMIT 1")
    suspend fun getActiveCheckpoint(): LoraCheckpoint?

    @Query("UPDATE lora_checkpoints SET is_active = 0")
    suspend fun deactivateAll()

    @Query("UPDATE lora_checkpoints SET is_active = 1 WHERE id = :id")
    suspend fun activate(id: Long)
}

// ============================================================
// Database
// ============================================================

@Database(
    entities = [MemoryEntity::class, KnowledgeEdge::class, LoraCheckpoint::class],
    views = [MemorySearchRow::class],
    version = 1,
    exportSchema = false
)
abstract class MemoryDatabase : RoomDatabase() {
    abstract fun memoryDao(): MemoryDao
    abstract fun knowledgeGraphDao(): KnowledgeGraphDao
    abstract fun loraDao(): LoraDao

    companion object {
        @Volatile
        private var INSTANCE: MemoryDatabase? = null

        fun getInstance(context: android.content.Context): MemoryDatabase {
            return INSTANCE ?: synchronized(this) {
                Room.databaseBuilder(
                    context.applicationContext,
                    MemoryDatabase::class.java,
                    "androbot_memory.db"
                )
                .setJournalMode(JournalMode.WAL)  // WAL mode for concurrent access
                .fallbackToDestructiveMigration()
                .build()
                .also { INSTANCE = it }
            }
        }
    }
}
