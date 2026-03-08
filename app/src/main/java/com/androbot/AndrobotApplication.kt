package com.androbot

import android.app.Application
import android.content.Intent
import android.util.Log
import com.androbot.service.AIBackgroundService

class AndrobotApplication : Application() {

    companion object {
        private const val TAG = "AndrobotApp"

        @Volatile
        lateinit var instance: AndrobotApplication
            private set
    }

    override fun onCreate() {
        super.onCreate()
        instance = this
        Log.i(TAG, "Androbot application started")

        // Auto-start the AI background service
        startAIService()
    }

    private fun startAIService() {
        try {
            val intent = Intent(this, AIBackgroundService::class.java).apply {
                action = AIBackgroundService.ACTION_START
            }
            startForegroundService(intent)
            Log.i(TAG, "AI Background Service started")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start AI service: ${e.message}")
        }
    }
}
