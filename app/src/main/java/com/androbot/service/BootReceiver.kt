package com.androbot.service

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.Build
import android.util.Log

/**
 * Boot Receiver - auto-starts the AI service after device boot.
 */
class BootReceiver : BroadcastReceiver() {

    companion object {
        private const val TAG = "BootReceiver"
    }

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == Intent.ACTION_BOOT_COMPLETED ||
            intent.action == Intent.ACTION_MY_PACKAGE_REPLACED) {

            Log.i(TAG, "Boot/update received - starting AI service")

            val serviceIntent = Intent(context, AIBackgroundService::class.java).apply {
                action = AIBackgroundService.ACTION_START
            }

            context.startForegroundService(serviceIntent)
        }
    }
}
