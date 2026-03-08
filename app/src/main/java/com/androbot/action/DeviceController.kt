package com.androbot.action

import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.util.Log
import com.androbot.perception.AccessibilityAgentService

/**
 * Device Controller
 *
 * Routes function-calling actions to the AccessibilityService
 * for physical device control:
 * - Click UI elements by text or view ID
 * - Type text into focused fields
 * - Scroll in any direction
 * - Open applications by package name
 * - Navigate back/home/recents
 */
class DeviceController(private val context: Context) {

    companion object {
        private const val TAG = "DeviceController"
    }

    private val accessibilityService: AccessibilityAgentService?
        get() = AccessibilityAgentService.getInstance()

    val isAccessibilityEnabled: Boolean
        get() = accessibilityService != null

    // ── UI Actions ────────────────────────────────────────────────────────────

    fun clickElement(text: String): Boolean {
        val service = accessibilityService ?: run {
            Log.w(TAG, "Accessibility service not available")
            return false
        }
        return service.clickElementByText(text)
    }

    fun typeText(text: String): Boolean {
        val service = accessibilityService ?: run {
            Log.w(TAG, "Accessibility service not available")
            return false
        }
        return service.typeText(text)
    }

    fun scroll(direction: String): Boolean {
        val service = accessibilityService ?: return false
        return service.scroll(direction)
    }

    fun readScreen(): String {
        val service = accessibilityService ?: return "[Accessibility service not enabled]"
        return service.captureFullUIText()
    }

    fun getClickableElements(): List<String> {
        return accessibilityService?.getClickableElements() ?: emptyList()
    }

    fun getCurrentApp(): String {
        return accessibilityService?.getCurrentPackageName() ?: "unknown"
    }

    // ── App Launching ─────────────────────────────────────────────────────────

    fun openApp(packageName: String): Boolean {
        return try {
            val pm = context.packageManager
            val launchIntent = pm.getLaunchIntentForPackage(packageName)

            if (launchIntent != null) {
                launchIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                context.startActivity(launchIntent)
                Log.i(TAG, "Opened app: $packageName")
                true
            } else {
                Log.w(TAG, "No launch intent for package: $packageName")
                false
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to open app $packageName: ${e.message}")
            false
        }
    }

    fun listInstalledApps(): List<AppInfo> {
        return try {
            val pm = context.packageManager
            val flags = PackageManager.GET_META_DATA
            pm.getInstalledApplications(flags)
                .filter { pm.getLaunchIntentForPackage(it.packageName) != null }
                .map { app ->
                    AppInfo(
                        packageName = app.packageName,
                        label       = pm.getApplicationLabel(app).toString()
                    )
                }
                .sortedBy { it.label }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to list apps: ${e.message}")
            emptyList()
        }
    }

    // ── Navigation ────────────────────────────────────────────────────────────

    fun pressBack(): Boolean {
        return accessibilityService?.performGlobalAction(
            AccessibilityAgentService.GLOBAL_ACTION_BACK
        ) ?: false
    }

    fun pressHome(): Boolean {
        return accessibilityService?.performGlobalAction(
            AccessibilityAgentService.GLOBAL_ACTION_HOME
        ) ?: false
    }

    fun pressRecents(): Boolean {
        return accessibilityService?.performGlobalAction(
            AccessibilityAgentService.GLOBAL_ACTION_RECENTS
        ) ?: false
    }

    fun pressNotifications(): Boolean {
        return accessibilityService?.performGlobalAction(
            AccessibilityAgentService.GLOBAL_ACTION_NOTIFICATIONS
        ) ?: false
    }

    data class AppInfo(val packageName: String, val label: String)
}
