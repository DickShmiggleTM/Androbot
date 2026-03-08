# Keep JNI classes
-keep class com.androbot.ai.** { *; }
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep Room entities
-keep class com.androbot.memory.** { *; }

# Keep Gson models
-keepattributes Signature
-keepattributes *Annotation*
-dontwarn sun.misc.**
-keep class com.google.gson.** { *; }

# Keep ObjectBox
-keep class io.objectbox.** { *; }
-dontwarn io.objectbox.**

# Keep WorkManager
-keep class androidx.work.** { *; }
