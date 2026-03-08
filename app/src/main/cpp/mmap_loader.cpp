#include "inference_engine.h"
#include <sys/stat.h>
#include <dirent.h>

// ============================================================
// GGUF Model Discovery & Validation Utilities
// ============================================================

extern "C" {

/**
 * Scans a directory for GGUF model files and returns their paths.
 * JNI: com.androbot.ai.ModelManager.nativeScanForModels
 */
JNIEXPORT jobjectArray JNICALL
Java_com_androbot_ai_ModelManager_nativeScanForModels(
        JNIEnv* env, jclass /* clazz */,
        jstring j_directory) {

    const char* dir_path = env->GetStringUTFChars(j_directory, nullptr);
    std::vector<std::string> model_files;

    DIR* dir = opendir(dir_path);
    if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string name(entry->d_name);
            if (name.size() > 5 &&
                name.substr(name.size() - 5) == ".gguf") {
                model_files.push_back(std::string(dir_path) + "/" + name);
            }
        }
        closedir(dir);
    } else {
        LOGW("Could not open directory: %s (errno=%d)", dir_path, errno);
    }

    env->ReleaseStringUTFChars(j_directory, dir_path);

    jobjectArray result = env->NewObjectArray(
            static_cast<jsize>(model_files.size()),
            env->FindClass("java/lang/String"),
            nullptr);

    for (size_t i = 0; i < model_files.size(); i++) {
        jstring path = env->NewStringUTF(model_files[i].c_str());
        env->SetObjectArrayElement(result, static_cast<jsize>(i), path);
        env->DeleteLocalRef(path);
    }

    LOGI("Found %zu GGUF model(s) in %s", model_files.size(),
         (dir_path ? dir_path : "null"));
    return result;
}

/**
 * Validates a GGUF file and returns metadata as JSON string.
 * JNI: com.androbot.ai.ModelManager.nativeValidateModel
 */
JNIEXPORT jstring JNICALL
Java_com_androbot_ai_ModelManager_nativeValidateModel(
        JNIEnv* env, jclass /* clazz */,
        jstring j_path) {

    const char* path = env->GetStringUTFChars(j_path, nullptr);

    // Quick validation: open and read GGUF header
    int fd = open(path, O_RDONLY);
    std::string json = "{\"valid\": false, \"error\": \"unknown\"}";

    if (fd >= 0) {
        uint32_t magic = 0;
        uint32_t version = 0;
        uint64_t n_tensors = 0;

        if (read(fd, &magic, 4) == 4 &&
            read(fd, &version, 4) == 4 &&
            read(fd, &n_tensors, 8) == 8) {

            if (magic == GGUF_MAGIC) {
                struct stat st;
                fstat(fd, &st);
                char buf[512];
                snprintf(buf, sizeof(buf),
                         "{\"valid\": true, \"version\": %u, "
                         "\"n_tensors\": %llu, "
                         "\"size_mb\": %.1f}",
                         version,
                         (unsigned long long)n_tensors,
                         (double)st.st_size / (1024.0 * 1024.0));
                json = buf;
                LOGI("Valid GGUF: %s (v%u, %llu tensors, %.1fMB)",
                     path, version, (unsigned long long)n_tensors,
                     (double)st.st_size / (1024.0 * 1024.0));
            } else {
                char buf[128];
                snprintf(buf, sizeof(buf),
                         "{\"valid\": false, \"error\": \"bad magic: 0x%08X\"}",
                         magic);
                json = buf;
                LOGW("Invalid GGUF magic in file: %s", path);
            }
        }
        close(fd);
    } else {
        json = "{\"valid\": false, \"error\": \"cannot open file\"}";
        LOGE("Cannot open file for validation: %s", path);
    }

    env->ReleaseStringUTFChars(j_path, path);
    return env->NewStringUTF(json.c_str());
}

/**
 * Returns available free RAM in bytes.
 * JNI: com.androbot.ai.ModelManager.nativeGetFreeRamBytes
 */
JNIEXPORT jlong JNICALL
Java_com_androbot_ai_ModelManager_nativeGetFreeRamBytes(
        JNIEnv* /* env */, jclass /* clazz */) {
    // Read /proc/meminfo
    FILE* f = fopen("/proc/meminfo", "r");
    long long free_kb = 0, available_kb = 0;

    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (sscanf(line, "MemAvailable: %lld kB", &available_kb) == 1) break;
            if (sscanf(line, "MemFree: %lld kB", &free_kb) == 1) {}
        }
        fclose(f);
    }

    long long result = available_kb > 0 ? available_kb : free_kb;
    LOGI("Free/available RAM: %lld KB (%.1f MB)", result, (double)result / 1024.0);
    return static_cast<jlong>(result * 1024LL);
}

} // extern "C"
