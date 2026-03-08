#include "inference_engine.h"
#include <jni.h>
#include <string>
#include <vector>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

// ============================================================
// JNI bridge between Kotlin and C++ inference engine
// Class: com.androbot.ai.InferenceEngine
// ============================================================

extern "C" {

// ----- Model Loading -----

JNIEXPORT jboolean JNICALL
Java_com_androbot_ai_InferenceEngine_nativeLoadModel(
        JNIEnv* env, jobject /* obj */,
        jstring model_path,
        jint    n_ctx,
        jint    n_threads) {

    const char* path = env->GetStringUTFChars(model_path, nullptr);
    bool result = getGlobalEngine().loadModel(
            std::string(path),
            static_cast<int>(n_ctx),
            static_cast<int>(n_threads));
    env->ReleaseStringUTFChars(model_path, path);
    return static_cast<jboolean>(result);
}

JNIEXPORT void JNICALL
Java_com_androbot_ai_InferenceEngine_nativeUnloadModel(
        JNIEnv* /* env */, jobject /* obj */) {
    getGlobalEngine().unloadModel();
}

JNIEXPORT jboolean JNICALL
Java_com_androbot_ai_InferenceEngine_nativeIsModelLoaded(
        JNIEnv* /* env */, jobject /* obj */) {
    return static_cast<jboolean>(getGlobalEngine().isModelLoaded());
}

// ----- Token Generation (streaming) -----

JNIEXPORT void JNICALL
Java_com_androbot_ai_InferenceEngine_nativeGenerate(
        JNIEnv* env, jobject obj,
        jstring j_prompt,
        jint    max_new_tokens,
        jfloat  temperature,
        jfloat  top_p,
        jint    top_k) {

    const char* prompt = env->GetStringUTFChars(j_prompt, nullptr);
    std::string prompt_str(prompt);
    env->ReleaseStringUTFChars(j_prompt, prompt);

    // Get callback method reference
    jclass   clazz      = env->GetObjectClass(obj);
    jmethodID on_token  = env->GetMethodID(clazz, "onTokenGenerated",
                                            "(Ljava/lang/String;Z)V");
    if (!on_token) {
        LOGE("Could not find onTokenGenerated callback method");
        return;
    }

    // JavaVM needed to attach thread if engine spawns worker threads
    JavaVM* jvm = nullptr;
    env->GetJavaVM(&jvm);
    jobject obj_global = env->NewGlobalRef(obj);

    getGlobalEngine().generate(
        prompt_str,
        [env, obj_global, on_token](const std::string& token, bool is_done) {
            jstring j_token = env->NewStringUTF(token.c_str());
            env->CallVoidMethod(obj_global, on_token, j_token,
                                static_cast<jboolean>(is_done));
            env->DeleteLocalRef(j_token);
        },
        static_cast<int>(max_new_tokens),
        static_cast<float>(temperature),
        static_cast<float>(top_p),
        static_cast<int>(top_k)
    );

    env->DeleteGlobalRef(obj_global);
}

JNIEXPORT void JNICALL
Java_com_androbot_ai_InferenceEngine_nativeStopGeneration(
        JNIEnv* /* env */, jobject /* obj */) {
    getGlobalEngine().stopGeneration();
}

// ----- Embedding -----

JNIEXPORT jfloatArray JNICALL
Java_com_androbot_ai_InferenceEngine_nativeEmbed(
        JNIEnv* env, jobject /* obj */,
        jstring j_text) {

    const char* text = env->GetStringUTFChars(j_text, nullptr);
    std::vector<float> embedding = getGlobalEngine().embed(std::string(text));
    env->ReleaseStringUTFChars(j_text, text);

    jfloatArray result = env->NewFloatArray(static_cast<jsize>(embedding.size()));
    if (result) {
        env->SetFloatArrayRegion(result, 0,
                                  static_cast<jsize>(embedding.size()),
                                  embedding.data());
    }
    return result;
}

// ----- Memory Diagnostics -----

JNIEXPORT jlong JNICALL
Java_com_androbot_ai_InferenceEngine_nativeGetKvCacheBytes(
        JNIEnv* /* env */, jobject /* obj */) {
    return static_cast<jlong>(getGlobalEngine().getKVCacheRAMBytes());
}

JNIEXPORT jlong JNICALL
Java_com_androbot_ai_InferenceEngine_nativeGetMmapBytes(
        JNIEnv* /* env */, jobject /* obj */) {
    return static_cast<jlong>(getGlobalEngine().getModelMmapBytes());
}

JNIEXPORT jint JNICALL
Java_com_androbot_ai_InferenceEngine_nativeGetContextLength(
        JNIEnv* /* env */, jobject /* obj */) {
    return static_cast<jint>(getGlobalEngine().getContextLength());
}

JNIEXPORT jboolean JNICALL
Java_com_androbot_ai_InferenceEngine_nativeNeedsContextEviction(
        JNIEnv* /* env */, jobject /* obj */) {
    return static_cast<jboolean>(getGlobalEngine().needsContextEviction());
}

JNIEXPORT void JNICALL
Java_com_androbot_ai_InferenceEngine_nativeClearContext(
        JNIEnv* /* env */, jobject /* obj */) {
    getGlobalEngine().clearContext();
}

// ----- Model Config -----

JNIEXPORT jint JNICALL
Java_com_androbot_ai_InferenceEngine_nativeGetVocabSize(
        JNIEnv* /* env */, jobject /* obj */) {
    return static_cast<jint>(getGlobalEngine().config.n_vocab);
}

JNIEXPORT jint JNICALL
Java_com_androbot_ai_InferenceEngine_nativeGetEmbedDim(
        JNIEnv* /* env */, jobject /* obj */) {
    return static_cast<jint>(getGlobalEngine().config.n_embd);
}

JNIEXPORT jstring JNICALL
Java_com_androbot_ai_InferenceEngine_nativeGetArchitecture(
        JNIEnv* env, jobject /* obj */) {
    return env->NewStringUTF(getGlobalEngine().config.arch.c_str());
}

} // extern "C"
