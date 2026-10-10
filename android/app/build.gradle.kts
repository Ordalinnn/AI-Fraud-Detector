plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "kz.aifrauddetector"
    compileSdk = 35

    defaultConfig {
        applicationId = "kz.aifrauddetector"
        minSdk = 26
        targetSdk = 35
        versionCode = (System.getenv("GITHUB_RUN_NUMBER") ?: "1").toInt()
        versionName = "0.1.${System.getenv("GITHUB_RUN_NUMBER") ?: "0"}"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    // The model is plain text; keep it uncompressed-agnostic and readable.
    androidResources {
        noCompress += "tsv"
    }
    testOptions {
        unitTests.isReturnDefaultValues = true
        unitTests.isIncludeAndroidResources = true // Robolectric: real resources/assets
    }
}

dependencies {
    // No AndroidX / third-party runtime dependencies on purpose: the app is
    // small, auditable, and has no network library at all.
    testImplementation("junit:junit:4.13.2")
    // Test-only: runs the notification listener and the main screen on the
    // JVM with a simulated Android, so CI checks the whole flow end to end.
    testImplementation("org.robolectric:robolectric:4.14.1")
}
