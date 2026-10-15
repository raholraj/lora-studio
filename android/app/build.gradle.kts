plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "app.lorastudio"
    compileSdk = 35

    defaultConfig {
        applicationId = "app.lorastudio"
        minSdk = 29
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
    }

    // Two apps from one project:
    //   full = Colab mode + "This phone" mode (needs the MediaPipe library, arm64 phones only)
    //   colab = Colab mode only ("Lite", small, no native code)
    flavorDimensions += "mode"
    productFlavors {
        create("colab") {
            dimension = "mode"
            applicationIdSuffix = ".lite"
            versionNameSuffix = "-lite"
            resValue("string", "app_name", "LoRA Studio Lite")
        }
        create("full") {
            dimension = "mode"
            resValue("string", "app_name", "LoRA Studio")
            ndk { abiFilters.add("arm64-v8a") }
        }
    }

    // The same web app that runs in the browser is packed into the APK from ../../webapp
    sourceSets {
        getByName("main") {
            assets.srcDir("../../webapp")
        }
    }

    // Fixed signing key kept in the repo so every build has the same signature and phones accept updates.
    // It is a hobby key for apps shared outside the Play Store. Do not use it for the Play Store.
    signingConfigs {
        create("release") {
            storeFile = file("lorastudio.keystore")
            storePassword = "lorastudio"
            keyAlias = "lorastudio"
            keyPassword = "lorastudio"
        }
    }
    buildTypes {
        release {
            signingConfig = signingConfigs.getByName("release")
            isMinifyEnabled = false
            isShrinkResources = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    lint {
        checkReleaseBuilds = false
        abortOnError = false
    }
    packaging {
        resources.excludes += "/META-INF/{AL2.0,LGPL2.1}"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.webkit:webkit:1.12.1")
    // On-phone image generation (Google MediaPipe). Only in the Full app.
    "fullImplementation"("com.google.mediapipe:tasks-vision-image-generator:latest.release")
}
