package app.lorastudio

import android.graphics.Bitmap
import java.io.File

class CancelledException : RuntimeException("Cancelled")

/** Makes one image on the phone. The Full app has a MediaPipe version, the Lite app has an empty one. */
interface Generator {
    val supported: Boolean

    fun generate(
        modelDir: File,
        prompt: String,
        iterations: Int,
        seed: Int,
        onLoading: () -> Unit,
        onProgress: (Float) -> Unit,
        isCancelled: () -> Boolean
    ): Bitmap

    fun close()
}
