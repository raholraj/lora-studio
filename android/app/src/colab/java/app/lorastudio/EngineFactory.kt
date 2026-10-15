package app.lorastudio

import android.content.Context
import android.graphics.Bitmap
import java.io.File

/** Lite app: no on-phone generation. */
object EngineFactory {
    @Suppress("UNUSED_PARAMETER")
    fun create(context: Context): Generator = object : Generator {
        override val supported: Boolean = false

        override fun generate(
            modelDir: File,
            prompt: String,
            iterations: Int,
            seed: Int,
            onLoading: () -> Unit,
            onProgress: (Float) -> Unit,
            isCancelled: () -> Boolean
        ): Bitmap = throw UnsupportedOperationException("This app cannot make photos on the phone.")

        override fun close() {}
    }
}
