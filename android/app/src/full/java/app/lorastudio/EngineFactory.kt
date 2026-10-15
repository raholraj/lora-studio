package app.lorastudio

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapExtractor
import com.google.mediapipe.tasks.vision.imagegenerator.ImageGenerator
import java.io.File

/** Full app: photos are made on the phone with Google MediaPipe (Stable Diffusion 1.5 converted for phones). */
object EngineFactory {
    fun create(context: Context): Generator = MediaPipeGenerator(context.applicationContext)
}

class MediaPipeGenerator(private val context: Context) : Generator {
    override val supported: Boolean = true

    private var generator: ImageGenerator? = null
    private var loadedFrom: String? = null

    @Synchronized
    override fun generate(
        modelDir: File,
        prompt: String,
        iterations: Int,
        seed: Int,
        onLoading: () -> Unit,
        onProgress: (Float) -> Unit,
        isCancelled: () -> Boolean
    ): Bitmap {
        val path = modelDir.absolutePath
        if (generator == null || loadedFrom != path) {
            close()
            onLoading()
            val options = ImageGenerator.ImageGeneratorOptions.builder()
                .setImageGeneratorModelDirectory(path)
                .build()
            generator = ImageGenerator.createFromOptions(context, options)
            loadedFrom = path
        }
        val g = generator ?: throw IllegalStateException("The model could not be loaded.")

        g.setInputs(prompt, iterations, seed)
        var bitmap: Bitmap? = null
        for (step in 0 until iterations) {
            if (isCancelled()) throw CancelledException()
            val last = step == iterations - 1
            val result = g.execute(last)
            onProgress((step + 1f) / iterations)
            if (last) {
                val image = result?.generatedImage() ?: throw IllegalStateException("The model returned no image.")
                bitmap = BitmapExtractor.extract(image)
            }
        }
        return bitmap ?: throw IllegalStateException("No image was made.")
    }

    @Synchronized
    override fun close() {
        try {
            generator?.close()
        } catch (ignored: Throwable) {
        }
        generator = null
        loadedFrom = null
    }
}
