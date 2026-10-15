package app.lorastudio

import android.app.ActivityManager
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Environment
import android.provider.MediaStore
import android.provider.OpenableColumns
import android.util.Base64
import android.util.Log
import android.webkit.JavascriptInterface
import android.webkit.WebView
import androidx.core.content.FileProvider
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlin.random.Random

/**
 * What the web page can call as window.NativeStudio.
 * Answers come back to the page through window.__studioNative(json).
 */
class StudioBridge(private val activity: MainActivity, private val web: WebView) {
    private val tag = "StudioBridge"
    private val store = ModelStore(activity)
    private val engine: Generator = EngineFactory.create(activity)
    private val worker = Executors.newSingleThreadExecutor()
    private val io = Executors.newSingleThreadExecutor()
    private val busy = AtomicBoolean(false)
    private val installing = AtomicBoolean(false)
    private val jobCounter = AtomicInteger(0)
    @Volatile private var cancelFlag = false
    @Volatile private var closed = false
    @Volatile private var pendingTrigger = ""

    private fun emit(obj: JSONObject) {
        if (closed) return
        val js = "window.__studioNative && window.__studioNative(" + JSONObject.quote(obj.toString()) + ")"
        web.post { web.evaluateJavascript(js, null) }
    }

    private fun modelEvent(state: String, p: Float, message: String) {
        emit(JSONObject().put("type", "model").put("state", state).put("p", p.toDouble()).put("message", message))
    }

    private fun err(message: String): String = JSONObject().put("error", message).toString()

    // ------------------------------------------------------------------ info

    @JavascriptInterface
    fun info(): String {
        val am = activity.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mem = ActivityManager.MemoryInfo()
        am.getMemoryInfo(mem)
        return JSONObject()
            .put("native", true)
            .put("onDevice", engine.supported)
            .put("modelReady", store.ready())
            .put("modelMB", store.sizeMB())
            .put("trigger", store.trigger)
            .put("ramMB", mem.totalMem / (1024L * 1024L))
            .toString()
    }

    // ----------------------------------------------------------- model files

    @JavascriptInterface
    fun importModel(trigger: String) {
        if (!engine.supported) return
        pendingTrigger = trigger.trim()
        activity.runOnUiThread { activity.pickModelFile() }
    }

    /** Called by MainActivity after the person picked a zip file. */
    fun importFromUri(uri: Uri) {
        if (!engine.supported) return
        if (busy.get() || !installing.compareAndSet(false, true)) {
            modelEvent("error", 0f, "Please wait until the current job finishes.")
            return
        }
        io.execute {
            activity.keepScreenOn(true)
            try {
                modelEvent("working", 0f, "Copying the model…")
                engine.close()
                var total = -1L
                try {
                    activity.contentResolver.query(uri, arrayOf(OpenableColumns.SIZE), null, null, null)?.use { c ->
                        if (c.moveToFirst() && !c.isNull(0)) total = c.getLong(0)
                    }
                } catch (ignored: Exception) {
                }
                val input = activity.contentResolver.openInputStream(uri)
                    ?: throw IllegalStateException("The file could not be opened.")
                input.use {
                    store.installZip(it, total) { p -> modelEvent("working", p, "Copying the model… ${(p * 100).toInt()}%") }
                }
                finishInstall()
            } catch (e: Throwable) {
                Log.e(tag, "import failed", e)
                modelEvent("error", 0f, "Could not import the model: ${e.message ?: "unknown error"}")
            } finally {
                installing.set(false)
                activity.keepScreenOn(false)
            }
        }
    }

    @JavascriptInterface
    fun downloadModel(url: String, trigger: String) {
        if (!engine.supported) return
        if (!url.startsWith("https://")) {
            modelEvent("error", 0f, "Only https links are allowed.")
            return
        }
        if (busy.get() || !installing.compareAndSet(false, true)) {
            modelEvent("error", 0f, "Please wait until the current job finishes.")
            return
        }
        pendingTrigger = trigger.trim()
        io.execute {
            activity.keepScreenOn(true)
            var conn: HttpURLConnection? = null
            try {
                modelEvent("working", 0f, "Connecting…")
                engine.close()
                conn = URL(url).openConnection() as HttpURLConnection
                conn.connectTimeout = 20000
                conn.readTimeout = 30000
                conn.instanceFollowRedirects = true
                val code = conn.responseCode
                if (code !in 200..299) throw IllegalStateException("The server answered with code $code.")
                val total = conn.contentLengthLong
                conn.inputStream.use {
                    store.installZip(it, total) { p -> modelEvent("working", p, "Downloading… ${(p * 100).toInt()}%") }
                }
                finishInstall()
            } catch (e: Throwable) {
                Log.e(tag, "download failed", e)
                modelEvent("error", 0f, "Could not download the model: ${e.message ?: "unknown error"}")
            } finally {
                conn?.disconnect()
                installing.set(false)
                activity.keepScreenOn(false)
            }
        }
    }

    private fun finishInstall() {
        if (pendingTrigger.isNotBlank()) store.trigger = pendingTrigger
        modelEvent("done", 1f, "Model ready.")
    }

    @JavascriptInterface
    fun deleteModel(): String {
        if (busy.get() || installing.get()) return "busy"
        // closing can wait for a running job, so never do it on the page's thread
        io.execute {
            engine.close()
            store.delete()
            modelEvent("done", 1f, "Model deleted.")
        }
        return "ok"
    }

    // ------------------------------------------------------------ generating

    @JavascriptInterface
    fun generate(json: String): String {
        if (!engine.supported) return err("This app cannot make photos on the phone.")
        if (!store.ready()) return err("Import the model first.")
        if (installing.get()) return err("The model is still being copied.")
        if (!busy.compareAndSet(false, true)) return err("A photo is already being made.")

        val req = try {
            JSONObject(json)
        } catch (e: Exception) {
            busy.set(false)
            return err("The request could not be read.")
        }

        var prompt = req.optString("prompt").trim()
        if (prompt.isEmpty()) {
            busy.set(false)
            return err("The prompt is empty.")
        }
        // Safety word filter removed
        val trigger = store.trigger
        if (trigger.isNotBlank() && !prompt.contains(trigger, ignoreCase = true)) {
            prompt = "$trigger, $prompt"
        }
        prompt = "$prompt, photorealistic, natural light, sharp focus"

        val iterations = req.optInt("steps", 20).coerceIn(10, 50)
        val rawSeed = req.optLong("seed", -1L)
        val seed = if (rawSeed < 0) Random.nextInt(0, Int.MAX_VALUE) else (rawSeed % Int.MAX_VALUE).toInt()
        val id = "g" + jobCounter.incrementAndGet()
        cancelFlag = false
        activity.keepScreenOn(true)
        worker.execute { runJob(id, prompt, iterations, seed) }
        return JSONObject().put("id", id).toString()
    }

    private fun runJob(id: String, prompt: String, iterations: Int, seed: Int) {
        try {
            val bitmap = engine.generate(
                store.dir, prompt, iterations, seed,
                onLoading = { emit(JSONObject().put("type", "loading").put("id", id)) },
                onProgress = { p -> emit(JSONObject().put("type", "progress").put("id", id).put("p", p.toDouble())) },
                isCancelled = { cancelFlag }
            )
            val out = ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.JPEG, 93, out)
            val b64 = Base64.encodeToString(out.toByteArray(), Base64.NO_WRAP)
            emit(
                JSONObject().put("type", "done").put("id", id).put("seed", seed)
                    .put("prompt", prompt).put("image", b64)
            )
        } catch (e: CancelledException) {
            emit(JSONObject().put("type", "cancelled").put("id", id))
        } catch (e: OutOfMemoryError) {
            engine.close()
            emit(
                JSONObject().put("type", "error").put("id", id)
                    .put("message", "The phone ran out of memory. Close other apps and try again.")
            )
        } catch (e: Throwable) {
            Log.e(tag, "generate failed", e)
            emit(
                JSONObject().put("type", "error").put("id", id)
                    .put("message", "Could not make the photo: ${e.message ?: "unknown error"}")
            )
        } finally {
            busy.set(false)
            activity.keepScreenOn(false)
        }
    }

    @JavascriptInterface
    fun cancel() {
        cancelFlag = true
    }

    // ----------------------------------------------------------- save, share

    private fun safeName(name: String): String =
        name.replace(Regex("[^A-Za-z0-9._-]"), "_").ifBlank { "photo.jpg" }

    @JavascriptInterface
    fun saveImage(b64: String, name: String): String {
        return try {
            val bytes = Base64.decode(b64, Base64.DEFAULT)
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, safeName(name))
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/LoRA Studio")
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
            val resolver = activity.contentResolver
            val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                ?: throw IllegalStateException("No storage")
            try {
                val stream = resolver.openOutputStream(uri) ?: throw IllegalStateException("No stream")
                stream.use { it.write(bytes) }
                values.clear()
                values.put(MediaStore.Images.Media.IS_PENDING, 0)
                resolver.update(uri, values, null, null)
            } catch (e: Throwable) {
                try {
                    resolver.delete(uri, null, null)
                } catch (ignored: Throwable) {
                }
                throw e
            }
            "ok"
        } catch (e: Throwable) {
            Log.e(tag, "save failed", e)
            "Could not save the photo."
        }
    }

    @JavascriptInterface
    fun shareImage(b64: String, name: String): String {
        return try {
            val bytes = Base64.decode(b64, Base64.DEFAULT)
            val dir = File(activity.cacheDir, "share")
            dir.mkdirs()
            val file = File(dir, safeName(name))
            file.writeBytes(bytes)
            val uri = FileProvider.getUriForFile(activity, activity.packageName + ".files", file)
            val send = Intent(Intent.ACTION_SEND).apply {
                type = "image/jpeg"
                putExtra(Intent.EXTRA_STREAM, uri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            activity.runOnUiThread { activity.startActivity(Intent.createChooser(send, "Share photo")) }
            "ok"
        } catch (e: Throwable) {
            Log.e(tag, "share failed", e)
            "Could not share the photo."
        }
    }

    fun close() {
        closed = true
        cancelFlag = true
        worker.shutdownNow()
        // engine.close() waits for a running job, so it must not run on the main thread
        Thread { engine.close() }.start()
        io.shutdown()
    }
}
