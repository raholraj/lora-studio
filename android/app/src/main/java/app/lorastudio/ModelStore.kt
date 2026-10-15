package app.lorastudio

import android.content.Context
import java.io.File
import java.io.InputStream
import java.util.zip.ZipInputStream

/** Keeps the converted model files on the phone (app private storage). */
class ModelStore(context: Context) {
    private val appContext = context.applicationContext
    val dir: File = File(appContext.filesDir, "model")
    private val prefs = appContext.getSharedPreferences("studio", Context.MODE_PRIVATE)

    var trigger: String
        get() = prefs.getString("trigger", "") ?: ""
        set(value) {
            prefs.edit().putString("trigger", value).apply()
        }

    fun ready(): Boolean = dir.isDirectory && (dir.listFiles()?.isNotEmpty() == true)

    fun sizeMB(): Long {
        if (!ready()) return 0L
        return dir.walkTopDown().filter { it.isFile }.sumOf { it.length() } / (1024L * 1024L)
    }

    fun delete() {
        dir.deleteRecursively()
    }

    /**
     * Unpacks a zip (read as a stream, so nothing big is stored twice) into the model folder.
     * progress gets values from 0 to 1 (always 0 when totalBytes is unknown).
     */
    fun installZip(input: InputStream, totalBytes: Long, progress: (Float) -> Unit) {
        val tmp = File(appContext.filesDir, "model_tmp")
        tmp.deleteRecursively()
        tmp.mkdirs()
        val root = tmp.canonicalFile
        val counted = CountingInputStream(input)
        var lastReport = 0L
        try {
            ZipInputStream(counted.buffered(1 shl 16)).use { zip ->
                val buf = ByteArray(1 shl 16)
                var entry = zip.nextEntry
                while (entry != null) {
                    val name = entry.name
                    if (!name.startsWith("__MACOSX") && !name.endsWith(".DS_Store")) {
                        val out = File(tmp, name).canonicalFile
                        if (out != root && !out.path.startsWith(root.path + File.separator)) {
                            throw SecurityException("The zip has a bad file path.")
                        }
                        if (out == root) {
                            // an entry like "./" : nothing to write
                        } else if (entry.isDirectory) {
                            out.mkdirs()
                        } else {
                            out.parentFile?.mkdirs()
                            out.outputStream().use { o ->
                                while (true) {
                                    val n = zip.read(buf)
                                    if (n < 0) break
                                    o.write(buf, 0, n)
                                    val now = System.currentTimeMillis()
                                    if (now - lastReport > 400) {
                                        lastReport = now
                                        progress(if (totalBytes > 0) (counted.count.toFloat() / totalBytes).coerceIn(0f, 1f) else 0f)
                                    }
                                }
                            }
                        }
                    }
                    entry = zip.nextEntry
                }
            }

            // If everything sits inside one folder, use that folder as the model folder.
            val children = tmp.listFiles()?.toList().orEmpty()
            val modelRoot = if (children.size == 1 && children[0].isDirectory) children[0] else tmp
            if (modelRoot.walkTopDown().none { it.isFile }) {
                throw IllegalStateException("The zip has no files in it.")
            }
            dir.deleteRecursively()
            if (!modelRoot.renameTo(dir)) {
                modelRoot.copyRecursively(dir, overwrite = true)
            }
        } finally {
            tmp.deleteRecursively()
        }
    }
}

private class CountingInputStream(private val inner: InputStream) : InputStream() {
    var count = 0L
        private set

    override fun read(): Int {
        val b = inner.read()
        if (b >= 0) count++
        return b
    }

    override fun read(b: ByteArray, off: Int, len: Int): Int {
        val n = inner.read(b, off, len)
        if (n > 0) count += n
        return n
    }

    override fun close() = inner.close()
}
