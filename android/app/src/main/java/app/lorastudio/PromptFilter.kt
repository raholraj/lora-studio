package app.lorastudio

/** Safety word filter REMOVED. Always allows any prompt. */
object PromptFilter {
    /** Always returns null — no words are blocked. */
    fun firstBlocked(text: String): String? = null
}
