package kz.aifrauddetector

/**
 * Pure (Android-free, unit-testable) helpers for turning notification text
 * into the messages worth checking.
 */
object Messages {

    /** Messaging apps watched, grouped as shown on the settings screen. */
    val APP_GROUPS: Map<String, Set<String>> = linkedMapOf(
        "whatsapp" to setOf("com.whatsapp", "com.whatsapp.w4b"),
        "telegram" to setOf("org.telegram.messenger", "org.telegram.messenger.web", "org.thunderdog.challegram",
            "org.telegram.plus"),
        "sms" to setOf("com.google.android.apps.messaging", "com.samsung.android.messaging", "com.android.mms",
            "com.android.messaging", "com.huawei.message", "com.oneplus.mms", "com.miui.sms"),
        "other" to setOf("com.viber.voip", "com.instagram.android", "com.facebook.orca",
            "org.thoughtcrime.securesms", "com.imo.android.imoim"),
    )

    fun groupOf(packageName: String): String? =
        APP_GROUPS.entries.firstOrNull { packageName in it.value }?.key

    /** Messages shorter than this carry too little text to judge. */
    const val MIN_CHARS = 12

    // "3 new messages", "2 новых сообщения" etc. - app summaries, not messages.
    private val SUMMARY = Regex(
        """^\s*\d+\s+(new\s+messages?|новы[хе]?\s+сообщени[йяе]|жаңа\s+хабарлама).*""",
        RegexOption.IGNORE_CASE,
    )

    fun isWorthChecking(text: String): Boolean {
        val t = text.trim()
        return t.length >= MIN_CHARS && !SUMMARY.matches(t)
    }

    /**
     * Messaging apps re-post the whole conversation in every notification
     * (MessagingStyle). Remembers which texts were already checked so each
     * message is scored - and warned about - only once.
     */
    class SeenCache(private val capacity: Int = 500) {
        private val seen = object : LinkedHashMap<String, Unit>(capacity, 0.75f, true) {
            override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, Unit>?) = size > capacity
        }

        @Synchronized
        fun firstTime(packageName: String, text: String): Boolean {
            val key = packageName + "\u0000" + text.trim()
            if (seen.containsKey(key)) return false
            seen[key] = Unit
            return true
        }
    }

    /** New, checkable texts among a notification's candidate texts, in order. */
    fun newTexts(packageName: String, candidates: List<CharSequence?>, cache: SeenCache): List<String> =
        candidates.mapNotNull { it?.toString()?.trim() }
            .filter { it.isNotEmpty() }
            .distinct()
            .filter { isWorthChecking(it) && cache.firstTime(packageName, it) }
}
