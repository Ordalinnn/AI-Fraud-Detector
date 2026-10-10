package kz.aifrauddetector

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class MessagesTest {

    @Test
    fun knownMessengersAreGrouped() {
        assertEquals("whatsapp", Messages.groupOf("com.whatsapp"))
        assertEquals("telegram", Messages.groupOf("org.telegram.messenger"))
        assertEquals("sms", Messages.groupOf("com.google.android.apps.messaging"))
        assertNull(Messages.groupOf("com.android.chrome"))
    }

    @Test
    fun summariesAndShortTextsAreSkipped() {
        assertFalse(Messages.isWorthChecking("ok"))
        assertFalse(Messages.isWorthChecking("3 new messages"))
        assertFalse(Messages.isWorthChecking("2 новых сообщения"))
        assertTrue(Messages.isWorthChecking("Ваша карта заблокирована"))
    }

    @Test
    fun conversationRepostedInEveryNotificationIsCheckedOnce() {
        val cache = Messages.SeenCache()
        val first = Messages.newTexts("com.whatsapp", listOf("Привет, как дела у тебя?"), cache)
        val second = Messages.newTexts("com.whatsapp",
            listOf("Привет, как дела у тебя?", "Назови код из SMS срочно"), cache)
        assertEquals(listOf("Привет, как дела у тебя?"), first)
        assertEquals(listOf("Назови код из SMS срочно"), second)
    }

    @Test
    fun cacheForgetsOldestBeyondCapacity() {
        val cache = Messages.SeenCache(capacity = 2)
        assertTrue(cache.firstTime("p", "a"))
        assertTrue(cache.firstTime("p", "b"))
        assertTrue(cache.firstTime("p", "c"))
        assertTrue(cache.firstTime("p", "a")) // evicted, so seen as new again
    }
}
