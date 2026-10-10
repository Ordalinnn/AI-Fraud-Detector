package kz.aifrauddetector

import android.app.Notification
import android.app.NotificationManager
import android.app.Person
import android.content.Context
import android.os.Process
import android.service.notification.StatusBarNotification
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.Shadows.shadowOf
import org.robolectric.annotation.Config

/**
 * End-to-end on a simulated Android (Robolectric): a WhatsApp-style
 * notification arrives -> the listener extracts the text -> scores it on
 * the device -> posts (or doesn't post) a warning.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class FlowTest {

    private lateinit var ctx: Context
    private lateinit var watcher: NotificationWatcher
    private lateinit var nm: NotificationManager

    @Before
    fun setUp() {
        ctx = RuntimeEnvironment.getApplication()
        watcher = Robolectric.buildService(NotificationWatcher::class.java).create().get()
        nm = ctx.getSystemService(NotificationManager::class.java)
        Store(ctx).clearAlerts()
    }

    private fun messaging(vararg texts: String): Notification {
        val me = Person.Builder().setName("Me").build()
        val them = Person.Builder().setName("+7 700 000 00 00").build()
        val style = Notification.MessagingStyle(me)
        texts.forEachIndexed { i, t -> style.addMessage(Notification.MessagingStyle.Message(t, i.toLong(), them)) }
        return Notification.Builder(ctx, "x").setSmallIcon(R.drawable.ic_stat_shield).setStyle(style).build()
    }

    private fun plain(text: String): Notification =
        Notification.Builder(ctx, "x").setSmallIcon(R.drawable.ic_stat_shield)
            .setContentTitle("Kaspi").setContentText(text).build()

    @Suppress("DEPRECATION")
    private fun post(pkg: String, n: Notification) = watcher.onNotificationPosted(
        StatusBarNotification(pkg, pkg, 1, null, 10123, 0, 0, n, Process.myUserHandle(), System.currentTimeMillis()))

    private fun warnings() = shadowOf(nm).allNotifications.filter {
        it.extras.getCharSequence(Notification.EXTRA_TITLE)?.startsWith("⚠️") == true
    }

    @Test
    fun scamInWhatsAppProducesOneWarning() {
        val scam = "Здравствуйте, это служба безопасности банка. Ваша карта заблокирована, назовите код из SMS"
        post("com.whatsapp", messaging("Привет, как дела?", scam))
        assertEquals(1, warnings().size)
        val a = Store(ctx).alerts().single()
        assertEquals("WhatsApp", a.app)
        assertEquals(scam, a.text)
        assertTrue(a.probability >= 0.5)

        // WhatsApp re-posts the whole conversation with the next message:
        post("com.whatsapp", messaging("Привет, как дела?", scam, "Ок, жду"))
        assertEquals("same scam must not be warned about twice", 1, Store(ctx).alerts().size)
    }

    @Test
    fun ordinaryMessagesProduceNoWarning() {
        post("com.whatsapp", messaging("Привет! Урок математики перенесли на пятницу, не забудь тетрадь"))
        post("org.telegram.messenger", messaging("Мама, я буду дома в восемь, купить хлеба?"))
        assertEquals(0, warnings().size)
    }

    @Test
    fun plainSmsNotificationIsChecked() {
        post("com.google.android.apps.messaging",
            plain("Срочно! Ваша карта заблокирована. Подтвердите данные по ссылке http://kaspi-login.xyz и введите код"))
        assertEquals(1, warnings().size)
    }

    @Test
    fun unwatchedAppsAndDisabledGroupsAreIgnored() {
        val scam = "Ваша карта заблокирована, срочно назовите код из SMS сотруднику банка"
        post("com.android.chrome", plain(scam))
        Store(ctx).setGroupEnabled("telegram", false)
        post("org.telegram.messenger", messaging(scam))
        assertEquals(0, warnings().size)
        Store(ctx).setGroupEnabled("telegram", true)
    }

    @Test
    fun mainScreenOpensAndDemoWarningWorks() {
        val activity = Robolectric.buildActivity(MainActivity::class.java).setup().get()
        val demo = activity.getString(R.string.btn_demo)
        val button = findButton(activity.window.decorView, demo)
        button.performClick()
        assertEquals(1, warnings().size)
    }

    private fun findButton(v: android.view.View, label: String): android.widget.Button {
        if (v is android.widget.Button && v.text == label) return v
        if (v is android.view.ViewGroup) for (i in 0 until v.childCount) {
            runCatching { return findButton(v.getChildAt(i), label) }
        }
        throw NoSuchElementException(label)
    }
}
