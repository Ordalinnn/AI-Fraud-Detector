package kz.aifrauddetector

import android.app.Notification
import android.os.Bundle
import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import android.util.Log

/**
 * Reads the text of incoming notifications from the messaging apps the
 * user selected, scores each new message on the phone, and posts a warning
 * when the risk is above the chosen sensitivity. Nothing leaves the device:
 * the app doesn't even have the INTERNET permission.
 */
class NotificationWatcher : NotificationListenerService() {

    private val cache = Messages.SeenCache()

    override fun onNotificationPosted(sbn: StatusBarNotification) {
        try {
            handle(sbn)
        } catch (e: Exception) {
            Log.w(TAG, "could not check notification", e) // never crash the listener
        }
    }

    private fun handle(sbn: StatusBarNotification) {
        if (sbn.packageName == packageName) return
        val group = Messages.groupOf(sbn.packageName) ?: return
        val store = Store(this)
        if (!store.isGroupEnabled(group)) return
        val n = sbn.notification
        if (n.flags and Notification.FLAG_GROUP_SUMMARY != 0) return

        val texts = Messages.newTexts(sbn.packageName, candidateTexts(n.extras), cache)
        if (texts.isEmpty()) return
        val scorer = Scorer.get(this)
        val threshold = store.sensitivity.threshold
        for (text in texts) {
            val r = scorer.analyze(text)
            if (r.probability >= threshold) {
                val alert = Store.Alert(
                    id = (sbn.packageName + text).hashCode(),
                    time = System.currentTimeMillis(),
                    app = Alerts.appLabel(this, group),
                    text = text,
                    probability = r.probability,
                    words = r.suspiciousWords,
                )
                store.addAlert(alert)
                Alerts.show(this, alert)
            }
        }
    }

    /** Message texts carried by a notification. MessagingStyle (WhatsApp,
     *  Telegram, Google Messages) lists every message of the conversation;
     *  others put the text in EXTRA_BIG_TEXT / EXTRA_TEXT / EXTRA_TEXT_LINES. */
    private fun candidateTexts(extras: Bundle): List<CharSequence?> {
        val out = ArrayList<CharSequence?>()
        @Suppress("DEPRECATION")
        val messages = extras.getParcelableArray(Notification.EXTRA_MESSAGES)
        if (messages != null) {
            for (m in messages) (m as? Bundle)?.getCharSequence("text")?.let { out.add(it) }
        }
        if (out.isEmpty()) {
            out.add(extras.getCharSequence(Notification.EXTRA_BIG_TEXT) ?: extras.getCharSequence(Notification.EXTRA_TEXT))
            extras.getCharSequenceArray(Notification.EXTRA_TEXT_LINES)?.let { out.addAll(it) }
        }
        return out
    }

    companion object { private const val TAG = "FraudWatcher" }
}
