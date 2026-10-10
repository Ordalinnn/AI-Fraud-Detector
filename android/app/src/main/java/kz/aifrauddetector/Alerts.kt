package kz.aifrauddetector

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent

/** Posts the "looks like fraud" warning notification. */
object Alerts {
    private const val CHANNEL = "fraud_warnings"

    fun ensureChannel(context: Context) {
        val nm = context.getSystemService(NotificationManager::class.java)
        if (nm.getNotificationChannel(CHANNEL) == null) {
            val ch = NotificationChannel(CHANNEL, context.getString(R.string.channel_name),
                NotificationManager.IMPORTANCE_HIGH)
            ch.description = context.getString(R.string.channel_desc)
            nm.createNotificationChannel(ch)
        }
    }

    fun appLabel(context: Context, group: String?): String = when (group) {
        "whatsapp" -> "WhatsApp"
        "telegram" -> "Telegram"
        "sms" -> "SMS"
        else -> context.getString(R.string.app_group_other)
    }

    fun show(context: Context, alert: Store.Alert) {
        ensureChannel(context)
        val open = Intent(context, MainActivity::class.java)
            .putExtra(MainActivity.EXTRA_ALERT_ID, alert.id)
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
        val pi = PendingIntent.getActivity(context, alert.id, open,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        val pct = Math.round(alert.probability * 100)
        val title = context.getString(R.string.alert_title, pct)
        val snippet = alert.text.replace('\n', ' ').take(140)
        val body = context.getString(R.string.alert_body, alert.app, snippet)
        val n = Notification.Builder(context, CHANNEL)
            .setSmallIcon(R.drawable.ic_stat_shield)
            .setContentTitle(title)
            .setContentText(body)
            .setStyle(Notification.BigTextStyle().bigText(body + "\n\n" + context.getString(R.string.alert_advice)))
            .setContentIntent(pi)
            .setAutoCancel(true)
            .setCategory(Notification.CATEGORY_MESSAGE)
            .build()
        context.getSystemService(NotificationManager::class.java).notify(alert.id, n)
    }
}
