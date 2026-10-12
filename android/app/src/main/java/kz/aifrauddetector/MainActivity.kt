package kz.aifrauddetector

import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.content.ComponentName
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Typeface
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import java.net.URLEncoder
import java.text.DateFormat
import java.util.Date

/**
 * Settings + manual check + recent warnings, built from plain Android views
 * (no layout XML, no libraries) so the whole app stays small and readable.
 */
class MainActivity : Activity() {

    private lateinit var store: Store
    private lateinit var root: LinearLayout
    private val dp by lazy { resources.displayMetrics.density }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = Store(this)
        Alerts.ensureChannel(this)
        root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(px(20), px(24), px(20), px(32))
        }
        setContentView(ScrollView(this).apply { addView(root) })
        if (Build.VERSION.SDK_INT >= 33 &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 1)
        }
        showAlertFromIntent(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        showAlertFromIntent(intent)
    }

    override fun onResume() {
        super.onResume()
        render() // refresh: the user may be back from the system settings screen
    }

    private fun render() {
        root.removeAllViews()
        title(getString(R.string.app_name), 26f)
        text(getString(R.string.subtitle), muted = true)

        // ---- status ----
        section()
        val on = listenerEnabled()
        text(getString(if (on) R.string.status_on else R.string.status_off), size = 18f, bold = true,
            color = if (on) GREEN else RED)
        if (!on) button(getString(R.string.btn_grant_access)) {
            startActivity(Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS))
        }
        if (Build.VERSION.SDK_INT >= 33 &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            text(getString(R.string.status_notif_blocked), color = RED)
            button(getString(R.string.btn_allow_notifications)) {
                requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 1)
            }
        }
        text(getString(R.string.battery_title), muted = true)
        button(getString(R.string.btn_battery)) {
            startActivity(Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS))
        }
        button(getString(R.string.btn_demo)) { showDemoAlert() }

        // ---- sensitivity ----
        section()
        title(getString(R.string.sensitivity_title), 18f)
        val group = RadioGroup(this)
        val labels = mapOf(Sensitivity.HIGH to R.string.sens_high, Sensitivity.NORMAL to R.string.sens_normal,
            Sensitivity.LOW to R.string.sens_low)
        for ((s, label) in labels) {
            group.addView(RadioButton(this).apply {
                id = View.generateViewId()
                text = getString(label)
                isChecked = store.sensitivity == s
                setOnCheckedChangeListener { _, checked -> if (checked) store.sensitivity = s }
            })
        }
        root.addView(group)

        // ---- apps ----
        section()
        title(getString(R.string.apps_title), 18f)
        for (g in Messages.APP_GROUPS.keys) {
            root.addView(CheckBox(this).apply {
                text = Alerts.appLabel(this@MainActivity, g)
                isChecked = store.isGroupEnabled(g)
                setOnCheckedChangeListener { _, c -> store.setGroupEnabled(g, c) }
            })
        }

        // ---- manual check ----
        section()
        title(getString(R.string.manual_title), 18f)
        val input = EditText(this).apply {
            hint = getString(R.string.manual_hint)
            minLines = 3
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
            gravity = Gravity.TOP
        }
        root.addView(input)
        val result = TextView(this).apply { textSize = 16f; setPadding(0, px(8), 0, 0) }
        // "The model is wrong": saved on this phone, exported from the section below
        val wrong = Button(this).apply { text = getString(R.string.mistake_wrong); visibility = View.GONE }
        button(getString(R.string.btn_check)) {
            val t = input.text.toString()
            if (t.isNotBlank()) {
                val r = Scorer.get(this).analyze(t)
                val fraud = r.probability >= store.sensitivity.threshold
                result.text = getString(R.string.manual_result, Math.round(r.probability * 100),
                    getString(if (fraud) R.string.verdict_fraud else R.string.verdict_safe)) +
                    if (r.suspiciousWords.isNotEmpty()) "\n" + getString(R.string.words_label) + " " +
                        r.suspiciousWords.joinToString(", ") else ""
                result.setTextColor(if (fraud) RED else GREEN)
                wrong.visibility = View.VISIBLE
                wrong.setOnClickListener {
                    reportMistake(t, fraud, r.probability, "app: manual check")
                    wrong.visibility = View.GONE
                }
            }
        }
        root.addView(result)
        root.addView(wrong)

        // ---- recent warnings ----
        section()
        title(getString(R.string.alerts_title), 18f)
        val alerts = store.alerts()
        if (alerts.isEmpty()) text(getString(R.string.alerts_empty), muted = true)
        val fmt = DateFormat.getDateTimeInstance(DateFormat.SHORT, DateFormat.SHORT)
        for (a in alerts) {
            root.addView(TextView(this).apply {
                text = "⚠️ ${Math.round(a.probability * 100)}% · ${a.app} · ${fmt.format(Date(a.time))}\n" +
                    a.text.replace('\n', ' ').take(120)
                textSize = 15f
                setPadding(0, px(8), 0, px(8))
                setOnClickListener { showAlert(a) }
            })
        }
        if (alerts.isNotEmpty()) button(getString(R.string.btn_clear)) { store.clearAlerts(); render() }

        // ---- marked mistakes ----
        val mistakes = store.mistakes()
        if (mistakes.isNotEmpty()) {
            section()
            title(getString(R.string.mistakes_title, mistakes.size), 18f)
            text(getString(R.string.mistakes_note), muted = true)
            button(getString(R.string.btn_share_csv)) {
                val send = Intent(Intent.ACTION_SEND).apply {
                    type = "text/plain"
                    putExtra(Intent.EXTRA_SUBJECT, "AI Fraud Detector: mistakes")
                    putExtra(Intent.EXTRA_TEXT, Mistakes.toCsv(store.mistakes()))
                }
                startActivity(Intent.createChooser(send, getString(R.string.btn_share_csv)))
            }
            button(getString(R.string.btn_clear_mistakes)) { store.clearMistakes(); render() }
        }

        section()
        text(getString(R.string.privacy_note), muted = true)
    }

    // ---------- alerts ----------

    private fun showAlertFromIntent(intent: Intent?) {
        val id = intent?.getIntExtra(EXTRA_ALERT_ID, 0) ?: 0
        if (id != 0) store.alerts().firstOrNull { it.id == id }?.let { showAlert(it) }
    }

    private fun showAlert(a: Store.Alert) {
        val msg = buildString {
            append(getString(R.string.alert_title, Math.round(a.probability * 100))).append("\n\n")
            append(a.app).append(":\n«").append(a.text).append("»\n\n")
            if (a.words.isNotEmpty()) append(getString(R.string.words_label)).append(" ")
                .append(a.words.joinToString(", ")).append("\n\n")
            append(getString(R.string.alert_advice)).append("\n\n").append(getString(R.string.web_note))
        }
        AlertDialog.Builder(this)
            .setTitle(R.string.detail_title)
            .setMessage(msg)
            .setPositiveButton(R.string.btn_close, null)
            .setNegativeButton(R.string.mistake_not_fraud) { _, _ ->
                reportMistake(a.text, true, a.probability, "app: ${a.app}")
            }
            .setNeutralButton(R.string.btn_open_web) { _, _ ->
                val url = WEB_APP_URL + "?text=" + URLEncoder.encode(a.text, "UTF-8")
                startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)))
            }
            .show()
    }

    private fun showDemoAlert() {
        val text = getString(R.string.demo_text)
        val r = Scorer.get(this).analyze(text)
        val a = Store.Alert(id = DEMO_ID, time = System.currentTimeMillis(), app = "SMS (demo)", text = text,
            probability = r.probability, words = r.suspiciousWords)
        store.addAlert(a)
        Alerts.show(this, a)
        render()
    }

    private fun reportMistake(text: String, predictedFraud: Boolean, probability: Double, source: String) {
        store.addMistake(Mistakes.makeEntry(text, predictedFraud, probability, source))
        Toast.makeText(this, R.string.mistake_thanks, Toast.LENGTH_SHORT).show()
        render()
    }

    private fun listenerEnabled(): Boolean {
        val flat = Settings.Secure.getString(contentResolver, "enabled_notification_listeners") ?: return false
        val me = ComponentName(this, NotificationWatcher::class.java)
        return flat.split(':').mapNotNull { ComponentName.unflattenFromString(it) }.any { it == me }
    }

    // ---------- tiny view helpers ----------

    private fun px(v: Int) = (v * dp).toInt()

    private fun title(s: String, size: Float) = text(s, size = size, bold = true)

    private fun text(s: String, size: Float = 15f, bold: Boolean = false, muted: Boolean = false, color: Int? = null) {
        root.addView(TextView(this).apply {
            text = s
            textSize = size
            if (bold) setTypeface(typeface, Typeface.BOLD)
            setTextColor(color ?: if (muted) Color.GRAY else Color.BLACK)
            setPadding(0, px(4), 0, px(4))
        })
    }

    private fun button(label: String, onClick: () -> Unit) {
        root.addView(Button(this).apply { text = label; setOnClickListener { onClick() } })
    }

    private fun section() {
        root.addView(View(this).apply {
            setBackgroundColor(0xFFE2E8F0.toInt())
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, px(1)).apply {
                topMargin = px(16); bottomMargin = px(12)
            }
        })
    }

    companion object {
        const val EXTRA_ALERT_ID = "alert_id"
        private const val DEMO_ID = 4242
        private const val WEB_APP_URL = "https://ai-fraud-detector-p6jkftqrssphfcsxxepipo.streamlit.app/"
        private val GREEN = Color.rgb(22, 163, 74)
        private val RED = Color.rgb(220, 38, 38)
    }
}
