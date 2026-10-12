package kz.aifrauddetector

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

/** Settings and the short list of recent warnings, kept only on this phone. */
class Store(context: Context) {
    private val prefs = context.applicationContext.getSharedPreferences("settings", Context.MODE_PRIVATE)

    var sensitivity: Sensitivity
        get() = Sensitivity.fromName(prefs.getString("sensitivity", null))
        set(v) { prefs.edit().putString("sensitivity", v.name).apply() }

    fun isGroupEnabled(group: String): Boolean = prefs.getBoolean("group_$group", true)
    fun setGroupEnabled(group: String, on: Boolean) { prefs.edit().putBoolean("group_$group", on).apply() }

    data class Alert(val id: Int, val time: Long, val app: String, val text: String,
                     val probability: Double, val words: List<String>)

    @Synchronized
    fun alerts(): List<Alert> {
        val arr = JSONArray(prefs.getString("alerts", "[]"))
        return (0 until arr.length()).map { i ->
            val o = arr.getJSONObject(i)
            val w = o.optJSONArray("words") ?: JSONArray()
            Alert(o.getInt("id"), o.getLong("time"), o.getString("app"), o.getString("text"),
                o.getDouble("p"), (0 until w.length()).map { w.getString(it) })
        }
    }

    @Synchronized
    fun addAlert(a: Alert) {
        val list = (listOf(a) + alerts()).take(MAX_ALERTS)
        val arr = JSONArray()
        for (x in list) {
            arr.put(JSONObject().put("id", x.id).put("time", x.time).put("app", x.app)
                .put("text", x.text).put("p", x.probability).put("words", JSONArray(x.words)))
        }
        prefs.edit().putString("alerts", arr.toString()).apply()
    }

    @Synchronized
    fun clearAlerts() { prefs.edit().remove("alerts").apply() }

    // ---- "the model was wrong" reports (Mistakes.kt), only on this phone ----

    @Synchronized
    fun mistakes(): List<Mistakes.Entry> {
        val arr = JSONArray(prefs.getString("mistakes", "[]"))
        return (0 until arr.length()).map { i ->
            val o = arr.getJSONObject(i)
            Mistakes.Entry(o.getString("text"), o.getInt("label"), o.getString("language"),
                o.getString("source"), o.getString("date"))
        }
    }

    @Synchronized
    fun addMistake(e: Mistakes.Entry): Int {
        val list = Mistakes.addTo(mistakes(), e)
        val arr = JSONArray()
        for (x in list) {
            arr.put(JSONObject().put("text", x.text).put("label", x.label).put("language", x.language)
                .put("source", x.source).put("date", x.date))
        }
        prefs.edit().putString("mistakes", arr.toString()).apply()
        return list.size
    }

    @Synchronized
    fun clearMistakes() { prefs.edit().remove("mistakes").apply() }

    companion object { const val MAX_ALERTS = 20 }
}
