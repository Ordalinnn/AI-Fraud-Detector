package kz.aifrauddetector

import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.File
import java.util.Date

class MistakesTest {

    @Test
    fun reportStoresTheCorrectedLabelAndLanguage() {
        val a = Mistakes.makeEntry("Привет, урок перенесли", predictedFraud = true, probability = 0.71, source = "app: WhatsApp", now = Date(0))
        assertEquals(0, a.label)
        assertEquals("ru", a.language)
        assertEquals("app: WhatsApp: model said fraud 71%", a.source)
        assertEquals(1, Mistakes.makeEntry("Сіздің картаңыз бұғатталды", false, 0.2, "x").label)
        assertEquals("kk", Mistakes.detectLanguage("Сіздің картаңыз"))
        assertEquals("en", Mistakes.detectLanguage("Your parcel"))
    }

    @Test
    fun newestFirstOnePerTextCapped() {
        var list = emptyList<Mistakes.Entry>()
        list = Mistakes.addTo(list, Mistakes.makeEntry("a message", true, 0.9, "x"))
        list = Mistakes.addTo(list, Mistakes.makeEntry("b message", true, 0.9, "x"))
        list = Mistakes.addTo(list, Mistakes.makeEntry("a message", false, 0.1, "x"))
        assertEquals(listOf("a message" to 1, "b message" to 0), list.map { it.text to it.label })
        repeat(600) { list = Mistakes.addTo(list, Mistakes.makeEntry("m$it", true, 0.9, "x")) }
        assertEquals(Mistakes.MAX, list.size)
    }

    @Test
    fun csvMatchesTheTeamTemplateAndTheExtension() {
        val header = File("../../data/real/team_real_messages_TEMPLATE.csv").readLines()[0].trim()
        val csv = Mistakes.toCsv(listOf(
            Mistakes.Entry("Карта, \"заблокирована\"\nсрочно", 1, "ru", "web.whatsapp.com", "2026-10-12"),
            Mistakes.Entry("=HYPERLINK(1)", 0, "en", "popup", "2026-10-12"),
        ))
        val lines = csv.split("\n")
        assertEquals(header, lines[0])
        assertEquals("\"Карта, \"\"заблокирована\"\"", lines[1])
        assertEquals("срочно\",1,ru,,web.whatsapp.com,2026-10-12", lines[2])
        assertEquals("'=HYPERLINK(1),0,en,,popup,2026-10-12", lines[3])
    }
}
