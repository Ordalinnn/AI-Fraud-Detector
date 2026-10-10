package kz.aifrauddetector

/**
 * Warning thresholds. Measured for this exact model by
 * scripts/export_mobile_model.py (reports/mobile_model.json):
 *   0.5  - catches ~93% of scams in the jury test; ~1% of real ordinary SMS warned about
 *   0.8  - catches ~61%; ~0.2% of real ordinary SMS warned about
 *   0.35 - more sensitive than 0.5, more false alarms
 */
enum class Sensitivity(val threshold: Double) {
    HIGH(0.35),
    NORMAL(0.5),
    LOW(0.8);

    companion object {
        fun fromName(name: String?): Sensitivity = entries.firstOrNull { it.name == name } ?: NORMAL
    }
}
