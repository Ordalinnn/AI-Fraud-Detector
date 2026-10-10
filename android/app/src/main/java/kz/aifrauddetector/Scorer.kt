package kz.aifrauddetector

import android.content.Context

/** One model instance per process (the file is ~0.9 MB; loading takes ~0.1 s). */
object Scorer {
    @Volatile private var instance: FraudScorer? = null

    fun get(context: Context): FraudScorer =
        instance ?: synchronized(this) {
            instance ?: context.applicationContext.assets.open("char_model.tsv").use { FraudScorer.load(it) }
                .also { instance = it }
        }
}
