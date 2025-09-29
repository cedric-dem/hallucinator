package com.example.hallucinator

import android.os.Bundle
import android.util.Log
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.example.EncoderApplicator
import java.io.IOException
import kotlin.math.min
import kotlin.random.Random

class OnlyEncoderActivity : AppCompatActivity() {
    private var encoderApplicator: EncoderApplicator? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_only_encoder)

        val greetingText: TextView = findViewById(R.id.greeting_text)
        greetingText.text = loadModelStatusMessage()

        setupBottomNavigation(R.id.navigation_main)
    }

    override fun onDestroy() {
        encoderApplicator?.close()
        encoderApplicator = null
        super.onDestroy()
    }

    private fun loadModelStatusMessage(): String {
        return try {
            val applicator = EncoderApplicator(this).also { encoderApplicator = it }
            val inputShape = applicator.inputShape
            val outputShape = applicator.outputShape

            val randomInput = createRandomInput(inputShape)
            val output = applicator.apply(randomInput)
            logOutputPreview(output)

            getString(
                R.string.model_status_success,
                inputShape.contentToString(),
                outputShape.contentToString()
            )
        } catch (error: IOException) {
            getString(R.string.model_status_error, error.localizedMessage ?: error.toString())
        } catch (error: IllegalArgumentException) {
            getString(R.string.model_status_error, error.localizedMessage ?: error.toString())
        }
    }

    private fun createRandomInput(shape: IntArray): FloatArray {
        val elementCount = shape.fold(1) { acc, dimension -> acc * dimension }
        return FloatArray(elementCount) { Random.nextFloat() }
    }

    private fun logOutputPreview(output: FloatArray) {
        val previewLength = min(10, output.size)
        val preview = output.copyOfRange(0, previewLength).joinToString(prefix = "[", postfix = "]")
        val message = "Random encoder output preview: $preview"
        Log.d(TAG, message)
        println(message)
    }

    companion object {
        private const val TAG = "OnlyEncoderActivity"
    }
}