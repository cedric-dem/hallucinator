package com.example.hallucinator


import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.DecoderApplicator
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale
import kotlin.random.Random

class OnlyDecoderActivity : AppCompatActivity() {
    private var decoderApplicator: DecoderApplicator? = null
    private var currentInput: FloatArray? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_only_decoder)

        setupBottomNavigation(R.id.navigation_only_decoder)

        val generateButton = findViewById<Button>(R.id.button_generate_array)
        val applyButton = findViewById<Button>(R.id.button_apply_decoder)
        val generatedArrayText = findViewById<TextView>(R.id.text_generated_array)
        val decoderOutputImage = findViewById<ImageView>(R.id.image_decoder_output)

        decoderApplicator = try {
            DecoderApplicator(this)
        } catch (error: Exception) {
            val message = error.localizedMessage ?: "Unknown error"
            Toast.makeText(this, getString(R.string.model_status_error, message), Toast.LENGTH_LONG).show()
            null
        }

        generateButton.setOnClickListener {
            val array = FloatArray(INPUT_SIZE) { Random.nextFloat() * RANDOM_BOUND }
            currentInput = array
            generatedArrayText.text = formatFloatArray(array)
            decoderOutputImage.setImageDrawable(null)
        }

        applyButton.setOnClickListener {
            val applicator = decoderApplicator
            if (applicator == null) {
                Toast.makeText(
                    this,
                    getString(R.string.model_status_error, "decoder out error"),
                    Toast.LENGTH_LONG
                ).show()
                return@setOnClickListener
            }

            val input = currentInput
            if (input == null) {
                Toast.makeText(this, "input here", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            applyButton.isEnabled = false
            decoderOutputImage.setImageDrawable(null)

            lifecycleScope.launch {
                try {
                    val result = withContext(Dispatchers.Default) {
                        applicator.apply(input)
                    }
                    val bitmap = convertDecoderOutputToBitmap(result)
                    decoderOutputImage.setImageBitmap(bitmap)
                } catch (error: Exception) {
                    decoderOutputImage.setImageDrawable(null)
                    val message = error.localizedMessage ?: "decoder error"
                    Toast.makeText(
                        this@OnlyDecoderActivity,
                        "apply error " +  message,
                        Toast.LENGTH_LONG
                    ).show()
                } finally {
                    applyButton.isEnabled = true
                }
            }
        }
    }

    override fun onDestroy() {
        decoderApplicator?.close()
        decoderApplicator = null
        super.onDestroy()
    }

    private fun formatFloatArray(values: FloatArray): String {
        return values.joinToString(prefix = "[", postfix = "]", separator = ", ") {
            String.format(Locale.US, "%.4f", it)
        }
    }

    private fun convertDecoderOutputToBitmap(values: FloatArray): Bitmap {
        require(values.size == DECODER_OUTPUT_SIZE) {
            "Decoder output size ${values.size} does not match expected ${DECODER_OUTPUT_SIZE}"
        }

        val pixels = IntArray(DECODER_IMAGE_WIDTH * DECODER_IMAGE_HEIGHT)
        var index = 0
        for (position in pixels.indices) {
            val red = convertChannel(values[index++])
            val green = convertChannel(values[index++])
            val blue = convertChannel(values[index++])
            pixels[position] = Color.rgb(red, green, blue)
        }

        return Bitmap.createBitmap(
            pixels,
            DECODER_IMAGE_WIDTH,
            DECODER_IMAGE_HEIGHT,
            Bitmap.Config.ARGB_8888
        )
    }

    private fun convertChannel(value: Float): Int {
        val scaled = (value.coerceIn(0f, 1f) * 255f + 0.5f).toInt()
        return scaled.coerceIn(0, 255)
    }

    companion object {
        private const val INPUT_SIZE = 6272
        private const val RANDOM_BOUND = 10f
        private const val DECODER_IMAGE_WIDTH = 224
        private const val DECODER_IMAGE_HEIGHT = 224
        private const val DECODER_IMAGE_CHANNELS = 3
        private const val DECODER_OUTPUT_SIZE = DECODER_IMAGE_WIDTH * DECODER_IMAGE_HEIGHT * DECODER_IMAGE_CHANNELS
    }
}