package com.example.hallucinator

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.example.EncoderApplicator
import java.io.IOException
import kotlin.math.min

class OnlyEncoderActivity : AppCompatActivity() {
    private var encoderApplicator: EncoderApplicator? = null
    private lateinit var imagePreview: ImageView
    private lateinit var selectImageButton: Button
    private lateinit var applyModelButton: Button
    private lateinit var modelOutputText: TextView

    private var modelInputBitmap: Bitmap? = null

    private val selectImageLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            if (uri == null) {
                return@registerForActivityResult
            }

            try {
                val originalBitmap = loadBitmapFromUri(uri)
                if (originalBitmap != null) {
                    val resizedBitmap = Bitmap.createScaledBitmap(
                        originalBitmap,
                        ModelConfig.MODEL_IMAGE_SIZE,
                        ModelConfig.MODEL_IMAGE_SIZE,
                        true
                    )
                    if (resizedBitmap != originalBitmap) {
                        originalBitmap.recycle()
                    }
                    modelInputBitmap?.recycle()
                    modelInputBitmap = resizedBitmap
                    imagePreview.setImageBitmap(resizedBitmap)
                    modelOutputText.text = "Apply the model to see output"
                    applyModelButton.isEnabled = true
                } else {
                    showError("Error 404")
                }
            } catch (error: IOException) {
                showError("Error 403 " + error.toString())
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_only_encoder)

        imagePreview = findViewById(R.id.image_preview)
        selectImageButton = findViewById(R.id.button_select_image)
        applyModelButton = findViewById(R.id.button_apply_model)
        modelOutputText = findViewById(R.id.text_model_output)

        selectImageButton.setOnClickListener { selectImageLauncher.launch("image/*") }
        applyModelButton.setOnClickListener { applyModel() }

        setupBottomNavigation(R.id.navigation_main)
    }

    override fun onDestroy() {
        encoderApplicator?.close()
        encoderApplicator = null
        modelInputBitmap?.recycle()
        modelInputBitmap = null
        super.onDestroy()
    }

    private fun applyModel() {
        val bitmap = modelInputBitmap
        if (bitmap == null) {
            modelOutputText.text = "Bitmap is null"
            return
        }

        try {
            val applicator = encoderApplicator ?: EncoderApplicator(this).also { encoderApplicator = it }
            val inputShape = applicator.inputShape
            val inputTensor = createModelInputFromBitmap(bitmap, inputShape)
            val output = applicator.apply(inputTensor)
            val formattedOutput = output.toString()
            modelOutputText.text = formattedOutput
            logOutputPreview(output)
        } catch (error: Exception) {
            showError("Error 402 " + error.toString())
        }
    }

    private fun createModelInputFromBitmap(bitmap: Bitmap, inputShape: IntArray): FloatArray {
        val expectedSize = inputShape.fold(1) { acc, dimension -> acc * dimension }
        require(expectedSize > 0) { "Invalid input shape: ${inputShape.contentToString()}" }

        val batchSize = inputShape.firstOrNull() ?: 1
        require(batchSize == 1) { "Only batch size of 1 is supported but was $batchSize" }

        val (height, width, channels, channelFirst) = when {
            inputShape.size == 4 && inputShape[3] == 3 -> Quadruple(inputShape[1], inputShape[2], inputShape[3], false)
            inputShape.size == 4 && inputShape[1] == 3 -> Quadruple(inputShape[2], inputShape[3], inputShape[1], true)
            inputShape.size == 3 && inputShape[2] == 3 -> Quadruple(inputShape[0], inputShape[1], inputShape[2], false)
            inputShape.size == 3 && inputShape[0] == 3 -> Quadruple(inputShape[1], inputShape[2], inputShape[0], true)
            else -> throw IllegalArgumentException("Unsupported input shape: ${inputShape.contentToString()}")
        }

        require(channels == 3) { "Expected 3 channels but was $channels" }
        require(height == ModelConfig.MODEL_IMAGE_SIZE && width == ModelConfig.MODEL_IMAGE_SIZE) {
            "Model expects $height x $width but activity resizes to $ModelConfig.MODEL_IMAGE_SIZE x $ModelConfig.MODEL_IMAGE_SIZE"
        }

        val pixels = IntArray(ModelConfig.MODEL_IMAGE_SIZE * ModelConfig.MODEL_IMAGE_SIZE)
        bitmap.getPixels(pixels, 0, ModelConfig.MODEL_IMAGE_SIZE, 0, 0, ModelConfig.MODEL_IMAGE_SIZE, ModelConfig.MODEL_IMAGE_SIZE)

        val output = FloatArray(expectedSize)
        val normalizationFactor = 1f / 255f
        var pixelIndex = 0

        if (channelFirst) {
            val channelSize = ModelConfig.MODEL_IMAGE_SIZE * ModelConfig.MODEL_IMAGE_SIZE
            for (y in 0 until ModelConfig.MODEL_IMAGE_SIZE) {
                for (x in 0 until ModelConfig.MODEL_IMAGE_SIZE) {
                    val pixel = pixels[pixelIndex++]
                    val r = ((pixel shr 16) and 0xFF) * normalizationFactor
                    val g = ((pixel shr 8) and 0xFF) * normalizationFactor
                    val b = (pixel and 0xFF) * normalizationFactor

                    val baseIndex = y * ModelConfig.MODEL_IMAGE_SIZE + x
                    output[baseIndex] = r
                    output[channelSize + baseIndex] = g
                    output[channelSize * 2 + baseIndex] = b
                }
            }
        } else {
            var outputIndex = 0
            for (y in 0 until ModelConfig.MODEL_IMAGE_SIZE) {
                for (x in 0 until ModelConfig.MODEL_IMAGE_SIZE) {
                    val pixel = pixels[pixelIndex++]
                    output[outputIndex++] = ((pixel shr 16) and 0xFF) * normalizationFactor
                    output[outputIndex++] = ((pixel shr 8) and 0xFF) * normalizationFactor
                    output[outputIndex++] = (pixel and 0xFF) * normalizationFactor
                }
            }
        }

        return output
    }

    private fun loadBitmapFromUri(uri: Uri): Bitmap? {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val source = ImageDecoder.createSource(contentResolver, uri)
            ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                decoder.isMutableRequired = true
            }
        } else {
            @Suppress("DEPRECATION")
            MediaStore.Images.Media.getBitmap(contentResolver, uri)
        }
    }

    private fun showError(message: String) {
        modelOutputText.text = message
        Log.e(TAG, message)
    }

    private fun setOutput(message: String) {
        modelOutputText.text = message
    }


    private fun logOutputPreview(output: FloatArray) {

        val previewLength = min(100, output.size)
        val previewBuilder = StringBuilder(previewLength * 8 + 3)
        previewBuilder.append('[')
        for (index in 0 until previewLength) {
            if (index > 0) {
                previewBuilder.append(", ")
            }
            previewBuilder.append(output[index])
        }
        if (output.size > previewLength) {
            previewBuilder.append(", …")
        }
        previewBuilder.append(']')

        val output_text ="output is size "+output.size.toString() + "\nboundaries  : "+ output.minOrNull()+  " and "+ output.maxOrNull() +  "\noutput preview : "+previewBuilder.toString()
        Log.d(TAG, output_text)
        setOutput(output_text)
    }

    companion object {
        private const val TAG = "OnlyEncoderActivity"
    }
}

private data class Quadruple(
    val height: Int,
    val width: Int,
    val channels: Int,
    val channelFirst: Boolean
)