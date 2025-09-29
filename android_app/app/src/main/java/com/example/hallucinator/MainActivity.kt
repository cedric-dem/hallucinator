package com.example.hallucinator

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {
    private var interpreter: Interpreter? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val greetingText: TextView = findViewById(R.id.greeting_text)
        greetingText.text = loadModelStatusMessage()

        setupBottomNavigation(R.id.navigation_main)
    }

    override fun onDestroy() {
        interpreter?.close()
        interpreter = null
        super.onDestroy()
    }

    private fun loadModelStatusMessage(): String {
        return try {
            interpreter = loadInterpreter()
            val inputShape = interpreter?.getInputTensor(0)?.shape()?.contentToString()
            val outputShape = interpreter?.getOutputTensor(0)?.shape()?.contentToString()
            getString(
                R.string.model_status_success,
                inputShape ?: getString(R.string.model_status_unknown_shape),
                outputShape ?: getString(R.string.model_status_unknown_shape)
            )
        } catch (error: IOException) {
            getString(R.string.model_status_error, error.localizedMessage ?: error.toString())
        } catch (error: IllegalArgumentException) {
            getString(R.string.model_status_error, error.localizedMessage ?: error.toString())
        }
    }

    private fun loadInterpreter(): Interpreter {
        val modelBytes = assets.open(MODEL_ASSET_NAME).use { inputStream ->
            inputStream.readBytes()
        }
        val buffer = ByteBuffer
            .allocateDirect(modelBytes.size)
            .order(ByteOrder.nativeOrder())
        buffer.put(modelBytes)
        buffer.rewind()
        return Interpreter(buffer)
    }

    companion object {
        private const val MODEL_ASSET_NAME = "model_encoder.tflite"
    }
}