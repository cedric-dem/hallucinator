package com.example.hallucinator

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val greetingText: TextView = findViewById(R.id.greeting_text)
        greetingText.text = getString(R.string.greeting_message, "Android")
    }
}