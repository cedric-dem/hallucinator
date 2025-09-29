package com.example.hallucinator

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

class EncoderAndDecoderActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_decoder_and_encoder)

        setupBottomNavigation(R.id.navigation_encoder_decoder)
    }
}