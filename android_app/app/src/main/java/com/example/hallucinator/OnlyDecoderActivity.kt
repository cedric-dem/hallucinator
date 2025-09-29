package com.example.hallucinator

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

class OnlyDecoderActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_only_decoder)

        setupBottomNavigation(R.id.navigation_only_decoder)
    }
}