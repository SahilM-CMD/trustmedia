package com.example.trustmediamobile

import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Response
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.GET
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import retrofit2.http.Query

data class DetectImageResponse(
    val success: Boolean,
    val type: String?,
    val fake_score: Double?,
    val suspicion_level: String?
)

data class DetectVideoResponse(
    val success: Boolean,
    val type: String?,
    val fake_score: Double?,
    val suspicion_level: String?
)

data class HistoryItem(
    val id: Int,
    val filename: String,
    val type: String,
    val fake_score: Double,
    val suspicion_level: String,
    val timestamp: String
)

data class HistoryResponse(
    val success: Boolean,
    val detections: List<HistoryItem>
)

interface ApiService {

    @GET("/health")
    suspend fun healthCheck(): Response<Unit>

    @Multipart
    @POST("/detect-image")
    suspend fun detectImage(
        @Part file: MultipartBody.Part
    ): DetectImageResponse

    @Multipart
    @POST("/detect-video")
    suspend fun detectVideo(
        @Part file: MultipartBody.Part
    ): DetectVideoResponse

    // ---------- NEW: detection history endpoint ----------
    @GET("/detection-history")
    suspend fun getHistory(
        @Query("limit") limit: Int = 20
    ): HistoryResponse

    companion object {
        fun create(): ApiService {
            val logging = HttpLoggingInterceptor().apply {
                level = HttpLoggingInterceptor.Level.BODY
            }
            val client = OkHttpClient.Builder()
                .addInterceptor(logging)
                .build()

            // If using the Android EMULATOR with Flask on same PC:
            //   baseUrl = "http://10.0.2.2:5000/"
            // If using a REAL PHONE on same Wi‑Fi:
            //   baseUrl = "http://<YOUR_PC_IP>:5000/"
            val baseUrl = "http://172.30.34.208:5000/"

            return Retrofit.Builder()
                .baseUrl(baseUrl)
                .client(client)
                .addConverterFactory(GsonConverterFactory.create())
                .build()
                .create(ApiService::class.java)
        }
    }
}
