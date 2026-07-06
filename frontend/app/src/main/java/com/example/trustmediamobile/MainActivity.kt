package com.example.trustmediamobile

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.trustmediamobile.ui.theme.TrustMediaMobileTheme
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import java.io.File
import java.io.FileOutputStream
import okhttp3.RequestBody.Companion.asRequestBody
import coil.compose.AsyncImage

class MainActivity : ComponentActivity() {

    private val viewModel: MainViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            TrustMediaMobileTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = Color.Transparent
                ) {
                    TrustMediaScreen(viewModel = viewModel)
                }
            }
        }
    }
}

enum class Mode { IMAGE, VIDEO }

class MainViewModel : ViewModel() {

    private val api = ApiService.create()

    var statusText by mutableStateOf("Select an image or video to analyze.")
        private set

    var currentMode by mutableStateOf(Mode.IMAGE)

    var selectedMediaUri by mutableStateOf<Uri?>(null)
        private set

    var serverStatusText by mutableStateOf("Flask server: checking...")
        private set

    var serverStatusColor by mutableStateOf(Color.Yellow)
        private set

    // ---------- NEW: history state ----------
    var historyItems = mutableStateListOf<HistoryItem>()
        private set

    var isHistoryLoading by mutableStateOf(false)
        private set

    init {
        startServerStatusLoop()
        // Optional: load history once on startup
        fetchHistory()
    }

    private fun startServerStatusLoop() {
        viewModelScope.launch {
            while (true) {
                try {
                    val resp = api.healthCheck()
                    if (resp.isSuccessful) {
                        serverStatusText = "Flask server: ON"
                        serverStatusColor = Color(0xFF4CAF50) // green
                    } else {
                        serverStatusText = "Flask server: OFF"
                        serverStatusColor = Color(0xFFF44336) // red
                    }
                } catch (e: Exception) {
                    serverStatusText = "Flask server: OFF"
                    serverStatusColor = Color(0xFFF44336) // red
                }

                kotlinx.coroutines.delay(5000L)
            }
        }
    }

    fun setMode(mode: Mode) {
        currentMode = mode
    }

    fun setSelectedUri(uri: Uri) {
        selectedMediaUri = uri
    }

    // ---------- NEW: fetch history from backend ----------
    fun fetchHistory() {
        viewModelScope.launch {
            isHistoryLoading = true
            try {
                val resp = api.getHistory(limit = 20)
                if (resp.success) {
                    historyItems.clear()
                    historyItems.addAll(resp.detections)
                }
            } catch (e: Exception) {
                // ignore for now or log if needed
            } finally {
                isHistoryLoading = false
            }
        }
    }

    fun uploadAndDetect(activity: Activity, uri: Uri) {
        statusText = "Uploading & analyzing..."
        viewModelScope.launch {
            try {
                setSelectedUri(uri)

                val file = copyUriToTempFile(activity, uri)
                val mime = activity.contentResolver.getType(uri) ?: "application/octet-stream"
                val requestBody = file.asRequestBody(mime.toMediaTypeOrNull())
                val part = MultipartBody.Part.createFormData("file", file.name, requestBody)

                when (currentMode) {
                    Mode.IMAGE -> {
                        val resp = api.detectImage(part)
                        if (resp.success) {
                            val domain = resp.type?.uppercase() ?: "UNKNOWN"
                            val scorePercent = (resp.fake_score ?: 0.0) * 100

                            val statusLine = when (resp.suspicion_level) {
                                "HIGH_SUSPICION" -> "Scan result: HIGH RISK of deepfake"
                                "MEDIUM_SUSPICION" -> "Scan result: MEDIUM RISK of deepfake"
                                "LOW_SUSPICION" -> "Scan result: LOW RISK of deepfake"
                                else -> "Scan result: UNKNOWN RISK"
                            }

                            val analysisLine =
                                "Analyzed $domain content for manipulation artifacts."

                            statusText =
                                "IMAGE · ${domain.uppercase()} ANALYSIS\n" +
                                        "$statusLine\n" +
                                        "Estimated fake probability: ${
                                            String.format(
                                                "%.1f",
                                                scorePercent
                                            )
                                        }%\n" +
                                        analysisLine

                            // refresh history after new scan
                            fetchHistory()
                        } else {
                            statusText = "Image detection failed."
                        }
                    }

                    Mode.VIDEO -> {
                        val resp = api.detectVideo(part)
                        if (resp.success) {
                            val scorePercent = (resp.fake_score ?: 0.0) * 100

                            val statusLine = when (resp.suspicion_level) {
                                "HIGH_SUSPICION" -> "Scan result: HIGH RISK of deepfake"
                                "MEDIUM_SUSPICION" -> "Scan result: MEDIUM RISK of deepfake"
                                "LOW_SUSPICION" -> "Scan result: LOW RISK of deepfake"
                                else -> "Scan result: UNKNOWN RISK"
                            }

                            val analysisLine =
                                "Analyzed VIDEO content for manipulation artifacts."

                            statusText =
                                "VIDEO ANALYSIS\n" +
                                        "$statusLine\n" +
                                        "Estimated fake probability: ${
                                            String.format(
                                                "%.1f",
                                                scorePercent
                                            )
                                        }%\n" +
                                        analysisLine

                            // refresh history after new scan
                            fetchHistory()
                        } else {
                            statusText = "Video detection failed."
                        }
                    }
                }

            } catch (e: Exception) {
                statusText = "Error: ${e.message}"
            }
        }
    }

    private fun copyUriToTempFile(activity: Activity, uri: Uri): File {
        val fileName = queryName(activity, uri)
        val inputStream = activity.contentResolver.openInputStream(uri)!!
        val file = File(activity.cacheDir, fileName)
        val outputStream = FileOutputStream(file)
        inputStream.copyTo(outputStream)
        inputStream.close()
        outputStream.close()
        return file
    }

    private fun queryName(activity: Activity, uri: Uri): String {
        var name = "upload_file"
        val cursor = activity.contentResolver.query(uri, null, null, null, null)
        cursor?.use {
            val nameIndex = it.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (it.moveToFirst() && nameIndex != -1) {
                name = it.getString(nameIndex)
            }
        }
        return name
    }
}

@Composable
fun TrustMediaScreen(viewModel: MainViewModel) {
    val context = LocalContext.current
    val activity = context as Activity

    var pendingMode by remember { mutableStateOf(Mode.IMAGE) }

    val pickFileLauncher =
        rememberLauncherForActivityResult(
            contract = ActivityResultContracts.StartActivityForResult()
        ) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                val uri = result.data?.data
                if (uri != null) {
                    viewModel.setMode(pendingMode)
                    viewModel.uploadAndDetect(activity, uri)
                }
            }
        }

    fun pickFile(mode: Mode) {
        pendingMode = mode
        val intent = Intent(Intent.ACTION_GET_CONTENT).apply {
            type = if (mode == Mode.IMAGE) "image/*" else "video/*"
        }
        pickFileLauncher.launch(Intent.createChooser(intent, "Select media"))
    }

    Box(
        modifier = Modifier.fillMaxSize()
    ) {
        // 1) Background image
        Image(
            painter = painterResource(id = R.drawable.bg_trustmedia),
            contentDescription = null,
            modifier = Modifier.fillMaxSize(),
            contentScale = ContentScale.Crop
        )

        // 2) Main content
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(0.dp),
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(24.dp),
                verticalArrangement = Arrangement.Top,
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Spacer(modifier = Modifier.height(32.dp))

                Text(
                    text = "TrustMedia Deepfake Detector",
                    style = MaterialTheme.typography.titleLarge,
                    modifier = Modifier.padding(bottom = 24.dp),
                    textAlign = TextAlign.Center,
                    color = Color.White
                )

                Button(
                    onClick = { pickFile(Mode.IMAGE) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 12.dp)
                ) {
                    Text(text = "Pick Image")
                }

                Button(
                    onClick = { pickFile(Mode.VIDEO) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 12.dp)
                ) {
                    Text(text = "Pick Video")
                }

                // ---------- NEW: Refresh history button ----------
                OutlinedButton(
                    onClick = { viewModel.fetchHistory() },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 16.dp)
                ) {
                    Text(text = if (viewModel.isHistoryLoading) "Loading history..." else "Refresh History")
                }

                // Preview selected media
                viewModel.selectedMediaUri?.let { uri ->
                    if (viewModel.currentMode == Mode.IMAGE) {
                        Text(
                            text = "Selected image:",
                            color = Color.White,
                            style = MaterialTheme.typography.bodySmall
                        )

                        AsyncImage(
                            model = uri,
                            contentDescription = "Selected image",
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(200.dp)
                                .padding(top = 8.dp),
                            contentScale = ContentScale.Crop
                        )
                    } else {
                        Text(
                            text = "Video selected (preview not shown).",
                            color = Color.White,
                            style = MaterialTheme.typography.bodySmall,
                            textAlign = TextAlign.Center
                        )
                    }

                    Spacer(modifier = Modifier.height(16.dp))
                }

                if (viewModel.statusText.isNotBlank()) {
                    ScanResultCard(statusText = viewModel.statusText)
                }

                Spacer(modifier = Modifier.height(16.dp))

                // ---------- NEW: History list ----------
                HistoryList(
                    items = viewModel.historyItems,
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f, fill = true)
                )
            }
        }

        // 3) Top-left server status
        Text(
            text = viewModel.serverStatusText,
            color = viewModel.serverStatusColor,
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(12.dp)
        )
    }
}

@Composable
fun ScanResultCard(statusText: String) {
    val riskColor = when {
        "HIGH RISK" in statusText -> Color(0xFFB00020)
        "MEDIUM RISK" in statusText -> Color(0xFFFFA000)
        "LOW RISK" in statusText -> Color(0xFF388E3C)
        else -> Color(0xFF757575)
    }

    val percent = Regex("""([0-9]+(\.[0-9]+)?)%""")
        .find(statusText)
        ?.value ?: ""

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = 8.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 6.dp),
        colors = CardDefaults.cardColors(
            containerColor = Color.Transparent,
            contentColor = Color.White
        )
    ) {
        Column(
            modifier = Modifier
                .padding(16.dp)
                .fillMaxWidth(),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Box(
                contentAlignment = Alignment.Center,
                modifier = Modifier
                    .size(96.dp)
                    .border(
                        width = 4.dp,
                        color = riskColor,
                        shape = CircleShape
                    )
            ) {
                Text(
                    text = percent,
                    style = MaterialTheme.typography.titleMedium,
                    color = riskColor
                )
            }

            Spacer(modifier = Modifier.height(12.dp))

            Text(
                text = statusText,
                textAlign = TextAlign.Center,
                style = MaterialTheme.typography.bodyMedium,
                color = Color.White
            )

            Spacer(modifier = Modifier.height(4.dp))

            Text(
                text = "This is a risk estimate. The model can be wrong.",
                textAlign = TextAlign.Center,
                style = MaterialTheme.typography.bodySmall,
                color = Color.Gray
            )
        }
    }
}

// ---------- NEW: History list composable ----------
@Composable
fun HistoryList(
    items: List<HistoryItem>,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier,
    ) {
        Text(
            text = "Recent scans",
            color = Color.White,
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.padding(bottom = 8.dp)
        )

        if (items.isEmpty()) {
            Text(
                text = "No scans yet.",
                color = Color.LightGray,
                style = MaterialTheme.typography.bodySmall
            )
        } else {
            LazyColumn {
                items(items) { item ->
                    HistoryRow(item = item)
                }
            }
        }
    }
}

@Composable
fun HistoryRow(item: HistoryItem) {
    val riskColor = when (item.suspicion_level) {
        "HIGH_SUSPICION" -> Color(0xFFB00020)
        "MEDIUM_SUSPICION" -> Color(0xFFFFA000)
        "LOW_SUSPICION" -> Color(0xFF388E3C)
        else -> Color(0xFF757575)
    }

    val scorePercent = item.fake_score * 100.0

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        colors = CardDefaults.cardColors(
            containerColor = Color(0x66000000),
            contentColor = Color.White
        )
    ) {
        Column(modifier = Modifier.padding(8.dp)) {
            Text(
                text = "${item.filename} (${item.type})",
                style = MaterialTheme.typography.bodyMedium,
                color = Color.White
            )
            Text(
                text = "Fake probability: ${String.format("%.1f", scorePercent)}%",
                style = MaterialTheme.typography.bodySmall,
                color = riskColor
            )
            Text(
                text = "Level: ${item.suspicion_level}",
                style = MaterialTheme.typography.bodySmall,
                color = Color.LightGray
            )
            Text(
                text = item.timestamp,
                style = MaterialTheme.typography.bodySmall,
                color = Color.Gray
            )
        }
    }
}
