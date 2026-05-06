# Launch the Streamlit app and open it in Chrome (not the OS-default browser).
# Usage:  .\launch.ps1
$ErrorActionPreference = "Stop"

$port = 8501
$url  = "http://localhost:$port"

# Find Chrome on this machine.
$candidates = @(
    "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
    "$env:LocalAppData\Google\Chrome\Application\chrome.exe"
)
$chrome = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $chrome) {
    Write-Warning "Chrome not found in standard locations. Falling back to default browser."
}

# Start Streamlit headless (won't auto-open the OS-default browser).
$streamlitArgs = @(
    "-m", "streamlit", "run", "app.py",
    "--server.headless", "true",
    "--server.port", $port,
    "--browser.gatherUsageStats", "false"
)
$proc = Start-Process -FilePath ".\.venv\Scripts\python.exe" `
                      -ArgumentList $streamlitArgs `
                      -PassThru -NoNewWindow

# Wait until the server is responding before opening the browser.
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Milliseconds 500
    try {
        $r = Invoke-WebRequest -Uri $url -TimeoutSec 1 -UseBasicParsing -ErrorAction Stop
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
}
if (-not $ready) {
    Write-Warning "Streamlit did not respond within 15s; opening anyway."
}

if ($chrome) {
    Start-Process -FilePath $chrome -ArgumentList $url
} else {
    Start-Process $url
}

Write-Host ""
Write-Host "Streamlit running at $url"
Write-Host "Press Ctrl+C in this window to stop the server."
Wait-Process -Id $proc.Id
