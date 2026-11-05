<#
.SYNOPSIS
    Start (or stop) the whole OCR stack on Windows: Docker services, vLLM, backend,
    frontend, and Maker.

.DESCRIPTION
    Native PowerShell replacement for run_project.sh. The bash script only ever started
    the Python backend and assumed everything else was already up; it also had two
    Windows-specific defects that this script exists to fix:

      1. PORT CLEARING. run_project.sh used `lsof -ti :PORT`, which does not reliably see
         native Windows processes. Failed cleanups left dead sockets bound to :8000, and
         with several sockets on one port Windows answers requests inconsistently — the
         symptom is a server that passes /health while serving stale data from a process
         you thought you killed. This script uses Get-NetTCPConnection + Stop-Process, and
         reports any socket it cannot clear (an orphaned kernel entry needs a reboot).

      2. PATH TRANSLATION. Git Bash exports POSIX paths ("/d/OCR-data") but the backend is
         a native python.exe, and MSYS does not translate environment variables — Python
         resolved that to "D:\d\OCR-data", silently created it, and every Corpus preview
         404'd against an empty store. PowerShell passes native paths, so the class of bug
         cannot occur here.

    Also starts what run_project.sh never did: Docker Desktop, the pgvector database, the
    vLLM server, and the frontend container.

.PARAMETER Stop
    Stop the stack (backend, Maker, and this project's containers) and exit.

.PARAMETER SkipVllm
    Don't start vLLM. The Thinker SQL lane returns "analytical model unavailable"; OCR,
    Corpus, and the prose lane are unaffected. Frees ~19 GB VRAM for OCR.

.PARAMETER SkipMaker
    Don't start deckworks (:8002). Reader/Thinker/Corpus are unaffected.

.PARAMETER BackendPort
    Port for the backend API. Default 8000. Useful to sidestep a port left unusable by
    orphaned sockets — note the frontend build targets :8000, so a different port serves
    the API only (good for diagnostics, not for normal use).

.PARAMETER Force
    Start even when the backend port still carries an orphaned socket. Off by default
    because that state produces a server which passes /health while answering some
    requests from a dead process — a failure that looks like corrupted data, not a port
    problem. Use only if you know the orphan is harmless.

.PARAMETER ModelsDir
    Override the weights directory. Default: <repo_parent>\OCR-models

.PARAMETER DataDir
    Override the runtime data directory. Default: <repo_parent>\OCR-data

.EXAMPLE
    .\run_project.ps1
.EXAMPLE
    .\run_project.ps1 -SkipVllm       # OCR only, maximum VRAM headroom
.EXAMPLE
    .\run_project.ps1 -Stop
.EXAMPLE
    .\run_project.ps1 -BackendPort 8010    # diagnose around a poisoned :8000
#>

[CmdletBinding()]
param(
    [switch]$Stop,
    [switch]$SkipVllm,
    [switch]$SkipMaker,
    [switch]$Force,
    [int]$BackendPort = 8000,
    [string]$ModelsDir,
    [string]$DataDir
)

$ErrorActionPreference = 'Stop'

# ── Layout ───────────────────────────────────────────────────────────────────
$ProjectDir    = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectParent = Split-Path -Parent $ProjectDir
$VenvPy        = Join-Path $ProjectDir ".venv\Scripts\python.exe"

# Weights and runtime data live OUTSIDE the repo so it stays copyable (117 GB -> ~5 GB).
# Defaults are siblings of the project dir, derived from its own location rather than a
# hardcoded D:\ — a moved or renamed checkout still resolves. See backend/ocr_paths.py.
if (-not $ModelsDir) { $ModelsDir = Join-Path $ProjectParent "OCR-models" }
if (-not $DataDir)   { $DataDir   = Join-Path $ProjectParent "OCR-data" }
$LogDir = Join-Path $DataDir "logs"

$FrontendPort  = 3001
$VllmPort      = 8001
$DeckworksPort = 8002

# Container names as created previously; this script starts them, never re-creates them.
$PgContainer       = "ocr-rag-pg"
$VllmContainer     = "vllm-qwen3"
$FrontendContainer = "ocr-frontend-dev"

$PopplerBin = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages\oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe\poppler-25.07.0\Library\bin"
$DockerExe  = Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\Docker Desktop.exe"

function Invoke-Native {
    <#
      Run a native .exe and return its exit code, without letting its stderr become a
      PowerShell error.

      Windows PowerShell 5.1 wraps every stderr line from a native command in an
      ErrorRecord when you pipe `2>&1`, and with $ErrorActionPreference='Stop' that turns
      a harmless warning into a fatal script error. `docker info` emits
      "WARNING: No blkio throttle.read_bps_device support" on this machine and exits 0 —
      which killed this script until this helper existed. Send stderr to $null (NOT to the
      success stream) and judge the command by $LASTEXITCODE alone.
    #>
    param([Parameter(Mandatory)][string]$Exe, [string[]]$Args = @(), [string]$LogFile)

    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        if ($LogFile) { & $Exe @Args 2>$null | Out-File -FilePath $LogFile -Append -Encoding utf8 }
        else          { & $Exe @Args 2>$null | Out-Null }
        return $LASTEXITCODE
    } finally { $ErrorActionPreference = $prev }
}

function Write-Step($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Write-Ok($m)   { Write-Host "    OK   $m" -ForegroundColor Green }
function Write-Warn2($m){ Write-Host "    WARN $m" -ForegroundColor Yellow }
function Write-Err($m)  { Write-Host "    FAIL $m" -ForegroundColor Red }

# ── Port handling ────────────────────────────────────────────────────────────
function Clear-Port {
    <#
      Free a TCP port, and be honest when it cannot be freed.

      Distinguishes a live process (killable) from an orphaned kernel socket — a listener
      whose owning process no longer exists. Windows keeps those until reboot, and because
      several sockets can sit on one port, an orphan alongside a healthy server makes
      request routing nondeterministic. Silently ignoring that is what made the original
      bash version so confusing to debug, so we surface it loudly instead.
    #>
    param([int]$Port)

    $conns = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
    if (-not $conns) { return $true }

    foreach ($ownerPid in ($conns.OwningProcess | Select-Object -Unique)) {
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$ownerPid" -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "    clearing :$Port (PID $ownerPid)"
            Stop-Process -Id $ownerPid -Force -ErrorAction SilentlyContinue
        } else {
            Write-Warn2 ":$Port has an ORPHANED socket (PID $ownerPid, process already gone)"
        }
    }
    Start-Sleep -Seconds 2

    $left = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
    if (-not $left) { return $true }

    $orphans = @($left.OwningProcess | Select-Object -Unique |
                 Where-Object { -not (Get-CimInstance Win32_Process -Filter "ProcessId=$_" -ErrorAction SilentlyContinue) })
    if ($orphans.Count -gt 0) {
        Write-Err ":$Port still bound by orphaned socket(s): $($orphans -join ', ')"
        Write-Host "         These survive until REBOOT. A server started now may answer" -ForegroundColor Red
        Write-Host "         inconsistently (healthy /health, stale data). Reboot to clear." -ForegroundColor Red
        return $false
    }
    return $true
}

function Wait-Http {
    param([string]$Url, [int]$TimeoutSec = 180, [string]$Label = "service")
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -Uri $Url -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
            if ($r.StatusCode -eq 200) { return $true }
        } catch { }
        Start-Sleep -Seconds 3
    }
    return $false
}

function Test-Container {
    param([string]$Name)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'SilentlyContinue'
    try {
        $s = & docker inspect -f '{{.State.Status}}' $Name 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $s) { return "missing" }
        return $s.Trim()
    } catch {
        return "missing"
    } finally {
        $ErrorActionPreference = $prev
    }
}

function Start-ContainerIfNeeded {
    param([string]$Name, [string]$Label)
    $state = Test-Container $Name
    switch ($state) {
        "running" { Write-Ok "$Label already running"; return $true }
        "missing" { Write-Warn2 "$Label container '$Name' does not exist - skipping"; return $false }
        default {
            Write-Host "    starting $Label ($Name)..."
            & docker start $Name | Out-Null
            if ($LASTEXITCODE -eq 0) { Write-Ok "$Label started"; return $true }
            Write-Err "$Label failed to start"; return $false
        }
    }
}

# ── Stop mode ────────────────────────────────────────────────────────────────
if ($Stop) {
    Write-Step "Stopping OCR stack"
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*server_with_task_management*" -or $_.CommandLine -like "*deckworks*" } |
        ForEach-Object { Write-Host "    kill PID $($_.ProcessId)"; Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 3
    foreach ($p in @($BackendPort, $DeckworksPort)) { Clear-Port $p | Out-Null }
    foreach ($c in @($FrontendContainer, $VllmContainer)) {
        if ((Test-Container $c) -eq "running") { Write-Host "    stopping $c"; & docker stop $c | Out-Null }
    }
    Write-Host ""
    Write-Ok "Stopped. Postgres ($PgContainer) left running - it holds the corpus."
    Write-Host "    Stop it too with:  docker stop $PgContainer"
    exit 0
}

# ── Banner ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "==================================================" -ForegroundColor White
Write-Host "  OCR PROJECT - $ProjectDir" -ForegroundColor White
Write-Host "==================================================" -ForegroundColor White
Write-Host "  models : $ModelsDir"
Write-Host "  data   : $DataDir"
Write-Host "  logs   : $LogDir"
Write-Host ""

# ── Preflight ────────────────────────────────────────────────────────────────
Write-Step "Preflight"

if (-not (Test-Path $VenvPy)) {
    Write-Err "venv Python not found: $VenvPy"
    Write-Host "         Create it with:  python -m venv .venv; .\.venv\Scripts\pip install -r backend\requirements.txt"
    exit 1
}
Write-Ok "venv Python"

if (-not (Test-Path (Join-Path $ProjectDir "backend\server_with_task_management.py"))) {
    Write-Err "backend not found under $ProjectDir\backend"; exit 1
}
Write-Ok "backend source"

# Fail fast and specifically: a missing weights dir otherwise surfaces as an opaque
# import-time FileNotFoundError inside the backend log.
if (-not (Test-Path $ModelsDir)) {
    Write-Err "models directory not found: $ModelsDir"
    Write-Host "         Weights live OUTSIDE the repo. Point -ModelsDir at them, or place"
    Write-Host "         them at the default location. See documentation\README.md."
    exit 1
}
$missing = @()
foreach ($m in @("models--allenai--olmOCR-2-7B-1025", "models--Qwen--Qwen2.5-VL-7B-Instruct",
                 "table-structure-recognition", "table-transformer-detection")) {
    if (-not (Test-Path (Join-Path $ModelsDir $m))) { $missing += $m }
}
if ($missing.Count -gt 0) {
    Write-Err "models directory is missing: $($missing -join ', ')"
    exit 1
}
Write-Ok "all 4 model directories present"

foreach ($d in @($DataDir, $LogDir, (Join-Path $DataDir "temp"))) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}
Write-Ok "data directories"

# pdftoppm renders every page; without it OCR fails per-page with WinError 2.
if (Test-Path $PopplerBin) {
    if ($env:Path -notlike "*$PopplerBin*") { $env:Path = "$env:Path;$PopplerBin" }
    Write-Ok "poppler (pdftoppm)"
} elseif (Get-Command pdftoppm -ErrorAction SilentlyContinue) {
    Write-Ok "poppler (pdftoppm, already on PATH)"
} else {
    Write-Err "poppler not found - PDF rendering will fail on every page"
    Write-Host "         Install with:  winget install oschwartz10612.Poppler"
    exit 1
}

# ── Docker ───────────────────────────────────────────────────────────────────
Write-Step "Docker services"

$dockerRc = Invoke-Native "docker" @("info")
if ($dockerRc -ne 0) {
    if (Test-Path $DockerExe) {
        Write-Host "    Docker engine not responding - launching Docker Desktop..."
        Start-Process $DockerExe
        $ok = $false
        foreach ($i in 1..40) {
            Start-Sleep -Seconds 5
            if ((Invoke-Native "docker" @("info")) -eq 0) { $ok = $true; break }
        }
        if ($ok) { Write-Ok "Docker engine ready" }
        else { Write-Err "Docker did not come up in 200s"; exit 1 }
    } else {
        Write-Err "Docker engine not running and Docker Desktop not found at $DockerExe"; exit 1
    }
} else {
    Write-Ok "Docker engine ready"
}

# Postgres holds the corpus; without it ingest is disabled and the Corpus tab is empty.
if (-not (Start-ContainerIfNeeded $PgContainer "Postgres/pgvector")) {
    Write-Err "Postgres unavailable - corpus and RAG lanes will not work"
}

if (-not $SkipVllm) {
    if (Start-ContainerIfNeeded $VllmContainer "vLLM (Qwen3-14B-AWQ)") {
        Write-Host "    waiting for vLLM to load weights (~2 min on first start)..."
        if (Wait-Http "http://localhost:$VllmPort/v1/models" 300 "vLLM") { Write-Ok "vLLM serving on :$VllmPort" }
        else { Write-Warn2 "vLLM not ready yet - SQL lane will error until it is (docker logs $VllmContainer)" }
    }
} else {
    Write-Warn2 "vLLM skipped (-SkipVllm): Thinker SQL lane will report 'model unavailable'"
}

Start-ContainerIfNeeded $FrontendContainer "Frontend (:$FrontendPort)" | Out-Null

# ── Environment ──────────────────────────────────────────────────────────────
# Native Windows paths throughout - no MSYS translation hazard (see header).
$env:OCR_MODELS_DIR      = $ModelsDir
$env:OCR_DATA_DIR        = $DataDir
# The backend reads its listen port from OCR_PORT (server_with_task_management.py), so
# -BackendPort is inert unless it is exported here — without this the process silently
# binds 8000 while every health check polls the port you asked for.
$env:OCR_PORT            = "$BackendPort"
$env:PYTHONIOENCODING    = "utf-8"
$env:RAG_INGEST_ENABLED  = "1"
$env:RAG_LANES           = "both"
$env:RAG_DATABASE_URL    = "postgresql://ocr:ocrpass@localhost:5432/ocr_rag"
$env:RAG_DATABASE_URL_RO = "postgresql://ocr_ro:ocrpass_ro@localhost:5432/ocr_rag"
$env:RAG_LLM_URL            = if ($SkipVllm) { "" } else { "http://localhost:$VllmPort/v1" }
$env:RAG_LLM_MODEL          = "qwen3-14b-awq"
$env:RAG_SQL_LOCAL_FALLBACK = "1"
# Maker (deckworks) reuses the SAME vLLM server - it must never load its own GPU model.
$env:AI_API_KEY          = "local"
$env:AI_BASE_URL         = "http://localhost:$VllmPort/v1"
$env:AI_MODEL            = "qwen3-14b-awq"
$env:OCR_API_BASE        = "http://localhost:$BackendPort"

# With vLLM absent, 'both' makes every Thinker answer carry a SQL error even when the
# prose lane answered fine; 'prose' degrades cleanly instead.
if ($SkipVllm) { Write-Host "    RAG_LANES=prose (no vLLM)" }

# ── Backend ──────────────────────────────────────────────────────────────────
Write-Step "Backend (:$BackendPort)"

if (-not (Clear-Port $BackendPort)) {
    if ($Force) {
        Write-Warn2 "-Force given: starting anyway. Treat any odd/stale response with suspicion."
    } else {
        Write-Host ""
        Write-Err "Refusing to start: :$BackendPort has an orphaned socket."
        Write-Host "         Starting anyway would give a server that passes /health while"
        Write-Host "         answering some requests from a dead process - it looks like data"
        Write-Host "         corruption, not a port fault. Options:"
        Write-Host "           - reboot (only way to clear an orphaned socket), then re-run"
        Write-Host "           - .\run_project.ps1 -BackendPort 8010   (API only; UI targets 8000)"
        Write-Host "           - .\run_project.ps1 -Force              (accept the risk)"
        exit 1
    }
}

$backendLog = Join-Path $LogDir "backend_run.log"
$backend = Start-Process -FilePath $VenvPy `
    -ArgumentList "backend\server_with_task_management.py" `
    -WorkingDirectory $ProjectDir `
    -RedirectStandardOutput $backendLog `
    -RedirectStandardError (Join-Path $LogDir "backend_err.log") `
    -PassThru -WindowStyle Hidden
Write-Host "    backend PID $($backend.Id)  (log: $backendLog)"

Write-Host "    waiting for models to load (~1-2 min cold)..."
if (Wait-Http "http://localhost:$BackendPort/health" 300 "backend") {
    Write-Ok "backend healthy on :$BackendPort"
} else {
    Write-Err "backend did not become healthy"
    Write-Host "    --- last 20 log lines ---"
    if (Test-Path $backendLog) { Get-Content $backendLog -Tail 20 | ForEach-Object { "      $_" } }
    if (Test-Path (Join-Path $LogDir "backend_err.log")) { Get-Content (Join-Path $LogDir "backend_err.log") -Tail 20 | ForEach-Object { "      $_" } }
    exit 1
}

# ── Maker (deckworks) ────────────────────────────────────────────────────────
$deckworksDir = Join-Path $ProjectDir "deckworks"
if (-not $SkipMaker -and (Test-Path $deckworksDir) -and (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Step "Maker / deckworks (:$DeckworksPort)"
    Clear-Port $DeckworksPort | Out-Null
    $dwLog = Join-Path $LogDir "deckworks_run.log"
    Push-Location $deckworksDir
    try {
        Invoke-Native "uv" @("sync") -LogFile $dwLog | Out-Null
        # The OCR frontend embeds this build in an iframe, so a dev server isn't needed.
        if (-not (Test-Path (Join-Path $deckworksDir "frontend\dist\index.html"))) {
            Write-Host "    building Maker console (first run only)..."
            Push-Location (Join-Path $deckworksDir "frontend")
            Invoke-Native "npm" @("install")   -LogFile $dwLog | Out-Null
            Invoke-Native "npm" @("run","build") -LogFile $dwLog | Out-Null
            Pop-Location
        }
        $dw = Start-Process -FilePath "uv" `
            -ArgumentList "run","uvicorn","deckworks.api.app:app","--host","0.0.0.0","--port","$DeckworksPort" `
            -WorkingDirectory $deckworksDir -PassThru -WindowStyle Hidden
        Write-Host "    maker PID $($dw.Id)  (log: $dwLog)"
        if (Wait-Http "http://localhost:$DeckworksPort/api/health" 60 "maker") { Write-Ok "Maker ready" }
        else { Write-Warn2 "Maker not ready - Reader/Thinker/Corpus still work (see $dwLog)" }
    } finally { Pop-Location }
} elseif ($SkipMaker) {
    Write-Warn2 "Maker skipped (-SkipMaker)"
} else {
    Write-Warn2 "deckworks/ or uv missing - skipping Maker"
}

# ── Summary ──────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "==================================================" -ForegroundColor White
Write-Host "  Stack status" -ForegroundColor White
Write-Host "==================================================" -ForegroundColor White

function Report($label, $url) {
    try {
        $r = Invoke-WebRequest -Uri $url -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
        Write-Host ("  {0,-22} UP    {1}" -f $label, $url) -ForegroundColor Green
    } catch {
        Write-Host ("  {0,-22} DOWN  {1}" -f $label, $url) -ForegroundColor Yellow
    }
}
Report "Frontend"        "http://localhost:$FrontendPort"
Report "Backend"         "http://localhost:$BackendPort/health"
Report "vLLM"            "http://localhost:$VllmPort/v1/models"
Report "Maker"           "http://localhost:$DeckworksPort/api/health"

# Corpus size is the fastest signal that Postgres AND the external data dir are wired up.
try {
    $docs = Invoke-WebRequest -Uri "http://localhost:$BackendPort/rag/documents" -TimeoutSec 10 -UseBasicParsing -ErrorAction Stop
    $n = ($docs.Content | ConvertFrom-Json).documents.Count
    Write-Host ("  {0,-22} {1} document(s)" -f "Corpus", $n) -ForegroundColor Green
} catch {
    Write-Host ("  {0,-22} unavailable" -f "Corpus") -ForegroundColor Yellow
}

Write-Host ""
Write-Host "  Open:  http://localhost:$FrontendPort"
Write-Host "  Stop:  .\run_project.ps1 -Stop"
Write-Host ""

if ($backend -and (-not $backend.HasExited)) {
    Write-Host "  Stack running. Press Ctrl+C or run .\run_project.ps1 -Stop to terminate."
    try {
        Wait-Process -Id $backend.Id
    } catch { }
}
