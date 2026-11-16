param(
 [string]$ProductionConfig = "$PSScriptRoot\config\production-pipeline.json",
 [ValidateRange(20,300)][int]$Seconds = 300,
 [switch]$Headless
)
$ErrorActionPreference = "Stop"
$c = Get-Content -Raw $ProductionConfig | ConvertFrom-Json
if ($c.state -ne "active") { throw "Production selection is not active" }
if ($c.receiver -ne "172.21.30.230" -or $c.video_port -ne 5010 -or $c.metadata_port -ne 5012) { throw "Selected receiver/ports differ from this approved runtime" }
$package = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot $c.package))
$python = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot $c.python))
$credentials = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot $c.credentials))
$manifest = Join-Path $package "MANIFEST.json"
if ((Get-FileHash -Algorithm SHA256 $manifest).Hash.ToLower() -ne $c.manifest_sha256) { throw "Selected production manifest differs" }
$m = Get-Content -Raw $manifest | ConvertFrom-Json
if ($m.package_id -ne $c.release_id -or $m.release_channel -ne "production" -or !$m.feature_roi -or $m.cpp_input_kernel -ne "parallel8" -or $m.cpp_library_sha256 -ne $c.cpp_library_sha256) { throw "Selected production runtime/flags differ" }
$runArgs = @{ Python = $python; Credentials = $credentials; Seconds = $Seconds }
if ($Headless) { $runArgs.Headless = $true }
& (Join-Path $package "run.ps1") @runArgs
if ($LASTEXITCODE -ne 0) { throw "Production launch failed; no automatic fallback or other-process restart attempted" }
