<#
.SYNOPSIS
  Run every test suite (calculator + all labs) and print a pass/fail table.

.DESCRIPTION
  Each suite runs in its own pytest process so a missing dependency shows up as an ERROR
  row instead of aborting the whole run. Use -Python for a specific interpreter and -Only
  to filter suites by substring.

.EXAMPLE
  ./scripts/run-all-tests.ps1
  ./scripts/run-all-tests.ps1 -Only cs149
#>
param(
    [string]$Python = "python",
    [string]$Only = ""
)

$root = Split-Path -Parent $PSScriptRoot
$suites = [ordered]@{
    "tools mlsys_calc"              = "tools/tests"
    "cs149 asst1 cpu-parallel"      = "solutions/cs149/asst1-cpu-parallel/tests"
    "cs149 asst2 task-scheduler"    = "solutions/cs149/asst2-task-scheduler/tests"
    "cs149 asst3 scan-render"       = "solutions/cs149/asst3-scan-render/tests"
    "cs149 asst4 dnn-accelerator"   = "solutions/cs149/asst4-dnn-accelerator/tests"
    "cs149 asst5 big-graphs"        = "solutions/cs149/asst5-big-graphs/tests"
    "cs229s project1 kernels"       = "solutions/cs229s/project1-transformer-kernels/tests"
    "cs229s project2 kv-spec"       = "solutions/cs229s/project2-kv-speculative/tests"
    "cmu15849 distributed-training" = "solutions/cmu15849/distributed-training/tests"
    "cmu15849 serving-memory"       = "solutions/cmu15849/serving-memory/tests"
    "cmu15849 compiler-compression" = "solutions/cmu15849/compiler-compression/tests"
}

$rows = @()
foreach ($name in $suites.Keys) {
    if ($Only -and -not $name.Contains($Only)) { continue }
    $dir = Join-Path $root $suites[$name]
    if (-not (Test-Path $dir)) { $rows += [pscustomobject]@{Suite=$name; Result="MISSING"; Summary=""}; continue }
    Push-Location (Split-Path $dir)
    $out = & $Python -m pytest -q -p no:cacheprovider (Split-Path -Leaf $dir) 2>&1 | Out-String
    $code = $LASTEXITCODE
    Pop-Location
    $summary = ($out -split "`n" | Where-Object { $_ -match "passed|failed|error" } | Select-Object -Last 1)
    $summary = if ($summary) { $summary.Trim() } else { "" }
    $result = if ($code -eq 0) { "PASS" } elseif ($out -match "ModuleNotFoundError|ImportError") { "ERROR (deps)" } else { "FAIL" }
    $rows += [pscustomobject]@{Suite=$name; Result=$result; Summary=$summary}
}
$rows | Format-Table -AutoSize
if ($rows.Result -contains "FAIL" -or $rows.Result -contains "MISSING") { exit 1 }
