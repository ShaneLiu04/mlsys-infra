<#
.SYNOPSIS
  Structural checks: notes validator, relative links in every Markdown file, lab layout,
  LaTeX reports (PDF present, figures resolve, every reference cited), and a scan for
  secrets that must never be committed.
#>
param([string]$Python = "python")

$root = Split-Path -Parent $PSScriptRoot
$fail = 0

Write-Host "== notes =="
& $Python (Join-Path $root "notes/check_notes.py")
if ($LASTEXITCODE -ne 0) { $fail++ }

Write-Host "== markdown links =="
$broken = @()
Get-ChildItem $root -Recurse -Filter *.md -File |
    Where-Object { $_.FullName -notmatch "[\\/](assignments|templates|\.git|\.pytest_cache)[\\/]" } |
    ForEach-Object {
        $file = $_
        $text = Get-Content $file.FullName -Raw -Encoding utf8
        foreach ($m in [regex]::Matches($text, '\]\(([^)#\s]+)(#[^)]*)?\)')) {
            $target = $m.Groups[1].Value
            if ($target -match '^(https?|mailto):') { continue }
            $resolved = Join-Path $file.DirectoryName ([uri]::UnescapeDataString($target))
            if (-not (Test-Path $resolved)) { $broken += "$($file.FullName.Substring($root.Length + 1)) -> $target" }
        }
    }
if ($broken.Count) { $broken | ForEach-Object { Write-Host "  broken: $_" }; $fail++ } else { Write-Host "  OK" }

Write-Host "== lab layout =="
$labs = Get-ChildItem (Join-Path $root "solutions") -Directory | ForEach-Object { Get-ChildItem $_.FullName -Directory }
foreach ($lab in $labs) {
    foreach ($need in "README.md", "tests", "scripts/run_experiments.py", "artifacts/raw", "artifacts/figures") {
        if (-not (Test-Path (Join-Path $lab.FullName $need))) {
            Write-Host "  $($lab.Name): missing $need"; $fail++
        }
    }
}
Write-Host "  $($labs.Count) labs checked"
$commonH = "solutions/cs149/asst1-cpu-parallel/cpp/common/common.h", "solutions/cs149/asst2-task-scheduler/cpp/common/common.h" |
    ForEach-Object { (Get-FileHash (Join-Path $root $_)).Hash }
if ($commonH[0] -ne $commonH[1]) { Write-Host "  cs149 asst1/asst2 cpp/common/common.h differ (keep the copies identical)"; $fail++ }

Write-Host "== reports =="
& $Python (Join-Path $root "scripts/check_reports.py")
if ($LASTEXITCODE -ne 0) { $fail++ }

Write-Host "== published numbers =="
& $Python (Join-Path $root "scripts/check_claims.py")
if ($LASTEXITCODE -ne 0) { $fail++ }

Write-Host "== secrets =="
$hits = & git -C $root grep -nIE "(gitee\.com/[^ ]*:[^ ]*@|password\s*=|[0-9a-f]{32})" -- ':!*.csv' ':!*.json' 2>$null
if ($hits) { $hits | ForEach-Object { Write-Host "  $_" }; $fail++ } else { Write-Host "  OK" }

if ($fail) { Write-Host "`nFAILED ($fail)"; exit 1 }
Write-Host "`nALL CHECKS PASSED"
exit 0
