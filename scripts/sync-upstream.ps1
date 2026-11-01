<#
.SYNOPSIS
  Clone or update the official CS149 starter repositories into assignments/ (read-only mirror).

.DESCRIPTION
  Each repo is shallow-cloned once and fast-forwarded afterwards. The HEAD commit of every
  repo is appended to UPSTREAM.md so the mirrored snapshot is traceable. Use -Proxy for a
  one-off HTTP proxy; the global git config is never modified.

.EXAMPLE
  ./scripts/sync-upstream.ps1
  ./scripts/sync-upstream.ps1 -Proxy http://proxy.example.com:8080
#>
param([string]$Proxy = "")

$root = Split-Path -Parent $PSScriptRoot
$repos = [ordered]@{
    "cs149/asst1"          = "https://github.com/stanford-cs149/asst1.git"
    "cs149/asst2"          = "https://github.com/stanford-cs149/asst2.git"
    "cs149/asst3"          = "https://github.com/stanford-cs149/asst3.git"
    "cs149/asst4-trainium" = "https://github.com/stanford-cs149/asst4-trainium.git"
    "cs149/biggraphs-ec"   = "https://github.com/stanford-cs149/biggraphs-ec.git"
    "cs149/cs149gpt"       = "https://github.com/stanford-cs149/cs149gpt.git"
}
$gitArgs = @()
if ($Proxy) { $gitArgs = @("-c", "http.proxy=$Proxy", "-c", "https.proxy=$Proxy") }

$rows = @()
foreach ($name in $repos.Keys) {
    $dest = Join-Path $root "assignments/$name"
    if (Test-Path (Join-Path $dest ".git")) {
        & git @gitArgs -C $dest pull --ff-only --quiet
    } else {
        New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
        & git @gitArgs clone --depth 1 --quiet $repos[$name] $dest
    }
    if ($LASTEXITCODE -ne 0) { Write-Warning "failed: $name"; continue }
    $sha = (& git -C $dest rev-parse --short HEAD).Trim()
    Remove-Item -Recurse -Force (Join-Path $dest ".git")
    $rows += "| $name | $($repos[$name]) | $sha | $(Get-Date -Format yyyy-MM-dd) |"
}

if ($rows.Count -gt 0) {
    $upstream = Join-Path $root "UPSTREAM.md"
    $text = Get-Content $upstream -Raw -Encoding utf8
    if ($text -notmatch "## 已同步") {
        Add-Content $upstream -Encoding utf8 "`n## 已同步`n`n| 目录 | 来源 | commit | 日期 |`n| --- | --- | --- | --- |"
    }
    Add-Content $upstream -Encoding utf8 ($rows -join "`n")
}
$rows
