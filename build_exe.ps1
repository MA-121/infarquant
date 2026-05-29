# Usage: .\build_exe.ps1 -> single portable dist\InfarQuant.exe; -OneFolder for a faster folder build; -Verify to self-test the bundle after building (run inside the activated `infarquant` conda env).
param([switch]$OneFolder, [switch]$Verify)

$mode = if ($OneFolder) { "--onedir" } else { "--onefile" }

python -m PyInstaller --noconfirm --clean `
  $mode `
  --windowed `
  --name InfarQuant `
  --icon docs\infarquant_icon.ico `
  --paths src `
  --collect-all imagecodecs `
  --add-data "docs\infarquant_logo.png;docs" `
  launch.py

if ($LASTEXITCODE -ne 0) { Write-Output "BUILD FAILED (exit $LASTEXITCODE)"; exit $LASTEXITCODE }

if ($Verify) {
  $exe = if ($OneFolder) { "dist\InfarQuant\InfarQuant.exe" } else { "dist\InfarQuant.exe" }
  $res = Join-Path $env:TEMP "infarquant_selftest.txt"
  Remove-Item $res -Force -ErrorAction SilentlyContinue
  Write-Output "== Verifying bundle: $exe --selftest =="
  $proc = Start-Process -FilePath $exe -ArgumentList "--selftest", "`"$res`"" -Wait -PassThru
  $out = if (Test-Path $res) { Get-Content $res -Raw } else { "(no result written; exit $($proc.ExitCode))" }
  Write-Output $out
  Remove-Item $res -Force -ErrorAction SilentlyContinue
  if ($out -match "SELFTEST OK") { Write-Output "BUNDLE VERIFY: PASS" }
  else { Write-Output "BUNDLE VERIFY: FAIL"; exit 1 }
}
