# Usage: .\build_exe.ps1 -> single portable dist\InfarQuant.exe; add -OneFolder for a faster dist\InfarQuant\ folder build (run inside the activated `infarquant` conda env).
param([switch]$OneFolder)

$mode = if ($OneFolder) { "--onedir" } else { "--onefile" }

python -m PyInstaller --noconfirm --clean `
  $mode `
  --windowed `
  --name InfarQuant `
  --icon docs\infarquant_icon.ico `
  --paths src `
  --add-data "docs\infarquant_logo.png;docs" `
  launch.py
