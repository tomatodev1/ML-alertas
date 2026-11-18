# Corrida semanal completa: modelo -> export_dashboard/ -> data.json del dashboard.
# Uso (desde c:\proyectos\ML alertas):   powershell -File actualizar_dashboard.ps1
# Antes: copiar dataIncidentes.xlsx, dataFormulario.xlsx y reportes nuevos a src\data\.
# Después: commit + push de data\data.json en c:\proyectos\dashboard\dashboard-antamina.

$ErrorActionPreference = "Stop"
$py = "C:\Users\Operador_01\AppData\Local\Programs\Python\Python312\python.exe"
$ml = $PSScriptRoot
$dash = "C:\proyectos\dashboard\dashboard-antamina"

Set-Location $ml
& $py export_dashboard.py --actualizar-fuentes
if ($LASTEXITCODE -ne 0) { throw "export_dashboard.py falló: el paquete no cumple el contrato. No se toca el dashboard." }

Set-Location $dash
& $py scripts\export_sheet_antamina.py --desde-carpeta (Join-Path $ml "export_dashboard")
if ($LASTEXITCODE -ne 0) { throw "export_sheet_antamina.py falló." }

Write-Host ""
Write-Host "Listo. Para publicar, desde $dash :"
Write-Host '  git add data/data.json'
Write-Host '  git commit -m "scoring <semana>"'
Write-Host '  git push'
Set-Location $ml
