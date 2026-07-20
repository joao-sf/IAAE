param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Path,

    [Parameter(Mandatory = $false)]
    [string]$Title = "IAAE — Relatório",

    [Parameter(Mandatory = $false)]
    [string]$FilterColumn,

    [Parameter(Mandatory = $false)]
    [string]$FilterValue
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Path)) {
    throw "Arquivo não encontrado: $Path"
}

$extension = [System.IO.Path]::GetExtension($Path).ToLowerInvariant()

switch ($extension) {
    ".psv" {
        $data = Import-Csv -LiteralPath $Path -Delimiter "|"
    }
    ".csv" {
        Write-Warning "CSV detectado. Para relatórios do IAAE, prefira PSV."
        $data = Import-Csv -LiteralPath $Path -Delimiter "|"
    }
    default {
        throw "Formato não suportado: $extension. Use .psv ou .csv."
    }
}

if ($FilterColumn -and $FilterValue) {
    if (-not ($data | Get-Member -Name $FilterColumn -MemberType NoteProperty)) {
        throw "Coluna de filtro não encontrada: $FilterColumn"
    }

    $data = $data | Where-Object {
        $_.$FilterColumn -eq $FilterValue
    }
}

if (-not $data) {
    Write-Host "Nenhum registro encontrado."
    exit 0
}

$data | Out-GridView -Title $Title
