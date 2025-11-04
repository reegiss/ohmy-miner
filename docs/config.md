# Configuração do miner (miner.conf)

O `ohmy-miner` aceita um arquivo de configuração opcional via `--config` (padrão: `miner.conf`).

- Formatos suportados:
  - JSON (recomendado)
  - key=value (compat simples)
- Precedência de parâmetros (mais forte para mais fraco):
  1. CLI (flags como `--url`, `--user`)
  2. Variáveis de ambiente `OMM_*` (ex.: `OMM_URL`, `OMM_USER`)
  3. Arquivo de configuração (`miner.conf`)

## Exemplo JSON

Veja `docs/miner.conf.example`:

```json
{
  "algo": "qhash",
  "url": "qubitcoin.luckypool.io:8610",
  "user": "bc1qacadts4usj2tjljwdemfu44a2tq47hch33fc6f.RIG-1",
  "pass": "x"
}
```

## Validações (schema leve)

- `algo`: se presente, deve ser "qhash".
- `url`:
  - Formato `host:port`.
  - `host` com labels separados por `.`; cada label 1..63 chars, apenas `[A-Za-z0-9-]`, sem começar/terminar com `-`; hostname total ≤ 253.
  - `port` numérica no intervalo `1..65535`.
- `user`, `pass`: strings.

Erros de validação são impressos no stderr e o arquivo é ignorado; o miner segue com CLI/env.

## Formato key=value (alternativo)

```ini
algo=qhash
url=qubitcoin.luckypool.io:8610
user=bc1q...RIG-1
pass=x
```

## Dicas

- Para testes rápidos, você pode usar `start.sh` e sobrescrever via env:
  - `OMM_URL`, `OMM_USER`, `OMM_PASS`, `OMM_ALGO`.
- Em produção, prefira o JSON para validações mais seguras.
