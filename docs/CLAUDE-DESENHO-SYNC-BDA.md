# Sincronização de buffers (BDA) e o custo por draw: o que os dados mostram e o desenho proposto

Autor: sessão Claude Code (Sonnet 5.5). Data: 2026-09-30. Base: árvore de trabalho sobre `3379c96c`.
Cena: Crash Bandicoot 4 (PPSA02433), gameplay lento (~13–14 fps, jogo em 3840×2160), RX 9070 XT, `--redzone`.
Este documento é uma proposta. Nada aqui foi implementado nem validado em jogo.

## Pergunta

A cena lenta tem ~11,8 mil falhas de escrita e ~17,7 mil uploads de buffer por segundo. Dá para tratar
regiões inteiras de buffers dinâmicos como "quentes" e cortar esse custo?

## O que os dados dizem (run `kyty-bench-20260930-230231`, segundos 70–90, hang trace `transfers.csv`)

- 14,8 mil uploads por segundo (18,3 mil nos segundos estáveis), **~300 MB/s**, ~20 MB e ~1.300 uploads por
  frame. Motivo registrado: `read-binding` (493 mil das 503 mil linhas).
- **Tamanho:** 60% dos uploads têm exatamente 16 KiB; 14% têm 4 KiB; 6% 12 KiB, 5% 32 KiB, 4% 8 KiB; o
  resto são outros múltiplos de página.
- **Espalhamento:** 529 regiões de 2 MiB distintas; as 6 maiores têm só 22% dos uploads e 17% dos bytes
  (em regiões de 64 MiB, 117 distintas, as 6 maiores têm 37%). Não são poucos buffers circulares.
- **Repetição:** 7.606 endereços de início distintos em 20 s; 1.549 (20%) são reenviados em pelo menos
  metade dos segundos; ~270 em 17 dos 20.
- **Fundíveis:** 29% das linhas começam exatamente onde outro upload do mesmo segundo terminou.

Leitura: são dados realmente sujos (o jogo reescreve ~20 MB por frame), em pedaços pequenos e espalhados
pelo heap. O custo está no **número de operações**, não nos bytes (300 MB/s é nada para o barramento).

## O que o perfil mostra (thread do emulador que cuida da GPU, run sem preset)

`ExecutePreparedDraw` 52% da thread → `PrepareGraphicsBindings` 35% → **`RenderContext::PrepareBda` 22,7%**
(100% `SynchronizeBdaBuffers → SynchronizeBdaBuffersNow`, 93% em `SynchronizeBuffersInRange`). Dentro de
`SynchronizeBuffer`: 33% `RegionManager::UpdateProtection`, 19% `HangTrace::RecordTransfer` (custo do próprio
trace, ~3,5% da thread), resto de contabilidade de upload. Fora do BDA: `PrepareDrawRenderState` ~14% da
thread, `FindBuffers`/`RebindBuffers` ~10%, `ObtainBuffer` ~3,5%, `AcquireRenderTargets` ~2%.

## O que já existe no código (e rodou no run com o preset)

`bufferCache.cpp`, `SynchronizeBdaBuffers` e `SynchronizeBdaBuffersNow`:

1. **Atalho por época** (`KYTY_BDA_SYNC_EPOCH`, padrão ligado): pula a passagem se nada avançou a época.
2. **Incremental** (`KYTY_BDA_INCREMENTAL_SYNC=1`, **opt-in**; está no `u59-preset.json`): só repete a varredura
   quando a época de falhas mudou.
3. **Páginas quentes** (`KYTY_BDA_HOT_SYNC`, padrão ligado *quando o incremental está ligado*): páginas que
   falham repetidamente deixam de ser protegidas e são comparadas com uma cópia.
4. **Log de faixas sujas** (`KYTY_BDA_DIRTY_LOG`, padrão ligado *quando o incremental está ligado*): em vez de
   varrer todos os buffers, a passagem visita só as faixas que o rastreador de falhas registrou.

Ou seja, **o preset já ligava 2, 3 e 4**. Efeito medido (segundos lentos, mesma cena aproximada, **uma
amostra por configuração, posição do jogador não controlada**): CPU da thread da GPU 82% → 67%, fps 12,9 → 14,5
(+12%), com **as mesmas** contagens de falhas e uploads por segundo (10,8 mil contra 11,5 mil; 17,0 mil contra 17,2 mil).
O que o preset economizou foi a varredura, não o trabalho por página suja.

## Por que "promover regiões inteiras a quentes" não é o caminho

- Seriam **centenas** de regiões espalhadas, não algumas poucas: hoje as páginas quentes são tratadas por
  comparação com cópia a cada passagem, e o custo cresce com o número delas. A passagem roda por época, não por
  frame (várias por frame).
- Os uploads são dados novos a cada frame (~20 MB), então a comparação com a cópia encontraria diferença
  quase sempre: não evita o upload, só troca a falha por leitura.
- A alternativa de rastreio sem falhas (`MEM_WRITE_WATCH` / `GetWriteWatch`) não serve: ela só existe para
  memória privada, e o espaço de endereços do guest usa mapeamentos de arquivo para aliases de memória direta.
  (Inferência pela estrutura do espaço de endereços, **não verificada**.)

## Onde ainda há trabalho: custo fixo por pedaço sujo e por draw

O custo restante é por operação. Candidatos, do mais seguro ao mais arriscado:

| # | Mudança | Ganho esperado | Risco | Como validar |
|---|---|---|---|---|
| 1 | **Ligar o preset por padrão** para os jogos (copiar o `u59-preset.json` para o lado do `launcher.exe`, ou o launcher aplicá-lo) | +10–15% (medido uma vez) | baixo, é o caminho previsto pelo projeto | `bench_boot.ps1 -PresetFile`, 3 repetições |
| 2 | **Fundir uploads adjacentes** na mesma passagem em uma só cópia e uma só chamada de proteção (29% já adjacentes; 60% dos uploads têm 4 páginas) | reduz operações; ganho não estimado | médio: tamanho de cópia e barreiras | testes de `SynchronizeBuffer` com faixas; comparar contagem de cópias |
| 3 | **Agrupar `VirtualProtect`** por faixa contígua depois da passagem (`UpdateProtection` é 33% de `SynchronizeBuffer`; ~17,9 mil chamadas/s) | até ~7% da thread (0,33 × 18,6% × fração agrupável), pouco | médio | contagem de chamadas de proteção por segundo no trace |
| 4 | **Estado por draw em cache** (`PrepareDrawRenderState` ~14%, `FindBuffers`/`RebindBuffers` ~10%): já há memos por época (`KYTY_BINDING_EPOCH_MEMO`) | não estimado | médio-alto | perfil com pilhas depois do preset |
| 5 | **Sincronizar por frame**, não por época, buffers que o jogo só reescreve entre frames | alto, em teoria | **alto**: pode perder dados que um draw lê no meio do frame | só com modos de verificação existentes (`*_VERIFY`) |

Nenhum ganho acima está medido. O item 1 é o único com um número (+12%, uma amostra).

## Falta medir antes de implementar

1. **Perfil com pilhas do run com o preset ligado** (as pilhas só existem para o run sem preset): mostra o que
   sobra depois da varredura incremental e permite escolher entre 2, 3 e 4 com número.
2. **Taxa de passagens, saltos e estouros do log por segundo.** Hoje esses contadores
   (`BdaSyncPasses`, `BdaSyncSkips`, `BdaSyncLogPasses`, `BdaSyncLogOverflows`, ...) só existem como eventos do
   Tracy e não aparecem no `summary.csv`. Expô-los no hang trace é uma mudança pequena e de baixo risco e
   responde "a varredura completa ainda acontece com que frequência?".
3. **O limite pode não ser só a thread da GPU.** Com o preset a thread da GPU gastou menos CPU, mas o fps só
   subiu 12%; `RenderThread 1` ficou em ~91–93% nos dois. Cada redução de trabalho na thread da GPU parece
   render proporcionalmente menos fps (aqui ~0,7 por ponto de CPU), consistente com um pipeline em que a
   thread de render espera a thread da GPU. Não provado.

## Atualização: perfil com pilhas, com e sem o preset (2026-09-30)

Dois runs com o amostrador de pilhas, **um por configuração, cenas parecidas mas não iguais** (sem preset:
`kyty-bench-20260930-231818`, 2182 amostras por thread; com preset: `kyty-bench-20260930-234145`, 432 amostras).
Fração da thread do emulador que cuida da GPU:

| Função | sem preset | com preset |
|---|---|---|
| `ExecutePreparedDraw` | 52,1% | 36,3% |
| `PrepareGraphicsBindings` | 34,6% | 23,6% |
| `RenderContext::PrepareBda` | 22,7% | 16,0% |
| `SynchronizeBuffersInRange` | 21,2% | 13,9% |
| `ExecPredication` → `BufferFlushAndWait` | **6,4%** | **20,0%** |

- O preset reduz o custo da varredura (`PrepareBda` −30% em fração da thread), como esperado. Dentro dele
  aparecem `SynchronizeBdaDirtied` (8,7% do tempo de `SynchronizeBdaBuffersNow`) e `SynchronizeBdaHotRanges` (5,8%): o
  custo do próprio modo incremental; o resto (84,1%) continua sendo `SynchronizeBuffersInRange`.
- **Achado novo:** o pacote de predicação de memória (`op 3` com espera, `CommandProcessor::ExecPredication`) manda
  todo o trabalho gravado para a GPU e **bloqueia a thread até a GPU terminar** (`BufferFlushAndWait` →
  `MasterSemaphore::Wait`). É o único chamador ativo de `BufferFlushAndWait` (o outro está comentado). Com o preset
  essa espera passou a ser 1/5 da thread. Em hardware real o processador de comandos avalia o predicado sem ida e
  volta à CPU; aqui cada pacote destes derruba o paralelismo CPU↔GPU (a GPU fica ociosa enquanto a CPU grava e a CPU
  espera enquanto a GPU executa). Isso combina com "GPU ~18% ocupada, CPU saturada".
- Cuidado: uma amostra por configuração, durações diferentes, e a causa no jogo (qual recurso emite o pacote)
  continua desconhecida.

Instrumentação adicionada: `summary.csv` ganhou, no fim da linha, `pred_flush_waits` e `pred_flush_wait_us`;
`tools/bench_boot.ps1` mostra `pred_waits_s` e `pred_wait_ms_s`; `tools/analyze_samples.py` ganhou `--callers` (quem
chama uma função) e `--blame` (primeiro frame do emulador sob uma espera em DLL).

### Medição (run `kyty-bench-20260930-235726`, preset, 146 s, um run só)

Segundos agrupados por fps, do `summary.csv`:

| Segundos | fps | esperas por frame | ms por espera | bloqueado por frame | GPU ocupada |
|---|---|---|---|---|---|
| fps ≥ 55 (37 s) | 59,6 | 2,2 | 0,29 | 0,6 ms (4% do tempo) | 36 ms/s |
| fps 15–25 (19 s, cena lenta) | 18,1 | **13,0** | **1,00** | **13,0 ms (24% do tempo)** | 359 ms/s (36%) |

- Na cena lenta a thread da GPU fica bloqueada em ~13 de cada ~55 ms por frame esperando a GPU, em ~13 pacotes de
  predicação por frame. A GPU está só ~36% ocupada: o bloqueio impede CPU e GPU de trabalharem em paralelo.
- Teto teórico de ganho ao eliminar o bloqueio: ~13 ms a menos por frame, ou seja ~18 → ~24 fps (+30%), se o resto
  não mudar e a GPU acompanhar (48% ocupada a 24 fps). Não é 60 fps.
- **Há uma segunda causa de lentidão neste run: compilação de pipelines.** Os segundos com 0–5 fps (100–102, 113–115,
  124, 140–146) têm de 300 a 1000 ms/s de `compile_gfx_pipeline_us` e `compile_stall_us`. O bench restaura o cache de
  pipelines a cada run, então isso reaparece sempre, como na primeira vez de quem joga. O perfil de pilhas deste run
  (disparado em ~140–146 s) caiu nessa fase: 85% da thread da GPU dentro de `amdvlk64.dll`, `GetGraphicsPipeline`
  com 38% de `ExecutePreparedDraw`. **Não serve para a cena lenta estável.**

Se os números confirmarem (aqui confirmaram), as saídas possíveis são: (a) avaliar o
predicado na GPU com `VK_EXT_conditional_rendering` (a extensão não é usada hoje em lugar nenhum do código);
(b) esperar só quando a escrita do predicado ainda está em voo (exige rastrear quem escreve naquele endereço);
(c) ignorar o predicado quando ele só serve de otimização (desenha tudo): risco visual/correção, e mais draws na
thread que já é o gargalo. Nenhuma está implementada.

## Recomendação

1. Usar o preset já (item 1), com DCC desligado para o Crash.
2. Rodar mais um bench **com o preset e o amostrador de pilhas**, para ver o custo restante.
3. Expor os contadores de BDA no `summary.csv` (mudança pequena em `hangTrace`).
4. Só então decidir entre fundir uploads/proteções (2 e 3) e o cache por draw (4). Evitar o item 5.
