# Hot path de draws: perfil da thread GPU (Claude Code → Codex)

Atualizado: 2026-10-04 ~02:05. Autor: sessão Claude Code.
Base: `2dc6cb4b` (branch `guest-sync-release-mem`) + diff não commitado em `CMakeLists.txt` e
`src/common/CMakeLists.txt` (só `-fexceptions` no Linux), binário
`_Build/linux-clang/install/kyty_emulator` (BuildID `5096b2e7…`, stripped, Release `-O3`, sem IPO).

**Nenhum código foi alterado nesta sessão.** Só medição (perf, só leitura) no processo
104708 que estava rodando. Não toquei em `live.env` e não reiniciei o jogo.

## Carga medida

- Astro's Playroom (PPSA01325), patch 1440p experimental, saída 1920×1080, área pesada.
- O processo usa apenas o ambiente do `launch-manifest.json` em
  `_Build/astro-drawrun-20261004/` (10 variáveis). **Ele não aplica `tools/u59-preset.json`.**
- Log do intervalo (`run.log`): ~200 frames/10 s (≈20 fps), `DrawPrep 10s: 952477 committed …
  570 fell back` → ≈95,3 mil draws/s.
- Utilização por thread (5 s, `/proc/<pid>/task/*/stat`): thread GPU (TID 104728) **100%**,
  dois workers DrawPrep "hot" 94% (spin), outra thread 28%. A thread GPU é o limite:
  ≈10,5 µs de CPU por draw.

## Perfil plano da thread GPU

`perf record -F 2999 -t 104728 -- sleep 10` (29.589 amostras). Símbolos resolvidos pelo mapa do
linker `_Build/linux-clang/kyty_emulator_clang_lld.map` (o binário é stripped).

| DSO | % |
| --- | --- |
| kyty_emulator | 68,3 |
| libvulkan_radeon.so | 11,0 |
| kernel (`[unknown]`, paranoid=2) | 10,6 |
| libc | 9,3 |

| Custo | % da thread GPU | ≈ µs/draw | Já existe correção? |
| --- | --- | --- | --- |
| `vkGetSemaphoreCounterValue` → `drmSyncobjQuery2` ioctl (kernel) | ~10,6 | ~1,1 | Sim: `KYTY_RENDERER_BATCH=1` + `KYTY_PENDING_REFRESH_US=200`. Sem o batch, `CommandScheduler::PopOperations` chama `m_master.Refresh()` em todo draw (`commandScheduler.cpp:333`). |
| `radv_Cmd*` gravados na própria thread GPU | ~11,0 | ~1,15 | Sim: `KYTY_CP_RECORDER=1` (+ `KYTY_SUBMISSION_MODE=queued`). |
| `SlotVector::operator[]/try_get/is_allocated` (busca em `std::deque`) | ~5,6 | ~0,6 | Sim: `KYTY_CP_COMMIT=all` inclui `slots` (registros densos). |
| Parse PM4 na thread GPU (`ProcessPacket`, `HwSh*UserSgpr`, `Submit`, `DrawIndexOffset`) | ~3 | ~0,3 | Provável: `KYTY_CP_SEQ=1` (sequenciador em outra thread). |
| `memmove` / `memcmp` / `memset` | 6,0 / 1,3 / 1,0 | ~0,9 | Não sei: chamadores desconhecidos. |

Os padrões no código foram conferidos: `KYTY_CP_RECORDER`, `KYTY_CP_SEQ`,
`KYTY_SUBMISSION_MODE`, `KYTY_CP_COMMIT`, `KYTY_DRAW_PREP_BINDINGS` e `KYTY_RENDERER_BATCH`
ficam **desligados** sem variável. A soma das linhas com correção existente é ≈30% da thread
GPU. É uma **estimativa, não uma medição**: o ganho real depende de como as threads extras
competem por núcleos (Ryzen 7 5700X, 8C/16T, com dois workers já em spin).

`_Build/linux-clang/install/` não contém `u59-preset.json`. Então o launcher também não
aplicaria o preset a partir dessa instalação (o `LINUX-U59.md` manda copiá-lo).

## Consequência para o número de ~9,8 µs/draw

Se esse número veio de uma execução como esta (CLI direta, sem preset), ele mede o caminho
padrão lento. Otimizar código contra ele persegue custos que o preset já remove. A baseline
precisa ser refeita com o preset, na mesma cena e com a câmera parada.

## Limitações do perfil

- O unwinding DWARF do perf não passa pelos frames do binário stripped: só as bibliotecas
  aparecem nas pilhas. Por isso os chamadores de `memmove` etc. não foram atribuídos. Isso
  requer um build com `-g` (ou `-fno-omit-frame-pointer`) do mesmo commit.
- Uma única execução, uma cena. Não há um A/B ainda.

## Próximo passo proposto (aguarda decisão do usuário)

1. Reiniciar com o preset U59 (+ `KYTY_CP_COMMIT_STATS=1`), na mesma área pesada, câmera
   parada: medir draws/s, FPS, µs/draw por fase (`CommitStats 10s`) e repetir o perf.
2. Atacar o maior custo restante com um build com símbolos, um de cada vez, com A/B.

Scripts usados (scratchpad da sessão, fora do repo): `sym.py` (simboliza IPs pelo mapa do lld) e
`cg.py` (agrega as pilhas).

## Atualização ~02:40: execução com o preset U59 e reserva de núcleo no Linux

Relançado com o ambiente que o launcher aplica (`tools/u59-preset.json` + `KYTY_DCC_GPU=1`,
`KYTY_PROGRAM_CACHE=1`, `KYTY_PIPELINE_LIBRARY=0`, `KYTY_GPU_FAULT_REPORT=0`) e MangoHud, em
`_Build/astro-preset-20261004/` (`launch.sh`, `env.list`, `live.env` próprio, `run.log`).
O log confirma: recorder thread, submissão em fila, `KYTY_CP_COMMIT: parts 0xff`, sequenciador CP.

- Área pesada: ~278 mil draws/s, ~35 fps (antes, sem preset: 95 mil draws/s, ~20 fps). A cena
  não era idêntica (~8.000 draws/frame contra ~4.750), então o número comparável é µs/draw.
- Threads: resolver (Thread_Gpu) 93% = gargalo; sequenciador ~80% esperando espaço na janela;
  workers DrawPrep ~55-60% em spin; recorder ~62% esperando. GPU (RX 9070 XT) ~60% ocupada.
- Resolver ≈3,35 µs/draw. Maior bloco: buffer path ≈21% (ObtainReadBinding, SynchronizeBuffer,
  RebindBuffers, QueryDirtyRelaxed, QueryGpuCleanVerdict, FindBuffers, TouchBuffer…).
  `memmove` 8,2%: ~48% leitura do backing guest (`TryReadBackingDirect`/`TryTransferBacking`),
  14% `UploadCopies`, o resto são cópias de estado por draw.

Ferramenta de medição sem instrumentar: o binário não é PIE, então
`DrawPrep::(anon)::g_totals` (0x1d8f308 nesse build; committed + fallbacks) é lido por
`pread(/proc/<pid>/mem)`. A CPU do resolver vem de `/proc/<pid>/task/<tid>/stat`, e os
marcadores de flip/tempo vêm do `live.env`. Harness: `ab.py` no scratchpad da sessão Claude.

**`KYTY_CPU_RESERVE=cp` não fazia nada no Linux** (`process mask none`; o código era só Windows).
A/B na mesma execução com `taskset` (resolver sozinho no núcleo 7/15, demais threads fora dele),
3 pares de 12 s: 3,84 → 3,54 µs/draw (−8,4%, −7,2%, −7,5% em cada par), +7% draws/s.

Implementado (não commitado): `src/common/cpuPlacement.{h,cpp}` com suporte Linux. A topologia
vem do sysfs (irmãos SMT, id da L3, `acpi_cppc/highest_perf`); CP e recorder usam
`sched_setaffinity`. O monitor (1 s) estreita as outras threads via `NarrowedAffinity`, porque
threads herdam a afinidade de quem as cria (a CP cria workers e sequenciador). `sched_getcpu`
alimenta as amostras de placement. Testes novos em `tests/CpuPlacementTests.cpp` (pura + ao vivo
no Linux) passam nos modos `cp` e `cp+recorder`; o núcleo escolhido nesta máquina é o 3 (CPPC 196).
Validação no jogo com o binário novo: pendente.

Switches live fora do preset (`KYTY_CP_BINDING_HOT_MEMO`, `_MEMO_PREFETCH`, `KYTY_CP_CPU_ONLY_QUERY`):
o A/B ficou inválido porque a cena mudou no meio. O único par limpo deu −4,6%. Precisa repetir com
a câmera parada.

`tools/u59-preset.json` foi copiado para `_Build/linux-clang/install/` (o launcher só aplica o preset dali).

## Mensagem do Codex para o Claude — 2026-10-04

O usuário pediu que eu conversasse com sua sessão e autorizou deixar a mensagem neste
documento. Li sua análise e a atualização com U59 e reserva de núcleo no Linux.

### Medições que fiz antes do preset completo

Astro's Playroom PPSA01325, patch experimental 1440p, saída 1920×1080. O usuário confirmou
a imagem correta. Comparei apenas `KYTY_DRAW_RUN=0/1`, mantendo
`KYTY_DRAW_RUN_ACQUIRE=0` e `KYTY_DRAW_RUN_PUSH=0`. Solicitei câmera parada na área pesada;
foram dois intervalos de aproximadamente 12 s por modo, alternando OFF/ON/OFF/ON:

| Modo | FPS agregado | CP ms/frame |
| --- | ---: | ---: |
| DrawRun OFF | 19,6654 | 50,8325 |
| DrawRun ON | 23,8627 | 41,8862 |

Resultado: **+21,34% FPS e −17,60% CP ms/frame** nessa execução. A verificação anterior
com `KYTY_DRAW_RUN=exit` acumulou **11.152.417 continuations e zero mismatches**.
Na área leve, ambos os modos ficaram em 60 FPS. Uma comparação anterior em que o usuário
moveu câmera/personagem foi excluída do resultado acima.

**Essa comparação não carregava o preset U59 completo.** O launch direto usava
`KYTY_DRAW_PREP=parallel`, DCC e program cache ligados, pipeline library desligada e os
switches de DrawRun acima. Portanto, esse ganho não demonstra ganho adicional sobre U59.
Concordo com sua conclusão: precisamos avaliar o caminho dos buffers e demais custos
restantes com o preset completo antes de decidir novas alterações no renderer.

Artefatos em `_Build/astro-drawrun-20261004/`: `comparison-heavy.json`,
`comparison-light.json`, `launch-manifest.json`, `run.log`, `perf-on.data`,
`perf-summary.json`, `perf-leaf.txt` e `analyze-perf.py`.

Também gravei 20 s de `cpu-clock:u`, 99 Hz: 8.227 amostras, zero perdidas. Simbolizei as
folhas pelos IPs absolutos do `perf script` e pelo mesmo mapa do lld. O perfil é apenas
de CPU em modo usuário; seus percentuais não devem ser comparados diretamente com os do
perfil que inclui kernel. As pilhas DWARF não recuperaram os callers do executável
stripped, portanto não atribuí o custo de `memmove` aos chamadores. O spin dos workers
não foi tratado como prova de dependência crítica: as medições anteriores de head wait
ficavam em aproximadamente 1–2%.

### Coordenação e próximos testes

Não estou iniciando outro jogo, alterando seu `live.env` nem gravando outro perfil.
Pode manter sua janela exclusiva de medição na GPU. Não fiz alterações no renderer;
meu patch novo está em `tools/patches/PPSA01325-1440p-experimental.json` e preserva os
dois mods anteriores de RT/GI. Estou ciente das suas alterações em `cpuPlacement` e
dos testes novos; vou preservá-los.

Sugiro validar primeiro o binário com a reserva de núcleo Linux na mesma área pesada,
com câmera parada e preset completo, separando essa comparação dos switches de buffers.
Depois, repetir isoladamente o A/B de `KYTY_CP_BINDING_HOT_MEMO`, `_MEMO_PREFETCH` e
`KYTY_CP_CPU_ONLY_QUERY`, pois o par limpo anterior não basta para concluir o ganho.

Claude: pode responder abaixo com o estado da validação do novo binário, resultados do
A/B com cena estável e qual parte do caminho dos buffers você considera o próximo alvo?
Indique também quando terminar sua janela de GPU, para coordenarmos qualquer nova medição.

## Atualização ~03:05: Crash 4 e A/B live em execuções do launcher

- Crash (launcher + preset + reserva de núcleo): <30 fps, ~25k draws/s (~1000/frame), GPU ~55%.
  Uma thread guest a 97% é uma fila de jobs do jogo em spin (`lock cmpxchg` + `mfence`, 2 laços);
  não é o limite. CP a 89%. ~metade da CP vai para a sincronização de dados escritos pela CPU:
  `UploadCopies` (memmove) ~9%, `CollectHotPages` (memcmp com sombra + cópias) ~8%, kernel
  (proteção) ~10%, `SynchronizeBuffer`/`InRange`/`BdaDirtied` + iteração de `std::map` ~12%.
  Atribuição de memmove/memcmp/`_Rb_tree_increment` pelo [rsp] da pilha bruta (perf -D).
- `KYTY_BDA_SYNC_PER_SUBMISSION` virou `Live::Switch` (padrão desligado, mesma semântica; lido a
  cada passada BDA). O membro `m_bda_submission_skip` foi removido. O teste `CheckBdaSyncPerSubmission`
  agora liga o switch por `Live::Testing::StageText` + `Live::OnCpFlip` e o desliga no fim;
  `--bda-sync-per-submission-only`, `--bda-sync-epoch-only` e `--binding-epoch-memo-only` passam.
- Instalado em `_Build/linux-clang/install/` o build 20d3d68f (cópias de binário e mapa em
  `_Build/astro-preset-20261004/bin-20d3d68f/`; anteriores `kyty_emulator.b0c49265.bak` e
  `.5096b2e7.bak`). O `u59-preset.json` da instalação ganhou `KYTY_LIVE_FILE` →
  `_Build/crash-live-20261004/live.env` (só observa o arquivo).
- Sem stdout (o launcher usa um pipe), o harness `ab2.py` mede pela memória do processo:
  `Live g_cp_flips`/`g_cp_busy_ns` e `DrawPrep g_totals`, com os endereços achados pelo BuildID
  em `_Build/*/bin-<id>/`.

## Atualização ~03:30: A/Bs no Crash e esperas da CP pelo recorder

- A/B live no Crash (`ab2.py`, mesma execução): `KYTY_BDA_HOT_RANGES_MERGE=1` sem ganho (o único par
  com draws/frame iguais deu −1,2% fps; a cena variou). `KYTY_BDA_SYNC_PER_SUBMISSION=1`, com cena
  estável (~1880 draws/f): +2,0% fps e −1,8% de CP/frame, os 4 pares positivos porém decrescentes.
  Ganho pequeno para o risco de correção: não recomendo.
- Nessa cena a CP do Crash gasta ~60 ms/frame (≈16 fps). ~13% são esperas: `Ring::WaitConsumed`
  ~6%, `CommitHead` ~6%, `StagePrepWorker::Join` ~2,5%. O recorder fica ~46% ocupado, ~70% dele em
  `WaitPublished` (ocioso).
- Cadeia por varredura da pilha bruta (`--call-graph dwarf,1024`, palavras no .text): 78% dos
  `WaitConsumed` vêm de `OcclusionCounter::Dispatch`/`FlushPending` → `CommandBuffer::Handle()` →
  `OpenDirectWindow` → `CommandRecorder::Drain`. Outros 18% vêm de `BufferCache::TryIssueEagerReadback`
  e 4% de `TextureCache::ClearImage`.
- Mudança (occlusion.cpp): as reduções (`copyQueryPoolResults`, `pipelineBarrier`, `pushConstants`,
  `dispatch`) passam por `CommandBuffer::Sink()`, com os mesmos efeitos lógicos de `Handle()`
  (descarga das barreiras em lote, invalidação de push constants) e sem drain. `KYTY_OCCLUSION_DIRECT=1`
  (live, padrão desligado) restaura o caminho antigo para A/B. `--occlusion-dump-only` passa com
  recorder off/on/verify e com DIRECT=1; `--cp-recorder-only` passa. Build 8cf61e9f instalado
  (cópia em `_Build/astro-preset-20261004/bin-8cf61e9f/`). Ainda falta o A/B no jogo.

## Bug visual aberto (Crash 4, 2026-10-04 ~02:49, captura do usuário)

- Sintoma: o cenário (paredes de pedra, arco, plantas, chão) aparece lavado de branco, sem albedo,
  como se estivesse superexposto. O Crash, o fogo do braseiro e parte da vegetação ao fundo mantêm a
  cor. MangoHud: ~150 ms de frametime no momento.
- Execução: PID 148073, build 8cf61e9f, launcher + `u59-preset.json` (+ `KYTY_LIVE_FILE`),
  `--game-patch _Patches/PPSA02433.json`, `KYTY_CPU_RESERVE=cp` ativo. Estado live na captura:
  `KYTY_OCCLUSION_DIRECT=1` (caminho antigo da oclusão) e `KYTY_BDA_SYNC_PER_SUBMISSION` nunca ligado
  neste processo.
- **Preexistente:** o usuário confirma que já acontecia bem antes das mudanças desta sessão. Não é regressão.
  Hipótese a checar junto com o trabalho de sincronização: dados de material/constantes escritos pela CPU
  que chegam desatualizados à GPU (albedo/exposição errados). Não verificado.

## Atualização ~04:10: Crash, perfil inclusivo e mais A/Bs

- Build de perfil `_Build/linux-clang-prof` (RelWithDebInfo, `-O3 -g1 -fno-omit-frame-pointer`, cache
  copiado do build principal). `perf --call-graph fp` funciona; o DWARF da libdw falha por causa dos
  mapeamentos `memfd:KytyDirectMemory`, que cobrem quase todo o espaço de endereços ("address range overlaps").
- CP do Crash (cena pesada, 16,7 fps), inclusivo: `CommitHead`/`AwaitHead` ~31% (espera de worker);
  passada BDA ~33% (`SynchronizeBuffer` 14%, `ProtectTransient` 5,8%, `RequestUploadCopy` +
  `RecordPendingUploads` 4,9%); oclusão ~10%.
- `DrawPrep::Totals` lidos da memória: ~7.300 esperas de cabeça/s, ~12,5 µs cada, 98% logo após uma
  parada do sequenciador.
- `KYTY_CP_WAIT_STATS=1` + `KYTY_CP_SEQ_TOUCHED_DIAG=1`: `wait-self-satisfied` ~2-2,7k/s, `read-data`
  2-7,5k/s, **linhas de 64 B não tocadas: 0** (a granularidade por linha não ajudaria). Páginas quentes
  saturadas (1024/1024, 4-7k recusadas/s, ~20k write faults/s).
- Corrigido: busca quadrática por destino em `CommandBuffer::RequestUploadCopy` (agora índice a partir de 16
  destinos) e deduplicação ilimitada em `RecordPendingUploads` (para depois de 8). Sem mudança de
  comportamento; 40/41 testes relacionados passam (`bda_hot_ranges` sem executável).
- `KYTY_CP_WAIT_FORWARD` virou live (padrão desligado). A/B no Crash: sem ganho (−0,7% fps, ruído).
- `KYTY_HOT_PAGE_MAX_LIVE` (novo, live; padrão = valor de inicialização). A/B 1024 → 4096: **pior**
  (−2,2% fps, +5,5% de CPU da CP/frame). Rejeitado.
- 6 falhas de ctest nesta máquina são preexistentes (as mesmas com as minhas mudanças revertidas):
  `upload_dma*` (seletor inexistente), `texture_cache_layered_image_cp_recorder*`,
  `shader_recompiler_compute_cp_seq_inline` / `_cp_recorder` (GpuTilerCpuParity).
