# Ordem de tarefas: desempenho e estabilidade do fork

Autor: sessão Claude Code (Sonnet 5.5). Data: 2026-09-30.
Base: `3379c96c` mais a árvore de trabalho suja descrita na seção "Árvore e donos".
Máquina de medição: Ryzen 7 5700X, Radeon RX 9070 XT, Windows 11. Jogos: Astro's Playroom
(PPSA01325) e Crash Bandicoot 4 (PPSA02433).

Este arquivo é só desta sessão. Outras sessões devem criar o próprio arquivo e não editar este.

## Regras de execução

- Sem subir jogo até o usuário liberar. Build e teste unitário estão liberados.
- Medição de jogo exige janela exclusiva: outra sessão rodando jogo na mesma GPU polui os números.
  `tools/bench_boot.ps1` recusa rodar se `kyty_emulator.exe` já estiver aberto.
- Um dono por arquivo (tabela abaixo). Ninguém commita sem o usuário pedir.
- Builds desta sessão usam `_Build/claude-tests` (ignorado pelo git), nunca `_Build/windows`.

## Estado

| # | Tarefa | Estado |
|---|---|---|
| A1 | Config do Crash: logs `Silent` e `program_cache_enabled=true` (Astro: biblioteca de pipelines ligada) | feito, em `C:\ProgramData\Kyty\Kyty.ini` |
| A2 | Divisão de commits e donos | proposta abaixo, aguarda confirmação |
| A3 | `tools/bench_boot.ps1` | feito; modo `-AnalyzeOnly` validado com traces reais, modo de execução só com sintaxe verificada |
| A4 | Pool de handles para leituras do APR | código e teste prontos, compila; **não validado em jogo** |
| A5 | Diagnóstico do `DeviceLost` (`VK_EXT_device_fault`) | **da outra sessão**: `ReportDeviceFault` nas criações de pipeline (`graphicContext.h`, `shaders.cpp`, `vulkanWindow.cpp`) |
| A5b | Relatório de falha da GPU em todos os pontos que detectam `DeviceLost` | feito com OK do usuário: `deviceLostReport.h` (uma vez, threads esperam até 3 s, sem reentrada), chamado em `RequireVulkanSuccess`, `MasterSemaphore::Refresh/Wait` e `vkAcquireNextImageKHR`; teste CTest `device_lost_report`. Emulador completo compilado e linkado em `_Build/claude-tests` (22:29), copiado para `_Build/windows/install-claude` (exe com A4 e A5b; `--help` abre). O relatório tem interruptor próprio, `KYTY_GPU_FAULT_REPORT=1` (launcher: opção "GPU fault report", campo `gpu_fault_report_enabled`, ligada no Crash em `Kyty.ini`). **Não ligar ao `--graphics-debug-dump`**: ele despeja cada pacote PM4 e derrubou o Crash para ~5 fps (log de 1,85 GB em poucos minutos). Launcher e exe novos em `install-claude`. **Não validado em jogo** |
| A6 | Testes de correção para AMD (SAVEEXEC, FP16, wave32 em wave64) | **bloqueado**: a suíte de shaders do fork não inicializa nesta GPU (ver "Achados do A6") |
| A7 | Fazer o harness `shader_recompiler_compute_tests` rodar em GPU só com `attachment_feedback_loop_layout` (os testes de compute não precisam de rasterização) | proposto; destrava A6 e valida o recompilador na Radeon |
| B1 | Baseline: preset ligado/desligado × Astro/Crash × 3 repetições | pendente, precisa de janela de jogo |
| B2 | `DeviceLost`: reproduzir, achar o comando que trava, corrigir | pendente, depende de A5 e janela |
| B3 | Validar A4 em jogo | pendente, janela |
| B4 | Astro escuro e pontinhos verdes (com e sem patch na cena do print; RenderDoc) | pendente, janela |
| B5 | Feedback loop de attachment em driver só com a extensão de layout (flags estáticas no pipeline, como o upstream fez para AMD) | hipótese a verificar, janela; ver "Achados do A6" |
| C1 | Criação de pipeline fora de `m_mutex` com compilação única por chave | condicionado a B1 |
| C2 | Prewarm: journal de shaders + replay paralelo, junto do `programDiskCache` | condicionado a B1 |

Fora do plano: compilação em background que pula o draw (o jogo precisa do draw na hora); migrar
para Rust; integrar o driver do PS5_Vulkan (GPL-3.0-or-later contra GPL-2.0-only).

## O que foi medido (e o que não foi)

Todas as medições abaixo são de boot (cerca de 60 s), com hang trace, em cópia congelada do
emulador. Nenhuma é de gameplay.

- **Astro, `ErrorDeviceLost` no boot** com `KYTY_PIPELINE_LIBRARY=0` ou sem a variável; roda com `=1`.
  Com patch ou sem patch dá o mesmo. O patch "NO ray-traced GI" não é a causa do crash.
- **Crash, `ErrorDeviceLost`** com `KYTY_PIPELINE_LIBRARY=1` (shader `0x9bf4cf7e4ad6231d`, wave64);
  roda com `=0`. Ou seja, a biblioteca só desloca onde cai. O sintoma é de uma trava de GPU anterior,
  que só aparece na próxima criação de pipeline. Isso é timing, não um defeito da biblioteca.
- O upstream (`KytyPS5-main`, `--pipeline-libraries true/false`) roda o Astro nas duas formas nesta
  máquina. GPU e driver aguentam; a causa está no fork.
- **Crash, fps por janela de 10 s** (logs ligados ou desligados dá o mesmo): 60, 60, 60, ~53, ~25.
  A queda vem de rajadas de compilação de shaders numa única thread (até 157 programas num segundo,
  ~0,7 s) e de leituras do APR.
- `KYTY_PROGRAM_CACHE=1` no segundo run: tradução 1060 ms para 38 ms, mas o fps cai nos mesmos pontos.
  Tradução não é o custo dominante.
- **APR:** 32 mil leituras na MainThread em 60 s, 22,6 mil de até 4 KB, picos de 3,6 mil por segundo.
  `ReadHostFileToGuest` abria e fechava o arquivo a cada leitura.
- **Preset de otimização:** `KYTY_CP_RECORDER`, `KYTY_CP_SEQ` e `KYTY_SUBMISSION_MODE=queued` ficam
  desligados sem variável. Só o `u59-preset.json` ao lado do `launcher.exe` os liga, e o
  `install-video-fix` não tem esse arquivo. Tudo acima rodou **sem** o preset.

## A4: o que mudou e o que ele rende

- `src/libs/aprHostFilePool.h` (novo): pool de handles abertos por caminho. Um handle pertence a um
  lease por vez, leituras concorrentes do mesmo arquivo abrem handles próprios, no máximo 64 caminhos
  e 8 handles ociosos por caminho.
- `src/libs/libAmpr.cpp`: `ReadHostFileToGuest` usa o pool, e leituras de até 64 KiB reutilizam um
  buffer por thread. **A cópia pelo buffer do host continua**: ler direto na memória do guest não foi
  feito porque a página pode estar protegida para coerência CPU/GPU (o `KernelPread` chama
  `Memory::InvalidateMemory` antes de ler por isso). Isso só dá para validar com jogo.
  A linha `Common::PathFromUtf8(...)` que já estava na árvore foi mantida.
- `tests/AprHostFilePoolTests.cpp`, teste CTest `apr_host_file_pool` (contabilidade com tipo falso,
  conteúdo correto sob 8 threads com evicção forçada, EOF). `--bench` imprime o custo por leitura.
- **Microbenchmark** (arquivo de 64 MiB com cache quente, 20 mil leituras de 222 bytes): 40,5 us por
  leitura abrindo a cada vez, 5,2 us com o handle reaproveitado (cerca de 7,8 vezes).
- **Expectativa honesta:** a economia é de ~35 us por leitura. Num segundo com 3,7 mil leituras isso é
  ~0,13 s. Ajuda, mas não explica sozinho quedas de 60 para 14 fps. Leituras reais de um `.pak` de
  30 GB podem ter latência de disco que o pool não remove.

## Crash em `tracy::rpmalloc_thread_finalize` (Crash Bandicoot 4, 2026-09-30)

Mensagem do usuário: `Unhandled host exception: type=1 code=3221225477 pc=0x00000001407dd0c3
access=1 address=0x0000000002460000`, vinda do handler em `runtimeLinker.cpp:969`.

- `0xC0000005`, `access=1` é **leitura** (enum `Unknown, Read, Write, Execute`).
- `pc` está dentro do próprio `kyty_emulator.exe` (base `0x140000000`), não em código do jogo. Com o PDB
  do build de 21:32 (`llvm-symbolizer`), cai em `tracy::rpmalloc_thread_finalize(int)`. O disassembly
  mostra `movq (%rdi), %rdi` num laço de lista encadeada (`node = node->next`), com o nó em
  `0x2460000`, que não é mais memória válida. Ou seja, o heap interno do Tracy leu um span já liberado.
- No Windows o Tracy registra um destrutor de FLS que chama essa função quando uma thread (ou fiber)
  termina. O Tracy só é iniciado com `profiler_enabled` ligado (`profiler.cpp:1481`).
- **Atualização (22:12): a causa raiz não é o Tracy.** Dois crashes novos do mesmo tipo
  (`pc=0x1407deb18`, leitura em `0x0`, que o PDB resolve para
  `tracy::moodycamel::ConcurrentQueue<QueueItem>::~ConcurrentQueue`; e o mesmo `pc=0x1407dd0c3` lendo
  `0x3500000`) estão em `guest-audio.log` (22:09), com a cadeia completa, a ~47 s do início do run
  (`[00:00:46.999]`):
  1. `GPU DCC inspection: metadata=0x20652b0000 ... format=97` (a única do log; é a última atividade da GPU);
  2. `submit upload DMA copies failed: ErrorDeviceLost (-4)` (`vulkanCommon.cpp:122`);
  3. `Fatal Error ... masterSemaphore.cpp:37`, depois `vkAcquireNextImageKHR failed: ErrorDeviceLost`
     (`swapchain.cpp:584`);
  4. o `EXIT` chama `EmergencyShutdown`, que executa `Profiler::Shutdown` e desmonta o Tracy com outras
     threads vivas, e aí vêm os `Unhandled host exception` dentro do Tracy. **Esses são secundários.**
- **Causa real: `ErrorDeviceLost` (o mesmo problema do boot do Astro).** Neste run: biblioteca de
  pipelines desligada, `Upload DMA: on` (fila de transferência, família 2), `DCC materialization: GPU
  validation and conditional clear`, `Vulkan device fault reporting: disabled`.
- **Lacuna de diagnóstico:** `ReportDeviceFault` (`shaders.cpp`, outra sessão) só é chamado quando a
  perda aparece na **criação de pipeline**. Esta foi pega por `RequireVulkanSuccess`, que não imprime
  nada da GPU, e a extensão só é ativada com `--graphics-debug-dump` (launcher: env
  `KYTY_CAPTURE_SHADER_DETAILS=1`). **Feito (A5b):** o relatório agora também sai em
  `vulkanCommon.cpp` (`RequireVulkanSuccess`), `masterSemaphore.cpp` (`Refresh` e `Wait`) e
  `swapchain.cpp` (`vkAcquireNextImageKHR`). Quando a extensão não está ativa, o log diz como ativar.
  Edições nos arquivos da outra sessão: `shaders.cpp` (`ReportDeviceFault` deixou de ser `static`, mais a
  mensagem de aviso) e `vulkanWindow.cpp` (registro e remoção do hook).
- **Suspeitos para A/B (um por vez, mesma cena):** `KYTY_DCC_GPU=0` (a config do Crash no `Kyty.ini` já
  tem DCC desligado; este run estava com ele ligado) e `KYTY_UPLOAD_DMA=0`. Só correlação até aqui.
- **A linha de comando do run não bate com o `Kyty.ini`**: `--printf-output-file guest-audio.log`,
  `--shader-log-direction Console`, `--redzone`, `--profile`, `--amd-cpu`. Foi iniciada pelo launcher
  (pid 10740), mas com valores que o ini do Crash não tem. O shader log em `Console` gera logs enormes.
- **Falta (separado):** `EmergencyShutdown` não deveria desmontar o Tracy com threads vivas; baixa
  prioridade, só esconde a mensagem útil sob um segundo erro.
- **Mitigação aplicada (só do sintoma secundário):** `profiler_enabled=false` só no Crash (`Kyty.ini`,
  backup no scratchpad). É o que `docs/EXPERIMENTAL.md` já pede para runs limpos. Não evita o
  `ErrorDeviceLost`. `tools/bench_boot.ps1` agora também roda **sem**
  `--profile`, a menos que se passe `-Profile`.
- **Ressalva sobre as medições deste documento:** todas foram feitas com `--profile` ligado (a config
  do usuário). O Tracy custa tempo, então os números de fps podem melhorar sem ele; refazer em B1.
- Variável nova em relação ao último run estável do Crash: `program_cache_enabled=true` (A1). Se o
  crash voltar com o Tracy desligado, testar com ele desligado.

## Crash do `AkRoomVerb.prx` reproduzido (bench, 22:44, a ~62 s)

O jogo "fechou sozinho" em `tools/bench_boot.ps1` (exit code 321). É o mesmo crash do `guest-abort.log`
de 20:13, agora com o contexto completo (`%TEMP%\kyty-bench-20260930-224254\clean-1.log`):

- `thread: AK::EventManager` (thread do Wwise, `os_thread_id = 19380`), `pc = AkRoomVerb.prx+0xdb73`,
  `movl %r11d, 0x10(%rdi)` com `rdi = 0`, `red_zone_protection=false`.
- A função `0xd9e0` do plugin (chamada de `0x901d`; endereço de retorno `0x9022` na pilha) guarda
  `rdi`, `r8`, `r9d` e `rcx` **abaixo do `rsp`** (red zone: `-0x10`, `-0x18`, `-0x1c`, `-0x8(%rsp)`,
  em `0xdb33..0xdb42`) e recarrega `rdi` de `-0x10(%rsp)` em `0xdbf2`. O `rdi` era válido na entrada
  (a primeira instrução lê `0x10(%rdi)`). O dump da red zone no log mostra `-0x10(%rsp)` e `-0x18(%rsp)`
  **zerados** com `-0x8(%rsp)` intacto.
- Leitura: algo zerou a red zone do guest durante a função (no Windows, uma exceção ou APC no mesmo
  thread escreve abaixo do `rsp`). **Inferência, não medida.** Este run não tinha `--redzone`
  (a proteção é opt-in; `eboot.bin` tem 1155 funções com red zone e `patched=0`).
- **Teste:** repetir com `-ExtraArgs '--redzone'` (checkbox "red zone protection" no launcher).
- Dado de desempenho do mesmo run: a thread mais ocupada era a própria `AK::EventManager` (82% de um
  núcleo em média), CPU total de ~1,8 núcleos. De 53 s até o crash, 50–95 programas compilando por
  segundo, `gfx_wait` de 1–2 s e `flips` ~0.

## Bench do Crash com `--redzone` (22:48, 106 s)

- Passou dos 62 s sem o crash do `AkRoomVerb` (um único run: indício a favor da red zone, não prova) e
  **não** reproduziu o `ErrorDeviceLost` de 47 s. Terminou com `unsupported sampled depth image`
  (`descriptors.cpp:290`, `EXIT` deliberado): `resource=1 encoding=1 view=0`, só o check de view falha;
  imagem `D32_SFLOAT_S8_UINT` acompanhada pelo emulador, lida pelo descritor do jogo como RGBA8 sRGB
  (`view_format=43`, endereço `0x20843a0000`, `0x1fe0000` bytes). Leitura do código, não medido: reuso
  da mesma memória para duas finalidades.
- **Gameplay lento (60–106 s, 12,5 fps):** `RenderThread 1` 97% de um núcleo, thread de compilação do
  emulador (a do trabalho da GPU guest) **86%** (7% na parte fluida), `MainThread` 7%. Fila gráfica do
  guest ocupada ~1000 ms/s, GPU real ocupada 179 ms/s, ociosa 841 ms/s (`gpu_starved` 333 ms/s). Ou
  seja, limite de CPU no lado do emulador, não da GPU.
- **Contadores por segundo na fase lenta:** 11,8 mil falhas de escrita (`mem_write_faults`), 17,9 mil
  chamadas de proteção de página, 17,7 mil uploads de buffer, `compile_stall` 154 ms/s, 11 programas e
  8 pipelines compilados por segundo, 37 readbacks. Compilação sozinha explica uma fração (~125 ms/s de
  trabalho, ~150 ms/s de stall); a rotina de rastreamento de memória é o destaque.
- `AK::EventManager` gasta ~95% de um núcleo o tempo todo (também no trecho fluido): espera ocupada.

## Bench do Crash com o preset (22:53, 138 s, saída limpa) e perfil de thread

- O preset (`tools/u59-preset.json`, DCC desligado) **não melhorou a cena lenta**: nos segundos lentos
  (1–30 fps depois de 55 s) 12,9 fps no baseline contra 14,5 com o preset, com os mesmos contadores
  (~11 mil falhas de escrita/s, ~17 mil uploads de buffer/s). `RenderThread 1` 91–93% e a thread de
  compilação do emulador 67–82% nos dois. Diferença entre posições do jogador não está controlada.
- Faltava saber **onde dentro das duas threads** o tempo vai. Novas ferramentas:
  - `tools/thread_sampler.cpp`: amostrador sem admin (suspende a thread ~250 vezes/s e anota o `rip`).
    Compilar com `clang-cl /EHsc /O1 tools\thread_sampler.cpp`; cópia em `install-claude`.
  - `tools/analyze_samples.py`: perfil por thread; funções do emulador pelo PDB (`llvm-symbolizer`),
    DLLs do Windows pelo export mais próximo (exceção, proteção de página e espera separadas), código do
    guest por módulo (precisa de `-Printf File` para os nomes de thread e módulos).
  - `tools/bench_boot.ps1` dispara o amostrador sozinho depois de `-SlowSeconds` (8) segundos abaixo de
    30 fps (a partir de `-SlowAfter` = 50 s), por `-SampleSeconds` (20), nas 5 threads mais ocupadas.
  - Validado com um emulador de mentira (fluxo completo) e com endereços reais contra o
    `llvm-symbolizer`. **Não rodado com o jogo.**

## Primeiro perfil da cena lenta do Crash (23:02, 20 s a 250 Hz, `--redzone`, sem preset, ~14 fps)

Run de 240 s, sem crash. Perfil por thread (`profile-profile-1.txt`):

- **`RenderThread 1`** (UE4): 71% das amostras em código do jogo, **53% numa única função**
  (`eboot.bin+0x14e88b0`, chamada em laço de `0x14ea460`): pop de lista lock-free (índice de 26 bits,
  elementos de 24 bytes, 3 `mfence` por tentativa), isto é, a thread consulta uma fila de tarefas. 21%
  bloqueada (`ZwWaitForAlertByThreadId`), 6% emulador, **~1% proteção de página, ~0% exceções**. A suspeita
  anterior de que o tratamento de falhas de escrita consumia a thread de render **foi refutada**.
- **Thread do emulador que cuida da GPU** (host, 66% de CPU): perfil plano, **30% esperando**;
  `ZwProtectVirtualMemory` 4,8%, esperas do driver (`D3DKMT WaitForSynchronizationObjectFromCpu` 3,7%,
  `SubmitCommandToHwQueue` 2,1%), `BufferCache::SynchronizeBuffersInRange` 3,4%, `DrawPrep::CommitHead` 2,9%.
- **Workers de `DrawPrep`**: ~62% ociosos.
- **`AK::EventManager`**: espera ocupada, ~95% de um núcleo (`KernelGetProcessTime` 10%,
  `RtlQueryPerformanceCounter` 11%, laço do jogo em `eboot+0x465fc95`); desperdício de CPU, não o limite do fps.
- **Leitura:** nenhuma thread é um limite duro de CPU; o pipeline parece limitado por dependência/latência
  (a render thread espera trabalho ou fence; a thread da GPU espera 30%; GPU real ~80% ociosa). **Falta
  saber o que a render thread espera.** Sem pilhas de chamada isso não aparece.
- **Ferramentas:** `tools/thread_sampler.cpp` agora captura pilhas (código do emulador pelas tabelas de
  unwind via DbgHelp, código do guest pela cadeia de `rbp`) e `tools/analyze_samples.py` mostra a contagem
  **inclusiva** e os caminhos de chamada mais frequentes. Validado com uma cadeia de funções montada à mão
  fora de qualquer módulo (endereços de retorno corretos). `tools/bench_boot.ps1` ganhou `-PresentMode`
  (o log diz que `Mailbox` cai para `Fifo`, vsync).

## Perfil com pilhas da cena lenta (23:18, ~8,7 s de amostras, tid da GPU = thread `GuestGpu::ThreadRun`)

Atenção: este run teve o fps derrubado pelo próprio amostrador (7–9 fps; corrigido depois: a pilha agora
é copiada com a thread suspensa e desenrolada depois). Os percentuais são fatias das amostras de cada thread.

- **Thread do emulador que cuida da GPU:** `CommandProcessor::ProcessPacket` 74%, `DrawPrep::Engine::
  CommitHead→Commit` 63%, **`RenderExecutor::ExecutePreparedDraw` 52%**. Dentro dele, 66% é
  `PrepareGraphicsBindings` (35% da thread), e **65,8% disso é `RenderContext::PrepareBda` = 22,7% da
  thread inteira**. `PrepareBda` é 100% `SynchronizeBdaBuffers → SynchronizeBdaBuffersNow` (o caminho
  completo, não o atalho por época), 93% em `SynchronizeBuffersInRange`, e dentro de `SynchronizeBuffer`:
  33% `RegionManager::UpdateProtection` (proteção de página), 19% `HangTrace::RecordTransfer` (custo do
  próprio hang trace, ~3,5% da thread), o resto de contabilidade de uploads.
- **Workers de `DrawPrep`:** 99% parados nesta configuração.
- **`RenderThread 1`:** 90% em código do jogo, laço da fila de tarefas, sem espera.
- **`AK::EventManager`:** espera ocupada vinda de `KernelWaitEventFlag` (22%) e `AudioOut2ContextGetQueueLevel`
  (14%); consome um núcleo, não limita o fps.
- **Resolução do jogo:** janela 1280×720 (config); o jogo renderiza em **3840×2160** (buffer de vídeo
  registrado pelo jogo; 917 imagens de 3840×2160 no log). Não há opção de escala interna no fork; o
  upstream só tem `--gpu-timestamp-scale` (estica o tempo de GPU que o jogo mede, para o resolution
  dinâmico dele). GPU real ~18% ocupada na cena lenta: reduzir resolução não ataca o gargalo.
- **Hipótese de trabalho (não medida):** as ~11,8 mil falhas de escrita por segundo (~840 por frame) vêm de
  o jogo reescrever, a cada frame, páginas de buffers dinâmicos que o emulador mantém protegidas; cada
  página custa falha + upload + nova proteção. O mecanismo de "páginas quentes" existe
  (`KYTY_BDA_HOT_SYNC`) mas as contagens não mudaram com o preset.

## Achados do A6

Sonda Vulkan própria (programa pequeno que carrega `vulkan-1.dll`, sem SDK) na RX 9070 XT,
Vulkan 1.4.349, driver `0x0080018b`:

- `VK_EXT_attachment_feedback_loop_layout`: **sim**. `VK_EXT_attachment_feedback_loop_dynamic_state`: **não**.
- `VK_EXT_device_fault` (feature `deviceFault` ligada), `VK_AMD_buffer_marker` e
  `VK_EXT_device_address_binding_report`: sim. Úteis para o `DeviceLost` (A5/B2).
- `VK_EXT_graphics_pipeline_library` e `VK_EXT_pipeline_creation_cache_control`: sim.

Consequências verificadas no código:

1. **A suíte de shaders do fork não roda nesta GPU.** O `VulkanHarness`
   (`tests/ShaderRecompilerComputeTests.cpp`, ~25730) exige as duas extensões de feedback e aborta com
   `production rasterization features are not supported` (a tentativa de rodar está no log desta
   sessão; o build do alvo funcionou). Então não existe validação do recompilador do fork nesta GPU.
   A única execução registrada nas notas do repo é a do OpenCode, numa RTX 2060. O
   `performance-amd.md` do upstream diz ter sido escrito numa RX 9070 XT e relata falhas que só
   apareceram na suíte completa em Radeon (relato deles, não reproduzido aqui).
2. **No emulador, `attachment_feedback_loop_enabled` exige as duas extensões**
   (`vulkanWindow.cpp`, ~791), então fica `false` aqui (o log do boot diz
   `Vulkan depth feedback support: false`). Sem isso o fork não usa o layout de feedback, o uso de
   imagem `eAttachmentFeedbackLoopEXT` nem o estado dinâmico. O upstream passou a usar flags estáticas
   no pipeline quando só o layout existe. **Não sei se isso causa o escuro ou os pontinhos verdes do
   Astro**; é uma hipótese a testar (B5), não uma causa demonstrada.

Os três itens de aritmética/indexação do upstream, lidos contra o código do fork:

- **`V_FRACT_F16`**: o fork converte para f32, aplica `FPFract32` e converte de volta, sem clamp.
  Pela conta (não executada): para entrada `-2^-12`, `1 - 2^-12` fica exatamente no meio entre
  `0x3bff` e `0x3c00`, e o arredondamento para par dá 1,0 (`0x3c00`). O upstream considera o
  resultado correto como `0x3bff`, citando uma seção do manual RDNA2 da AMD. Sem oráculo de hardware
  não dá para dizer qual lado está certo; não alterar sem medição (o PS5_Vulkan, que mede no
  console, seria o oráculo).
- **`ADD_TID`**: `BufferLane` faz `lane & 63` fixo. Wave32 do guest em subgrupo host de 64 não
  reduziria a 0..31. Nesta GPU o pipeline pede subgrupo 32 para wave32 (`size_control=true`), então
  não deve haver efeito aqui; vale um teste de código gerado quando o A7 existir.
- **Constante inline 1/(2π) em 16 bits**: o fork converte o f32 `0.15915494` pela conversão do host,
  que em arredondamento ao mais próximo dá `0x3118`, o valor arquitetural. Sem falha demonstrada.
  Fixar `0x3118` removeria a dependência do modo de arredondamento do host (endurecimento barato).

## Árvore e donos (proposta A2, a confirmar)

Itens sem commit hoje, agrupados por tema. "Dono provável" vem de horário de edição e do conteúdo,
não de confirmação.

| Tema | Arquivos | Dono provável |
|---|---|---|
| Vídeo | `src/libs/avPlayer.cpp`, `src/libs/avPlayerSync.h`, `tests/AvPlayerSyncTests.cpp` | a confirmar |
| Rede | `src/libs/network.cpp`, `src/libs/network.h`, `tests/HttpWaitRequestTests.cpp` | a confirmar |
| Launcher (DCC, cache de programas e biblioteca de pipelines por jogo) | `configuration_edit_dialog.ui`, `configuration.h`, `configurationEditDialog.cpp`, `mainDialog.cpp` | a confirmar |
| Loader / caminho UTF-8 | `src/loader/runtimeLinker.cpp`, linha `PathFromUtf8` em `libAmpr.cpp` | a confirmar |
| APR | `src/libs/aprHostFilePool.h`, resto do diff de `libAmpr.cpp`, `tests/AprHostFilePoolTests.cpp` | esta sessão |
| Diagnóstico de `DeviceLost` | `graphicContext.h`, `shaders.cpp`, `vulkanWindow.cpp`, `renderCompute.cpp` | outra sessão |
| Ferramenta de benchmark | `tools/bench_boot.ps1` | esta sessão |
| Este plano | `docs/CLAUDE-ORDEM-DE-TAREFAS.md` | esta sessão |

Pontos de atenção:

- `CMakeLists.txt` tem hunks de três temas (vídeo, rede, APR). Cada commit deve levar o hunk do seu
  teste junto com as fontes, senão o build quebra no meio da história (`git add -p`).
- `docs/performance-amd.md` e `docs/perf-research/` são cópias de docs do upstream `KytyPS5-main`,
  ainda sem rastreio. Decidir se entram (com a origem citada) ou ficam fora.
- `guest-abort.log` (72 MB), `guest-audio.log` (84 MB) e `guest-astro-gpu.log` não estão no
  `.gitignore`. Sugestão: ignorar `guest-*.log`. Nunca commitar.

## Como usar o que ficou pronto

- Benchmark (precisa de janela de jogo):
  `.\tools\bench_boot.ps1 -InstallDir <install> -GameDir <dir do jogo> -TitleId PPSA02433 -Seconds 60 -Repeat 3 -PresetFile tools\u59-preset.json -Variant 'base:KYTY_DCC_GPU=0;KYTY_PROGRAM_CACHE=1;KYTY_PIPELINE_LIBRARY=0','preset'`
- Analisar um trace existente sem rodar nada: `.\tools\bench_boot.ps1 -AnalyzeOnly <pasta _HangTrace>`.
- Teste do pool: build do alvo `apr_host_file_pool_tests` e `ctest -R apr_host_file_pool`; custo por
  leitura com `apr_host_file_pool_tests --bench`.

## Limites desta nota

- Nenhuma mudança de código foi validada em jogo.
- Os números de fps são de boot, uma amostra por configuração (exceto onde há repetição indicada).
- A medição do "preset ligado" ainda não existe; o efeito dele nas threads de compilação é hipótese.
- O Context7 não estava disponível na sessão, então nada aqui usou documentação externa dele.


## Crash 4 com pipeline library: ErrorDeviceLost com relatório de falha (2026-10-01)

Run `kyty-bench-20261001-000316` (`KYTY_PIPELINE_LIBRARY=1`, preset, `KYTY_GPU_FAULT_REPORT=1`): o jogo "fechou sozinho"
em 10 s, código de saída 321 (`EXIT_HALT`), com `ErrorDeviceLost` reportado ao criar o 5º pipeline de compute
(shader `0x9bf4cf7e4ad6231d`, 52 palavras). **É o primeiro relatório de falha capturado**:

- `address: type=ReadInvalid reported=0x0000beefdeadb000 precision=4096`, sem informação de fornecedor.
  O endereço é o `0xBEEFDEADBEEF` arredondado para a página de 4 KiB (inferência): um ponteiro-veneno. O código do
  emulador não gera esse valor (só `0xDeadBeef00000007` em `libKernel.cpp`, o canário da pilha); a origem
  (jogo ou driver) é **desconhecida**.
- O erro é "pegajoso": aparece na próxima chamada Vulkan, então o trabalho culpado foi submetido antes do pipeline.
- Os 3 runs sem pipeline library (preset apenas) nesta sessão rodaram de 146 a 187 s sem `ErrorDeviceLost`; o único
  com a flag caiu em 10 s. Sugere, **não prova** (uma amostra), que a flag é arriscada no Crash. No Astro a flag só
  deslocava o momento da queda.
- Consequência: a pipeline library **não** é, por ora, a saída para os travamentos de compilação (36% dos segundos
  em ≤ 2 fps, 52 s de `GetGraphicsPipeline` no run sem a flag). Ver `CLAUDE-DESENHO-SYNC-BDA.md`.

Varredura de memória na queda (feita, **ainda não vista em ação dentro do emulador**): depois do relatório de falha,
`ReportDeviceFault` procura em memória do guest valores de 8 bytes cujos 48 bits baixos caem na página do endereço
inválido e imprime `--- Guest memory scan for the faulting address ... ---` com os trechos achados (endereço do guest,
offset no dmem, tamanho, valor). Lógica em `src/kernel/pointerScan.h` (teste `pointer_scan`), varredura em
`GuestBackingStore::ScanForAddressRange`. Detalhes que importam: o estado do working set é **por view** (testado: página
escrita só pela view do guest aparece `Valid=0` na view do backing), por isso varre as views do guest; só lê páginas com
entrada de tabela válida (não cria páginas nem dispara o rastreio de escrita); orçamento de 900 ms por endereço, no
máximo 2 endereços, para o processo não sair no meio. Páginas fora do working set não são cobertas.

**Segundo run (`kyty-bench-20261001-003058`) com a varredura:** a queda é **determinística** (2 de 2): mesmo endereço
(`0xbeefdeadb000`), mesmo shader (`0x9bf4cf7e4ad6231d`), mesmo ponto (5º pipeline de compute), em ~11 s e **antes de
qualquer frame**: `flips = 0`, `gpl_links = 0`, `gpl_libraries = 0`, 0 pipelines gráficos, só 4 de compute e 5
programas. Ou seja, nenhum pipeline gráfico passou pela pipeline library: a flag só mudou a criação do dispositivo
(extensões e features de pipeline library), não o caminho de link. A varredura rodou no emulador (20 ms, 0,15 de
0,35 GiB das mapeações lidos) e achou **0 ocorrências**: o valor não está nas páginas do guest que foram lidas
(páginas inacessíveis por rastreio de escrita ou fora do working set não são cobertas). Próximo passo barato:
repetir com `--graphics-debug-dump` (só ~11 s de log) para ver os dispatches com o hash do shader, e rodar o Astro
com o relatório de falha para ver se o endereço é o mesmo (o crash de boot do Astro também é `ErrorDeviceLost`).

Ferramentas: `thread_sampler.exe` dava violação de acesso em `memmove` dentro do desenrolar de pilha (dois runs com
amostras cortadas); agora cada desenrolar é protegido por SEH: a pilha daquela amostra é descartada (a folha fica) e a contagem aparece no fim.
A causa exata da violação de acesso não foi encontrada; o teste de 5 s a 250 Hz contra outro processo deu 0 falhas.
