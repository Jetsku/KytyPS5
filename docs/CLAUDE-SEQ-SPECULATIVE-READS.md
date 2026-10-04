# Sequenciador CP: preparar os draws durante as paradas (desenho)

Autor: sessão Claude Code, 2026-10-04. Base: `c62a6c5c`. É uma proposta: nada aqui foi implementado.
As medições estão em [CLAUDE-DRAW-HOTPATH-2026-10-04.md](CLAUDE-DRAW-HOTPATH-2026-10-04.md).

## Problema medido (Crash Bandicoot 4, cena pesada, ~16 fps, preset U59)

- A CP (resolver) gasta ~31% do tempo em `CommitHead`/`AwaitHead`, esperando o worker que prepara o
  draw da vez: ~5-7 mil esperas/s, ~12,5 µs cada (`DrawPrep::Totals`).
- 98% dessas esperas são do **primeiro draw publicado depois de uma parada** do sequenciador
  (`after_stop`). Na parada a janela esvazia. O sequenciador só publica o draw seguinte depois da
  resposta, e um worker "quente" o pega antes de a CP chegar. A CP então espera o preparo
  inteiro (preparar ela mesma custa ~20 µs, o que é pior).
- Paradas por tipo (`KYTY_CP_WAIT_STATS=1`, 10 s): `wait-self-satisfied` 18-27 mil, `read-data`
  21-75 mil, `condition` ~2,5 mil, `flip-wait` ~0,3 mil. O `read-data` são as leituras de
  `SET_*_REG_INDIRECT` (`ReadRegisterPairs` → `ReadGuestForFront` → `LockstepRead`) em páginas que
  a GPU tocou. `KYTY_CP_SEQ_TOUCHED_DIAG=1`: 0 dessas leituras caem em linhas de 64 B não tocadas.
- `KYTY_CP_WAIT_FORWARD=1` (não parar em `wait-self-satisfied`) não deu ganho: as esperas depois de
  paradas continuaram por causa das `read-data`.

Teto do ganho: o tempo dessas esperas, ~6-9% da CP no Crash (talvez +5-8% de fps). Isso é uma
estimativa, não uma medição.

## Restrições

1. **Vulkan:** um comando gravado não pode ser desfeito, só com o reset do command buffer inteiro
   (`vkResetCommandBuffer`, fora do estado pendente) ou do pool. Os pools exigem sincronização
   externa. Confirmado no Context7 (Vulkan-Docs, "Command Buffers > Lifecycle / Command Pools").
   Portanto a especulação acaba antes de qualquer `vkCmd*`, na camada do DrawPrep, cujos workers
   já nunca tocam Vulkan nem caches (`CommandScheduler::CheckActive` faz `EXIT_IF(IsWorkerThread())`).
2. **Ring de ops sem rollback:** uma op emitida é executada. As ops nunca podem depender de bytes não
   validados. O que pode ser descartado são slots da janela (`SkipSlots` → `SkipPublished`, sem `Commit`).
3. **Uma resposta por vez:** `Sequencer` tem um único slot de resposta (`m_answered`). Só uma op
   lockstep fica pendente por vez.
4. **Leitura sem falta:** o sequenciador pode ler a cópia de apoio (`GuestBackingStore::TryReadBackingDirect`,
   alias memfd) sem falta de página em qualquer proteção. Os bytes podem estar velhos em relação a
   dados GPU-dirty ou ainda não publicados. Servem só para especular.

## Mecanismo existente que o desenho reaproveita: P3c (`KYTY_CP_SEQ_PREFETCH`)

- Numa `WAIT_REG_MEM` sobre a própria label já satisfeita, `RunPrefetch` faz um parse adiantado numa
  cópia do estado do front (`CopyFrontStateForPrefetch`, `FrontMode::Prefetch`) enquanto o sequenciador
  espera a resposta. O parse sombra **não emite ops** (`SubmitPrefetch`), não altera o estado do front
  real e para em qualquer op lockstep ou leitura de bytes tocados.
- Cada draw que ele encontra vira um slot especulativo (`PublishSpeculative`) com uma chave: número de
  pacotes e hash XXH3 encadeado dos dwords dos pacotes **mais os bytes das tabelas de registradores**
  (`TrackAdoptInputs`) desde a espera.
- O parse real, depois da resposta, calcula a mesma chave. Se for igual, adota o slot (`TryAdopt`) e o
  draw já está preparado. Se for diferente, os slots são descartados (`DropSpeculative` + `SkipSlots`).
  Mesmos pacotes e mesmos bytes dão mesmos registradores, e a preparação ainda passa pelo certificado
  do DrawPrep no commit. A correção vem da construção.
- **Nunca foi medido em jogo.** Está fora do preset. Os contadores `prefetch_published/adopted/skipped`
  existem em `DrawPrep::Totals`, mas só os testes os leem.

## Desenho em fases

### Fase 0 — medir o P3c como está (sem código novo de lógica)

- Tornar `KYTY_CP_SEQ_PREFETCH` um `Live::Switch` (padrão desligado) e imprimir os contadores de
  prefetch na linha `DrawPrep 10s`. Os dois mudam só a observação e a seleção do modo.
- Rodar Astro e Crash com `KYTY_CP_SEQ_VERIFY=exit` + P3c ligado: correção em jogo primeiro.
- A/B live no Crash (`ab2.py`): fps, CP ocupada/frame, `head_waits_after_stop` e taxa de adoção.
- Esperado: cobre só as paradas `wait-self-satisfied`. O parse sombra para na primeira leitura tocada,
  então a cobertura depende de quantos draws vêm antes da próxima `read-data`.

### Fase 1 — especular através da leitura lockstep (o núcleo da proposta)

Em `ReadGuestForFront`, quando a faixa é tocada e por isso exige `LockstepRead`:

1. O front real emite a `LockstepRead` como hoje. O ring e as ops ficam iguais.
2. Antes de `AwaitAnswer`, como o P3c, roda um parse sombra a partir do pacote corrente. A diferença
   é que os bytes desta leitura vêm de `TryReadBackingDirect` (especulativos, sem falta). Se a leitura
   de apoio falhar (faixa sem memória direta), não especula.
3. O parse sombra publica slots especulativos com a chave de adoção, que já inclui os bytes das
   tabelas lidos, e para na próxima op lockstep ou na próxima leitura tocada (como hoje).
4. O parse real continua com os bytes **oficiais** da resposta. `ReadRegisterPairs` já os dobra em
   `TrackAdoptInputs`. Se forem iguais aos especulativos, as chaves batem e os draws são adotados já
   preparados. Se diferirem, nenhum slot é adotado e todos caem em `SkipSlots`, como no P3c.

Por que é correto: o front real nunca usa um byte não validado e só emite ops com dados oficiais.
Um slot especulativo só é commitado se a chave de adoção (pacotes + bytes lidos pelo front real)
for idêntica à do parse sombra. Um slot não adotado é descartado sem `Commit`. Nada especulativo
chega ao Vulkan nem aos caches.

O que muda no código:

- `ReadGuestForFront`: aciona o parse sombra antes de emitir/aguardar a `LockstepRead`, como
  `SubmitThread` faz para a espera de label (`RunPrefetch`).
- `ReadGuestForPrefetch`: no parse sombra, a leitura que **disparou** a especulação usa os bytes de
  apoio. As demais leituras tocadas continuam parando o parse sombra.
- O ponto de início do hash de adoção passa a ser a leitura, e não só a espera. O pacote
  `SET_*_REG_INDIRECT` corrente precisa entrar na chave do mesmo jeito nos dois fronts
  (o equivalente ao `m_adopt_skip_one`).
- `DropSpeculativeSlots` em toda op lockstep (gR:4294) continua valendo. Não há especulação
  atravessando duas paradas nesta fase.
- Contadores: especulações iniciadas por leitura, adotadas, descartadas por divergência de bytes e
  falhas de leitura de apoio.

### Fase 2 — encadear paradas (opcional, depois da Fase 1 medida)

Manter os slots especulativos através da próxima `LockstepRead` em vez de descartá-los, encadeando a
chave de adoção nos dois fronts. Permitiria especular vários trechos à frente. Exige revisar o
`DropSpeculativeSlots` em ops lockstep e o reinício do rastreamento de adoção. Só vale se a Fase 1
mostrar adoção alta e ainda sobrar espera.

### Fora do escopo

Prever o resultado de `Predication`/`CondExec`/`Branch` (~250/s). Desfazer ops emitidas (exigiria
rollback do ring e do estado do resolver). Gravar especulativamente em secondary command buffers.

## Validação e medição

- **Testes:** os testes do P3c existentes (`ShaderRecompilerComputeTests`, ~16290-16380) mais casos
  novos: leitura tocada com bytes iguais (adota), com bytes diferentes entre apoio e resposta (descarta e
  desenha certo), leitura de apoio impossível (não especula). Todos com `KYTY_CP_SEQ_VERIFY=exit`.
- **Em jogo:** Astro e Crash com `KYTY_CP_SEQ_VERIFY=exit` sem divergências e imagem conferida pelo
  usuário. Depois A/B live (`ab2.py`) com fps, CP ocupada/frame, CPU da CP/frame,
  `head_waits_after_stop` e adoção. A cena deve ser estável, com draws/frame iguais entre variantes.
- **Critério de aceitação:** as esperas da CP depois de paradas caem de forma consistente em todos os
  pares; nenhuma divergência no verify; fps igual ou maior.

## Riscos

- A taxa de adoção depende de os bytes de apoio já estarem certos quando o sequenciador lê. Se a GPU
  escreve essas tabelas e a publicação chega tarde, a adoção cai e o custo é só o parse sombra
  descartado. Medir antes de seguir para a Fase 2.
- O parse sombra ocupa o sequenciador durante a espera. Hoje ele passa ~70% do tempo esperando, então
  há folga.
- `m_predicate_gpu` não é copiado para o front sombra (observado no estudo). Isso é seguro hoje porque
  um predicado de GPU gera uma op lockstep que interrompe o parse sombra. Precisa continuar assim.

## Estado em 2026-10-04 ~05:00 (fim da sessão; continuar amanhã)

### Fase 0 feita (código NÃO commitado; o binário instalado `d84ab046` já a contém)

- `KYTY_CP_SEQ_PREFETCH` virou `Live::Switch` (`cpOps.cpp`; valor inválido no ambiente ainda encerra o
  programa na inicialização; valor inválido no arquivo live vira desligado).
- A linha `DrawPrep 10s` mostra `prefetch published/adopted/skipped` quando houve prefetch (`drawPrep.cpp`).
- **Bug do P3c achado e corrigido:** o front sombra lê tabelas de `SET_*_REG_INDIRECT` antes da espera
  terminar e encontrou uma tabela ainda zerada. O handler encerrou o emulador com
  `unknown sh reg at 000f5: 0x0` (`pm4Handlers.cpp:2128`, exit 65). Agora
  `CommandProcessor::AbandonPrefetchOnInvalidData()` interrompe só a especulação nos pontos de dado
  inválido dos handlers indiretos CX/SH/UC. Os outros fronts continuam com `EXIT`.
- Testes: os ctests de sequenciador/prefetch passam, exceto `cp_sequencer` (sem executável) e
  `shader_recompiler_compute_cp_seq_thread_prefetch_verify` (aborta antes, no `GpuTilerCpuParity`
  preexistente).

### Resultados em jogo (Crash 4, cena pesada, ~16,8 fps)

- `KYTY_CP_SEQ_PREFETCH=1` + `KYTY_CP_SEQ_VERIFY=exit`: ~2,5 min, **0 divergências**, imagem sem glitch
  novo (o branco preexistente continua). Adoção ~100% (31-43 mil slots/10 s, 0 descartados).
- A/B live P3c (4 pares de 10 s, cena estável ~1.865 draws/f): fps 16,80 → 16,87 (+0,4%, ruído),
  CP ocupada −0,5%. **Sem ganho.** As esperas depois de paradas caem um pouco nas janelas com P3c,
  mas não o bastante.

### PROBLEMA ABERTO: quedas `certunclean` para o caminho serial ~20x maiores

Contagens por 10 s na mesma cena (`DrawPrep 10s`):

| execução | build | fallbacks/10 s | observação |
| --- | --- | --- | --- |
| perfil fp (03:00) | bc0b551b (perfil) | ~3,8 mil (384/s, `Totals`) | antes da correção do batch de uploads |
| run-diag.log | 296bc833 (perfil) | ~1,4 mil `certunclean` | com a correção de uploads; `KYTY_CP_WAIT_STATS=1` + `KYTY_CP_SEQ_TOUCHED_DIAG=1` |
| run-ab.log | b7c00ecd (perfil) | **~34 mil** | + `WAIT_FORWARD` live + `HOT_PAGE_MAX_LIVE`; sem diagnósticos |
| run-verify2.log | d84ab046 | ~66 mil | P3c + `VERIFY=exit` |
| run-p3c-ab.log | d84ab046 | **~50 mil** | o mesmo com P3c 0 e 1 |

A ~20 µs por preparo serial, ~5 mil/s seriam ~10% da CP. Logs em `_Build/crash-live-20261004/`.

Hipóteses, em ordem de verificação:
1. **Timing, não regressão:** os diagnósticos (mutex por marca no `g_lines`) deixam o sequenciador mais
   lento. A preparação fica mais perto do commit e o certificado sobrevive. Sem eles, o sequenciador
   corre até 32 draws à frente e as transições de coerência (uploads de constantes reescritas todo frame)
   invalidam mais certificados. Testar: rodar `296bc833` sem os diagnósticos, ou `b7c00ecd`/`d84ab046`
   com eles.
2. **Regressão de uma mudança minha** entre `296bc833` e `b7c00ecd`: `KYTY_CP_WAIT_FORWARD` como
   `Live::Switch` (`graphicsRun.cpp`) e `MemoryTracker::HotMax()` copiando a `FaultPolicy` a cada
   falta (`memoryTracker.h`). Pela leitura do código as duas mantêm o comportamento padrão. A revisão
   de `RecordPendingUploads` também não achou uso da lista truncada fora do ramo não global. Testar:
   bissecção com os binários guardados em `_Build/astro-preset-20261004/bin-*`.
3. **Cena diferente:** comparar draws/frame e o trecho de jogo (todas as execuções foram "no mesmo ponto"
   segundo o usuário, ~1.860 draws/f).

Próximos passos:
- Resolver o item acima antes de qualquer outra medição: ele vale mais que o P3c.
- Commit da Fase 0 (código acima) depois da revisão.
- Fase 1 (especular através de `LockstepRead`): reavaliar. O P3c, que já cobre as esperas por label,
  não deu ganho. O teto previsto da Fase 1 (~6-9% da CP) precisa ser confirmado depois que os fallbacks
  `certunclean` forem entendidos.
