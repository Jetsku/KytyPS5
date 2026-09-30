# Handoff OpenCode → Claude Code

Atualizado: 2026-09-30. Autor: sessão OpenCode.
Base inspecionada: `4a50f8bd278a5c4e6e7023e6506bd4d8111b2e0f`.

## Objetivo e estado

Atualização OpenCode após implementação: build Windows/RTX 2060 concluído;
4/4 testes novos e 17/17 regressões focadas passaram. Corrigido dispatch/parser
de RELEASE_MEM nativo `0x49`. Detalhes em [GUEST-SYNC-TESTS.md](GUEST-SYNC-TESTS.md).
As seções seguintes registram a análise inicial. Sem commit/push ou benchmark.

O usuário pediu uma análise de possíveis gargalos **no código**, trouxe o commit
Senaxx `309ba4f5955bfd7e87a24a55600fd5dad351d629`, perguntou se Rust ajudaria e solicitou
compartilhamento de conhecimento com uma sessão Claude Code.

- Realizado: análise estática de caminhos do renderer, coerência de memória e I/O;
  leitura do diff e dos caminhos relevantes do commit externo.
- Não realizado nesta sessão: implementação, build, execução de testes ou benchmark.
- Nenhuma migração para Rust ou otimização foi aprovada para implementação.
- As prioridades abaixo são recomendações, não tarefas já assumidas por agentes.
- O arquivo preexistente e não rastreado `KYTYPS5_AGENT_PROMPT.md` trata de uma
  avaliação de PS5_Vulkan. Foi preservado; não deve ser confundido com esta tarefa.

## Achados estáticos

As referências são relativas a este diretório. As linhas correspondem à base
inspecionada; conferir os símbolos se a outra sessão modificar o código.
Os comportamentos foram encontrados no código, mas sua contribuição para o tempo
de frame/carregamento não foi medida nesta sessão.

| Área | Evidência | Hipótese de impacto / próximo exame |
| --- | --- | --- |
| AIO síncrono | [`KernelAioSubmitReadCommands`](../src/libs/libKernel.cpp#L2856) chama `KernelPread` sequencialmente e retorna o ID após terminar o lote. A escrita tem o mesmo desenho em L2894. | Bloqueia a thread de submissão e elimina sobreposição de I/O nos jogos que usam esse caminho. |
| Leitura serializada por descritor | [`KernelPread`](../src/kernel/fileSystem.cpp#L748) segura `file->mutex` durante invalidação de memória, busca de posição, leitura e restauração do cursor. | Apenas colocar workers no AIO não permite leituras paralelas pelo mesmo descritor. Avaliar leitura posicional e contratos de fechamento/memória. |
| Criação de pipelines sob lock | [`GetGraphicsPipeline`](../src/graphics/host_gpu/renderer/pipeline/pipelineCache.cpp#L4303) mantém `m_mutex` durante `CreatePipelineInternal`; compute faz o mesmo em L4398. Workers retornam `PlanLookup::Busy` em L4184–4196. | Stutter no miss e perda de preparação antecipada. Reutilizar o padrão de compilação em andamento/publicação do cache de programas; tirar criação do lock não elimina a espera do primeiro consumidor. |
| Espera e retrabalho de draw prep | [`CommitHead`](../src/graphics/host_gpu/renderer/drawPrep/drawPrep.cpp#L945) espera o primeiro item; [`AwaitHead`](../src/graphics/host_gpu/renderer/drawPrep/workerGate.h#L213) faz polling. [`RefreshShaders`](../src/graphics/host_gpu/renderer/renderDraw.cpp#L1874) refaz preparação quando o certificado é rejeitado. | Medir `DrawPrepCommitWait`, `DrawPrepValidate` e `DrawPrepFallback*`; preservar ordem de commit. Mais workers/janela maior não garantem ganho. |
| Readback com submissão e espera | [`ReadMemoryDrain`](../src/graphics/host_gpu/renderer/cache/bufferCache.cpp#L1186) considera janela de 512 KiB nas leituras e espera o tick corrente se houver download. [`CommandScheduler::Wait`](../src/graphics/host_gpu/renderer/commandScheduler.cpp#L257) submete quando esse tick ainda está em gravação. | Investigar `GpuWaitDrain` e recusas do side readback (`CurrentWriter`, `Unbounded`, etc.). A janela não significa que todos os seus bytes sejam sempre copiados. |
| Dependências amplas de compute | [`DispatchDirect`](../src/graphics/host_gpu/renderer/renderCompute.cpp#L408) e indireto (L540) inserem `ShaderAccessBarrier` após o dispatch; [`ShaderAccessBarrier`](../src/graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.cpp#L136) usa destino `eAllCommands`. | Pode limitar sobreposição de trabalho independente; batching reduz chamadas, não necessariamente o alcance das dependências. Exige prova de hazards para estreitar. |
| Lock longo de texturas | [`FindTexture`](../src/graphics/host_gpu/renderer/cache/textureCache.cpp#L3694) segura `m_lock` sobre residência, refresh e aquisição da view. [`InitializeImage`](../src/graphics/host_gpu/renderer/cache/textureCache.cpp#L2188) pode obter buffers e registrar upload; invalidação também adquire o lock (L4186). | Medir posse/espera por operação antes de separar etapas; preservar identidade, lifetime e versão da imagem. O lock usa parking, não é apenas spin ilimitado. |

## Commit externo e diferença importante do fork

Fonte: <https://github.com/Senaxx/KytyPS5/commit/309ba4f5955bfd7e87a24a55600fd5dad351d629>.

O diff adiciona um contador global incrementado na primeira fatia de uma submissão
guest e um controle em `PrepareBda` para evitar repetir a sincronização sob o mesmo
valor do contador, mesmo se o CPU-dirty epoch mudou. O contador é global, não um
identificador estável armazenado em cada submissão retomada.

**Resultados relatados pelo autor, não reproduzidos aqui:** cerca de 44.000 chamadas
de `SynchronizeBuffer` por frame, 0,8% com upload; aproximadamente 38% da thread GPU
do host no menu; menu 21,4 → 31,8 FPS; selva 8,8 → 9,7 FPS. Não são resultados deste
checkout nem evidência de um ganho adicional aqui. “Thread GPU” é trabalho de CPU
do emulador, não percentual de utilização física da GPU.

Neste fork, [`PrepareBda`](../src/graphics/host_gpu/renderer/renderContext.cpp#L217)
delega a `BufferCache::SynchronizeBdaBuffers`. Já existem:

- Skip por época de sincronização **e** geração estrutural dos buffers/mapeamentos
  ([bufferCache.cpp:3307](../src/graphics/host_gpu/renderer/cache/bufferCache.cpp#L3307)).
- Sincronização incremental, tratamento de hot pages e log de regiões modificadas
  ([bufferCache.cpp:3368](../src/graphics/host_gpu/renderer/cache/bufferCache.cpp#L3368)).
- Épocas que consideram waits, packets de escrita/invalidação, fatias e comandos de
  serviço ([syncEpoch.h](../src/graphics/host_gpu/syncEpoch.h)).
- `KYTY_BDA_INCREMENTAL_SYNC=1` no [preset U59](../tools/u59-preset.json).
- Modos de verificação e um teste `CheckBdaSyncEpoch` em
  `tests/ShaderRecompilerComputeTests.cpp` (localizado, não executado nesta sessão).

Não tratar o patch externo como transplante direto. Um caso relevante de regressão
é: primeiro uso BDA → CPU escreve dados e sinaliza uma label → wait guest ordena
essa escrita → novo uso BDA dentro da mesma submissão. É preciso garantir a
visibilidade dos dados nesse segundo uso. Trata-se de um cenário a validar, não
de uma falha reproduzida do commit externo nesta sessão.

## Rust: conclusão discutida, não decisão de migração

Reescrever os mesmos algoritmos em Rust não remove esperas, varreduras, barreiras
ou serialização. O benefício provável seria ownership/lifetimes mais explícitos
e prevenção de certas classes de erro de memória/concorrência.

Um executor de I/O com fronteira por lotes é candidato a piloto mais delimitado
que o renderer. Exige contrato de validade de memória guest, descritores,
conclusão, cancelamento e publicação de resultados. A linguagem não estabelece
sozinha esses contratos, nem a coerência CPU/GPU. Corrigir o AIO em C++ também é
uma opção e provavelmente requer menos integração.

## Próximos passos propostos

1. Para FPS: analisar custo residual de BDA e causas de full scan, preservando
   `SyncEpoch` e invalidação estrutural. Comparar contagem de buffers examinados
   com uploads efetivos.
2. Para piloto Rust: definir primeiro o contrato do AIO e confirmar que o workload
   alvo usa essas funções. Ainda depende da escolha do usuário.
3. Para alteração delimitada no renderer: estudar criação de pipelines fora do
   lock com deduplicação de compilações em andamento.

## Troca com a sessão Claude Code

Não há canal direto entre os chats; este arquivo permite troca pelo checkout.

- Ler este handoff e conferir a revisão/estado atual antes de assumir evidências.
- Para retorno, criar ou indicar uma nota própria (por exemplo,
  `docs/CLAUDE-FINDINGS.md`) com revisão, achados, testes executados, divergências
  e arquivos em edição. OpenCode poderá lê-la na próxima interação.
- Identificar autoria nas notas. Preferir um arquivo por sessão para evitar
  sobrescrita simultânea deste handoff.
- Registrar quem assumiu cada alteração antes de editar os mesmos arquivos;
  sugestões desta nota não significam que alguém iniciou a implementação.
- Não inferir concordância, teste concluído ou trabalho iniciado da outra sessão
  sem um retorno explícito dela.
