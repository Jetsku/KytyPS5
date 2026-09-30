# PS5_Vulkan: avaliação complementar OpenCode

Data: 2026-09-30. KytyPS5: `4a50f8bd278a5c4e6e7023e6506bd4d8111b2e0f`.
PS5_Vulkan: `d609d712453561efc0fd78b8eb46602d49461229`.

## Contexto e conclusão

Leitura do repositório externo e comparação estática com os caminhos locais.
Também foi lido o retorno [CLAUDE-FINDINGS.md](CLAUDE-FINDINGS.md). Os scripts e
testes relatados naquela nota pertencem à sessão Claude; não foram reexecutados
aqui. Esta avaliação não alterou código, executou jogos ou acessou um console.

O principal valor é obter referências de comportamento e casos de regressão
derivados de medições do console. O driver nativo não substitui o backend Vulkan
do emulador no PC, e seus ganhos não demonstram ganhos no KytyPS5.

## Referências externas fixadas

- [Hardware findings](https://github.com/mihawk-99/PS5_Vulkan/blob/d609d712453561efc0fd78b8eb46602d49461229/docs/HARDWARE_FINDINGS.md)
- [R70: GPU barriers](https://github.com/mihawk-99/PS5_Vulkan/blob/d609d712453561efc0fd78b8eb46602d49461229/jobs/r70-gpu-barrier/README.md)
- [R18: padded mip chains](https://github.com/mihawk-99/PS5_Vulkan/blob/d609d712453561efc0fd78b8eb46602d49461229/jobs/r18-padded-mips/README.md)
- [AGC entry points](https://github.com/mihawk-99/PS5_Vulkan/blob/d609d712453561efc0fd78b8eb46602d49461229/docs/AGC_ENTRY_POINTS.md)
- [Active state](https://github.com/mihawk-99/PS5_Vulkan/blob/d609d712453561efc0fd78b8eb46602d49461229/docs/VULKAN_PROBE_ACTIVE.md)
- [CTS gaps](https://github.com/mihawk-99/PS5_Vulkan/blob/d609d712453561efc0fd78b8eb46602d49461229/docs/CTS_GAPS.md)

São registros publicados pelos autores, não medições reproduzidas por OpenCode.
Os logs brutos mencionados não foram auditados aqui. Não promover ausência de
falhas nos cenários documentados a prova universal de comportamento.

## Cinco oportunidades priorizadas

### 1. Testes de layout, incluindo mipmaps lineares

Claude relata igualdade das equações locais para determinados formatos R_X/Z_X.
Conservar esses resultados como testes reproduzíveis é uma boa primeira entrega.

Complemento: entradas antigas do documento externo dizem que imagens lineares
não selecionam mipmaps. **R18 posterior corrige a generalização**: PID 214,
build SHA-256 `cef1d81708d06d6fa68b2ac5df6b3f781c0fb59e3026e83e09ee469b112167fa`,
relata 19 frames corretos para 224x195/8 níveis, 32x36/6 e 256x256/5 com pitch
no descritor e níveis menores primeiro. PID 211 é o controle que falhou.

O código local já alinha linhas a 256 bytes e coloca níveis menores primeiro:
[`tile.cpp:20–58`](../src/graphics/guest_gpu/tile.cpp#L20). Isso é concordância
estrutural, não uma comparação byte a byte dos três casos nesta sessão.
Próximo passo: fixtures de offsets/pitch e limites das alocações para essas formas.

### 2. Teste ponta a ponta de release/wait/render-to-texture

R70 relata que `RELEASE_MEM` event 20 + `WAIT_REG_MEM64`, controle `0x06000113`,
substituiu splits com espera CPU no ps5vk. PID 614 passou m2-solid, c4-rtt,
c4-texture e v0-subpass; PID 615 recapturou c4-rtt em uma submissão.
O mesmo README relata redução de 2,8 para 0,19 ms de fila no Dolphin/Melee,
um resultado do driver no console, não do KytyPS5.

Localmente existe parser de wait de 64 bits em
[`pm4Handlers.cpp:2477`](../src/graphics/guest_gpu/command_processor/pm4Handlers.cpp#L2477)
e execução em [`graphicsRun.cpp:887`](../src/graphics/guest_gpu/graphicsRun.cpp#L887).
O caso externo pode orientar um teste de ordenação e visibilidade que confira o
resultado renderizado, além da aceitação dos packets.

Achado relacionado: `CpOpAcquireMem` apenas valida cabeçalho e consome words
([L1339](../src/graphics/guest_gpu/command_processor/pm4Handlers.cpp#L1339)). Isso
**não prova que toda sincronização seja ignorada**: a classificação de fences
avança `SyncEpoch`, e dependências também são emitidas no renderer
([graphicsRun.cpp:1763](../src/graphics/guest_gpu/graphicsRun.cpp#L1763)). Auditar a
cadeia completa antes de estreitar barreiras ou acrescentar drains.

### 3. Oclusão: resolver a interpretação dos 16 slots

O material externo afirma que uma contagem equivale a 16 amostras. A nota Claude
questiona essa interpretação por possível leitura de um só slot de DB. O registro
externo PID 143 (`v0-timestamp-run5.log`) descreve o footprint amplo de ZPASS_DONE,
incluindo `0x800000000007e900`; não é uma prova isolada da granularidade global.

O KytyPS5 usa consultas precisas em
[`occlusion.cpp:135–160`](../src/graphics/host_gpu/renderer/occlusion.cpp#L135)
e reconhece layout de pares por DB. Não aplicar fator 16 sem teste que leia todos
os slots, subtraia begin/end, some os resultados e registre `DB_COUNT_CONTROL`.

### 4. NIDs e contrato observável dos helpers AGC

O documento externo declara que 15 de 156 labels não correspondem ao hash esperado.
Claude relata 6 já Unknown e 9 restantes inconsistentes. OpenCode conferiu as
entradas locais pertinentes, mas não recalculou os hashes nesta sessão.
Uma label incompatível com um nome presumido não prova que o binding pelo NID ou
a implementação esteja errado: pode ser alias, variante ou nome ainda desconhecido.

O flip local usa NOP privado de 6 words
([agc.cpp:4159](../src/libs/agc.cpp#L4159)); o console documenta 19 words escritas,
64 reservadas, com VideoOut aberto (PID 109, `flip-probe-3.log`). O NOP é uma
representação HLE interna. A diferença importa se o guest observa tamanho,
ponteiro retornado ou conteúdo do buffer; não autoriza substituição automática.

### 5. Formatos sRGB estreitos e ordem de canais

O fallback local de `k8Srgb`/`k8_8Srgb` é UNORM, explicitamente provisório
([vulkanCommon.cpp:74](../src/graphics/host_gpu/vulkanCommon.cpp#L74)).
PID 160 descreve curva antes dos seletores num caso de quatro componentes; não
estabelece sozinho o comportamento de formatos de um/dois componentes.
Pedir um caso específico e usar valores de canais distintos. Descrições externas
de formatos packed devem ser confrontadas com a especificação antes de copiar
conclusões sobre ordem dos bytes. Nenhuma alteração de formato foi proposta como
correção já demonstrada aqui.

## Limites e integração simples

- O estado ativo de 29/09 é posterior ao README: full-1 relata 1.569.390 pass,
  1.350.306 not supported, 60 quality warnings e um Crash atribuído pelo autor ao
  fechamento manual do título. Pendia rerun de 61 casos. Isso não é certificação
  Khronos; não apresentar o driver como standards-conformant por inferência.
- A coerência observada na memória do console não elimina transferências ou
  sincronização necessárias entre memória guest e uma GPU discreta no PC.
- KytyPS5 declara GPL-2.0-only e o código próprio de PS5_Vulkan GPL-3.0-or-later.
  Um transplante desse código não cabe nas licenças atuais. Implementações
  independentes baseadas em fatos e consulta a fontes upstream com licenças
  compatíveis são opções; verificar proveniência de fixtures e arquivos.
- Portar KytyPS5 para executar no PS5 seria outro projeto. Vulkan por si só não
  resolve execução guest, endereçamento, memória executável, page faults e suporte
  de plataforma. Claude registra obstáculos adicionais; viabilidade não foi
  demonstrada nesta avaliação.
- KISS: aproveitar `tests/`, casos pequenos e resultados esperados documentados.
  Manter driver e harness de console externos. Custo inicial: conferir evidências,
  criar fixtures independentes e integrar ao teste local; novas medições exigem
  coordenação com quem tem o console.

## Coordenação e recomendação

Implementação desta frente: [GUEST-SYNC-TESTS.md](GUEST-SYNC-TESTS.md), com cenários,
matriz CTest e bloqueio de validação por ausência de toolchain local.

Após a aprovação do usuário em 2026-09-30, OpenCode assumiu testes de regressão
release/wait/BDA. Alvos previstos: `tests/ShaderRecompilerComputeTests.cpp`, um
include de testes focados se necessário, registros CTest no `CMakeLists.txt` e
código de sincronização somente se uma falha for demonstrada.

Claude indicou tiling/NIDs como candidatos; seu início não foi confirmado nesta
sessão. Se ambas as frentes precisarem de `CMakeLists.txt` ou do runner de testes,
reler o diff corrente e preservar as adições da outra sessão.

**Status: viável como fonte de testes e validação; integração direta do driver não
é o próximo passo recomendado.**

## Perguntas para o mantenedor do PS5_Vulkan

1. Há um readback completo dos 16 pares de ZPASS_DONE, com begin/end e
   `DB_COUNT_CONTROL`, para distinguir contador por DB de granularidade global?
2. R70 pode ganhar um caso de CPU escreve dados → sinaliza label → GPU espera →
   shader lê os dados por endereço na mesma submissão, e uma variante com retomada?
3. Há resultados específicos para formatos sRGB de um/dois canais?
4. Quais são layout e contrato de retorno de `sceAgcGetRegisterDefaults2`,
   separados do getter não versionado usado nos probes?
5. É possível publicar evidência com hash do build e identificador de run além do
   PID, para evitar ambiguidade entre boots e entre as entradas históricas?
