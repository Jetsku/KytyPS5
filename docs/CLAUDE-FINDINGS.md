# Retorno Claude Code

Atualizado: 2026-09-30. Autor: sessão Claude Code.
Base inspecionada: `4a50f8bd` (a mesma do handoff). Base externa: `mihawk-99/PS5_Vulkan`
`d609d71`, lida num clone raso (50 commits) fora deste repositório, sem modificações.
Resposta a [SESSION-HANDOFF.md](SESSION-HANDOFF.md) e a
[PS5-VULKAN-OPENCODE-REVIEW.md](PS5-VULKAN-OPENCODE-REVIEW.md).

## Atualização: divisão de trabalho (2026-09-30)

O usuário dividiu as frentes: **Claude** transforma a comparação de tiling em
testes reproduzíveis e confere as labels de NID; **OpenCode** fica com os testes
de release/wait/BDA. As duas partes da Claude estão feitas, no commit `2984bf13`.

### Alterações realizadas (commit `2984bf13`)

| Arquivo | Mudança |
| --- | --- |
| `tests/TileHardwareReferenceTests.cpp` (novo) | Testa [tile.cpp](../src/graphics/guest_gpu/tile.cpp) contra mapas de hardware. Os oito mapas de bloco (64KB_R_X de 1/2/4/8/16B e 64KB_Z_X de 1/2/4B) são checados texel a texel em 4×4 tiles, com os twists entre blocos. A cadeia de mips RGBA8 256×256 de 5 níveis é checada contra os 5 endereços que o console buscou (runs pid 131–134): `0x2f0fc`, `0x13cfc`, `0xb4fc`, `0x58fc`, `0x8fc`, com o início do nível 2 em `0x8400`. Cada tabela cita a sua run. A referência está num formato próprio, bit de coordenada → bits de endereço; nenhum código do PS5_Vulkan foi copiado. |
| [CMakeLists.txt](../CMakeLists.txt) | **Duas inserções, sem nada removido.** (1) O alvo `tile_hardware_reference_tests` (`tile.cpp` + `gpu_format.cpp` + `common`), logo depois do bloco `binding_path_tests`. (2) `add_test(NAME tile_hardware_reference …)`, logo depois de `add_test(NAME binding_path …)`. |
| [.github/workflows/build.yml](../.github/workflows/build.yml) | Job Linux: `tile_hardware_reference_tests` entra no `--target` do Build e `tile_hardware_reference` entra no `-R` do ctest. |
| `tools/check_nid_labels.py` (novo) | Recalcula o NID de `sce<Label>` para cada `LIB_FUNC` de [libAgcDriver.cpp](../src/libs/libAgcDriver.cpp). Falha quando aparece uma label que não bate e não está explicada, ou quando uma exceção registrada fica obsoleta. As 15 exceções conhecidas estão listadas com o motivo: 6 `Unknown`, 7 labels descritivas e 2 segundos NIDs de handlers com nome. |
| [libAgcDriver.cpp](../src/libs/libAgcDriver.cpp) | Um comentário de 2 linhas antes de `LIB_DEFINE`, apontando para a verificação. |

**Decisão sobre as labels:** as funções não foram renomeadas. Busquei o nome real
das 15 NIDs por hash, testando combinações de prefixos (`sceAgc`, `Dcb`, `Cb`,
`Driver`…), variações do corpo e sufixos (`Internal`, `2`, `Ex`…). Nenhum nome foi
encontrado. O mais provável é que sejam símbolos C++ com nome *mangled*. A label
continua sendo a melhor descrição do que a função faz, e o nome só aparece em logs
(`PRINT_NAME` usa `__func__`). Isso concorda com a revisão OpenCode: uma label
incompatível não prova que o binding está errado.

### Testes efetivamente executados

- `python tools/check_nid_labels.py`: 156 entradas, 141 batem, 15 exceções
  conhecidas, 0 erros, exit 0.
- Teste negativo do script, numa cópia fora do repositório com uma label alterada
  e uma exceção removida: 2 erros e exit 1, como esperado.
- Resposta conhecida do hash: `sceAgcDriverSubmitDcb` → `UglJIZjGssM`.
- **`tile_hardware_reference_tests` compilado e executado no WSL.** Ambiente:
  Ubuntu 24.04, clang 18.1.3, cmake 3.28.3, Release. O build fica em
  `~/kyty-build`, fora do repositório. Resultados:
  - compilação sem erros e sem avisos em `TileHardwareReferenceTests.cpp`;
  - execução direta: `all cases passed`, exit 0;
  - `ctest -R '^tile_hardware_reference$'`: 1/1 passou, em 0,64 s.
- **Teste de mutação:** cada mutante altera um valor da referência e roda o teste.
  Todos os 4 falham com exit 134 e mensagem que aponta mapa e texel:
  1. twist Y do 16B removido;
  2. um bit do mapa 8B trocado;
  3. início do nível 2 em `0x8000` em vez de `0x8400`;
  4. um bit do mapa de depth 4B trocado.

  Depois dos mutantes, o arquivo foi restaurado (`cmp` confirma que é idêntico ao
  original) e o teste voltou a passar.
- **Receita do build local**, sem Qt e sem X11 no WSL:
  `-DKYTY_BUILD_LAUNCHER=OFF -DBUILD_TESTING=ON -DSDL_X11=OFF -DSDL_WAYLAND=OFF
  -DSDL_UNIX_CONSOLE_BUILD=ON`.

### Build Linux: `profiler.cpp` corrigido (aprovado pelo usuário)

- **Causa.** O [src/utils.cmake:99](../src/utils.cmake#L99) compila Clang e GCC com
  `-fno-exceptions`, e o `profiler.cpp` tinha dois `try` (vindos do `6a7f10ee`,
  2026-09-26). O clang no Linux rejeita (`cannot use 'try' with exceptions
  disabled`). No Windows o clang-cl recebe `/EHsc` (padrão do CMake, que o projeto
  mantém), então **lá os `try` funcionam** e o comportamento precisava ser
  preservado. Isso afetava todo alvo que linka `common`, inclusive os testes da
  OpenCode.
- **Correção** em [profiler.cpp](../src/common/profiler.cpp):
  - o corpo de `OpenLoadingLog` passou para `CreateLoadingLog`;
  - a criação da thread passou para `StartLoadingPublisher`;
  - as chamadas ficam em `try`/`catch` sob `#if defined(__cpp_exceptions)`, sem
    `try` no caso contrário. O Windows segue igual, e o Linux compila;
  - `std::filesystem::u8path` (obsoleto) foi trocado por `Common::PathFromUtf8`,
    a mesma conversão, o que elimina um aviso.

  Com `git diff -w`: 49 inserções, 24 remoções, e o corpo das funções não mudou.
- **Verificação num build Linux limpo, sem o contorno** (`~/kyty-build-plain`):
  - `common` compila sem erros e sem avisos no `profiler.cpp`;
  - dos 6 testes do job Linux do CI, 5 passam (`tile_hardware_reference`,
    `memory_tracker`, `page_manager`, `audio_out2_port`, `ime_dialog`);
  - o `virtual_memory_allocation`, que compila o emulador inteiro, **não compila**
    por dois outros erros abaixo.
- **Não verificado:** o ramo `__cpp_exceptions` no Windows, porque não há clang-cl
  nesta máquina.
- **Correção de uma suspeita anterior:** o `redZonePatcher.cpp` também tem `try`,
  mas compila no Linux sem erro (objeto isolado construído).

### Outros erros de build só no Linux (corrigidos no commit `2984bf13`)

Encontrados com `ninja -k 0` nos alvos do CI Linux. Não há outros erros nesses
alvos.

1. [bufferCache.cpp:314](../src/graphics/host_gpu/renderer/cache/bufferCache.cpp#L314)
   (`3240f48b`, 2026-09-27). A lambda de `WriteFaultWindow` devolve `uint64_t`
   (`Default`) num ramo e `unsigned long long` (`kib * 1024`, vindo de `strtoull`)
   em outro. No Windows os tipos coincidem; no Linux o `uint64_t` é
   `unsigned long`. Correção: declarar `-> uint64_t` na lambda.
2. [ProgramCodec.cpp:935](../src/graphics/shader/recompiler/ir/ProgramCodec.cpp#L935)
   (`02799e44`, 2026-09-28). `source.indirect_image.emplace()` não compila com a
   libstdc++. `DescriptorSource::IndirectImage`
   ([ShaderIR.h:499](../src/graphics/shader/recompiler/ir/ShaderIR.h#L499)) é
   uma struct aninhada com inicializadores padrão, usada em `std::optional` dentro
   da própria classe que a contém; nesse ponto o clang a considera "não
   construível por padrão". Correção provável: declarar a struct fora de
   `DescriptorSource`, mantendo o nome por `using`.
- **GitHub Actions inativo neste fork.** `gh run list` devolve 404 para o
  `build.yml` em `JonJon1992/KytyPS5-fork`. A inclusão no job Linux só tem efeito
  onde o Actions roda, e lá o job também vai esbarrar no problema acima.
- **Submódulos baixados.** Foram inicializados em `3rdparty/` (passo do README),
  pelo git do Windows. Os commits são os fixados pelo repositório.

### Pontos da revisão OpenCode

- **Tiling:** concordo. A sugestão de R18 (mips lineares, PID 214) fica como
  candidata; o teste atual cobre só a cadeia *tiled*.
- **Oclusão:** concordo em não aplicar o fator 16 sem ler os 16 pares begin/end,
  somá-los e registrar o `DB_COUNT_CONTROL`. A pergunta está nas duas notas.
- **Flip, sRGB e registradores:** mesma posição; nada muda sem medição dirigida.

## Estado da avaliação inicial

- **O que esta sessão fez:** avaliou o PS5_Vulkan, a tarefa de
  `KYTYPS5_AGENT_PROMPT.md`. Não tem relação com a análise de gargalos do handoff.
- **Correção de percurso:** uma versão desta nota chegou a ser anexada ao fim do
  `SESSION-HANDOFF.md` e foi retirada em seguida. O handoff está de novo com as 106
  linhas originais da sessão OpenCode.

## Comparação com os achados do handoff

Os escopos não se cruzam. Esta sessão **não investigou** AIO, `KernelPread`, cache
de pipelines, draw prep, readback, barreiras de compute, `TextureCache` nem BDA,
e não confirma nem contesta as hipóteses de impacto do handoff.

O que foi conferido:

- **Referências do handoff.** Todos os símbolos citados existem na base `4a50f8bd`,
  nas linhas indicadas ou dentro da função indicada. Dois links apontam para o meio
  da função e não para a assinatura:
  - `KernelPread` começa em [fileSystem.cpp:705](../src/kernel/fileSystem.cpp#L705); a L748 é a aquisição do mutex.
  - `InitializeImage` começa em [textureCache.cpp:2127](../src/graphics/host_gpu/renderer/cache/textureCache.cpp#L2127); a L2188 é um ponto interno.
- **PS5_Vulkan e as hipóteses de desempenho.** Nenhum fato medido lá se aplica a
  elas. O que o PS5_Vulkan registra sobre memória descreve o **console** e não o
  host PC: CPU e GPU coerentes nos dois sentidos, sem run id, e "cada submissão
  AGC começa com estado resetado", fonte só CTS. Isso não informa a coerência
  host↔Vulkan que o BDA e o readback tratam.

## Descobertas desta sessão

### Veredito

Não integrar o PS5_Vulkan e não copiar código dele. O KytyPS5 é GPL-2.0-only
([README.md:373](../README.md#L373)) e o PS5_Vulkan é GPL-3.0-or-later. Só fatos
(valores, equações, layouts) são reaproveitáveis, como oráculo de testes.

Divergência em relação ao prompt: o `HARDWARE_FINDINGS.md` não usa as tags
`[VERIFIED]`/`[INFERRED]`. A evidência é citada como pid, commit ou log. Os
`Klog_Logs/` não são versionados, e os pids se repetem entre boots:
`golden/c7-mip-tiled` é citado como pid 134, mas o próprio golden diz 214.

### Verificado por esta sessão

| Área | Resultado | Evidência do PS5_Vulkan |
| --- | --- | --- |
| Tiling 64KB_R_X, 1 amostra, 1/2/4/8/16B | **Idêntico** texel a texel em 4×4 tiles, incluindo os twists entre blocos ([tile.cpp:976](../src/graphics/guest_gpu/tile.cpp#L976)) | 1B pid 117; 4B `79b5ace`; 8B pid 337/R91; 16B pids 113–118; 2B só AddrLib (pid 244) |
| Tiling Z_X 1/2/4B | **Idêntico** em 3×3 tiles | stencil pid 192; D16 pid 129; D32F bateria `c5-depth` |
| NIDs ([libAgcDriver.cpp](../src/libs/libAgcDriver.cpp)) | Das 156 entradas, 141 labels geram o próprio NID e 15 não. Dessas, 6 já são `Unknown` e **9 estão erradas**: `AgcInit`, `AgcCreateInterpolantMapping`, `AgcCreateInterpolantMapping2`, `AgcGetDataPacketPayloadAddress`, `AgcDcbContextStateOp`, `AgcCondExecPatchSetEnd`, `AgcCondExecPatchSetCommandAddress`, e os segundos NIDs `AhGvpITrf4M` (SubmitDcb) e `+T8Xo6LtFJI` (SubmitMultiDcbs) | hash recalculado aqui |
| Flip | [AgcDcbSetFlip](../src/libs/agc.cpp#L4159) emite um NOP privado de 6 words. No console, o helper escreve 19 words e reserva 64 | pid 109 (`flip-probe-3.log`), que corrige as runs pid 117 e 115 |
| Clock de GPU | O KytyPS5 usa 100 MHz ([referenceClock.h](../src/graphics/host_gpu/renderer/referenceClock.h)); o console mediu 99,79 MHz, com a diferença atribuída à latência de submit | pid 143 |
| Contador de z-pass | Bit 63 como "ready", igual ao KytyPS5 ([graphicsRun.cpp:3421](../src/graphics/guest_gpu/graphicsRun.cpp#L3421)) | pid 143 |
| `BufferFormat` | A numeração do [gpu_defs.h](../src/graphics/guest_gpu/gpu_defs.h) foi validada no console para os formatos de vértice 8, 16 e 8888 bits | pids 110, 111, 112 |

### Hipóteses e divergências (não verificadas)

- **Oclusão.** O PS5_Vulkan conclui que 1 contagem equivale a 16 amostras (pid 111),
  mas a probe lê um único slot. O footprint da run pid 143 tem `0x7E900` (518.400)
  em cada slot por DB, e 16 × 518.400 = 8.294.400 pixels. Isso indica contagem por
  DB, que é o modelo atual do KytyPS5. **Não adotar o fator 16.**
- **MXCSR.** Segundo o registro, o console inicia o processo com `0x9fe0` (FTZ/DAZ).
  A fonte é um título homebrew e não há run id. O KytyPS5 não define o MXCSR do
  guest (só o manipula em
  [x64InstructionEmulator.cpp:703](../src/loader/x64InstructionEmulator.cpp#L703)).
- **sRGB de 1 e 2 canais.** `k8Srgb` e `k8_8Srgb` viram UNORM sem curva
  ([vulkanCommon.cpp:74](../src/graphics/host_gpu/vulkanCommon.cpp#L74)). O console
  aplica a curva nos 3 primeiros componentes buscados (pid 160), mas só formatos de
  4 canais foram medidos.
- **Não conferido:**
  - tiling nos modos Standard/S, PRT, 3D e MSAA, e a posição do mip tail;
  - ordem de canais de `A8B8G8R8` (pid 290).
- **KytyPS5 rodando no PS5: não vale.**
  - A faixa fixa `0x10_0000_0000`–`0xFB_FFFF_FFFF` colide com mapeamentos do kernel
    do console.
  - `dlopen` é recusado e `sceKernelDlsym` retorna ESRCH (pid 250).
  - O processo roda como uid 1.
  - Memória executável e tratamento de page fault não foram medidos.

## Testes e medições da avaliação inicial

Scripts Python temporários no scratchpad da sessão, fora do repositório. Foram
substituídos pelo teste C++ e por `tools/check_nid_labels.py` (ver a atualização
no topo).

- `tilecmp.py`: endereço 64KB_R_X do KytyPS5 contra `ps5vk_tiled_texel_offset`.
  0 diferenças em 1.048.576 / 524.288 / 262.144 / 131.072 / 65.536 texels
  (1/2/4/8/16B).
- `depthcmp.py`: Z_X do KytyPS5 contra as tabelas de depth e stencil do ps5vk.
  0 diferenças em 589.824 / 294.912 / 147.456 texels (1/2/4B).
- `nid.py`: hash recalculado das 156 labels de NID.
- Conferência por `sed`/`grep` dos símbolos e linhas citados no handoff.

## Arquivos em edição

Nenhuma edição em andamento. Commitados no `2984bf13`, além da tabela do topo:
[src/common/profiler.cpp](../src/common/profiler.cpp) (correção do build Linux,
concluída). O `agc.cpp` **não** foi tocado.

**Arquivo compartilhado:** o [CMakeLists.txt](../CMakeLists.txt) também é alvo da
OpenCode (registros CTest). As inserções da Claude estão depois de
`binding_path_tests` e depois de `add_test(NAME binding_path …)`. Antes de editar,
releia o diff atual e mantenha as duas inserções. O
`tests/ShaderRecompilerComputeTests.cpp` **não** foi tocado pela Claude.

## Próximo passo recomendado

1. **Decidir os dois erros restantes do build Linux** (`bufferCache.cpp:314` e
   `ProgramCodec.cpp:935`). Corrigidos no `2984bf13`: `-> uint64_t` na lambda e
   `emplace(DescriptorSource::IndirectImage {})`. Falta confirmar o build Linux completo.
2. Depois, na ordem, se o usuário quiser:
   1. fixtures de R18 para mips lineares (PID 214), como sugere a revisão OpenCode;
   2. `R8_SRGB`/`R8G8_SRGB` quando o host suportar;
   3. auditoria de ordem de canais em
      [textureCommon.cpp](../src/graphics/host_gpu/renderer/image/textureCommon.cpp);
   4. MXCSR `0x9fe0` atrás de uma opção desligada por padrão.

Nada disso conflita com a frente de release/wait/BDA da OpenCode.

## Perguntas para o PS5_Vulkan

1. Somando os 16 slots por DB de um `ZPASS_DONE`, o total dá as amostras exatas?
   Com qual `DB_COUNT_CONTROL`?
2. Qual ponteiro o `sceAgcDcbSetFlip` retorna? Quanto retorna o `*GetSize`
   correspondente?
3. Qual MXCSR a thread principal de um título retail vê?
4. `R8_SRGB` e `R8G8_SRGB` (IMG FORMAT 128/129) aplicam a curva no fetch?
5. Há mapas medidos para 64KB_S, 64KB_S_T e cor MSAA 2x/4x?
6. Qual é o layout do `sceAgcGetRegisterDefaults2`? O KytyPS5 modela 4 tabelas com
   contador em `0x38`; eles leem uma tabela com contador em `0x20`.
