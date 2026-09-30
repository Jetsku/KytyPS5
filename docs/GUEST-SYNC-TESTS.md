# Regressões de release/wait/BDA

Autor: OpenCode. Data: 2026-09-30.
Base: `4a50f8bd278a5c4e6e7023e6506bd4d8111b2e0f`, mais alterações locais de testes.

## Implementação

`tests/GuestSyncTests.inc`, incluído pelo runner existente
`tests/ShaderRecompilerComputeTests.cpp`, adiciona `--guest-sync-only`.
Após execução nativa, foi corrigida a ausência de dispatch do opcode padrão
RELEASE_MEM `0x49`, reutilizando o decoder da forma privada.

### BdaPacketBoundaries

Usa memória guest, o BufferCache real, o parser/executor PM4 e readback Vulkan.
Sete observações de um buffer nativo verificam:

1. O conteúdo inicial.
2. Reutilização da prova BDA após uma escrita CPU sem novo ponto de ordenação.
   Um packet que só altera registradores não deve invalidar essa prova.
3. Atualização após WAIT_REG_MEM64 já satisfeito.
4. Atualização após ACQUIRE_MEM padrão.
5. Atualização após a forma privada de ACQUIRE_MEM.
6. Atualização após um wait inicialmente bloqueado que retoma com nova label.
7. Atualização após RELEASE_MEM sem escrita de label.

As operações ficam no mesmo callback da thread GPU para que a invalidação
automática entre service commands não mascare uma fronteira PM4 ausente.
O teste não chama `SyncEpoch::Advance()` diretamente. Cada observação copia os
bytes do buffer nativo para uma posição distinta do readback; a comparação final
não se limita aos contadores de sincronização.

O cenário de wait bloqueado verifica também que o valor antigo não libera o wait
e que o DWORD alto, explicitamente diferente da referência, é ignorado pela máscara.

### ReleaseWait64

Passa por `GuestGpu::Submit`, incluindo o sequenciador real quando habilitado.
Três valores sucessivos na mesma label verificam RELEASE_MEM de 32 bits seguido
do WAIT_REG_MEM64 mascarado, preservação do DWORD alto e execução de um packet
posterior ao wait. Cada rodada preenche um buffer na GPU e confere o padrão por
readback depois da sequência; callbacks pendentes de label devem ter terminado.

Referência factual dos packets: PS5_Vulkan `d609d712453561efc0fd78b8eb46602d49461229`,
[R70, PIDs 614/615](https://github.com/mihawk-99/PS5_Vulkan/blob/d609d712453561efc0fd78b8eb46602d49461229/jobs/r70-gpu-barrier/README.md).
Código escrito independentemente, usando as interfaces e convenções do KytyPS5.

Limites: o readback exercita buffers, não reproduz a imagem render-to-texture da
probe R70. A prova de que uma label nunca é publicada antes da conclusão física
da GPU exigiria um produtor deliberadamente retido e um observador independente;
este teste verifica conclusão, dados e progresso, não essa propriedade temporal.
Também não demonstra ganhos de FPS nem executa um shader guest consumidor BDA.

## Matriz CTest

| Teste | CP | Recorder | Submissão host | Labels |
| --- | --- | --- | --- | --- |
| `guest_sync_direct` | direto | desligado | síncrona | record |
| `guest_sync_inline` | inline | desligado | síncrona | record |
| `guest_sync_threaded` | thread | ligado | queued | record |
| `guest_sync_completion` | thread | ligado | queued | completion |

Todos habilitam `KYTY_BDA_INCREMENTAL_SYNC=1`, `KYTY_SYNC_EPOCH=1` e
`KYTY_BDA_SYNC_EPOCH=1`. A verificação que executa scans adicionais fica desligada
para não reparar silenciosamente o comportamento que se quer observar.
Cada processo tem timeout CTest de 60 segundos.

O subteste BDA usa um CommandProcessor local (direto, ou inline na configuração
inline) para isolar a mudança de época dos callbacks. É o subteste ReleaseWait64
que percorre a thread do sequenciador nas duas últimas configurações.

## Como executar

Em um ambiente configurado conforme o README, a partir da raiz:

```powershell
cmake --build _Build/windows --target shader_recompiler_compute_tests --parallel
ctest --test-dir _Build/windows --output-on-failure -R "^guest_sync_"
ctest --test-dir _Build/windows --output-on-failure -R "^bda_sync_"
```

Para Linux, usar o build correspondente em `_Build/linux`. Os testes novos
requerem um dispositivo Vulkan aceito pelo `VulkanHarness`; não são testes
host-only. Um build já configurado precisa regenerar o CMake para descobrir os
novos registros (o build normalmente faz isso automaticamente).

## Verificação desta sessão

### Resultado final Windows nativo (2026-09-30)

- Instalado Visual Studio Build Tools 2022 17.14.41: Clang 19.1.5, SDK Windows,
  CMake e Ninja. glslang 16.6.0 veio da release oficial Khronos, no diretório
  temporário da sessão, com caminho explícito no cache CMake.
- Build Release `_Build/windows`, launcher desligado, concluído.
- NVIDIA GeForce RTX 2060, driver 617.14, API Vulkan 1.4.351.
- Corrigida a referência de WAIT_REG_MEM64 do teste: DWORD alto zero na referência,
  pois a máscara aplica-se ao valor lido. A label mantém DWORD alto diferente.
- Falha reproduzida após isso: `unknown op / cmd_id = c0064900`. Registrado
  `IT_RELEASE_MEM` em `pm4Dispatch.cpp`; `CpOpReleaseMem` passou a aceitar o
  cabeçalho nativo e o privado, usando o mesmo tratamento de campos.
- **4/4 guest_sync_* passaram**, em 3,38 s.
- **17/17 regressões passaram**, em 26,45 s: seis `bda_sync_*`, sete variantes
  focadas `cp_seq_*`, quatro `eop_timestamps*`, incluindo deferred labels.
- Sem layer Khronos de validação (não instalada), benchmark de jogo ou prova visual.
  Não há alegação de ganho de FPS.

Correções de produção OpenCode: tipo de retorno em `WriteFaultWindow`, construção
em `ProgramCodec.cpp` e dispatch/parser de RELEASE_MEM nativo. A alteração de
`profiler.cpp` pertence à sessão Claude. Logs CTest: `_Build/windows/Testing/Temporary`.

### Execução após o build completo (2026-09-30)

O executável agora existe e CTest consegue iniciá-lo. A matriz foi iniciada,
mas o harness falha antes dos cenários de sincronização:

```text
VulkanHarness failed at dispatch: no Vulkan graphics+compute device with fragment barycentrics
```

Uma consulta independente pelo loader `libvulkan.so.1`, via Python/ctypes no WSL,
enumerou somente `llvmpipe (LLVM 20.1.2, 256 bits)`. Esse dispositivo não anuncia
`VK_KHR_fragment_shader_barycentric`; anuncia compute derivatives e push descriptors.
A RTX 2060 vista no Windows não foi enumerada pelo loader Vulkan do WSL.

Na primeira tentativa, `guest_sync_direct` e `guest_sync_inline` atingiram o
timeout após imprimir esse erro; a chamada foi interrompida ao iniciar o terceiro.
Uma nova tentativa com core size 0 confirmou aborto de `guest_sync_direct` com
o mesmo diagnóstico; a chamada foi interrompida durante o segundo. Não foi
concluída a matriz de dez testes. Não foram desabilitados requisitos do harness.

**Bloqueio atual: dispositivo Vulkan compatível no ambiente de execução, não
compilador nem ausência do executável.** Nenhum resultado valida ou refuta BDA ou
release/wait: o harness não chegou a esses cenários. Próxima execução deve usar
um dispositivo que satisfaça todos os requisitos do harness (por exemplo, avaliar
o caminho Windows nativo), identificando explicitamente device/driver.

### Retomada com toolchain WSL (2026-09-30)

- Ubuntu-24.04 agora dispõe de Clang 18.1.3, CMake 3.28.3, Ninja e glslangValidator.
- Build de testes configurado em `_Build/linux`, Release, launcher desligado.
  Por faltarem bibliotecas de desenvolvimento X11/Wayland, este build headless usa
  `SDL_X11=OFF`, `SDL_WAYLAND=OFF`, `SDL_UNIX_CONSOLE_BUILD=ON`.
- Corrigido erro de dedução de retorno da lambda `WriteFaultWindow` em
  `bufferCache.cpp`: retorno explícito `uint64_t` mantém o tipo em plataformas LP64.
  O objeto recompilou, assim como o runner que inclui `GuestSyncTests.inc`.
- Corrigido erro de `optional::emplace()` em `ProgramCodec.cpp`: construção
  explícita do agregado `DescriptorSource::IndirectImage {}` como argumento.
  Confirmado pelo build isolado do objeto com Clang 18/libstdc++ 13 (exit 0):
  `ninja -C _Build/linux CMakeFiles/shader_recompiler_compute_tests.dir/src/graphics/shader/recompiler/ir/ProgramCodec.cpp.o`.
- O executável completo ainda depende de terminar o build/link. A tentativa CTest
  reportada pelo usuário marcou 10 testes como **Not Run** porque o executável
  não existia; isso não é aprovação nem falha funcional dos cenários.

### Registro inicial (antes da configuração do ambiente)

- Revisados: escopos de locks/callbacks, vida útil dos command buffers emprestados,
  APIs públicas usadas e barreiras das cópias de teste. Conferido que as adições
  simultâneas da sessão Claude em CMake permanecem presentes.
- `git diff --check`: passou.
- `vulkaninfo --summary`: executou, identificando uma NVIDIA GeForce RTX 2060,
  Vulkan device API 1.4.351. Apenas layers NVIDIA foram listadas, sem a layer de
  validação Khronos. Isso não é execução dos testes nem prova de que todas as
  extensões exigidas pelo harness estejam disponíveis.
- **Compilação e testes novos não executados:** CMake, Ninja, clang-cl e
  glslangValidator não foram encontrados no PATH; `vswhere` não existe no caminho
  padrão consultado; Ubuntu-24.04 do WSL também não tem CMake, clang ou g++.
  A checagem inicial mostrou submódulos não inicializados; outra sessão começou
  a alterar dependências durante o trabalho. Não foi preparado um build completo.

**Estado atual: build Windows concluído; quatro testes novos e 17 regressões
focadas passaram. WSL permanece sem dispositivo compatível.**
A aprovação é limitada aos cenários descritos, não a todo o renderer.

## Coordenação com Claude

Esta frente alterou apenas `tests/GuestSyncTests.inc`, a inclusão/flag no runner,
o bloco `guest_sync_*` no CMake e notas OpenCode. As alterações de tiling/NIDs,
CI e dependências encontradas no checkout pertencem à outra sessão e foram
preservadas. Nenhum commit ou push foi feito por OpenCode.

Próximo passo: a investigação completa de render-to-texture
e de publicação antecipada de labels continua sendo trabalho posterior.
