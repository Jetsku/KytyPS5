# Imports sem implementação (stub) — Marvel's Wolverine `PPSA03671`

Extraído de `_kyty.txt` (sessão do emulador **KytyPS5**, fork `Senaxx/KytyPS5` build `aa790b8`).
Nomes resolvidos cruzando os NIDs com a base comunitária `claimore22/ps5rs` (`data/nids.csv`, ~195k NIDs).

- **Total de imports sem implementação:** 203
- **Resolvidos por nome:** 202/203

> Observação: o Kyty **não embute** uma tabela NID→nome; ele monta os símbolos a partir das
> tabelas de exportação do próprio jogo (por isso o log mostra `NID fallback`). Os nomes abaixo
> vêm da base comunitária, não do emulador.

## Resumo por biblioteca (ordem de prioridade)

| Prioridade | Biblioteca | Qtd | Observação |
|---|---|---:|---|
| ALTA | `AgcDriver_v1` | 2 | driver grafico AGC (capture) |
| ALTA | `Agc_v1` | 1 | driver grafico AGC |
| ALTA | `Pad_v1` | 1 | controle |
| ALTA | `Posix_v1` | 3 | POSIX: sleep/fcntl/shutdown |
| ALTA | `SaveData_native_v1` | 2 | save/load |
| ALTA | `libkernel_v1` | 11 | kernel: syscalls basicas |
| MEDIA | `LibcInternalExt_v1` | 3 | libc interna (backtrace/TLS) |
| MEDIA | `Psml_mfsr2_v1` | 7 | upscaling FSR2 |
| BAIXA | `Coredump_v1` | 5 | crash dumps |
| BAIXA | `Http_v1` | 1 | HTTP |
| BAIXA | `NpCommerce_v1` | 2 | PSN loja |
| BAIXA | `NpCppWebApi_v1` | 147 | PSN C++ web API (sessoes/cloud) |
| BAIXA | `NpManager_v1` | 1 | PSN manager |
| BAIXA | `NpSessionSignaling_v1` | 3 | PSN sinalizacao de sessao |
| BAIXA | `NpTrophy2_v1` | 1 | trofeus |
| BAIXA | `NpWebApi2_v1` | 4 | PSN web API v2 |
| BAIXA | `PlayGoDialog_v1` | 6 | dialogo de instalacao |
| BAIXA | `PlayerInvitationDialog_v1` | 3 | dialogo de convite (NP) |

## Imports por biblioteca

### `AgcDriver_v1` — 2 imports  ·  prioridade **ALTA**

_driver grafico AGC (capture)_

| NID | Nome | Leitura | Tipo |
|---|---|---|---|
| `SAfhzJPcjuk` | `sceAgcDriverRequestCaptureStart` | `sceAgcDriverRequestCaptureStart` | C API |
| `FOwvmNlFLjM` | `sceAgcDriverRequestCaptureStop` | `sceAgcDriverRequestCaptureStop` | C API |

### `Agc_v1` — 1 imports  ·  prioridade **ALTA**

_driver grafico AGC_

| NID | Nome | Leitura | Tipo |
|---|---|---|---|
| `K2mciNVxUCE` | `sceAgcSetNop` | `sceAgcSetNop` | C API |

### `Pad_v1` — 1 imports  ·  prioridade **ALTA**

_controle_

| NID | Nome | Leitura | Tipo |
|---|---|---|---|
| `zBsgF0q8DIM` | `?` | `(nao encontrado na base)` | desconhecido |

### `Posix_v1` — 3 imports  ·  prioridade **ALTA**

_POSIX: sleep/fcntl/shutdown_

| NID | Nome | Leitura | Tipo |
|---|---|---|---|
| `8nY19bKoiZk` | `fcntl` | `fcntl` | C API |
| `TUuiYS2kE8s` | `shutdown` | `shutdown` | C API |
| `0wu33hunNdE` | `sleep` | `sleep` | C API |

### `SaveData_native_v1` — 2 imports  ·  prioridade **ALTA**

_save/load_

| NID | Nome | Leitura | Tipo |
|---|---|---|---|
| `e6y3WMBYbiM` | `sceSaveDataCancel` | `sceSaveDataCancel` | C API |
| `8EA5OMIL1lQ` | `sceSaveDataGetConvertProgress` | `sceSaveDataGetConvertProgress` | C API |

### `libkernel_v1` — 11 imports  ·  prioridade **ALTA**

_kernel: syscalls basicas_

| NID | Nome | Leitura | Tipo |
|---|---|---|---|
| `hHlZQUnlxSM` | `getrusage` | `getrusage` | C API |
| `c7ZnT7V1B98` | `rmdir` | `rmdir` | C API |
| `fgIsQ10xYVA` | `sceKernelChmod` | `sceKernelChmod` | C API |
| `UtszJWHrDcA` | `sceKernelFchmod` | `sceKernelFchmod` | C API |
| `-YTW+qXc3CQ` | `sceKernelInternalMemoryGetModuleSegmentInfo` | `sceKernelInternalMemoryGetModuleSegmentInfo` | C API |
| `3k6kx-zOOSQ` | `sceKernelMlock` | `sceKernelMlock` | C API |
| `XgVs4jRehcY` | `sceKernelSyncOnAddressWait16` | `sceKernelSyncOnAddressWait16` | C API |
| `dFym5b41JLU` | `sceKernelSyncOnAddressWait8` | `sceKernelSyncOnAddressWait8` | C API |
| `WlyEA-sLDf0` | `sceKernelTruncate` | `sceKernelTruncate` | C API |
| `0Cq8ipKr9n0` | `sceKernelUtimes` | `sceKernelUtimes` | C API |
| `VADc3MNQ3cM` | `signal` | `signal` | C API |

### `LibcInternalExt_v1` — 3 imports  ·  prioridade **MEDIA**

_libc interna (backtrace/TLS)_

| NID | Nome | Leitura | Tipo |
|---|---|---|---|
| `tB59hFLH3SA` | `_sceLibcInternalForceTlsDestructor` | `_sceLibcInternalForceTlsDestructor` | C API |
| `OQ-dzhlnM28` | `_sceLibcInternalThreadDtors` | `_sceLibcInternalThreadDtors` | C API |
| `EHsF2i9FXPM` | `sceLibcInternalBacktraceForGame` | `sceLibcInternalBacktraceForGame` | C API |

### `Psml_mfsr2_v1` — 7 imports  ·  prioridade **MEDIA**

_upscaling FSR2_

| NID | Nome | Leitura | Tipo |
|---|---|---|---|
| `m9JLPc3wOQw` | `scePsmlMfsr2CreateContext` | `scePsmlMfsr2CreateContext` | C API |
| `gMduXCLYrNg` | `scePsmlMfsr2CreateSharedResources` | `scePsmlMfsr2CreateSharedResources` | C API |
| `5z2gBlqxJ+0` | `scePsmlMfsr2GetContextInitRequirement` | `scePsmlMfsr2GetContextInitRequirement` | C API |
| `lrJwpLjKXRc` | `scePsmlMfsr2GetDispatchPackets` | `scePsmlMfsr2GetDispatchPackets` | C API |
| `ZLL31lzzxr4` | `scePsmlMfsr2GetDispatchPacketsSizeInDwords` | `scePsmlMfsr2GetDispatchPacketsSizeInDwords` | C API |
| `1ic5q-kdOsc` | `scePsmlMfsr2GetSharedResourcesInitRequirement` | `scePsmlMfsr2GetSharedResourcesInitRequirement` | C API |
| `o+NM86gEwFE` | `scePsmlMfsr2Init` | `scePsmlMfsr2Init` | C API |

### `Coredump_v1` — 5 imports  ·  prioridade **BAIXA**

_crash dumps_

| NID | Nome | Leitura | Tipo |
|---|---|---|---|
| `+YX0z-GUSNw` | `sceCoredumpAttachMemoryRegion` | `sceCoredumpAttachMemoryRegion` | C API |
| `kK0DUW1Ukgc` | `sceCoredumpGetStopInfoCpu` | `sceCoredumpGetStopInfoCpu` | C API |
| `Jrs7UUkGOFo` | `sceCoredumpGetStopInfoGpu` | `sceCoredumpGetStopInfoGpu` | C API |
| `Uxqkdta7wEg` | `sceCoredumpSetUserDataType` | `sceCoredumpSetUserDataType` | C API |
| `Dbbkj6YHWdo` | `sceCoredumpWriteUserData` | `sceCoredumpWriteUserData` | C API |

### `Http_v1` — 1 imports  ·  prioridade **BAIXA**

_HTTP_

| NID | Nome | Leitura | Tipo |
|---|---|---|---|
| `P5pdoykPYTk` | `sceHttpReadData` | `sceHttpReadData` | C API |

### `NpCommerce_v1` — 2 imports  ·  prioridade **BAIXA**

_PSN loja_

| NID | Nome | Leitura | Tipo |
|---|---|---|---|
| `r42bWcQbtZY` | `sceNpCommerceDialogGetResult` | `sceNpCommerceDialogGetResult` | C API |
| `DfSCDRA3EjY` | `sceNpCommerceDialogOpen` | `sceNpCommerceDialogOpen` | C API |

### `NpCppWebApi_v1` — 147 imports  ·  prioridade **BAIXA**

_PSN C++ web API (sessoes/cloud)_

| NID | Nome | Leitura | Tipo |
|---|---|---|---|
| `5QJRueHuTBo` | `_ZN3sce2Np9CppWebApi6Common13ConstIteratorINS2_12IntrusivePtrINS1_14SessionManager2V136UsersPlayerSessionsInvitationForReadEEEED2Ev` | `sce::Np::CppWebApi::Common::ConstIterator::~ConstIterator` | destrutor |
| `DrekXkm+oxg` | `_ZN3sce2Np9CppWebApi6Common15DefaultResponseC1Ev` | `sce::Np::CppWebApi::Common::DefaultResponse::DefaultResponse` | construtor |
| `cTLUQmCfWNQ` | `_ZN3sce2Np9CppWebApi6Common15DefaultResponseD1Ev` | `sce::Np::CppWebApi::Common::DefaultResponse::~DefaultResponse` | destrutor |
| `kpbW7Dr44Gc` | `_ZN3sce2Np9CppWebApi6Common21DownStreamTransactionINS2_12IntrusivePtrINS1_17TitleCloudStorage2V17DataApi27DownloadDataResponseHeadersEEEEC1Ev` | `sce::Np::CppWebApi::Common::DownStreamTransaction::DownStreamTransaction` | construtor |
| `0oIoKcDA0Fw` | `_ZN3sce2Np9CppWebApi6Common21DownStreamTransactionINS2_12IntrusivePtrINS1_17TitleCloudStorage2V17DataApi27DownloadDataResponseHeadersEEEE6finishEv` | `sce::Np::CppWebApi::Common::DownStreamTransaction::finish` | metodo |
| `kVGz48sqNJM` | `_ZN3sce2Np9CppWebApi6Common21DownStreamTransactionINS2_12IntrusivePtrINS1_17TitleCloudStorage2V17DataApi27DownloadDataResponseHeadersEEEE8readDataEPcm` | `sce::Np::CppWebApi::Common::DownStreamTransaction::readData` | metodo |
| `V2NelZNoJwI` | `_ZN3sce2Np9CppWebApi6Common21DownStreamTransactionINS2_12IntrusivePtrINS1_17TitleCloudStorage2V17DataApi27DownloadDataResponseHeadersEEEE5startEPNS2_10LibContextE` | `sce::Np::CppWebApi::Common::DownStreamTransaction::start` | metodo |
| `IAkGC4cv3zs` | `_ZN3sce2Np9CppWebApi6Common21DownStreamTransactionINS2_12IntrusivePtrINS1_17TitleCloudStorage2V17DataApi27DownloadDataResponseHeadersEEEED1Ev` | `sce::Np::CppWebApi::Common::DownStreamTransaction::~DownStreamTransaction` | destrutor |
| `8x++mBOUeso` | `_ZN3sce2Np9CppWebApi6Common10InitParamsC1Ev` | `sce::Np::CppWebApi::Common::InitParams::InitParams` | construtor |
| `52AlYvq+dmk` | `_ZN3sce2Np9CppWebApi6Common10InitParamsD1Ev` | `sce::Np::CppWebApi::Common::InitParams::~InitParams` | destrutor |
| `+DSNphhYdd4` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V115LocalizedStringEEC1ERS7_` | `sce::Np::CppWebApi::Common::IntrusivePtr::IntrusivePtr` | construtor |
| `203JffNGrbg` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V120RequestPlayerSessionEEC1Ev` | `sce::Np::CppWebApi::Common::IntrusivePtr::IntrusivePtr` | construtor |
| `3L1hfNaKKJE` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V151PostPlayerSessionsSessionIdMemberPlayersRequestBodyEEC1Ev` | `sce::Np::CppWebApi::Common::IntrusivePtr::IntrusivePtr` | construtor |
| `8VWyc8ke-uw` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V129PostPlayerSessionsRequestBodyEEC1Ev` | `sce::Np::CppWebApi::Common::IntrusivePtr::IntrusivePtr` | construtor |
| `anOM+-Ryi+o` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS2_6VectorINS2_6StringEEEEC1ERS7_` | `sce::Np::CppWebApi::Common::IntrusivePtr::IntrusivePtr` | construtor |
| `BLeUlxPwr5w` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V115LocalizedStringEEC1Ev` | `sce::Np::CppWebApi::Common::IntrusivePtr::IntrusivePtr` | construtor |
| `BWDILfRcisI` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V151PostPlayerSessionsSessionIdMemberPlayersRequestBodyEEC1ERS7_` | `sce::Np::CppWebApi::Common::IntrusivePtr::IntrusivePtr` | construtor |
| `d2JuGu+UcU8` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS2_6VectorINS2_6StringEEEEC1EPS6_PFvS8_EPNS2_10LibContextE` | `sce::Np::CppWebApi::Common::IntrusivePtr::IntrusivePtr` | construtor |
| `HfnTwwuLjAI` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V126RequestPlayerSessionPlayerEEC1Ev` | `sce::Np::CppWebApi::Common::IntrusivePtr::IntrusivePtr` | construtor |
| `iMcKgGtukyA` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V154GetUsersAccountIdPlayerSessionsInvitationsResponseBodyEEC1Ev` | `sce::Np::CppWebApi::Common::IntrusivePtr::IntrusivePtr` | construtor |
| `INgczqD0H+w` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V129PostPlayerSessionsRequestBodyEEC1ERS7_` | `sce::Np::CppWebApi::Common::IntrusivePtr::IntrusivePtr` | construtor |
| `N+uuxyqdvFo` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V132RequestPlayerSessionMemberPlayerEEC1Ev` | `sce::Np::CppWebApi::Common::IntrusivePtr::IntrusivePtr` | construtor |
| `NvV--345al4` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V124PlayerSessionPushContextEEC1Ev` | `sce::Np::CppWebApi::Common::IntrusivePtr::IntrusivePtr` | construtor |
| `oAuFlME7yNI` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V132RequestCreatePlayerSessionPlayerEEC1Ev` | `sce::Np::CppWebApi::Common::IntrusivePtr::IntrusivePtr` | construtor |
| `OVsvmF7lYPQ` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V130PostPlayerSessionsResponseBodyEEC1Ev` | `sce::Np::CppWebApi::Common::IntrusivePtr::IntrusivePtr` | construtor |
| `vRpYko5WTm4` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V152PostPlayerSessionsSessionIdMemberPlayersResponseBodyEEC1Ev` | `sce::Np::CppWebApi::Common::IntrusivePtr::IntrusivePtr` | construtor |
| `VVi2+kwIbCc` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V132RequestPlayerSessionMemberPlayerEEC1ERS7_` | `sce::Np::CppWebApi::Common::IntrusivePtr::IntrusivePtr` | construtor |
| `9p16KzQjIf4` | `_ZNK3sce2Np9CppWebApi6Common12IntrusivePtrINS2_6VectorINS3_INS1_14SessionManager2V136UsersPlayerSessionsInvitationForReadEEEEEEdeEv` | `sce::Np::CppWebApi::Common::IntrusivePtr::operator*` | operador |
| `NTELx+nQzhw` | `_ZNK3sce2Np9CppWebApi6Common12IntrusivePtrINS2_6VectorINS3_INS1_14SessionManager2V113PlayerSessionEEEEEEdeEv` | `sce::Np::CppWebApi::Common::IntrusivePtr::operator*` | operador |
| `+htBBgi85YA` | `_ZNK3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V120RequestPlayerSessionEEptEv` | `sce::Np::CppWebApi::Common::IntrusivePtr::operator->` | operador |
| `2V-RL56o5cU` | `_ZNK3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V126RequestPlayerSessionPlayerEEptEv` | `sce::Np::CppWebApi::Common::IntrusivePtr::operator->` | operador |
| `5RSKjWE2avA` | `_ZNK3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V113PlayerSessionEEptEv` | `sce::Np::CppWebApi::Common::IntrusivePtr::operator->` | operador |
| `EBjE6sitQCI` | `_ZNK3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V110FromMemberEEptEv` | `sce::Np::CppWebApi::Common::IntrusivePtr::operator->` | operador |
| `kI0wgtzMZC0` | `_ZNK3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V130PostPlayerSessionsResponseBodyEEptEv` | `sce::Np::CppWebApi::Common::IntrusivePtr::operator->` | operador |
| `PzLUwQXc7VM` | `_ZNK3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V154GetUsersAccountIdPlayerSessionsInvitationsResponseBodyEEptEv` | `sce::Np::CppWebApi::Common::IntrusivePtr::operator->` | operador |
| `vJrnbutABoE` | `_ZNK3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V136UsersPlayerSessionsInvitationForReadEEptEv` | `sce::Np::CppWebApi::Common::IntrusivePtr::operator->` | operador |
| `4Wq-RRWM+q0` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V151PostPlayerSessionsSessionIdMemberPlayersRequestBodyEED1Ev` | `sce::Np::CppWebApi::Common::IntrusivePtr::~IntrusivePtr` | destrutor |
| `7Bc6AIhycYA` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V132RequestCreatePlayerSessionPlayerEED1Ev` | `sce::Np::CppWebApi::Common::IntrusivePtr::~IntrusivePtr` | destrutor |
| `7jbnFXytxmQ` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V152PostPlayerSessionsSessionIdMemberPlayersResponseBodyEED1Ev` | `sce::Np::CppWebApi::Common::IntrusivePtr::~IntrusivePtr` | destrutor |
| `8cmupXHZgAw` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS2_6VectorINS3_INS1_14SessionManager2V136UsersPlayerSessionsInvitationForReadEEEEEED1Ev` | `sce::Np::CppWebApi::Common::IntrusivePtr::~IntrusivePtr` | destrutor |
| `AawHDpsapd4` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V126RequestPlayerSessionPlayerEED1Ev` | `sce::Np::CppWebApi::Common::IntrusivePtr::~IntrusivePtr` | destrutor |
| `EviQs3RBmoM` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V132RequestPlayerSessionMemberPlayerEED1Ev` | `sce::Np::CppWebApi::Common::IntrusivePtr::~IntrusivePtr` | destrutor |
| `gVfZ1j2bhPM` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V124PlayerSessionPushContextEED1Ev` | `sce::Np::CppWebApi::Common::IntrusivePtr::~IntrusivePtr` | destrutor |
| `gw7fieR8fC4` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V129PostPlayerSessionsRequestBodyEED1Ev` | `sce::Np::CppWebApi::Common::IntrusivePtr::~IntrusivePtr` | destrutor |
| `GwWn958m6g4` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V130PostPlayerSessionsResponseBodyEED1Ev` | `sce::Np::CppWebApi::Common::IntrusivePtr::~IntrusivePtr` | destrutor |
| `lCcoeldiMhM` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V154GetUsersAccountIdPlayerSessionsInvitationsResponseBodyEED1Ev` | `sce::Np::CppWebApi::Common::IntrusivePtr::~IntrusivePtr` | destrutor |
| `m7ykSknwnRg` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS2_6VectorINS2_6StringEEEED1Ev` | `sce::Np::CppWebApi::Common::IntrusivePtr::~IntrusivePtr` | destrutor |
| `NCe1Kgd6Jlo` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V120RequestPlayerSessionEED1Ev` | `sce::Np::CppWebApi::Common::IntrusivePtr::~IntrusivePtr` | destrutor |
| `OmJ5HIMLuNg` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V110FromMemberEED1Ev` | `sce::Np::CppWebApi::Common::IntrusivePtr::~IntrusivePtr` | destrutor |
| `V1jgl1b2Qog` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS2_6VectorINS3_INS1_14SessionManager2V113PlayerSessionEEEEEED1Ev` | `sce::Np::CppWebApi::Common::IntrusivePtr::~IntrusivePtr` | destrutor |
| `Zkxpy0zdodA` | `_ZN3sce2Np9CppWebApi6Common12IntrusivePtrINS1_14SessionManager2V115LocalizedStringEED1Ev` | `sce::Np::CppWebApi::Common::IntrusivePtr::~IntrusivePtr` | destrutor |
| `jYAXNHlcQAc` | `_ZNK3sce2Np9CppWebApi6Common8IteratorINS2_12IntrusivePtrINS1_14SessionManager2V136UsersPlayerSessionsInvitationForReadEEEEneERKS9_` | `sce::Np::CppWebApi::Common::Iterator::operator!=` | operador |
| `awTPhlC4368` | `_ZNK3sce2Np9CppWebApi6Common8IteratorINS2_12IntrusivePtrINS1_14SessionManager2V136UsersPlayerSessionsInvitationForReadEEEEdeEv` | `sce::Np::CppWebApi::Common::Iterator::operator*` | operador |
| `SJBKLQrHqFw` | `_ZN3sce2Np9CppWebApi6Common8IteratorINS2_12IntrusivePtrINS1_14SessionManager2V136UsersPlayerSessionsInvitationForReadEEEEppEv` | `sce::Np::CppWebApi::Common::Iterator::operator++` | operador |
| `Y295ygEccqk` | `_ZN3sce2Np9CppWebApi6Common10LibContextC1Ev` | `sce::Np::CppWebApi::Common::LibContext::LibContext` | construtor |
| `9y9YsMBbUq0` | `_ZN3sce2Np9CppWebApi6Common6StringC1EPNS2_10LibContextE` | `sce::Np::CppWebApi::Common::String::String` | construtor |
| `BgDtc93upnM` | `_ZN3sce2Np9CppWebApi6Common6String6appendEPKc` | `sce::Np::CppWebApi::Common::String::append` | metodo |
| `dVCh5JiPsSY` | `_ZNK3sce2Np9CppWebApi6Common6String5c_strEv` | `sce::Np::CppWebApi::Common::String::c_str` | metodo |
| `DNmJw98YzUU` | `_ZN3sce2Np9CppWebApi6Common6StringD1Ev` | `sce::Np::CppWebApi::Common::String::~String` | destrutor |
| `A3vS8-6YhBM` | `_ZN3sce2Np9CppWebApi6Common11TransactionINS2_15DefaultResponseENS2_12IntrusivePtrINS2_18ResponseHeaderBaseEEEEC1Ev` | `sce::Np::CppWebApi::Common::Transaction::Transaction` | construtor |
| `BQ-8I4FcCQs` | `_ZN3sce2Np9CppWebApi6Common11TransactionINS2_12IntrusivePtrINS1_14SessionManager2V154GetUsersAccountIdPlayerSessionsInvitationsResponseBodyEEENS4_INS2_18ResponseHeaderBaseEEEEC1Ev` | `sce::Np::CppWebApi::Common::Transaction::Transaction` | construtor |
| `F-KitvVOrpQ` | `_ZN3sce2Np9CppWebApi6Common11TransactionINS2_12IntrusivePtrINS1_14SessionManager2V130PostPlayerSessionsResponseBodyEEENS4_INS2_18ResponseHeaderBaseEEEEC1Ev` | `sce::Np::CppWebApi::Common::Transaction::Transaction` | construtor |
| `SK2UF3FYqM4` | `_ZN3sce2Np9CppWebApi6Common11TransactionINS2_12IntrusivePtrINS1_14SessionManager2V152PostPlayerSessionsSessionIdMemberPlayersResponseBodyEEENS4_INS2_18ResponseHeaderBaseEEEEC1Ev` | `sce::Np::CppWebApi::Common::Transaction::Transaction` | construtor |
| `26wN8Yjf9Fk` | `_ZNK3sce2Np9CppWebApi6Common11TransactionINS2_12IntrusivePtrINS1_14SessionManager2V152PostPlayerSessionsSessionIdMemberPlayersResponseBodyEEENS4_INS2_18ResponseHeaderBaseEEEE11getResponseERS8_` | `sce::Np::CppWebApi::Common::Transaction::getResponse` | acessor |
| `fJuX5gEyZwU` | `_ZNK3sce2Np9CppWebApi6Common11TransactionINS2_12IntrusivePtrINS1_14SessionManager2V130PostPlayerSessionsResponseBodyEEENS4_INS2_18ResponseHeaderBaseEEEE11getResponseERS8_` | `sce::Np::CppWebApi::Common::Transaction::getResponse` | acessor |
| `rpmRha1Eq9s` | `_ZNK3sce2Np9CppWebApi6Common11TransactionINS2_15DefaultResponseENS2_12IntrusivePtrINS2_18ResponseHeaderBaseEEEE11getResponseERS4_` | `sce::Np::CppWebApi::Common::Transaction::getResponse` | acessor |
| `SR76Quk0L9Y` | `_ZNK3sce2Np9CppWebApi6Common11TransactionINS2_12IntrusivePtrINS1_14SessionManager2V154GetUsersAccountIdPlayerSessionsInvitationsResponseBodyEEENS4_INS2_18ResponseHeaderBaseEEEE11getResponseERS8_` | `sce::Np::CppWebApi::Common::Transaction::getResponse` | acessor |
| `qMYMahQKgPQ` | `_ZN3sce2Np9CppWebApi6Common11TransactionINS2_12IntrusivePtrINS1_14SessionManager2V130PostPlayerSessionsResponseBodyEEENS4_INS2_18ResponseHeaderBaseEEEED1Ev` | `sce::Np::CppWebApi::Common::Transaction::~Transaction` | destrutor |
| `s4G45M9w8tc` | `_ZN3sce2Np9CppWebApi6Common11TransactionINS2_12IntrusivePtrINS1_14SessionManager2V154GetUsersAccountIdPlayerSessionsInvitationsResponseBodyEEENS4_INS2_18ResponseHeaderBaseEEEED1Ev` | `sce::Np::CppWebApi::Common::Transaction::~Transaction` | destrutor |
| `t8-zRyLWOXE` | `_ZN3sce2Np9CppWebApi6Common11TransactionINS2_12IntrusivePtrINS1_14SessionManager2V152PostPlayerSessionsSessionIdMemberPlayersResponseBodyEEENS4_INS2_18ResponseHeaderBaseEEEED1Ev` | `sce::Np::CppWebApi::Common::Transaction::~Transaction` | destrutor |
| `zrEp8lu3vAI` | `_ZN3sce2Np9CppWebApi6Common11TransactionINS2_15DefaultResponseENS2_12IntrusivePtrINS2_18ResponseHeaderBaseEEEED1Ev` | `sce::Np::CppWebApi::Common::Transaction::~Transaction` | destrutor |
| `ogc2cNOobW8` | `_ZN3sce2Np9CppWebApi6Common15TransactionBaseINS2_12IntrusivePtrINS1_14SessionManager2V152PostPlayerSessionsSessionIdMemberPlayersResponseBodyEEENS4_INS2_18ResponseHeaderBaseEEEE6finishEv` | `sce::Np::CppWebApi::Common::TransactionBase::finish` | metodo |
| `q+50oeoRpnY` | `_ZN3sce2Np9CppWebApi6Common15TransactionBaseINS2_15DefaultResponseENS2_12IntrusivePtrINS2_18ResponseHeaderBaseEEEE6finishEv` | `sce::Np::CppWebApi::Common::TransactionBase::finish` | metodo |
| `rPxhVLFJBm4` | `_ZN3sce2Np9CppWebApi6Common15TransactionBaseINS2_12IntrusivePtrINS1_14SessionManager2V154GetUsersAccountIdPlayerSessionsInvitationsResponseBodyEEENS4_INS2_18ResponseHeaderBaseEEEE6finishEv` | `sce::Np::CppWebApi::Common::TransactionBase::finish` | metodo |
| `VXm5RXt+uJ0` | `_ZN3sce2Np9CppWebApi6Common15TransactionBaseINS2_12IntrusivePtrINS1_14SessionManager2V130PostPlayerSessionsResponseBodyEEENS4_INS2_18ResponseHeaderBaseEEEE6finishEv` | `sce::Np::CppWebApi::Common::TransactionBase::finish` | metodo |
| `K8zhm2vYK+o` | `_ZN3sce2Np9CppWebApi6Common15TransactionBaseINS2_12IntrusivePtrINS1_14SessionManager2V152PostPlayerSessionsSessionIdMemberPlayersResponseBodyEEENS4_INS2_18ResponseHeaderBaseEEEE5startEPNS2_10LibContextE` | `sce::Np::CppWebApi::Common::TransactionBase::start` | metodo |
| `pleuuED1nRY` | `_ZN3sce2Np9CppWebApi6Common15TransactionBaseINS2_15DefaultResponseENS2_12IntrusivePtrINS2_18ResponseHeaderBaseEEEE5startEPNS2_10LibContextE` | `sce::Np::CppWebApi::Common::TransactionBase::start` | metodo |
| `XUeWyaje50Q` | `_ZN3sce2Np9CppWebApi6Common15TransactionBaseINS2_12IntrusivePtrINS1_14SessionManager2V154GetUsersAccountIdPlayerSessionsInvitationsResponseBodyEEENS4_INS2_18ResponseHeaderBaseEEEE5startEPNS2_10LibContextE` | `sce::Np::CppWebApi::Common::TransactionBase::start` | metodo |
| `ynvhoQekbwg` | `_ZN3sce2Np9CppWebApi6Common15TransactionBaseINS2_12IntrusivePtrINS1_14SessionManager2V130PostPlayerSessionsResponseBodyEEENS4_INS2_18ResponseHeaderBaseEEEE5startEPNS2_10LibContextE` | `sce::Np::CppWebApi::Common::TransactionBase::start` | metodo |
| `6K1g7+ztzAs` | `_ZN3sce2Np9CppWebApi6Common6VectorINS2_12IntrusivePtrINS1_14SessionManager2V132RequestCreatePlayerSessionPlayerEEEEC1EPNS2_10LibContextE` | `sce::Np::CppWebApi::Common::Vector::Vector` | construtor |
| `7VnK1K-gAEU` | `_ZN3sce2Np9CppWebApi6Common6VectorINS2_6StringEEC1EPNS2_10LibContextE` | `sce::Np::CppWebApi::Common::Vector::Vector` | construtor |
| `dqGIvYiZAso` | `_ZN3sce2Np9CppWebApi6Common6VectorINS2_12IntrusivePtrINS1_14SessionManager2V120RequestPlayerSessionEEEEC1EPNS2_10LibContextE` | `sce::Np::CppWebApi::Common::Vector::Vector` | construtor |
| `fwYdaeuRgrM` | `_ZN3sce2Np9CppWebApi6Common6VectorINS2_12IntrusivePtrINS1_14SessionManager2V126RequestPlayerSessionPlayerEEEEC1EPNS2_10LibContextE` | `sce::Np::CppWebApi::Common::Vector::Vector` | construtor |
| `tFe8YsdlKyc` | `_ZN3sce2Np9CppWebApi6Common6VectorINS2_12IntrusivePtrINS1_14SessionManager2V124PlayerSessionPushContextEEEEC1EPNS2_10LibContextE` | `sce::Np::CppWebApi::Common::Vector::Vector` | construtor |
| `88Rk5FV9QAs` | `_ZN3sce2Np9CppWebApi6Common6VectorINS2_12IntrusivePtrINS1_14SessionManager2V136UsersPlayerSessionsInvitationForReadEEEE5beginEv` | `sce::Np::CppWebApi::Common::Vector::begin` | metodo |
| `-4qn5SHAuoQ` | `_ZN3sce2Np9CppWebApi6Common6VectorINS2_12IntrusivePtrINS1_14SessionManager2V136UsersPlayerSessionsInvitationForReadEEEE3endEv` | `sce::Np::CppWebApi::Common::Vector::end` | metodo |
| `ZL5Znq6YsMY` | `_ZN3sce2Np9CppWebApi6Common6VectorINS2_12IntrusivePtrINS1_14SessionManager2V113PlayerSessionEEEEixEm` | `sce::Np::CppWebApi::Common::Vector::operator[]` | operador |
| `7X3wSywr7+0` | `_ZN3sce2Np9CppWebApi6Common6VectorINS2_12IntrusivePtrINS1_14SessionManager2V120RequestPlayerSessionEEEE8pushBackERKS8_` | `sce::Np::CppWebApi::Common::Vector::pushBack` | metodo |
| `BoakAg6TFe4` | `_ZN3sce2Np9CppWebApi6Common6VectorINS2_12IntrusivePtrINS1_14SessionManager2V132RequestCreatePlayerSessionPlayerEEEE8pushBackERKS8_` | `sce::Np::CppWebApi::Common::Vector::pushBack` | metodo |
| `iTIgJfwD0MY` | `_ZN3sce2Np9CppWebApi6Common6VectorINS2_6StringEE8pushBackERKS4_` | `sce::Np::CppWebApi::Common::Vector::pushBack` | metodo |
| `Jxct84SbIMg` | `_ZN3sce2Np9CppWebApi6Common6VectorINS2_12IntrusivePtrINS1_14SessionManager2V124PlayerSessionPushContextEEEE8pushBackERKS8_` | `sce::Np::CppWebApi::Common::Vector::pushBack` | metodo |
| `RzJr+LF13gI` | `_ZN3sce2Np9CppWebApi6Common6VectorINS2_12IntrusivePtrINS1_14SessionManager2V126RequestPlayerSessionPlayerEEEE8pushBackERKS8_` | `sce::Np::CppWebApi::Common::Vector::pushBack` | metodo |
| `53DH+ZDnTP4` | `_ZN3sce2Np9CppWebApi6Common6VectorINS2_12IntrusivePtrINS1_14SessionManager2V132RequestCreatePlayerSessionPlayerEEEED1Ev` | `sce::Np::CppWebApi::Common::Vector::~Vector` | destrutor |
| `83kEPQNWy8A` | `_ZN3sce2Np9CppWebApi6Common6VectorINS2_6StringEED1Ev` | `sce::Np::CppWebApi::Common::Vector::~Vector` | destrutor |
| `FQowxL1xbgc` | `_ZN3sce2Np9CppWebApi6Common6VectorINS2_12IntrusivePtrINS1_14SessionManager2V126RequestPlayerSessionPlayerEEEED1Ev` | `sce::Np::CppWebApi::Common::Vector::~Vector` | destrutor |
| `qtagaBorsKQ` | `_ZN3sce2Np9CppWebApi6Common6VectorINS2_12IntrusivePtrINS1_14SessionManager2V120RequestPlayerSessionEEEED1Ev` | `sce::Np::CppWebApi::Common::Vector::~Vector` | destrutor |
| `vFMRb8-gA7I` | `_ZN3sce2Np9CppWebApi6Common6VectorINS2_12IntrusivePtrINS1_14SessionManager2V124PlayerSessionPushContextEEEED1Ev` | `sce::Np::CppWebApi::Common::Vector::~Vector` | destrutor |
| `UYPxv8MIzGo` | `_ZN3sce2Np9CppWebApi6Common10initializeERKNS2_10InitParamsERNS2_10LibContextE` | `sce::Np::CppWebApi::Common::initialize` | metodo |
| `oTirsxQpqj0` | `_ZN3sce2Np9CppWebApi6Common9terminateERNS2_10LibContextE` | `sce::Np::CppWebApi::Common::terminate` | metodo |
| `0pwb8fOBwpQ` | `_ZNK3sce2Np9CppWebApi14SessionManager2V110FromMember11getOnlineIdEv` | `sce::Np::CppWebApi::SessionManager::V1::FromMember::getOnlineId` | acessor |
| `mqM7+dpduR0` | `_ZN3sce2Np9CppWebApi14SessionManager2V154GetUsersAccountIdPlayerSessionsInvitationsResponseBody14getInvitationsEv` | `sce::Np::CppWebApi::SessionManager::V1::GetUsersAccountIdPlayerSessionsInvitationsResponseBody::getInvitations` | acessor |
| `YZlVMCFs-Dw` | `_ZNK3sce2Np9CppWebApi14SessionManager2V154GetUsersAccountIdPlayerSessionsInvitationsResponseBody16invitationsIsSetEv` | `sce::Np::CppWebApi::SessionManager::V1::GetUsersAccountIdPlayerSessionsInvitationsResponseBody::invitationsIsSet` | acessor |
| `4V0H+brwuCU` | `_ZN3sce2Np9CppWebApi14SessionManager2V122LocalizedStringFactory6createEPNS1_6Common10LibContextEPKcNS_4Json6ObjectEPNS5_12IntrusivePtrINS3_15LocalizedStringEEE` | `sce::Np::CppWebApi::SessionManager::V1::LocalizedStringFactory::create` | metodo |
| `vjw99oC+SaU` | `_ZNK3sce2Np9CppWebApi14SessionManager2V113PlayerSession12getSessionIdEv` | `sce::Np::CppWebApi::SessionManager::V1::PlayerSession::getSessionId` | acessor |
| `gglj0cQ9K1E` | `_ZN3sce2Np9CppWebApi14SessionManager2V131PlayerSessionPushContextFactory6createEPNS1_6Common10LibContextEPKcPNS5_12IntrusivePtrINS3_24PlayerSessionPushContextEEE` | `sce::Np::CppWebApi::SessionManager::V1::PlayerSessionPushContextFactory::create` | metodo |
| `XkdGinZ2HtA` | `_ZN3sce2Np9CppWebApi14SessionManager2V117PlayerSessionsApi31ParameterToCreatePlayerSessionsC1Ev` | `sce::Np::CppWebApi::SessionManager::V1::PlayerSessionsApi::ParameterToCreatePlayerSessions::ParameterToCreatePlayerSessions` | construtor |
| `E8pPjlz943E` | `_ZN3sce2Np9CppWebApi14SessionManager2V117PlayerSessionsApi31ParameterToCreatePlayerSessions10initializeEPNS1_6Common10LibContextENS6_12IntrusivePtrINS3_29PostPlayerSessionsRequestBodyEEE` | `sce::Np::CppWebApi::SessionManager::V1::PlayerSessionsApi::ParameterToCreatePlayerSessions::initialize` | metodo |
| `LUvjGNEPg2Y` | `_ZN3sce2Np9CppWebApi14SessionManager2V117PlayerSessionsApi31ParameterToCreatePlayerSessionsD1Ev` | `sce::Np::CppWebApi::SessionManager::V1::PlayerSessionsApi::ParameterToCreatePlayerSessions::~ParameterToCreatePlayerSessions` | destrutor |
| `Q1WNakfGV08` | `_ZN3sce2Np9CppWebApi14SessionManager2V117PlayerSessionsApi38ParameterToGetPlayerSessionInvitationsC1Ev` | `sce::Np::CppWebApi::SessionManager::V1::PlayerSessionsApi::ParameterToGetPlayerSessionInvitations::ParameterToGetPlayerSessionInvitations` | construtor |
| `k1-l3-S82-Y` | `_ZN3sce2Np9CppWebApi14SessionManager2V117PlayerSessionsApi38ParameterToGetPlayerSessionInvitations10initializeEPNS1_6Common10LibContextEPKc` | `sce::Np::CppWebApi::SessionManager::V1::PlayerSessionsApi::ParameterToGetPlayerSessionInvitations::initialize` | metodo |
| `bTaEyd6uzzE` | `_ZN3sce2Np9CppWebApi14SessionManager2V117PlayerSessionsApi38ParameterToGetPlayerSessionInvitations9setfieldsENS1_6Common12IntrusivePtrINS6_6VectorINS6_6StringEEEEE` | `sce::Np::CppWebApi::SessionManager::V1::PlayerSessionsApi::ParameterToGetPlayerSessionInvitations::setfields` | acessor |
| `1hVd+ogdk94` | `_ZN3sce2Np9CppWebApi14SessionManager2V117PlayerSessionsApi38ParameterToGetPlayerSessionInvitations26setinvitationInvalidFilterEPKc` | `sce::Np::CppWebApi::SessionManager::V1::PlayerSessionsApi::ParameterToGetPlayerSessionInvitations::setinvitationInvalidFilter` | acessor |
| `STSh215IhuM` | `_ZN3sce2Np9CppWebApi14SessionManager2V117PlayerSessionsApi38ParameterToGetPlayerSessionInvitationsD1Ev` | `sce::Np::CppWebApi::SessionManager::V1::PlayerSessionsApi::ParameterToGetPlayerSessionInvitations::~ParameterToGetPlayerSessionInvitations` | destrutor |
| `exkGQSIrF+U` | `_ZN3sce2Np9CppWebApi14SessionManager2V117PlayerSessionsApi36ParameterToJoinPlayerSessionAsPlayerC1Ev` | `sce::Np::CppWebApi::SessionManager::V1::PlayerSessionsApi::ParameterToJoinPlayerSessionAsPlayer::ParameterToJoinPlayerSessionAsPlayer` | construtor |
| `XuVY56j7GM0` | `_ZN3sce2Np9CppWebApi14SessionManager2V117PlayerSessionsApi36ParameterToJoinPlayerSessionAsPlayer10initializeEPNS1_6Common10LibContextEPKcNS6_12IntrusivePtrINS3_51PostPlayerSessionsSessionIdMemberPlayersRequestBodyEEE` | `sce::Np::CppWebApi::SessionManager::V1::PlayerSessionsApi::ParameterToJoinPlayerSessionAsPlayer::initialize` | metodo |
| `E+lI7Bs1zlw` | `_ZN3sce2Np9CppWebApi14SessionManager2V117PlayerSessionsApi36ParameterToJoinPlayerSessionAsPlayer9terminateEv` | `sce::Np::CppWebApi::SessionManager::V1::PlayerSessionsApi::ParameterToJoinPlayerSessionAsPlayer::terminate` | metodo |
| `-QRPOVJU4aU` | `_ZN3sce2Np9CppWebApi14SessionManager2V117PlayerSessionsApi36ParameterToJoinPlayerSessionAsPlayerD1Ev` | `sce::Np::CppWebApi::SessionManager::V1::PlayerSessionsApi::ParameterToJoinPlayerSessionAsPlayer::~ParameterToJoinPlayerSessionAsPlayer` | destrutor |
| `GIBhqoeiiho` | `_ZN3sce2Np9CppWebApi14SessionManager2V117PlayerSessionsApi29ParameterToLeavePlayerSessionC1Ev` | `sce::Np::CppWebApi::SessionManager::V1::PlayerSessionsApi::ParameterToLeavePlayerSession::ParameterToLeavePlayerSession` | construtor |
| `ZZpDSZob2Bs` | `_ZN3sce2Np9CppWebApi14SessionManager2V117PlayerSessionsApi29ParameterToLeavePlayerSession10initializeEPNS1_6Common10LibContextEPKcSA_` | `sce::Np::CppWebApi::SessionManager::V1::PlayerSessionsApi::ParameterToLeavePlayerSession::initialize` | metodo |
| `40BxQHhkVGI` | `_ZN3sce2Np9CppWebApi14SessionManager2V117PlayerSessionsApi29ParameterToLeavePlayerSessionD1Ev` | `sce::Np::CppWebApi::SessionManager::V1::PlayerSessionsApi::ParameterToLeavePlayerSession::~ParameterToLeavePlayerSession` | destrutor |
| `eVU3cXIJz0w` | `_ZN3sce2Np9CppWebApi14SessionManager2V117PlayerSessionsApi20createPlayerSessionsEiRKNS4_31ParameterToCreatePlayerSessionsERNS1_6Common11TransactionINS8_12IntrusivePtrINS3_30PostPlayerSessionsResponseBodyEEENSA_INS8_18ResponseHeaderBaseEEEEE` | `sce::Np::CppWebApi::SessionManager::V1::PlayerSessionsApi::createPlayerSessions` | metodo |
| `LbhH9NeB93I` | `_ZN3sce2Np9CppWebApi14SessionManager2V117PlayerSessionsApi27getPlayerSessionInvitationsEiRKNS4_38ParameterToGetPlayerSessionInvitationsERNS1_6Common11TransactionINS8_12IntrusivePtrINS3_54GetUsersAccountIdPlayerSessionsInvitationsResponseBodyEEENSA_INS8_18ResponseHeaderBaseEEEEE` | `sce::Np::CppWebApi::SessionManager::V1::PlayerSessionsApi::getPlayerSessionInvitations` | acessor |
| `BY1jyWLZrcA` | `_ZN3sce2Np9CppWebApi14SessionManager2V117PlayerSessionsApi25joinPlayerSessionAsPlayerEiRKNS4_36ParameterToJoinPlayerSessionAsPlayerERNS1_6Common11TransactionINS8_12IntrusivePtrINS3_52PostPlayerSessionsSessionIdMemberPlayersResponseBodyEEENSA_INS8_18ResponseHeaderBaseEEEEE` | `sce::Np::CppWebApi::SessionManager::V1::PlayerSessionsApi::joinPlayerSessionAsPlayer` | metodo |
| `JiY4sfHkV34` | `_ZN3sce2Np9CppWebApi14SessionManager2V117PlayerSessionsApi18leavePlayerSessionEiRKNS4_29ParameterToLeavePlayerSessionERNS1_6Common11TransactionINS8_15DefaultResponseENS8_12IntrusivePtrINS8_18ResponseHeaderBaseEEEEE` | `sce::Np::CppWebApi::SessionManager::V1::PlayerSessionsApi::leavePlayerSession` | metodo |
| `yl3T9brJay4` | `_ZN3sce2Np9CppWebApi14SessionManager2V136PostPlayerSessionsRequestBodyFactory6createEPNS1_6Common10LibContextERKNS5_6VectorINS5_12IntrusivePtrINS3_20RequestPlayerSessionEEEEEPNS9_INS3_29PostPlayerSessionsRequestBodyEEE` | `sce::Np::CppWebApi::SessionManager::V1::PostPlayerSessionsRequestBodyFactory::create` | metodo |
| `JFkvxXYRVS0` | `_ZN3sce2Np9CppWebApi14SessionManager2V130PostPlayerSessionsResponseBody17getPlayerSessionsEv` | `sce::Np::CppWebApi::SessionManager::V1::PostPlayerSessionsResponseBody::getPlayerSessions` | acessor |
| `xjVhrq2KPJA` | `_ZN3sce2Np9CppWebApi14SessionManager2V158PostPlayerSessionsSessionIdMemberPlayersRequestBodyFactory6createEPNS1_6Common10LibContextERKNS5_6VectorINS5_12IntrusivePtrINS3_26RequestPlayerSessionPlayerEEEEEPNS9_INS3_51PostPlayerSessionsSessionIdMemberPlayersRequestBodyEEE` | `sce::Np::CppWebApi::SessionManager::V1::PostPlayerSessionsSessionIdMemberPlayersRequestBodyFactory::create` | metodo |
| `sRn6qePT54Q` | `_ZN3sce2Np9CppWebApi14SessionManager2V139RequestCreatePlayerSessionPlayerFactory6createEPNS1_6Common10LibContextEPKcS9_RKNS5_6VectorINS5_12IntrusivePtrINS3_24PlayerSessionPushContextEEEEEPNSB_INS3_32RequestCreatePlayerSessionPlayerEEE` | `sce::Np::CppWebApi::SessionManager::V1::RequestCreatePlayerSessionPlayerFactory::create` | metodo |
| `z+aXXhCwtKU` | `_ZN3sce2Np9CppWebApi14SessionManager2V120RequestPlayerSession28setExclusiveLeaderPrivilegesERKNS1_6Common6VectorINS5_6StringEEE` | `sce::Np::CppWebApi::SessionManager::V1::RequestPlayerSession::setExclusiveLeaderPrivileges` | acessor |
| `63mPNKdUOZY` | `_ZN3sce2Np9CppWebApi14SessionManager2V120RequestPlayerSession20setInvitableUserTypeERKNS3_17InvitableUserTypeE` | `sce::Np::CppWebApi::SessionManager::V1::RequestPlayerSession::setInvitableUserType` | acessor |
| `SQMjAiGtsiI` | `_ZN3sce2Np9CppWebApi14SessionManager2V120RequestPlayerSession19setJoinableUserTypeERKNS3_16JoinableUserTypeE` | `sce::Np::CppWebApi::SessionManager::V1::RequestPlayerSession::setJoinableUserType` | acessor |
| `+oqyRiwxW18` | `_ZN3sce2Np9CppWebApi14SessionManager2V120RequestPlayerSession16setMaxSpectatorsERKi` | `sce::Np::CppWebApi::SessionManager::V1::RequestPlayerSession::setMaxSpectators` | acessor |
| `fIzsqjyD5oY` | `_ZN3sce2Np9CppWebApi14SessionManager2V127RequestPlayerSessionFactory6createEPNS1_6Common10LibContextEiNS5_12IntrusivePtrINS3_32RequestPlayerSessionMemberPlayerEEERKNS5_6VectorINS5_6StringEEENS8_INS3_15LocalizedStringEEEPNS8_INS3_20RequestPlayerSessionEEE` | `sce::Np::CppWebApi::SessionManager::V1::RequestPlayerSessionFactory::create` | metodo |
| `-0Vv7HKZaBM` | `_ZN3sce2Np9CppWebApi14SessionManager2V139RequestPlayerSessionMemberPlayerFactory6createEPNS1_6Common10LibContextERKNS5_6VectorINS5_12IntrusivePtrINS3_32RequestCreatePlayerSessionPlayerEEEEEPNS9_INS3_32RequestPlayerSessionMemberPlayerEEE` | `sce::Np::CppWebApi::SessionManager::V1::RequestPlayerSessionMemberPlayerFactory::create` | metodo |
| `fI-bJo6WlXU` | `_ZN3sce2Np9CppWebApi14SessionManager2V126RequestPlayerSessionPlayer15setPushContextsERKNS1_6Common6VectorINS5_12IntrusivePtrINS3_24PlayerSessionPushContextEEEEE` | `sce::Np::CppWebApi::SessionManager::V1::RequestPlayerSessionPlayer::setPushContexts` | acessor |
| `e0n+XvGVWjY` | `_ZN3sce2Np9CppWebApi14SessionManager2V133RequestPlayerSessionPlayerFactory6createEPNS1_6Common10LibContextEPKcS9_PNS5_12IntrusivePtrINS3_26RequestPlayerSessionPlayerEEE` | `sce::Np::CppWebApi::SessionManager::V1::RequestPlayerSessionPlayerFactory::create` | metodo |
| `fDex2lyDL+I` | `_ZNK3sce2Np9CppWebApi14SessionManager2V136UsersPlayerSessionsInvitationForRead9fromIsSetEv` | `sce::Np::CppWebApi::SessionManager::V1::UsersPlayerSessionsInvitationForRead::fromIsSet` | acessor |
| `IBNzGcdOddw` | `_ZNK3sce2Np9CppWebApi14SessionManager2V136UsersPlayerSessionsInvitationForRead7getFromEv` | `sce::Np::CppWebApi::SessionManager::V1::UsersPlayerSessionsInvitationForRead::getFrom` | acessor |
| `HoXiuWuma4I` | `_ZNK3sce2Np9CppWebApi14SessionManager2V136UsersPlayerSessionsInvitationForRead12getSessionIdEv` | `sce::Np::CppWebApi::SessionManager::V1::UsersPlayerSessionsInvitationForRead::getSessionId` | acessor |
| `WuQAWT8kYek` | `_ZNK3sce2Np9CppWebApi14SessionManager2V136UsersPlayerSessionsInvitationForRead17invitationIdIsSetEv` | `sce::Np::CppWebApi::SessionManager::V1::UsersPlayerSessionsInvitationForRead::invitationIdIsSet` | acessor |
| `q9Ac0wYBOgQ` | `_ZNK3sce2Np9CppWebApi14SessionManager2V136UsersPlayerSessionsInvitationForRead22receivedTimestampIsSetEv` | `sce::Np::CppWebApi::SessionManager::V1::UsersPlayerSessionsInvitationForRead::receivedTimestampIsSet` | acessor |
| `VkqilTtSq7U` | `_ZNK3sce2Np9CppWebApi14SessionManager2V136UsersPlayerSessionsInvitationForRead14sessionIdIsSetEv` | `sce::Np::CppWebApi::SessionManager::V1::UsersPlayerSessionsInvitationForRead::sessionIdIsSet` | acessor |
| `E8BmtbByVKs` | `_ZN3sce2Np9CppWebApi17TitleCloudStorage2V17DataApi23ParameterToDownloadDataC1Ev` | `sce::Np::CppWebApi::TitleCloudStorage::V1::DataApi::ParameterToDownloadData::ParameterToDownloadData` | construtor |
| `hH7KHku1lqs` | `_ZN3sce2Np9CppWebApi17TitleCloudStorage2V17DataApi23ParameterToDownloadData10initializeEPNS1_6Common10LibContextEPKci` | `sce::Np::CppWebApi::TitleCloudStorage::V1::DataApi::ParameterToDownloadData::initialize` | metodo |
| `26SufKG5x4M` | `_ZN3sce2Np9CppWebApi17TitleCloudStorage2V17DataApi23ParameterToDownloadData9terminateEv` | `sce::Np::CppWebApi::TitleCloudStorage::V1::DataApi::ParameterToDownloadData::terminate` | metodo |
| `0UNp2Ey5bw4` | `_ZN3sce2Np9CppWebApi17TitleCloudStorage2V17DataApi23ParameterToDownloadDataD1Ev` | `sce::Np::CppWebApi::TitleCloudStorage::V1::DataApi::ParameterToDownloadData::~ParameterToDownloadData` | destrutor |
| `9YWXkd3AZAk` | `_ZN3sce2Np9CppWebApi17TitleCloudStorage2V17DataApi12downloadDataEiRKNS4_23ParameterToDownloadDataERNS1_6Common21DownStreamTransactionINS8_12IntrusivePtrINS4_27DownloadDataResponseHeadersEEEEE` | `sce::Np::CppWebApi::TitleCloudStorage::V1::DataApi::downloadData` | metodo |

### `NpManager_v1` — 1 imports  ·  prioridade **BAIXA**

_PSN manager_

| NID | Nome | Leitura | Tipo |
|---|---|---|---|
| `3Tcz5bNCfZQ` | `sceNpGetAccountLanguage2` | `sceNpGetAccountLanguage2` | C API |

### `NpSessionSignaling_v1` — 3 imports  ·  prioridade **BAIXA**

_PSN sinalizacao de sessao_

| NID | Nome | Leitura | Tipo |
|---|---|---|---|
| `cQkBH-pXhF0` | `sceNpSessionSignalingDeactivate` | `sceNpSessionSignalingDeactivate` | C API |
| `Z9Q9LzQDXf0` | `sceNpSessionSignalingDestroyContext` | `sceNpSessionSignalingDestroyContext` | C API |
| `CqJuNXo5yiM` | `sceNpSessionSignalingTerminate` | `sceNpSessionSignalingTerminate` | C API |

### `NpTrophy2_v1` — 1 imports  ·  prioridade **BAIXA**

_trofeus_

| NID | Nome | Leitura | Tipo |
|---|---|---|---|
| `wVqxM58sIKs` | `sceNpTrophy2UnregisterUnlockCallback` | `sceNpTrophy2UnregisterUnlockCallback` | C API |

### `NpWebApi2_v1` — 4 imports  ·  prioridade **BAIXA**

_PSN web API v2_

| NID | Nome | Leitura | Tipo |
|---|---|---|---|
| `NNVf18SlbT8` | `sceNpWebApi2PushEventCreatePushContext` | `sceNpWebApi2PushEventCreatePushContext` | C API |
| `KJdPcOGmK58` | `sceNpWebApi2PushEventDeleteFilter` | `sceNpWebApi2PushEventDeleteFilter` | C API |
| `lxtHJMwBsaU` | `sceNpWebApi2PushEventRegisterPushContextCallback` | `sceNpWebApi2PushEventRegisterPushContextCallback` | C API |
| `AAj9X+4aGYA` | `sceNpWebApi2PushEventStartPushContextCallback` | `sceNpWebApi2PushEventStartPushContextCallback` | C API |

### `PlayGoDialog_v1` — 6 imports  ·  prioridade **BAIXA**

_dialogo de instalacao_

| NID | Nome | Leitura | Tipo |
|---|---|---|---|
| `fbigNQiZpm0` | `scePlayGoDialogClose` | `scePlayGoDialogClose` | C API |
| `wx9TDplJKB4` | `scePlayGoDialogGetResult` | `scePlayGoDialogGetResult` | C API |
| `fECamTJKpsM` | `scePlayGoDialogInitialize` | `scePlayGoDialogInitialize` | C API |
| `kHd72ukqbxw` | `scePlayGoDialogOpen` | `scePlayGoDialogOpen` | C API |
| `okgIGdr5Iz0` | `scePlayGoDialogTerminate` | `scePlayGoDialogTerminate` | C API |
| `Yb60K7BST48` | `scePlayGoDialogUpdateStatus` | `scePlayGoDialogUpdateStatus` | C API |

### `PlayerInvitationDialog_v1` — 3 imports  ·  prioridade **BAIXA**

_dialogo de convite (NP)_

| NID | Nome | Leitura | Tipo |
|---|---|---|---|
| `AhqlQ8cngrk` | `scePlayerInvitationDialogGetResult` | `scePlayerInvitationDialogGetResult` | C API |
| `gDm5a6GSE94` | `scePlayerInvitationDialogTerminate` | `scePlayerInvitationDialogTerminate` | C API |
| `kFhuwHrIUqs` | `scePlayerInvitationDialogUpdateStatus` | `scePlayerInvitationDialogUpdateStatus` | C API |

## Não encontrados na base

- `zBsgF0q8DIM` (Pad_v1) — sem entrada em `nids.csv`
